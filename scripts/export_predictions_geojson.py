#!/usr/bin/env python
"""Export model predictions (and ground truth) as georeferenced GeoJSON.

WHY VECTORS RATHER THAN A BURNED-IN RASTER
------------------------------------------
A picture with outlines drawn on it fixes the score threshold at render time.
Vectors carry the score as an attribute, so in ArcGIS/QGIS the threshold becomes
a definition query -- slide it and watch precision trade against recall on real
geography, with no re-rendering and no second inference pass.

Predictions live in COCOEvaluator's instances_predictions.pth as RLE masks in
PIXEL coordinates. Each tile's GeoTIFF carries the affine transform mapping
pixel -> lon/lat, so the conversion is per-tile and exact. Output is EPSG:4326,
which is both the source CRS and what the GeoJSON spec requires.

    ./scripts/run.sh python scripts/export_predictions_geojson.py \
        --predictions outputs/spacenet2_r50fpn/inference/instances_predictions.pth \
        --gt --out-dir outputs/vector_review

Writes <aoi>_pred.geojson and (with --gt) <aoi>_gt.geojson, one pair per city.
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import rasterio
import torch
from pycocotools import mask as mask_util


def rings_from_mask(mask, epsilon):
    """Contours of a binary mask -> [(exterior, [holes...]), ...] in pixel xy.

    RETR_CCOMP gives a two-level hierarchy: outer boundaries, then their holes.
    Courtyards are rare in SN2, but a hole silently promoted to its own building
    would be a phantom instance -- the bug class the converter already had to
    fix once.
    """
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    holes = defaultdict(list)
    outers = []
    for i, c in enumerate(cnts):
        if epsilon > 0:
            c = cv2.approxPolyDP(c, epsilon, True)
        if len(c) < 3:
            continue
        pts = c.reshape(-1, 2).astype(np.float64)
        if hier[i][3] < 0:
            outers.append((i, pts))
        else:
            holes[hier[i][3]].append(pts)
    return [(pts, holes.get(i, [])) for i, pts in outers]


def polys_to_geometry(parts, transform):
    """[(exterior, holes)] in pixel xy -> a GeoJSON MultiPolygon in lon/lat."""
    def ring(pts):
        r = [list(transform * (float(x), float(y))) for x, y in pts]
        if r[0] != r[-1]:
            r.append(r[0])
        return r
    coords = []
    for ext, hs in parts:
        coords.append([ring(ext)] + [ring(h) for h in hs])
    if not coords:
        return None
    return {"type": "MultiPolygon", "coordinates": coords}


def gt_parts(segmentation):
    """COCO polygon lists (already pixel xy) -> [(exterior, [])] parts.

    The converter groups a split footprint into one annotation carrying several
    polygons, so every polygon here is an exterior of the same building.
    """
    out = []
    for seg in segmentation:
        pts = np.asarray(seg, dtype=np.float64).reshape(-1, 2)
        if len(pts) >= 3:
            out.append((pts, []))
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions",
                   default="outputs/spacenet2_r50fpn/inference/instances_predictions.pth")
    p.add_argument("--coco", default="data/spacenet2/coco/pooled_val.json")
    p.add_argument("--root", default="data/spacenet2")
    p.add_argument("--out-dir", default="outputs/vector_review")
    p.add_argument("--min-score", type=float, default=0.05,
                   help="drop predictions below this before writing. Keep it "
                        "well under the reporting threshold (0.544) so the "
                        "threshold stays adjustable in the GIS")
    p.add_argument("--simplify", type=float, default=0.5,
                   help="Douglas-Peucker tolerance in PIXELS. Masks are "
                        "staircased; 0.5 removes most of that at sub-pixel "
                        "cost. 0 disables")
    p.add_argument("--gt", action="store_true", help="also write ground truth")
    p.add_argument("--aoi", default=None, help="one AOI only")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.coco) as f:
        coco = json.load(f)
    info = {im["id"]: im for im in coco["images"]}
    gt_by_img = defaultdict(list)
    for a in coco["annotations"]:
        gt_by_img[a["image_id"]].append(a)

    # Tile transforms are read once per tile, from headers only.
    tcache = {}

    def transform_of(image_id):
        if image_id not in tcache:
            path = os.path.join(args.root, info[image_id]["file_name"])
            with rasterio.open(path) as src:
                tcache[image_id] = src.transform
        return tcache[image_id]

    preds = torch.load(args.predictions, weights_only=False)
    print("loaded %d prediction records" % len(preds))

    feats = defaultdict(list)
    n_written = n_dropped = 0
    for entry in preds:
        image_id = entry["image_id"]
        im = info.get(image_id)
        if im is None:
            continue
        aoi = im["aoi"]
        if args.aoi and aoi != args.aoi:
            continue
        transform = transform_of(image_id)
        tile = os.path.basename(im["file_name"])
        for inst in entry.get("instances", []):
            score = float(inst["score"])
            if score < args.min_score:
                n_dropped += 1
                continue
            segm = inst.get("segmentation")
            if segm is None:
                continue
            # COCOEvaluator stores RLE counts as str after json round-trips.
            if isinstance(segm.get("counts"), str):
                segm = dict(segm, counts=segm["counts"].encode("utf-8"))
            mask = mask_util.decode(segm)
            geom = polys_to_geometry(rings_from_mask(mask, args.simplify), transform)
            if geom is None:
                continue
            x, y, w, h = inst["bbox"]
            det_id = "%s_%06d" % (os.path.splitext(tile)[0], n_written + 1)
            feats[aoi].append({
                "type": "Feature",
                "id": det_id,
                "geometry": geom,
                "properties": {
                    "det_id": det_id,
                    "source": "pred", "score": round(score, 4), "tile": tile,
                    "area_px": int(mask.sum()),
                    "bbox_w_px": round(float(w), 1),
                    "bbox_h_px": round(float(h), 1),
                },
            })
            n_written += 1

    gt_feats = defaultdict(list)
    if args.gt:
        for image_id, anns in gt_by_img.items():
            im = info[image_id]
            aoi = im["aoi"]
            if args.aoi and aoi != args.aoi:
                continue
            transform = transform_of(image_id)
            tile = os.path.basename(im["file_name"])
            for a in anns:
                geom = polys_to_geometry(gt_parts(a["segmentation"]), transform)
                if geom is None:
                    continue
                gt_feats[aoi].append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {"source": "gt", "tile": tile,
                                   "area_px": round(float(a["area"]), 1)},
                })

    def dump(mapping, suffix):
        for aoi, fs in sorted(mapping.items()):
            out = os.path.join(args.out_dir, "%s_%s.geojson" % (aoi, suffix))
            with open(out, "w") as f:
                json.dump({"type": "FeatureCollection",
                           "name": "%s_%s" % (aoi, suffix),
                           "crs": {"type": "name",
                                   "properties": {
                                       "name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                           "features": fs}, f)
            print("%-46s %7d features  %6.1f MB"
                  % (out, len(fs), os.path.getsize(out) / 1e6))

    dump(feats, "pred")
    if args.gt:
        dump(gt_feats, "gt")
    print("predictions written %d, dropped below --min-score %d"
          % (n_written, n_dropped))


if __name__ == "__main__":
    main()
