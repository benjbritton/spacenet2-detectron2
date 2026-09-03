#!/usr/bin/env python
"""Convert Chactun per-class binary masks into COCO instance annotations.

THE DATASET
-----------
Chactun (Kokalj et al., Scientific Data 10:558, 2023, CC BY 4.0, figshare
10.6084/m9.figshare.22202395) ships 2094 tiles of airborne laser scanning
visualisations over central Yucatan, with manual annotations of ancient Maya
structures as SEMANTIC masks -- one binary raster per class per tile.

  lidar/tile_N_lidar.tif                 480x480, 3-band uint8, 0.5 m
                                         sky-view factor, positive openness, slope
  masks/tile_N_mask_building.tif         480x480, 1-band uint8
  masks/tile_N_mask_platform.tif
  masks/tile_N_mask_aguada.tif

THREE THINGS THAT WILL SILENTLY RUIN THIS CONVERSION
----------------------------------------------------
1. **The masks are INVERTED.** Object pixels are 0; background is 255. Reading
   them the obvious way, `mask > 0`, yields one tile-sized "instance" per class
   per tile and a model trained on nothing at all. Verified on 2094 tiles:
   buildings occupy about 2.5% of a tile where present, aguadas about 11%.

2. **Semantic masks are not instance masks, and this is not fixable here.**
   Adjacent structures that touch fuse into one connected component. Converted,
   plain connected components yield 7442 buildings against the 8275 the dataset
   paper reports as present in these 2094 records (Kokalj et al., Scientific
   Data 10:558, 2023, Table 6) -- a 10% undercount, entirely from merging.
   The widely quoted 9303 counts the whole 130 km2 annotated section rather
   than these tiles, and comparing against it overstates the loss.

   --split-touching applies a distance-transform watershed. IT DOES NOT WORK, and
   the sweep is recorded here so the next reader does not repeat it. Same 500
   tiles, buildings and their median footprint:

       connected components   1679   157 m2
       watershed d=8          2345   114 m2    x1.40, footprint collapses
       watershed d=12         1242   193 m2    x0.74
       watershed d=16         1279   157 m2    x0.76
       watershed d=22         1586   156 m2    x0.94
       watershed d=30         1678   157 m2    x1.00, converges to CC

   Recovering the merges needs about x1.25 WITH the median footprint intact. No
   setting does that: small radii over-split single structures and halve their
   size, larger radii lose components to peak suppression and converge back to
   connected components. The flag remains for anyone with a better idea; the
   default is connected components.

   **Why the undercount is tolerable.** It applies identically to train and val,
   so it is a property of the dataset rather than a train/test mismatch. What it
   costs is absolute instance counts and any claim about dense-cluster recall --
   both of which should be stated when reporting on this data, not discovered by
   a reader.

3. **Structures are cut by tile boundaries.** 3429 components of 9853 touch a tile
   edge and are therefore partial objects, and a structure spanning two tiles
   appears as two instances -- which is why platforms OVERcount, 2335 against the
   1996 present in these records, and aguadas worse still at 76 against 51,
   in the same conversion where buildings undercount. Aguadas suffer most
   because they are the largest class and cross tile edges most often. The two errors have
   opposite sign and do not cancel; they act on different classes because
   platforms are large enough to cross tiles while buildings are small enough to
   fuse with their neighbours.

   Edge instances are KEPT. A partially visible structure is a real detection
   target, and dropping them would teach the model that half-visible buildings
   are background. Every annotation carries an `edge_touching` flag so a
   downstream evaluation can exclude them without redoing the conversion --
   which is the right way to test whether they help or hurt, rather than
   deciding it here.

POLYGONS, NOT RLE
-----------------
Segmentations are emitted as polygons to match the existing pipeline, whose
config sets INPUT.MASK_FORMAT = "polygon". The cost is that interior holes are
lost, exactly as in geojson_to_coco.py. Pass --rle for exact masks instead, which
requires INPUT.MASK_FORMAT = "bitmask" in the training config.

AREA COMES FROM THE MASK
------------------------
COCOeval buckets small/medium/large on the `area` field. It is computed from the
component's true pixel count, never from the bounding box, which would inflate
every non-rectangular structure and corrupt the size buckets.
"""
import argparse
import json
import os
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import rasterio
from scipy import ndimage

CLASSES = ["building", "platform", "aguada"]          # category_id = index + 1
# Counts of objects PRESENT IN THESE 2094 RECORDS, from Kokalj et al.,
# Scientific Data 10:558 (2023), Table 6. Not the area totals -- 9303
# buildings and 2110 platforms are for the whole 130 km2 annotated section,
# and the 95 aguadas span the full 220 km2 survey. Comparing a conversion of
# these tiles against the area totals overstates the building loss and hides
# the aguada over-count entirely.
PAPER_COUNTS = {"building": 8275, "platform": 1996, "aguada": 51}
PX_AREA_M2 = 0.25                                     # 0.5 m pixels


def load_mask(path):
    """Foreground boolean from an inverted 8-bit mask. Object = 0."""
    with rasterio.open(path) as src:
        v = src.read(1)
    return v == 0


def split_touching(fg, min_distance=6):
    """Watershed on the distance transform, to separate fused structures.

    Peaks in the distance transform are structure centres; the watershed grows
    basins from them. min_distance suppresses spurious peaks inside one object,
    which would over-split a single large platform into several.
    """
    dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5)
    peaks = (dist >= min_distance) & (
        dist >= ndimage.maximum_filter(dist, size=2 * min_distance + 1) - 1e-6)
    markers, n = ndimage.label(peaks, structure=np.ones((3, 3), bool))
    if n <= 1:
        return ndimage.label(fg, structure=np.ones((3, 3), bool))
    ws = cv2.watershed(
        np.dstack([(dist / max(dist.max(), 1e-6) * 255).astype(np.uint8)] * 3),
        markers.astype(np.int32).copy())
    ws[~fg] = 0
    ws[ws < 0] = 0
    return ws, int(ws.max())


def polygons_from_component(component, stats):
    """External contours of one component, as flat COCO polygon lists."""
    cnts, _ = cv2.findContours(component.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        c = c.reshape(-1, 2)
        if c.shape[0] < 3:
            stats["dropped_contour_under_3_points"] += 1
            continue
        if cv2.contourArea(c.astype(np.float32)) < 4.0:
            stats["dropped_contour_under_4px"] += 1
            continue
        out.append([float(v) for xy in c for v in xy])
    return out


def convert(root, out_path, use_watershed, use_rle, min_area,
            min_distance=6, limit=None):
    lidar_dir = os.path.join(root, "lidar")
    mask_dir = os.path.join(root, "masks")
    tiles = sorted(int(n.split("_")[1]) for n in os.listdir(lidar_dir)
                   if n.endswith("_lidar.tif"))
    if limit:
        tiles = tiles[:limit]

    images, annotations = [], []
    ann_id = 1
    stats = Counter()
    per_class = Counter()
    edge_touching = Counter()
    areas = {c: [] for c in CLASSES}

    if use_rle:
        import pycocotools.mask as mask_util

    for image_id, t in enumerate(tiles, start=1):
        lidar_path = os.path.join(lidar_dir, "tile_%d_lidar.tif" % t)
        with rasterio.open(lidar_path) as src:
            h, w = src.height, src.width
        images.append({
            "id": image_id,
            "file_name": os.path.join("lidar", "tile_%d_lidar.tif" % t),
            "width": w, "height": h,
            "tile": t,
        })

        for ci, cls in enumerate(CLASSES, start=1):
            mpath = os.path.join(mask_dir, "tile_%d_mask_%s.tif" % (t, cls))
            if not os.path.isfile(mpath):
                stats["missing_mask_file"] += 1
                continue
            fg = load_mask(mpath)
            if not fg.any():
                continue

            if use_watershed:
                lab, n = split_touching(fg, min_distance)
            else:
                lab, n = ndimage.label(fg, structure=np.ones((3, 3), bool))

            for k in range(1, n + 1):
                comp = lab == k
                area_px = int(comp.sum())
                if area_px < min_area:
                    stats["dropped_below_min_area"] += 1
                    continue
                ys, xs = np.nonzero(comp)
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())

                if x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1:
                    edge_touching[cls] += 1

                if use_rle:
                    r = mask_util.encode(np.asfortranarray(comp.astype(np.uint8)))
                    r["counts"] = r["counts"].decode("ascii")
                    seg = r
                else:
                    seg = polygons_from_component(comp, stats)
                    if not seg:
                        stats["dropped_no_usable_contour"] += 1
                        continue

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": ci,
                    "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                    # true pixel count, never the bounding box
                    "area": area_px,
                    "segmentation": seg,
                    "iscrowd": 0,
                    "edge_touching": bool(x0 == 0 or y0 == 0
                                          or x1 == w - 1 or y1 == h - 1),
                })
                ann_id += 1
                per_class[cls] += 1
                areas[cls].append(area_px)

    coco = {
        "info": {
            "description": "Chactun ancient Maya structures, instances derived "
                           "from per-class semantic masks",
            "source": "Kokalj et al. 2023, Scientific Data 10:558, "
                      "doi 10.6084/m9.figshare.22202395",
            "licence": "CC BY 4.0",
            "date_created": datetime.now().isoformat(timespec="seconds"),
            "mask_polarity": "object=0, background=255 (inverted)",
            "instances_from": "watershed on distance transform" if use_watershed
                              else "connected components, 8-connectivity",
            "segmentation_format": "rle" if use_rle else "polygon",
            "min_area_px": min_area,
            "conversion_stats": dict(stats),
            "edge_touching": dict(edge_touching),
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES, 1)],
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f)

    print("tiles                 : %d" % len(images))
    print("instances             : %d" % len(annotations))
    print("mode                  : %s, %s"
          % ("watershed" if use_watershed else "connected components",
             "RLE" if use_rle else "polygon"))
    print()
    print("%-10s %10s %10s %8s %12s %12s %10s"
          % ("class", "found", "paper", "ratio", "median m2", "p90 m2", "edge-cut"))
    for c in CLASSES:
        a = np.array(areas[c]) if areas[c] else np.array([0])
        print("%-10s %10d %10d %8.2f %12.0f %12.0f %10d"
              % (c, per_class[c], PAPER_COUNTS[c],
                 per_class[c] / PAPER_COUNTS[c],
                 np.median(a) * PX_AREA_M2,
                 np.percentile(a, 90) * PX_AREA_M2, edge_touching[c]))
    if stats:
        print()
        for k in sorted(stats):
            print("  %-34s %d" % (k, stats[k]))
    print()
    print("wrote %s" % out_path)
    return coco


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/chactun",
                   help="directory containing lidar/ and masks/")
    p.add_argument("--out", default="data/chactun/coco/chactun_all.json")
    p.add_argument("--split-touching", action="store_true",
                   help="watershed to separate fused structures. Changes "
                        "instance counts, so it is opt-in")
    p.add_argument("--rle", action="store_true",
                   help="exact masks instead of polygons; needs "
                        "INPUT.MASK_FORMAT=bitmask when training")
    p.add_argument("--min-distance", type=int, default=6,
                   help="watershed peak suppression radius; larger values "
                        "split less aggressively")
    p.add_argument("--limit", type=int, default=None,
                   help="process only the first N tiles, for sweeps")
    p.add_argument("--min-area", type=int, default=9,
                   help="discard components below this pixel count")
    a = p.parse_args()
    convert(a.root, a.out, a.split_touching, a.rle, a.min_area,
            a.min_distance, a.limit)


if __name__ == "__main__":
    main()
