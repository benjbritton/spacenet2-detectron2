#!/usr/bin/env python
"""Run the Chactun-trained model on a G-LiHT transect it has never seen.

THE TEST
--------
Someone hands over their LiDAR and asks whether this model helps them find
structures. Nothing about the input matches training: different survey,
different sensor, different processing, 0.33 m instead of 0.5 m, and a blended
3-channel composite where the model was trained on three distinct
visualisations. If it is useful anyway, that is the property the tool needs.

TILING
------
By GROUND EXTENT, not pixel count. Measured on Chactun, naive fixed-pixel tiling
costs about two thirds of the resolution penalty because objects change size in
the network input; tiling in metres holds object size constant and removes it.
240 m tiles at 0.33 m is 727 px, which reproduces the object scale the model was
trained at.

Tiles overlap by 25%, because a structure cut by a tile boundary is exactly the
detection a candidate generator must not miss. Duplicates from the overlap are
removed by NMS in map coordinates afterwards.

NORMALISATION -- THE THING BEING MEASURED
-----------------------------------------
Two modes, run separately, because the difference between them IS the answer to
how much input-matching discipline a real user needs.

  fixed     Feed raw values. The model applies the Chactun constants it was
            trained with (mean 216.5/198.5/228.6). G1 sits near 124, so the
            input arrives 3 to 5 standard deviations off centre.
  matched   Rescale each tile so its valid pixels carry Chactun's mean and
            standard deviation, then let the model normalise as usual.

Nodata is excluded from the statistics and left at the background level, since
these transects are narrow strips and roughly half of a square tile is black.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

sys.path.insert(0, "/workspace/src")

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

CHACTUN_MEAN = np.array([216.527, 198.453, 228.612])
CHACTUN_SD = np.array([26.915, 16.698, 21.212])
CLASSES = ["building", "platform", "aguada"]
BASE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


def build(weights, cfg_name, min_size):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE))
    cfg.merge_from_file(os.path.join("/workspace/configs", cfg_name))
    cfg.MODEL.WEIGHTS = weights
    cfg.INPUT.MIN_SIZE_TEST = min_size
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05      # candidate generator
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    return cfg, model


def prep(tile, mode):
    """(H,W,3) uint8 -> float tensor the model can consume."""
    a = tile.astype(np.float32)
    if mode == "matched":
        valid = (tile > 0).any(axis=2)
        if valid.sum() > 100:
            for b in range(3):
                v = a[:, :, b][valid]
                sd = v.std()
                if sd > 1e-3:
                    a[:, :, b] = ((a[:, :, b] - v.mean()) / sd
                                  * CHACTUN_SD[b] + CHACTUN_MEAN[b])
    return np.clip(a, 0, 255)


def nms_map(dets, iou_thr=0.5):
    """Greedy NMS on map-coordinate boxes, to drop overlap duplicates."""
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: -d["score"])
    keep = []
    for d in dets:
        x0, y0, x1, y1 = d["bbox_map"]
        dup = False
        for k in keep:
            a0, b0, a1, b1 = k["bbox_map"]
            ix = max(0.0, min(x1, a1) - max(x0, a0))
            iy = max(0.0, min(y1, b1) - max(y0, b0))
            inter = ix * iy
            if inter <= 0:
                continue
            ua = (x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter
            if ua > 0 and inter / ua > iou_thr:
                dup = True
                break
        if not dup:
            keep.append(d)
    return keep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raster", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--config", default="chactun_D_maskrcnn_d4_augmentation.yaml")
    p.add_argument("--mode", choices=["fixed", "matched"], default="matched")
    p.add_argument("--tile-m", type=float, default=240.0)
    p.add_argument("--overlap", type=float, default=0.25)
    p.add_argument("--min-valid", type=float, default=0.15,
                   help="skip tiles with less valid data than this fraction")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    src = rasterio.open(a.raster)
    res = abs(src.transform.a)
    tile_px = int(round(a.tile_m / res))
    step = int(round(tile_px * (1 - a.overlap)))
    print("raster    : %s" % os.path.basename(a.raster))
    print("size      : %d x %d at %.3f m, crs %s" % (src.width, src.height, res, src.crs))
    print("tiling    : %.0f m = %d px, step %d px, overlap %.0f%%"
          % (a.tile_m, tile_px, step, 100 * a.overlap))
    print("mode      : %s" % a.mode)

    # 800 regardless of source resolution: every tile covers the same GROUND
    # extent, so resizing them all to 800 px reproduces the object scale the
    # model trained at. That is what ground-extent tiling buys.
    cfg, model = build(a.weights, a.config, 800)
    print("score thr : %.2f" % cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)

    dets = []
    n_tiles = n_used = 0
    for row in range(0, src.height, step):
        for col in range(0, src.width, step):
            w = min(tile_px, src.width - col)
            h = min(tile_px, src.height - row)
            if w < tile_px * 0.5 or h < tile_px * 0.5:
                continue
            n_tiles += 1
            arr = src.read(window=Window(col, row, w, h))
            tile = np.transpose(arr, (1, 2, 0))
            valid = (tile > 0).any(axis=2)
            if valid.mean() < a.min_valid:
                continue
            n_used += 1

            img = prep(tile, a.mode)
            # the mapper normally does this; bypassing it means doing it here
            scale = 800.0 / min(h, w)
            rimg = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                              interpolation=cv2.INTER_LINEAR)
            t = torch.as_tensor(np.ascontiguousarray(
                rimg.transpose(2, 0, 1))).to(cfg.MODEL.DEVICE)
            with torch.no_grad():
                # height/width are the ORIGINAL tile size, so detections come
                # back in tile pixel coordinates ready for the map transform
                out = model([{"image": t, "height": h, "width": w}])[0]
            inst = out["instances"].to("cpu")
            for i in range(len(inst)):
                b = inst.pred_boxes.tensor[i].numpy()
                # pixel -> map coordinates
                x0, y0 = src.transform * (col + b[0], row + b[1])
                x1, y1 = src.transform * (col + b[2], row + b[3])
                dets.append({
                    "score": float(inst.scores[i]),
                    "cls": CLASSES[int(inst.pred_classes[i])],
                    "bbox_map": [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                })
        if n_tiles and row % (step * 8) == 0:
            print("  row %d/%d, %d tiles used, %d raw detections"
                  % (row, src.height, n_used, len(dets)))

    print("tiles     : %d scanned, %d with data" % (n_tiles, n_used))
    print("detections: %d raw" % len(dets))
    kept = nms_map(dets)
    print("            %d after cross-tile NMS" % len(kept))

    counts = {}
    for d in kept:
        counts[d["cls"]] = counts.get(d["cls"], 0) + 1
    print("by class  : %s" % counts)
    area_km2 = n_used * (a.tile_m / 1000.0) ** 2 * (1 - a.overlap) ** 2
    print("area      : ~%.2f km2, %.0f detections/km2"
          % (area_km2, len(kept) / max(area_km2, 1e-6)))

    # Stable identifier per detection, plus the geometry summaries a join
    # needs. Sorted by descending score first so the numbering is reproducible
    # across runs rather than dependent on dictionary or NMS ordering.
    stem = os.path.splitext(os.path.basename(a.out))[0]
    kept = sorted(kept, key=lambda d: (-float(d["score"]),
                                       d["bbox_map"][0], d["bbox_map"][1]))
    feats = []
    for i, d in enumerate(kept, 1):
        x0, y0, x1, y1 = d["bbox_map"]
        det_id = "%s_%06d" % (stem, i)
        feats.append({
            "type": "Feature",
            "id": det_id,
            "properties": {"det_id": det_id,
                           "score": round(d["score"], 4), "class": d["cls"],
                           "mode": a.mode,
                           "centroid_x": round((x0 + x1) / 2.0, 3),
                           "centroid_y": round((y0 + y1) / 2.0, 3),
                           "bbox_w_m": round(abs(x1 - x0), 2),
                           "bbox_h_m": round(abs(y1 - y0), 2),
                           "bbox_area_m2": round(abs((x1 - x0) * (y1 - y0)), 2)},
            "geometry": {"type": "Polygon", "coordinates": [[
                [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
        })
    gj = {"type": "FeatureCollection",
          "crs": {"type": "name",
                  "properties": {"name": str(src.crs)}},
          "features": feats}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(gj, f)
    print("wrote %s" % a.out)
    src.close()


if __name__ == "__main__":
    main()
