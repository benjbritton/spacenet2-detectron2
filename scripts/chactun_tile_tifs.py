#!/usr/bin/env python
"""Write per-tile Chactun overlays as native-resolution 3-band TIFFs.

The multi-panel figure is downsampled for display. These are 480x480, the size
the model actually saw, with ground truth and detections burned in, so the boxes
can be inspected against the imagery at full resolution.

NOT georeferenced, and cannot be. Chactun tiles carry no CRS and no affine
transform -- rasterio returns the identity matrix -- so there is no real-world
position to write. The files carry that identity transform through, which means
they open anywhere as plain rasters but will not land on a map.
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import rasterio
from pycocotools import mask as maskutil

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)   # repo root, whatever it is called

# BGR order for cv2
COLOURS = {"building": (255, 229, 0), "platform": (10, 214, 255),
           "aguada": (85, 45, 255)}
GT_COLOUR = (255, 255, 255)
ROOT = ROOT + "/data/chactun"


def iou_box(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def dashed(img, p0, p1, colour, dash=6):
    """Draw a dashed rectangle, so ground truth reads apart from detections."""
    x0, y0 = p0
    x1, y1 = p1
    for x in range(x0, x1, dash * 2):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), colour, 1)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), colour, 1)
    for y in range(y0, y1, dash * 2):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), colour, 1)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), colour, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default=ROOT + "/data/chactun/coco/fold0_val.json")
    p.add_argument("--pred", default=ROOT + "/outputs/"
                                     "chactun_D_maskrcnn_d4_augmentation/"
                                     "fold0_seed0/inference/"
                                     "coco_instances_results.json")
    p.add_argument("--score", type=float, default=0.30)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--out-dir", default=ROOT + "/outputs/chactun_tiles")
    a = p.parse_args()

    gt = json.load(open(a.gt))
    names = {c["id"]: c["name"] for c in gt["categories"]}
    tile_of = {im["id"]: im["tile"] for im in gt["images"]}
    size_of = {im["id"]: (im["height"], im["width"]) for im in gt["images"]}

    gt_by = defaultdict(list)
    for x in gt["annotations"]:
        h, w = size_of[x["image_id"]]
        bb = maskutil.toBbox(maskutil.merge(
            maskutil.frPyObjects(x["segmentation"], h, w)))
        gt_by[x["image_id"]].append(
            {"box": [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]],
             "cls": names[x["category_id"]]})

    pr_by = defaultdict(list)
    for x in json.load(open(a.pred)):
        if x["score"] < a.score:
            continue
        s = x["segmentation"]
        if isinstance(s, dict):
            bb = maskutil.toBbox(s)
        else:
            h, w = size_of[x["image_id"]]
            bb = maskutil.toBbox(maskutil.merge(maskutil.frPyObjects(s, h, w)))
        pr_by[x["image_id"]].append(
            {"box": [bb[0], bb[1], bb[0] + bb[2], bb[1] + bb[3]],
             "cls": names[x["category_id"]], "score": x["score"]})

    scored = []
    for img_id, gts in gt_by.items():
        if len(gts) < 4:
            continue
        prs = pr_by.get(img_id, [])
        used, tp = set(), 0
        for pr in sorted(prs, key=lambda d: -d["score"]):
            best, bj = 0.0, -1
            for j, g in enumerate(gts):
                if j in used or g["cls"] != pr["cls"]:
                    continue
                v = iou_box(pr["box"], g["box"])
                if v > best:
                    best, bj = v, j
            if bj >= 0 and best >= 0.5:
                used.add(bj)
                tp += 1
        prec = tp / max(len(prs), 1)
        rec = tp / max(len(gts), 1)
        f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        scored.append((f1, img_id, len(gts), len(prs), tp))
    scored.sort(reverse=True)

    # spread across the quality range rather than showing only the good ones
    idx = np.linspace(0, len(scored) - 1, a.n).astype(int)
    picks = [scored[i] for i in idx]

    os.makedirs(a.out_dir, exist_ok=True)
    print("%-8s %-6s %8s %6s %6s %6s" % ("tile", "rank", "F1", "truth", "pred", "hit"))
    for rank, (f1, img_id, ngt, npr, tp) in enumerate(picks):
        t = tile_of[img_id]
        src_path = os.path.join(ROOT, "lidar", "tile_%d_lidar.tif" % t)
        with rasterio.open(src_path) as src:
            arr = src.read()
            prof = src.profile.copy()
        img = np.ascontiguousarray(np.transpose(arr, (1, 2, 0))[:, :, ::-1])

        for g in gt_by[img_id]:
            x0, y0, x1, y1 = [int(round(v)) for v in g["box"]]
            dashed(img, (x0, y0), (x1, y1), GT_COLOUR)
        for pr in pr_by.get(img_id, []):
            x0, y0, x1, y1 = [int(round(v)) for v in pr["box"]]
            cv2.rectangle(img, (x0, y0), (x1, y1), COLOURS[pr["cls"]], 2)

        rgb = np.transpose(img[:, :, ::-1], (2, 0, 1))
        prof.update(count=3, dtype="uint8", compress="deflate",
                    photometric="RGB")
        out = os.path.join(a.out_dir, "tile_%d_f1_%.2f.tif" % (t, f1))
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(rgb)
            dst.update_tags(f1="%.3f" % f1, truth=str(ngt), predicted=str(npr),
                            matched=str(tp), score_threshold=str(a.score),
                            georeferenced="no - Chactun ships no CRS or transform")
        print("%-8d %-6d %8.2f %6d %6d %6d" % (t, rank, f1, ngt, npr, tp))
    print()
    print("wrote %d tiles to %s" % (len(picks), a.out_dir))
    print("white dashed = annotation, solid = detection at score >= %.2f" % a.score)


if __name__ == "__main__":
    main()
