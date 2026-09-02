#!/usr/bin/env python
"""Show arm D's detections on Chactun itself, against the ground truth.

Every judgement of this model on its own data has been numeric. AP 42.87 and
92% recall at score 0.05 are not things anyone can picture, and the only
visual assessment so far was on G-LiHT, where it did poorly. This renders the
model on home ground so the numbers can be checked against an eye.

Tiles are chosen by how well the run went, not cherry-picked: the best, the
median and the worst by per-tile F1, so the spread is visible rather than the
highlights. Ground truth is drawn dashed white, predictions solid in class
colour, so misses and false positives are directly readable.
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pycocotools import mask as maskutil

COLOURS = {"building": "#00e5ff", "platform": "#ffd60a", "aguada": "#ff2d55"}
ROOT = "/w/data/chactun"


def iou_box(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", default="/w/data/chactun/coco/fold0_val.json")
    p.add_argument("--pred", default="/w/repos/benjbritton_FA26/outputs/"
                                     "chactun_D_maskrcnn_d4_augmentation/"
                                     "fold0_seed0/inference/"
                                     "coco_instances_results.json")
    p.add_argument("--score", type=float, default=0.30)
    p.add_argument("--out", default="/w/repos/benjbritton_FA26/figures/"
                                    "chactun_detections_on_chactun.png")
    a = p.parse_args()

    gt = json.load(open(a.gt))
    names = {c["id"]: c["name"] for c in gt["categories"]}
    tile_of = {im["id"]: im["tile"] for im in gt["images"]}
    size_of = {im["id"]: (im["height"], im["width"]) for im in gt["images"]}

    gt_by = defaultdict(list)
    for x in gt["annotations"]:
        h, w = size_of[x["image_id"]]
        rle = maskutil.merge(maskutil.frPyObjects(x["segmentation"], h, w))
        bb = maskutil.toBbox(rle)
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

    # per-tile F1 so the selection is honest rather than flattering
    scored = []
    for img_id, gts in gt_by.items():
        if len(gts) < 4:
            continue
        prs = pr_by.get(img_id, [])
        used = set()
        tp = 0
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
    if not scored:
        raise SystemExit("no tiles with enough ground truth")
    picks = [("best", scored[0]),
             ("median", scored[len(scored) // 2]),
             ("worst", scored[-1])]
    # two more from the upper middle, for a fuller picture
    if len(scored) > 8:
        picks.insert(1, ("upper quartile", scored[len(scored) // 4]))
        picks.insert(3, ("lower quartile", scored[3 * len(scored) // 4]))

    n = len(picks)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.4))
    if n == 1:
        axes = [axes]
    for ax, (label, (f1, img_id, ngt, npr, tp)) in zip(axes, picks):
        t = tile_of[img_id]
        with rasterio.open(os.path.join(ROOT, "lidar",
                                        "tile_%d_lidar.tif" % t)) as src:
            arr = src.read()
        rgb = np.transpose(arr, (1, 2, 0))
        ax.imshow(rgb)
        for g in gt_by[img_id]:
            x0, y0, x1, y1 = g["box"]
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="white",
                linewidth=1.6, linestyle="--"))
        for pr in pr_by.get(img_id, []):
            x0, y0, x1, y1 = pr["box"]
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False,
                edgecolor=COLOURS[pr["cls"]], linewidth=1.8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("%s tile: F1 %.2f\n%d truth, %d predicted, %d matched"
                     % (label, f1, ngt, npr, tp), fontsize=9)

    handles = [mpatches.Patch(color=v, label=k) for k, v in COLOURS.items()]
    handles.append(mpatches.Patch(facecolor="none", edgecolor="white",
                                  linestyle="--", label="ground truth"))
    axes[0].legend(handles=handles, loc="lower left", fontsize=7,
                   framealpha=0.75)
    fig.suptitle("Arm D on Chactun, score >= %.2f. Dashed white = annotation, "
                 "solid = detection." % a.score, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=135)
    print("wrote", a.out)
    print("tiles scored: %d, F1 range %.2f to %.2f"
          % (len(scored), scored[-1][0], scored[0][0]))


if __name__ == "__main__":
    main()
