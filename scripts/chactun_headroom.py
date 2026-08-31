#!/usr/bin/env python
"""Is Chactun's weak building AP a DETECTION failure or a MEASUREMENT ceiling?

The distinction decides whether any architecture change can help. If the model
is failing to find structures, a better detector is worth trying. If it is
finding them and being scored down for boundary disagreement the ground truth
cannot resolve, then no architecture fixes it and the honest move is to report
AP50 and stop treating AP(0.5:0.95) as the target.

Three things are computed per class:

1. WHAT A PERFECT DETECTOR COULD SCORE. The ground truth is raster-traced, so
   every boundary carries sub-pixel error. For a shape of area A and perimeter
   L, moving the whole boundary out by d pixels costs about d*L of area, giving
   IoU = A / (A + d*L). COCO averages AP over ten IoU thresholds from 0.50 to
   0.95; at each one, any instance whose ceiling falls below the threshold is
   an automatic miss even for a flawless model. Averaging the achievable recall
   over the ten thresholds bounds AP(0.5:0.95) from above.

   Reported for d = 0.5 px (raster tracing alone) and d = 1.0 px (tracing plus
   ordinary annotation slop), because the answer should be shown to be robust
   to that choice rather than resting on it.

2. WHAT THE MODEL ACTUALLY SCORED, per class, at IoU 0.50, 0.75, and averaged.

3. HEADROOM: actual against ceiling. Small headroom means the metric is
   saturated and architecture work is wasted; large headroom means real
   detection is being missed and a better model has somewhere to go.

This does NOT touch the GPU -- it rescores predictions already on disk.
"""
import json
import os
import sys
from collections import defaultdict

import contextlib
import io as _io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASSES = {1: "building", 2: "platform", 3: "aguada"}
IOUS = np.arange(0.5, 1.0, 0.05)
ARM_DIR = "/w/repos/benjbritton_FA26/outputs/chactun_A_maskrcnn_default_anchors"
COCO_DIR = "/w/data/chactun/coco"


def poly_area_perim(seg):
    A = L = 0.0
    for ring in seg:
        p = np.array(ring, float).reshape(-1, 2)
        if len(p) < 3:
            continue
        x, y = p[:, 0], p[:, 1]
        A += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        d = np.diff(np.vstack([p, p[:1]]), axis=0)
        L += np.hypot(d[:, 0], d[:, 1]).sum()
    return A, L


def ceiling_by_class(gt_path, d):
    """Max achievable AP(0.5:0.95) per class under d pixels of boundary error."""
    gt = json.load(open(gt_path))
    per = defaultdict(list)
    for a in gt["annotations"]:
        if not isinstance(a["segmentation"], list):
            continue
        A, L = poly_area_perim(a["segmentation"])
        if A > 0 and L > 0:
            per[a["category_id"]].append(A / (A + d * L))
    out = {}
    for cid, v in per.items():
        v = np.array(v)
        # recall achievable at each COCO threshold, then averaged as AP is
        out[cid] = float(np.mean([(v >= t).mean() for t in IOUS])) * 100.0
    return out


def score(gt_path, pred_path):
    """Per-class AP at 0.50, 0.75 and averaged, from saved predictions."""
    preds = json.load(open(pred_path))
    with contextlib.redirect_stdout(_io.StringIO()):
        gt = COCO(gt_path)
        dt = gt.loadRes([dict(p) for p in preds])
        e = COCOeval(gt, dt, "segm")
        e.evaluate()
        e.accumulate()
    # precision: [T, R, K, A, M] -- thresholds, recall, category, area, maxdets
    p = e.eval["precision"]
    cat_ids = list(gt.getCatIds())
    out = {}
    for k, cid in enumerate(cat_ids):
        sl = p[:, :, k, 0, 2]                      # all areas, maxDets=100
        avg = sl[sl > -1].mean() * 100 if (sl > -1).any() else float("nan")
        a50 = p[0, :, k, 0, 2]
        a75 = p[5, :, k, 0, 2]
        out[cid] = (
            a50[a50 > -1].mean() * 100 if (a50 > -1).any() else float("nan"),
            a75[a75 > -1].mean() * 100 if (a75 > -1).any() else float("nan"),
            avg,
        )
    return out


def main():
    folds = [f for f in range(5)
             if os.path.isfile(os.path.join(
                 ARM_DIR, "fold%d_seed0" % f, "inference",
                 "coco_instances_results.json"))]
    print("arm A folds with saved predictions: %s" % folds)
    if not folds:
        sys.exit("no predictions found")

    acc = defaultdict(lambda: defaultdict(list))
    for f in folds:
        gt = os.path.join(COCO_DIR, "fold%d_val.json" % f)
        pred = os.path.join(ARM_DIR, "fold%d_seed0" % f, "inference",
                            "coco_instances_results.json")
        s = score(gt, pred)
        c05 = ceiling_by_class(gt, 0.5)
        c10 = ceiling_by_class(gt, 1.0)
        for cid in CLASSES:
            if cid in s:
                acc[cid]["AP50"].append(s[cid][0])
                acc[cid]["AP75"].append(s[cid][1])
                acc[cid]["AP"].append(s[cid][2])
            if cid in c05:
                acc[cid]["ceil05"].append(c05[cid])
                acc[cid]["ceil10"].append(c10[cid])

    print()
    print("Averaged over %d folds. Ceiling = best AP(0.5:0.95) a PERFECT" % len(folds))
    print("detector could score, given d pixels of boundary uncertainty.")
    print()
    print("%-10s %8s %8s %8s %10s %10s %12s"
          % ("class", "AP50", "AP75", "AP", "ceil d=0.5", "ceil d=1.0",
             "headroom@1.0"))
    for cid, name in CLASSES.items():
        a = acc[cid]
        if not a["AP"]:
            continue
        ap = np.mean(a["AP"])
        c10 = np.mean(a["ceil10"])
        print("%-10s %8.2f %8.2f %8.2f %10.2f %10.2f %12s"
              % (name, np.mean(a["AP50"]), np.mean(a["AP75"]), ap,
                 np.mean(a["ceil05"]), c10,
                 "%+.2f" % (c10 - ap)))

    print()
    print("=== how to read this ===")
    print("  A class whose AP sits near its ceiling is metric-limited: the")
    print("  detector is finding the structures and losing points to boundary")
    print("  disagreement the labels cannot resolve. Architecture work on such")
    print("  a class is wasted, and AP50 is the number to report.")
    print()
    print("  A class with wide headroom is genuinely being missed, and a better")
    print("  detector has somewhere to go.")


if __name__ == "__main__":
    main()
