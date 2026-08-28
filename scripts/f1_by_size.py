#!/usr/bin/env python
"""Per-city F1 broken out by COCO size bucket: does size explain Khartoum?

THE QUESTION
------------
scripts/city_separability.py found Khartoum is 47.7% small instances against
Vegas's 29.0%, median footprint 1182 px vs 2327. Small objects score worse
everywhere, so that alone could produce the whole 0.895 -> 0.627 ordering.

This settles it by scoring WITHIN a size bucket. If Khartoum's small buildings
score like Vegas's small buildings, the city effect is composition and nothing
else. If they still lag, something beyond size is at work and the shadow /
boundary-contrast findings have something to explain.

BUCKET ASSIGNMENT
-----------------
Ground truth is bucketed by its own COCO area, which is what pycocotools does.
A false positive has no ground truth to inherit an area from, so it is bucketed
by its OWN predicted mask area -- the same convention COCOeval uses when it
filters detections by area range. Stated because it is a choice, not a law: a
false positive is a building the model invented, and its size is whatever the
model imagined.

    TP_b   ground truth in bucket b matched by a prediction scoring >= t
    FN_b   ground truth in bucket b matched by nothing
    FP_b   surviving prediction matching no ground truth, own area in bucket b

Matching runs once over all predictions at IoU 0.5 via
detlab.spacenet_f1.match_greedy_pairs, so these numbers are the same matching
the headline F1 used -- only the bookkeeping differs.

    ./scripts/run.sh python scripts/f1_by_size.py --threshold 0.544
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from pycocotools import mask as mask_util

from detlab.spacenet_f1 import match_greedy_pairs

CITIES = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]
BUCKETS = [("small", 0, 1024), ("medium", 1024, 9216), ("large", 9216, 1e18)]


def bucket_of(area):
    for name, lo, hi in BUCKETS:
        if lo <= area < hi:
            return name
    return "large"


def gt_rles(anns, h, w):
    """COCO polygon annotations -> one merged RLE each.

    Merged, not one RLE per polygon: the converter groups a footprint split at a
    tile edge into a single annotation carrying several polygons, and they are
    one building.
    """
    out = []
    for a in anns:
        rles = mask_util.frPyObjects(a["segmentation"], h, w)
        out.append(mask_util.merge(rles))
    return out


def f1(tp, fp, fn):
    p = tp / float(tp + fp) if tp + fp else 0.0
    r = tp / float(tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions",
                   default="outputs/spacenet2_r50fpn/inference/instances_predictions.pth")
    p.add_argument("--coco", default="data/spacenet2/coco/pooled_val.json")
    p.add_argument("--threshold", type=float, default=0.544,
                   help="score cutoff, selected on train (2026-08-27 entry)")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--out", default="outputs/city_analysis/f1_by_size.json")
    args = p.parse_args()

    with open(args.coco) as f:
        coco = json.load(f)
    info = {im["id"]: im for im in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    preds = torch.load(args.predictions, weights_only=False)
    pred_by_img = {e["image_id"]: e.get("instances", []) for e in preds}

    counts = {c: {b[0]: {"tp": 0, "fp": 0, "fn": 0} for b in BUCKETS}
              for c in CITIES}

    for image_id, im in info.items():
        city = im["aoi"]
        h, w = im["height"], im["width"]
        anns = anns_by_img.get(image_id, [])
        gts = gt_rles(anns, h, w)
        gt_bucket = [bucket_of(float(a["area"])) for a in anns]

        rles, scores, pareas = [], [], []
        for inst in pred_by_img.get(image_id, []):
            segm = inst.get("segmentation")
            if segm is None:
                continue
            if isinstance(segm.get("counts"), str):
                segm = dict(segm, counts=segm["counts"].encode("utf-8"))
            rles.append(segm)
            scores.append(float(inst["score"]))
            pareas.append(float(mask_util.area(segm)))

        pairs = match_greedy_pairs(rles, scores, gts, args.iou)

        hit = [False] * len(gts)
        for d, g in enumerate(pairs):
            if scores[d] < args.threshold:
                continue          # below threshold: not a detection at all
            if g >= 0:
                hit[g] = True
                counts[city][gt_bucket[g]]["tp"] += 1
            else:
                counts[city][bucket_of(pareas[d])]["fp"] += 1
        for g, ok in enumerate(hit):
            if not ok:
                counts[city][gt_bucket[g]]["fn"] += 1

    print("threshold %.3f, IoU %.2f\n" % (args.threshold, args.iou))
    print("%-16s %-7s %7s %6s %6s %8s %8s %8s"
          % ("city", "bucket", "n_gt", "F1", "prec", "recall", "TP", "FP"))
    out = {}
    for city in CITIES:
        out[city] = {}
        for name, _, _ in BUCKETS:
            c = counts[city][name]
            n_gt = c["tp"] + c["fn"]
            s, pr, rc = f1(c["tp"], c["fp"], c["fn"])
            out[city][name] = dict(c, n_gt=n_gt, f1=round(s, 4),
                                   precision=round(pr, 4), recall=round(rc, 4))
            print("%-16s %-7s %7d %6.3f %6.3f %8.3f %8d %8d"
                  % (city, name, n_gt, s, pr, rc, c["tp"], c["fp"]))
        print()

    print("F1 within bucket, city by city -- the comparison that matters:")
    print("%-8s %-16s %-16s %-16s %-16s" % ("bucket", *CITIES))
    for name, _, _ in BUCKETS:
        print("%-8s %-16.3f %-16.3f %-16.3f %-16.3f"
              % (name, *[out[c][name]["f1"] for c in CITIES]))

    # If size were the whole story, rescoring every city with Vegas's size mix
    # would flatten the ordering. This reweights each city's per-bucket F1 by
    # Vegas's bucket proportions -- crude, since it assumes bucket F1 is
    # independent of the mix, but it says how much of the gap composition buys.
    vshare = np.array([out["AOI_2_Vegas"][b[0]]["n_gt"] for b in BUCKETS],
                      dtype=float)
    vshare /= vshare.sum()
    print("\nStandardised to Vegas's size mix (%.0f/%.0f/%.0f small/med/large):"
          % tuple(vshare * 100))
    for city in CITIES:
        raw = np.array([out[city][b[0]]["n_gt"] for b in BUCKETS], dtype=float)
        f1s = np.array([out[city][b[0]]["f1"] for b in BUCKETS])
        actual = float((f1s * raw).sum() / raw.sum())
        std = float((f1s * vshare).sum())
        print("  %-16s actual %.3f   size-standardised %.3f   (%+0.3f)"
              % (city, actual, std, std - actual))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
