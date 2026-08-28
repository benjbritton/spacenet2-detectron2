#!/usr/bin/env python
"""Rescore each AOI across IoU thresholds, to split "not found" from "found but
outlined too loosely".

THE QUESTION
------------
Khartoum recall is 0.583 at IoU 0.5. That number cannot distinguish two very
different failures:

  a) the detector never proposed anything on that building -- a DETECTION
     failure, and the fix is in the backbone or the RPN;
  b) something was proposed in the right place but its mask overlapped the truth
     by less than half -- a GEOMETRY failure, and the fix is in the mask head or
     the input resolution.

Loosening the IoU requirement separates them. A building that was never found
stays missing at any threshold. A building found and outlined loosely reappears
as the bar drops. The recall recovered between IoU 0.5 and IoU 0.25 is therefore
an estimate of how much of the loss is geometry rather than detection.

Reading it needs the other cities for contrast: if every AOI recovers the same
share, that is a property of the model, not of Khartoum.

Matching is redone at each IoU -- greedy, score-ordered, one ground truth per
prediction -- because which prediction claims which truth genuinely changes when
the bar moves. The score threshold is held at the reporting value of 0.544
throughout so that only IoU varies.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from detlab.datasets import spacenet
from detlab.spacenet_f1 import _gt_rles, match_greedy

CITIES = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]
IOUS = [0.10, 0.25, 0.50, 0.75]


def load(pred_path, dataset):
    from detectron2.data import DatasetCatalog

    gt_by_id = {r["image_id"]: r for r in DatasetCatalog.get(dataset)}
    preds = torch.load(pred_path, weights_only=False)
    tiles = []
    n_gt_total = 0
    seen = set()
    for entry in preds:
        rec = gt_by_id.get(entry["image_id"])
        if rec is None:
            continue
        seen.add(entry["image_id"])
        gt = _gt_rles(rec)
        n_gt_total += len(gt)
        rles, scores = [], []
        for i in entry.get("instances", []):
            segm = i.get("segmentation")
            if segm is None:
                continue
            if isinstance(segm.get("counts"), str):
                segm = dict(segm, counts=segm["counts"].encode("utf-8"))
            rles.append(segm)
            scores.append(i["score"])
        tiles.append((rles, scores, gt))
    for image_id, rec in gt_by_id.items():
        if image_id not in seen:
            n_gt_total += len(_gt_rles(rec))
    return tiles, n_gt_total


def score(tiles, n_gt, iou, thr):
    tp = fp = 0
    for rles, scores, gt in tiles:
        # Threshold BEFORE matching here: at a fixed operating point the model
        # would only ever emit these detections, so low-score predictions must
        # not be allowed to claim ground truth that then goes uncounted.
        keep = [i for i, s in enumerate(scores) if s >= thr]
        r = [rles[i] for i in keep]
        s = [scores[i] for i in keep]
        for _, ok in match_greedy(r, s, gt, iou):
            if ok:
                tp += 1
            else:
                fp += 1
    prec = tp / float(tp + fp) if tp + fp else 0.0
    rec = tp / float(n_gt) if n_gt else 0.0
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return f1, prec, rec, tp, fp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="outputs/spacenet2_r50fpn")
    p.add_argument("--data-root", default="data/spacenet2")
    p.add_argument("--threshold", type=float, default=0.544)
    args = p.parse_args()

    spacenet.register_pooled(root=args.data_root)
    spacenet.register_val_per_aoi(root=args.data_root)

    print("score threshold held at %.3f; only IoU varies" % args.threshold)
    print()
    header = "%-16s" % "city" + "".join(("IoU %.2f" % i).rjust(11) for i in IOUS)
    rows = {}
    for city in CITIES:
        name = "spacenet2_val_%s" % city
        pth = os.path.join(args.run_dir, "inference", name,
                           "instances_predictions.pth")
        if not os.path.isfile(pth):
            print("missing:", pth)
            continue
        tiles, n_gt = load(pth, name)
        rows[city] = {i: score(tiles, n_gt, i, args.threshold) for i in IOUS}
        rows[city]["n_gt"] = n_gt

    for label, idx in (("RECALL", 2), ("PRECISION", 1), ("F1", 0)):
        print("=" * 62)
        print(label)
        print("=" * 62)
        print(header)
        for city in CITIES:
            if city not in rows:
                continue
            line = "%-16s" % city
            for i in IOUS:
                line += ("%.3f" % rows[city][i][idx]).rjust(11)
            print(line)
        print()

    print("=" * 62)
    print("RECALL RECOVERED BY LOOSENING IoU 0.50 -> 0.25")
    print("=" * 62)
    print("%-16s %10s %10s %10s %10s" %
          ("city", "R@0.50", "R@0.25", "gained", "share of miss"))
    for city in CITIES:
        if city not in rows:
            continue
        r50 = rows[city][0.50][2]
        r25 = rows[city][0.25][2]
        missed = 1.0 - r50
        share = (r25 - r50) / missed if missed > 0 else 0.0
        print("%-16s %10.3f %10.3f %10.3f %9.1f%%"
              % (city, r50, r25, r25 - r50, 100 * share))
    print()
    print("share of miss = of everything not recalled at IoU 0.50, the fraction")
    print("that WAS detected and lost on geometry alone.")


if __name__ == "__main__":
    main()
