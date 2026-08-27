#!/usr/bin/env python
"""Score SpaceNet F1 from predictions COCOEvaluator already saved.

COCOEvaluator writes instances_predictions.pth into its output folder whenever
one is set. That file holds every prediction with its RLE mask and score, which
is everything the F1 metric needs -- so a run that finished before the evaluator
existed can be scored without a second inference pass over 2118 tiles.

    ./scripts/run.sh python scripts/score_f1.py \\
        --predictions outputs/spacenet2_r50fpn/inference/instances_predictions.pth \\
        --dataset spacenet2_val
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from detlab.datasets import spacenet
from detlab.spacenet_f1 import _gt_rles, match_greedy, sweep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", default="data/spacenet2")
    p.add_argument("--iou", type=float, default=0.5)
    args = p.parse_args()

    from detectron2.data import DatasetCatalog

    spacenet.register_pooled(root=args.data_root)
    spacenet.register_val_per_aoi(root=args.data_root)
    gt_by_id = {r["image_id"]: r for r in DatasetCatalog.get(args.dataset)}

    preds = torch.load(args.predictions, weights_only=False)
    print("loaded %d prediction records from %s" % (len(preds), args.predictions))

    records = []
    n_gt = n_pred = 0
    missing = 0
    for entry in preds:
        image_id = entry["image_id"]
        rec = gt_by_id.get(image_id)
        if rec is None:
            missing += 1
            continue
        gt = _gt_rles(rec)
        n_gt += len(gt)

        inst = entry.get("instances", [])
        rles, scores = [], []
        for i in inst:
            segm = i.get("segmentation")
            if segm is None:
                continue
            # COCOEvaluator stores RLE counts as str after json round-trips.
            if isinstance(segm.get("counts"), str):
                segm = dict(segm, counts=segm["counts"].encode("utf-8"))
            rles.append(segm)
            scores.append(i["score"])
        n_pred += len(rles)
        records.extend(match_greedy(rles, scores, gt, args.iou))

    # Tiles with zero predictions never appear in the prediction file for some
    # writers, so their ground truth would go uncounted and inflate recall.
    counted = {e["image_id"] for e in preds}
    for image_id, rec in gt_by_id.items():
        if image_id not in counted:
            n_gt += len(_gt_rles(rec))

    if missing:
        print("WARNING: %d prediction records had no matching ground truth" % missing)

    res = sweep(records, n_gt)
    print()
    print("dataset            :", args.dataset)
    print("ground truth       :", n_gt)
    print("predictions        :", n_pred)
    print("IoU threshold      :", args.iou)
    print("F1 (best)          : %.4f  at score >= %.3f" %
          (res["f1_at_best"], res["best_threshold"]))
    print("  precision        : %.4f" % res["precision_at_best"])
    print("  recall           : %.4f" % res["recall_at_best"])
    print("F1 at score >= 0.5 : %.4f" % res["f1_at_0.5"])


if __name__ == "__main__":
    main()
