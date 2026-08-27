#!/usr/bin/env python
"""Full SpaceNet F1 report at an externally selected score threshold.

WHY A THRESHOLD HAS TO COME FROM SOMEWHERE ELSE
-----------------------------------------------
F1 is a single-operating-point metric. Reporting the best F1 over all thresholds
means reporting a value at a threshold chosen using the very set being scored --
a tuned hyperparameter presented as a result. The threshold must be selected on
other data.

Two candidate sources, both computed here so the difference is visible rather
than assumed:

  train     Unbiased with respect to val, but the model has memorised these
            tiles, so its predictions there are more confident and more accurate.
            The optimal threshold on train is therefore systematically offset
            from what is optimal on unseen ground.

  val half  Split val by image id; select on half A, report on half B. Unbiased
            AND distribution-matched, at the cost of halving the reporting set.
            Costs no extra inference.

If the two agree, the choice does not matter and that can be said with evidence.
If they disagree, the val-half figure is the more defensible one.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from detlab.datasets import spacenet
from detlab.spacenet_f1 import _gt_rles, match_greedy, sweep

CITIES = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]


def collect(pred_path, dataset, iou=0.5, keep=None):
    """(records, n_gt) for a prediction file, optionally restricted to `keep`."""
    from detectron2.data import DatasetCatalog

    gt_by_id = {r["image_id"]: r for r in DatasetCatalog.get(dataset)}
    preds = torch.load(pred_path, weights_only=False)

    records = []
    n_gt = 0
    seen = set()
    for entry in preds:
        image_id = entry["image_id"]
        if keep is not None and image_id not in keep:
            continue
        rec = gt_by_id.get(image_id)
        if rec is None:
            continue
        seen.add(image_id)
        gt = _gt_rles(rec)
        n_gt += len(gt)
        rles, scores = [], []
        for i in entry.get("instances", []):
            segm = i.get("segmentation")
            if segm is None:
                continue
            if isinstance(segm.get("counts"), str):
                segm = dict(segm, counts=segm["counts"].encode("utf-8"))
            rles.append(segm)
            scores.append(i["score"])
        records.extend(match_greedy(rles, scores, gt, iou))

    # Tiles with no predictions still contribute ground truth; omitting them
    # would inflate recall.
    for image_id, rec in gt_by_id.items():
        if image_id not in seen and (keep is None or image_id in keep):
            n_gt += len(_gt_rles(rec))
    return records, n_gt


def at(records, n_gt, thr):
    tp = sum(1 for s, ok in records if s >= thr and ok)
    fp = sum(1 for s, ok in records if s >= thr and not ok)
    prec = tp / float(tp + fp) if tp + fp else 0.0
    rec = tp / float(n_gt) if n_gt else 0.0
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return f1, prec, rec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="outputs/spacenet2_r50fpn")
    p.add_argument("--train-predictions",
                   default="outputs/thresh_select/inference/instances_predictions.pth")
    p.add_argument("--data-root", default="data/spacenet2")
    args = p.parse_args()

    spacenet.register_pooled(root=args.data_root)
    spacenet.register_val_per_aoi(root=args.data_root)

    # --- threshold from train -------------------------------------------------
    tr_rec, tr_gt = collect(args.train_predictions, "spacenet2_train")
    tr = sweep(tr_rec, tr_gt)
    thr_train = tr["best_threshold"]
    print("threshold selected on TRAIN (%d gt): %.3f   [train F1 %.4f]"
          % (tr_gt, thr_train, tr["f1_at_best"]))

    # --- threshold from one half of val --------------------------------------
    from detectron2.data import DatasetCatalog
    val_ids = sorted(r["image_id"] for r in DatasetCatalog.get("spacenet2_val"))
    half_a = set(val_ids[0::2])
    half_b = set(val_ids[1::2])
    pooled_pred = os.path.join(args.run_dir, "inference", "instances_predictions.pth")

    a_rec, a_gt = collect(pooled_pred, "spacenet2_val", keep=half_a)
    thr_half = sweep(a_rec, a_gt)["best_threshold"]
    b_rec, b_gt = collect(pooled_pred, "spacenet2_val", keep=half_b)
    print("threshold selected on VAL HALF A (%d gt): %.3f" % (a_gt, thr_half))
    print()

    f1b, pb, rb = at(b_rec, b_gt, thr_half)
    f1b2, _, _ = at(b_rec, b_gt, thr_train)
    print("held-out val half B (%d gt):" % b_gt)
    print("   at val-half threshold %.3f : F1 %.4f  (P %.4f  R %.4f)"
          % (thr_half, f1b, pb, rb))
    print("   at train threshold    %.3f : F1 %.4f" % (thr_train, f1b2))
    print()

    # --- full val and per AOI at the train-selected threshold -----------------
    full_rec, full_gt = collect(pooled_pred, "spacenet2_val")
    fs = sweep(full_rec, full_gt)
    f1, pr, rc = at(full_rec, full_gt, thr_train)
    print("FULL VAL (%d gt)" % full_gt)
    print("   at train threshold %.3f : F1 %.4f  (P %.4f  R %.4f)" % (thr_train, f1, pr, rc))
    print("   best over sweep          : F1 %.4f at %.3f   <- NOT reportable"
          % (fs["f1_at_best"], fs["best_threshold"]))
    print()

    print("PER AOI at the train-selected threshold %.3f" % thr_train)
    macro = []
    for city in CITIES:
        name = "spacenet2_val_%s" % city
        pth = os.path.join(args.run_dir, "inference", name, "instances_predictions.pth")
        if not os.path.isfile(pth):
            print("   %-16s (no predictions)" % city)
            continue
        rec, gt = collect(pth, name)
        f1c, pc, rcc = at(rec, gt, thr_train)
        macro.append(f1c)
        print("   %-16s F1 %.4f   P %.4f   R %.4f   (%d gt)"
              % (city, f1c, pc, rcc, gt))
    if macro:
        print("   %-16s %.4f" % ("MACRO MEAN", sum(macro) / len(macro)))


if __name__ == "__main__":
    main()
