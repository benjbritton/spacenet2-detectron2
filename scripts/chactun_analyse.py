#!/usr/bin/env python
"""Analyse the Chactun A/B/C matrix: 21 runs, three arms, five folds, three seeds.

WHY PAIRED TESTS
----------------
Folds differ from each other far more than arms differ from each other -- arm A
alone spans 36.30 to 42.61 across folds. Comparing arm means against pooled
variance would drown any real effect in fold difficulty. Every arm ran on the
SAME five folds, so the comparison is paired: difference per fold, then a test
on those differences. That removes fold difficulty entirely.

WHY THE SEED SPREAD MATTERS SEPARATELY
--------------------------------------
A paired difference is only interesting if it exceeds the noise of rerunning the
same configuration. Fold 0 was run at three seeds per arm, so that noise floor is
measured rather than assumed, and any arm difference is reported against it.

POOLED CROSS-VALIDATION
-----------------------
Per-fold aguada AP swings from 16.73 to 37.92 because a fold holds only 15 or 16
aguadas. Averaging those per-fold numbers is not the same as evaluating the class
properly. The folds PARTITION the dataset, so concatenating the five folds'
predictions covers all 2094 tiles exactly once, and scoring that against the full
ground truth evaluates every one of the 76 aguadas exactly once.

Caveat recorded with the result: the concatenated predictions come from five
different models whose score distributions are not identically calibrated. That
is inherent to cross-validated pooling and is the standard practice, but it means
the pooled figure is not the score of any single deployable model.
"""
import contextlib
import io as _io
import json
import os
from collections import OrderedDict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import stats

REPO = "/w/repos/benjbritton_FA26"
FULL_GT = "/w/data/chactun/coco/chactun_cc.json"
COCO_DIR = "/w/data/chactun/coco"
ARMS = OrderedDict([
    ("A", ("outputs/chactun_A_maskrcnn_default_anchors", "Mask R-CNN, default anchors")),
    ("B", ("outputs/chactun_B_maskrcnn_shifted_anchors", "Mask R-CNN, shifted anchors")),
    ("C", ("outputs/chactun_C_cascade_shifted_anchors", "Cascade, shifted anchors")),
])
KEYS = ["segm/AP", "segm/AP50", "segm/AP75",
        "segm/AP-building", "segm/AP-platform", "segm/AP-aguada"]
SHORT = {"segm/AP": "segm AP", "segm/AP50": "AP50", "segm/AP75": "AP75",
         "segm/AP-building": "building", "segm/AP-platform": "platform",
         "segm/AP-aguada": "aguada"}


def final_metrics(d):
    p = os.path.join(REPO, d, "metrics.json")
    rows = [json.loads(l) for l in open(p)]
    ev = [r for r in rows if "segm/AP" in r]
    return ev[-1] if ev else None


def collect():
    folds, seeds = {}, {}
    for arm, (root, _) in ARMS.items():
        folds[arm] = [final_metrics(os.path.join(root, "fold%d_seed0" % f))
                      for f in range(5)]
        seeds[arm] = [final_metrics(os.path.join(root, "fold0_seed%d" % s))
                      for s in range(3)]
    return folds, seeds


def table(title, data, label_fn):
    print("=== %s ===" % title)
    print("%-28s %9s %8s %8s %9s %9s %8s"
          % ("", "segm AP", "AP50", "AP75", "building", "platform", "aguada"))
    for lab, rows in data:
        vals = [np.mean([r[k] for r in rows]) for k in KEYS]
        sds = [np.std([r[k] for r in rows], ddof=1) for k in KEYS]
        print("%-28s %9.2f %8.2f %8.2f %9.2f %9.2f %8.2f"
              % (lab, *vals))
        print("%-28s %9s %8s %8s %9s %9s %8s"
              % ("  sd", *["+/-%.2f" % s for s in sds]))
    print()


def paired(folds, seeds, a, b):
    print("=== %s vs %s (paired by fold, n=5) ===" % (a, b))
    print("%-12s %10s %10s %10s %8s %8s %10s"
          % ("metric", "%s mean" % a, "%s mean" % b, "diff", "sd", "t",
             "seed noise"))
    for k in KEYS:
        av = np.array([r[k] for r in folds[a]])
        bv = np.array([r[k] for r in folds[b]])
        d = bv - av
        t, p = stats.ttest_rel(bv, av)
        # noise floor: sd across the three fold-0 seeds, averaged over both arms
        sn = np.mean([np.std([r[k] for r in seeds[a]], ddof=1),
                      np.std([r[k] for r in seeds[b]], ddof=1)])
        flag = ""
        if p < 0.05:
            flag = "  SIGNIFICANT p=%.3f" % p
        elif abs(d.mean()) < sn:
            flag = "  below seed noise"
        print("%-12s %10.2f %10.2f %+10.2f %8.2f %8.2f %10.2f%s"
              % (SHORT[k], av.mean(), bv.mean(), d.mean(), d.std(ddof=1),
                 t, sn, flag))
    print()


def pooled(arm):
    """Concatenate the five folds' predictions and score against the full GT."""
    root = ARMS[arm][0]
    preds = []
    for f in range(5):
        p = os.path.join(REPO, root, "fold%d_seed0" % f, "inference",
                         "coco_instances_results.json")
        preds.extend(json.load(open(p)))
    with contextlib.redirect_stdout(_io.StringIO()):
        gt = COCO(FULL_GT)
        dt = gt.loadRes([dict(x) for x in preds])
        e = COCOeval(gt, dt, "segm")
        e.evaluate(); e.accumulate()
    prec = e.eval["precision"]
    cats = list(gt.getCatIds())
    names = {c["id"]: c["name"] for c in gt.dataset["categories"]}
    out = {}
    for i, cid in enumerate(cats):
        sl = prec[:, :, i, 0, 2]
        a50 = prec[0, :, i, 0, 2]
        out[names[cid]] = (
            sl[sl > -1].mean() * 100 if (sl > -1).any() else float("nan"),
            a50[a50 > -1].mean() * 100 if (a50 > -1).any() else float("nan"))
    allsl = prec[:, :, :, 0, 2]
    out["ALL"] = (allsl[allsl > -1].mean() * 100, float("nan"))
    return out, len(preds)


def main():
    folds, seeds = collect()

    table("Fold spread, seed 0, n=5",
          [(ARMS[a][1] + " (%s)" % a, folds[a]) for a in ARMS], None)
    table("Seed spread, fold 0, n=3",
          [(ARMS[a][1] + " (%s)" % a, seeds[a]) for a in ARMS], None)

    paired(folds, seeds, "A", "B")
    paired(folds, seeds, "B", "C")
    paired(folds, seeds, "A", "C")

    print("=== pooled cross-validation: every instance scored exactly once ===")
    print("%-6s %10s %10s %10s %10s %10s"
          % ("arm", "preds", "ALL AP", "building", "platform", "aguada"))
    for a in ARMS:
        o, n = pooled(a)
        print("%-6s %10d %10.2f %10.2f %10.2f %10.2f"
              % (a, n, o["ALL"][0], o["building"][0], o["platform"][0],
                 o["aguada"][0]))
    print()
    print("  aguada here is all 76 instances, not a 15-instance fold slice.")
    print("  Predictions come from five different models, so this is a")
    print("  cross-validated estimate, not the score of one deployable model.")


if __name__ == "__main__":
    main()
