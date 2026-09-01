#!/usr/bin/env python
"""Is aguada failing at DETECTION or at CLASSIFICATION?

The two demand opposite fixes and AP cannot tell them apart, because AP folds
localisation and labelling into one number.

  - If aguadas are never localised, the problem is detection: the model does not
    see them as objects at all, and the fix is more or better exemplars.
  - If aguadas are localised but labelled platform, the problem is
    discrimination between two classes that look alike in relief, and the fix is
    a classifier over crops, or a merged super-class with sub-classification
    afterwards -- which uses all 9853 instances to learn "structure" and asks
    the hard question only where it matters.

For each ground-truth instance, this finds the best-overlapping prediction of
ANY class. Localised means IoU >= 0.5 with something. Correct means that
something also carried the right label. The gap between the two is the
classification loss, and it is invisible in per-class AP.

Runs on saved predictions. No GPU.
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
from pycocotools import mask as maskutil
from pycocotools.coco import COCO

REPO = "/w/repos/benjbritton_FA26"
FULL_GT = "/w/data/chactun/coco/chactun_cc.json"
ARMS = {
    "A": "outputs/chactun_A_maskrcnn_default_anchors",
    "D": "outputs/chactun_D_maskrcnn_d4_augmentation",
}
SCORE_MIN = 0.5          # a detection nobody would act on is not a detection


def load_preds(arm):
    out = []
    for f in range(5):
        p = os.path.join(REPO, ARMS[arm], "fold%d_seed0" % f, "inference",
                         "coco_instances_results.json")
        if os.path.isfile(p):
            out.extend(json.load(open(p)))
    return [x for x in out if x.get("score", 1.0) >= SCORE_MIN]


def analyse(arm, gt):
    preds = load_preds(arm)
    by_img = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)

    names = {c["id"]: c["name"] for c in gt.dataset["categories"]}
    stat = {n: Counter() for n in names.values()}
    confusion = defaultdict(Counter)

    for img_id in gt.getImgIds():
        anns = gt.loadAnns(gt.getAnnIds(imgIds=img_id))
        if not anns:
            continue
        dets = by_img.get(img_id, [])
        info = gt.loadImgs(img_id)[0]
        h, w = info["height"], info["width"]

        if dets:
            drle = []
            for d in dets:
                s = d["segmentation"]
                if isinstance(s, dict):
                    drle.append(s)
                else:
                    drle.append(maskutil.merge(
                        maskutil.frPyObjects(s, h, w)))
        for a in anns:
            gname = names[a["category_id"]]
            stat[gname]["total"] += 1
            if not dets:
                continue
            grle = maskutil.merge(maskutil.frPyObjects(a["segmentation"], h, w))
            ious = maskutil.iou(drle, [grle], [0]).reshape(-1)
            j = int(np.argmax(ious))
            if ious[j] >= 0.5:
                stat[gname]["localised"] += 1
                pname = names[dets[j]["category_id"]]
                if pname == gname:
                    stat[gname]["correct"] += 1
                else:
                    confusion[gname][pname] += 1

    print("--- arm %s (predictions scored >= %.2f) ---" % (arm, SCORE_MIN))
    print("%-10s %8s %11s %10s %12s %12s"
          % ("class", "GT", "localised", "correct", "localised %", "correct %"))
    for n in ["building", "platform", "aguada"]:
        s = stat[n]
        tot = max(s["total"], 1)
        print("%-10s %8d %11d %10d %11.1f%% %11.1f%%"
              % (n, s["total"], s["localised"], s["correct"],
                 100 * s["localised"] / tot, 100 * s["correct"] / tot))
    print()
    for n in ["aguada", "platform", "building"]:
        if confusion[n]:
            tot = max(stat[n]["total"], 1)
            lost = sum(confusion[n].values())
            print("  %s localised but mislabelled: %d of %d (%.1f%% of all %s)"
                  % (n, lost, stat[n]["total"], 100 * lost / tot, n))
            for k, v in confusion[n].most_common():
                print("      called %-10s %4d" % (k, v))
    print()
    return stat


def main():
    gt = COCO(FULL_GT)
    print()
    a = analyse("A", gt)
    d = analyse("D", gt)

    print("=== reading ===")
    for n in ["aguada"]:
        la = 100 * a[n]["localised"] / max(a[n]["total"], 1)
        ca = 100 * a[n]["correct"] / max(a[n]["total"], 1)
        ld = 100 * d[n]["localised"] / max(d[n]["total"], 1)
        cd = 100 * d[n]["correct"] / max(d[n]["total"], 1)
        print("  %s localisation: A %.1f%% -> D %.1f%%" % (n, la, ld))
        print("  %s correct label: A %.1f%% -> D %.1f%%" % (n, ca, cd))
        gap = ld - cd
        print()
        if gap > 15:
            print("  A large localised-minus-correct gap (%.1f points) means the"
                  % gap)
            print("  model FINDS aguadas and calls them something else. The fix is")
            print("  discrimination -- a super-class detector plus a crop")
            print("  classifier -- not more detection capacity.")
        elif ld < 50:
            print("  Aguadas are mostly not found at all (%.1f%% localised)."
                  % ld)
            print("  The fix is exemplars, and augmentation cannot create them.")
        else:
            print("  Localisation and labelling are close, so the loss is")
            print("  elsewhere: score ranking or duplicate suppression.")


if __name__ == "__main__":
    main()
