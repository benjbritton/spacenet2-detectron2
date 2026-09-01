#!/usr/bin/env python
"""What a candidate generator actually delivers: recall against false positives per km2.

WHY NOT AP
----------
Every number in this milestone is COCO AP, which integrates over all score
thresholds and rewards precision at high confidence. That is not what this tool
is for. A regional candidate generator runs at a LOW threshold, floods a human
expert with candidates, and is judged on whether it missed real settlement --
false positives are triage cost, not failure. AP can rank two models in the
opposite order to how they would serve that job.

So this reports, per score threshold: what fraction of real structures were
found, and how many spurious detections per square kilometre that cost.

AREA. Tiles are 480 x 480 at 0.5 m, so each covers 240 m x 240 m = 0.0576 km2.
The five folds partition 2094 tiles, so pooled predictions cover 120.6 km2 and
every structure is evaluated exactly once.

A detection counts as a hit if it overlaps a ground-truth instance of the SAME
class at IoU >= 0.5. Predictions matching nothing are false positives. Duplicate
detections of one structure count once as a hit and the extras as false
positives, since an expert triaging a map sees every box.
"""
import json
import os
from collections import defaultdict

import numpy as np
from pycocotools import mask as maskutil
from pycocotools.coco import COCO

REPO = "/w/repos/benjbritton_FA26"
FULL_GT = "/w/data/chactun/coco/chactun_cc.json"
ARMS = {
    "A": "outputs/chactun_A_maskrcnn_default_anchors",
    "D": "outputs/chactun_D_maskrcnn_d4_augmentation",
    "F": "outputs/chactun_F_maskrcnn_hires960",
}
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70]
TILE_KM2 = (480 * 0.5 / 1000.0) ** 2          # 0.0576 km2


def load_preds(arm):
    out = []
    for f in range(5):
        p = os.path.join(REPO, ARMS[arm], "fold%d_seed0" % f, "inference",
                         "coco_instances_results.json")
        if os.path.isfile(p):
            out.extend(json.load(open(p)))
    return out


def evaluate(arm, gt, names):
    preds = load_preds(arm)
    by_img = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)

    n_img = len(gt.getImgIds())
    area_km2 = n_img * TILE_KM2

    # per threshold: hits per class, false positives
    hits = {t: defaultdict(int) for t in THRESHOLDS}
    fps = {t: 0 for t in THRESHOLDS}
    total = defaultdict(int)

    for img_id in gt.getImgIds():
        anns = gt.loadAnns(gt.getAnnIds(imgIds=img_id))
        info = gt.loadImgs(img_id)[0]
        h, w = info["height"], info["width"]
        for a in anns:
            total[names[a["category_id"]]] += 1

        dets = sorted(by_img.get(img_id, []), key=lambda d: -d["score"])
        if not dets and not anns:
            continue

        grle, gcls = [], []
        for a in anns:
            grle.append(maskutil.merge(maskutil.frPyObjects(a["segmentation"], h, w)))
            gcls.append(a["category_id"])

        drle = []
        for d in dets:
            s = d["segmentation"]
            drle.append(s if isinstance(s, dict)
                        else maskutil.merge(maskutil.frPyObjects(s, h, w)))

        if drle and grle:
            iou = maskutil.iou(drle, grle, [0] * len(grle))
        else:
            iou = np.zeros((len(drle), len(grle)))

        for t in THRESHOLDS:
            taken = set()
            for i, d in enumerate(dets):
                if d["score"] < t:
                    continue
                best, bj = 0.0, -1
                for j in range(len(grle)):
                    if j in taken or gcls[j] != d["category_id"]:
                        continue
                    if iou[i, j] > best:
                        best, bj = iou[i, j], j
                if bj >= 0 and best >= 0.5:
                    taken.add(bj)
                    hits[t][names[d["category_id"]]] += 1
                else:
                    fps[t] += 1

    print("--- arm %s --- %d tiles, %.1f km2" % (arm, n_img, area_km2))
    print("%-8s %10s %10s %10s %10s %14s"
          % ("score", "building", "platform", "aguada", "recall all", "FP per km2"))
    rows = []
    for t in THRESHOLDS:
        rec = {c: 100.0 * hits[t][c] / max(total[c], 1)
               for c in ("building", "platform", "aguada")}
        allrec = 100.0 * sum(hits[t].values()) / max(sum(total.values()), 1)
        fpkm = fps[t] / area_km2
        rows.append((t, rec, allrec, fpkm))
        print("%-8.2f %9.1f%% %9.1f%% %9.1f%% %9.1f%% %14.1f"
              % (t, rec["building"], rec["platform"], rec["aguada"], allrec, fpkm))
    print()
    return rows


def main():
    gt = COCO(FULL_GT)
    names = {c["id"]: c["name"] for c in gt.dataset["categories"]}
    print()
    out = {}
    for arm in ARMS:
        if os.path.isdir(os.path.join(REPO, ARMS[arm])):
            out[arm] = evaluate(arm, gt, names)

    print("=== how to read this ===")
    print("  For a candidate generator the row that matters is a LOW threshold:")
    print("  what fraction of real structures reach the expert, and how much")
    print("  triage that costs per square kilometre.")
    print()
    if "A" in out and "D" in out:
        for t_idx, t in enumerate(THRESHOLDS):
            a = out["A"][t_idx]
            d = out["D"][t_idx]
            if abs(t - 0.10) < 1e-9:
                print("  At score 0.10: arm A recall %.1f%% at %.0f FP/km2;"
                      % (a[2], a[3]))
                print("                 arm D recall %.1f%% at %.0f FP/km2."
                      % (d[2], d[3]))
                print("  Whichever finds more real structures for a comparable")
                print("  triage burden is the better regional tool, regardless")
                print("  of which has the higher AP.")


if __name__ == "__main__":
    main()
