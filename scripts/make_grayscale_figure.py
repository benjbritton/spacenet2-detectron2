#!/usr/bin/env python
"""Side-by-side figure for the grayscale ablation: same tile, both arms.

WHY THIS FIGURE AND NOT A SINGLE OVERLAY
----------------------------------------
The claim under test is that removing chroma changes almost nothing. A single
overlay of one model on colour input cannot show that -- it shows only that the
detector works. The evidence is the COMPARISON: identical tile, identical
ground truth, one panel as the colour model saw and scored it, the other as the
grayscale model saw and scored it. If the claim holds the two panels look
nearly the same, and that similarity is the argument.

The input rendering is produced by the TRAINING Stretch class, not by a display
routine, so the left panel is the colour arm's actual network input and the
right panel is the grayscale arm's -- unweighted mean of the three stretched
channels, replicated. A separate display stretch would make the panels look
different for reasons that have nothing to do with the ablation.

COLOURS, matching scripts/overlay_geotiff.py
    yellow   ground truth footprint
    cyan     predicted footprint at or above --threshold

    ./scripts/run.sh python scripts/make_grayscale_figure.py
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from detlab.datasets.spacenet import Stretch

YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)


def draw(bgr, coco_img, anns, instances, threshold):
    for a in anns:
        for seg in a["segmentation"]:
            pts = np.asarray(seg, dtype=np.float64).reshape(-1, 2)
            cv2.polylines(bgr, [np.round(pts).astype(np.int32)], True,
                          YELLOW, 1, cv2.LINE_AA)
    n = 0
    for inst in instances:
        if float(inst["score"]) < threshold:
            continue
        segm = inst.get("segmentation")
        if segm is None:
            continue
        if isinstance(segm.get("counts"), str):
            segm = dict(segm, counts=segm["counts"].encode("utf-8"))
        mask = mask_util.decode(segm)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(bgr, cnts, -1, CYAN, 1, cv2.LINE_AA)
        n += 1
    return n


def caption(panel, text, sub):
    h, w = panel.shape[:2]
    bar = np.zeros((46, w, 3), np.uint8)
    cv2.putText(bar, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (170, 170, 170), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tile", default="SN2_buildings_train_AOI_3_Paris_PS-RGB_img785.tif")
    p.add_argument("--coco", default="data/spacenet2/coco/pooled_val.json")
    p.add_argument("--root", default="data/spacenet2")
    p.add_argument("--colour-predictions",
                   default="outputs/spacenet2_r50fpn/inference/instances_predictions.pth")
    p.add_argument("--gray-predictions",
                   default="outputs/spacenet2_r50fpn_gray/inference/instances_predictions.pth")
    p.add_argument("--threshold", type=float, default=0.544)
    p.add_argument("--out", default="posts/figures/grayscale_side_by_side.png")
    a = p.parse_args()

    coco = json.load(open(a.coco))
    img = next((im for im in coco["images"]
                if os.path.basename(im["file_name"]) == a.tile), None)
    if img is None:
        raise SystemExit("tile not in %s: %s" % (a.coco, a.tile))
    anns = [an for an in coco["annotations"] if an["image_id"] == img["id"]]
    path = os.path.join(a.root, img["file_name"])
    print("tile %s  id %d  %d ground-truth buildings"
          % (a.tile, img["id"], len(anns)))

    def preds_for(pth):
        raw = torch.load(pth, map_location="cpu", weights_only=False)
        for r in raw:
            if r.get("image_id") == img["id"]:
                return r.get("instances", [])
        return []

    panels = []
    for label, gray, predpath in (
            ("COLOUR input, colour-trained model", False, a.colour_predictions),
            ("GRAYSCALE input, grayscale-trained model", True, a.gray_predictions)):
        rgb = Stretch(mode="per_image", grayscale=gray).load(path)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        n = draw(bgr, img, anns, preds_for(predpath), a.threshold)
        print("  %-42s %d detections at >= %.3f" % (label, n, a.threshold))
        panels.append(caption(bgr, label,
                              "%d detections at score >= %.3f   yellow = truth, cyan = predicted"
                              % (n, a.threshold)))

    gap = np.full((panels[0].shape[0], 8, 3), 30, np.uint8)
    fig = np.hstack([panels[0], gap, panels[1]])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, fig)
    print("wrote %s  (%d x %d)" % (a.out, fig.shape[1], fig.shape[0]))


if __name__ == "__main__":
    main()
