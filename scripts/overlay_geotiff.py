#!/usr/bin/env python
"""Burn ground truth and predictions onto tiles, written back as GeoTIFFs.

The companion to export_predictions_geojson.py. That one keeps the score
adjustable; this one produces a self-contained picture that lands in the right
place on a map and needs no join, no symbology and no threshold decision by the
viewer -- the right artefact for a report figure or a quick sanity sweep.

WHAT THE COLOURS MEAN (BGR order internally, described here as seen)
    yellow   ground truth footprint
    cyan     predicted footprint at or above --threshold
    magenta  predicted bounding box (with --boxes)
Overlap reads as green-ish where cyan sits on yellow, so a well-matched tile
looks green and errors stand out in pure yellow (missed) or pure cyan (false).

The stretch here is DISPLAY ONLY and deliberately independent of the training
preprocessing in detlab.datasets.spacenet -- this is for eyes, not for the
network, and confusing the two is how a rendering choice turns into a claim.

    ./scripts/run.sh python scripts/overlay_geotiff.py \
        --predictions outputs/spacenet2_r50fpn/inference/instances_predictions.pth \
        --threshold 0.544 --per-city 4 --boxes
"""
import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import rasterio
import torch
from pycocotools import mask as mask_util

YELLOW = (0, 255, 255)     # ground truth
CYAN = (255, 255, 0)       # prediction mask
MAGENTA = (255, 0, 255)    # prediction bbox


def display_rgb(path, low=2.0, high=98.0):
    """UInt16 tile -> uint8 BGR for drawing, plus its profile.

    Percentile stretch per band. The imagery is 11-bit in a 16-bit container
    (max 2047 of 65535), so an unstretched read renders essentially black.
    """
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32)
        profile = src.profile
    lo = np.percentile(arr, low, axis=(1, 2), keepdims=True)
    hi = np.percentile(arr, high, axis=(1, 2), keepdims=True)
    rgb = np.clip((arr - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb = (rgb * 255).astype(np.uint8).transpose(1, 2, 0)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), profile


def draw_gt(bgr, anns):
    for a in anns:
        for seg in a["segmentation"]:
            pts = np.asarray(seg, dtype=np.float64).reshape(-1, 2)
            cv2.polylines(bgr, [np.round(pts).astype(np.int32)], True,
                          YELLOW, 1, cv2.LINE_AA)


def draw_preds(bgr, instances, threshold, boxes):
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
        if boxes:
            x, y, w, h = inst["bbox"]
            cv2.rectangle(bgr, (int(x), int(y)), (int(x + w), int(y + h)),
                          MAGENTA, 1)
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions",
                   default="outputs/spacenet2_r50fpn/inference/instances_predictions.pth")
    p.add_argument("--coco", default="data/spacenet2/coco/pooled_val.json")
    p.add_argument("--root", default="data/spacenet2")
    p.add_argument("--out-dir", default="outputs/overlay_geotiff")
    p.add_argument("--threshold", type=float, default=0.544,
                   help="score cutoff for drawing a prediction. Default is the "
                        "train-selected reporting threshold, so the picture "
                        "shows what the reported F1 actually scored")
    p.add_argument("--per-city", type=int, default=4,
                   help="tiles per AOI, spread across the building-count range")
    p.add_argument("--aoi", default=None, help="one AOI only")
    p.add_argument("--tiles", default=None,
                   help="comma-separated tile filenames, overrides --per-city")
    p.add_argument("--boxes", action="store_true", help="also draw bboxes")
    p.add_argument("--no-gt", action="store_true", help="predictions only")
    p.add_argument("--png", action="store_true",
                   help="also write a plain PNG beside each GeoTIFF. Windows "
                        "Photos and Explorer previews do not render GeoTIFF "
                        "reliably; the PNG is for eyeballing, the GeoTIFF for "
                        "the GIS")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.coco) as f:
        coco = json.load(f)
    info = {im["id"]: im for im in coco["images"]}
    gt_by_img = defaultdict(list)
    for a in coco["annotations"]:
        gt_by_img[a["image_id"]].append(a)

    preds = torch.load(args.predictions, weights_only=False)
    pred_by_img = {e["image_id"]: e.get("instances", []) for e in preds}

    # Choose tiles: evenly spaced through each city's building-count ranking, so
    # the sample spans dense and sparse rather than showing four of the same.
    wanted = []
    if args.tiles:
        names = {t.strip() for t in args.tiles.split(",") if t.strip()}
        wanted = [i for i, im in info.items()
                  if os.path.basename(im["file_name"]) in names]
    else:
        by_aoi = defaultdict(list)
        for image_id, im in info.items():
            if args.aoi and im["aoi"] != args.aoi:
                continue
            by_aoi[im["aoi"]].append((len(gt_by_img.get(image_id, [])), image_id))
        for aoi in sorted(by_aoi):
            ranked = sorted(by_aoi[aoi], reverse=True)
            k = min(args.per_city, len(ranked))
            idx = np.linspace(0, len(ranked) - 1, k).round().astype(int)
            wanted.extend(ranked[i][1] for i in idx)

    print("%-52s %5s %5s" % ("output", "gt", "pred"))
    for image_id in wanted:
        im = info[image_id]
        path = os.path.join(args.root, im["file_name"])
        bgr, profile = display_rgb(path)

        anns = gt_by_img.get(image_id, [])
        if not args.no_gt:
            draw_gt(bgr, anns)
        n_pred = draw_preds(bgr, pred_by_img.get(image_id, []),
                            args.threshold, args.boxes)

        # Back to RGB band order, and inherit the source georeferencing so the
        # overlay lands in place rather than as a floating picture.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        profile.update(dtype="uint8", count=3, compress="lzw",
                       photometric="rgb", nodata=None)
        out = os.path.join(
            args.out_dir,
            "%s_overlay.tif" % os.path.basename(im["file_name"])[:-4])
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(rgb)
        if args.png:
            cv2.imwrite(out[:-4] + ".png", bgr)
        print("%-52s %5d %5d" % (os.path.basename(out), len(anns), n_pred))

    print("wrote %d GeoTIFFs to %s  (threshold %.3f)"
          % (len(wanted), args.out_dir, args.threshold))


if __name__ == "__main__":
    main()
