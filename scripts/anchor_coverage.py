#!/usr/bin/env python
"""Two questions the blog post leaves open, both cheap to settle.

1. ANCHOR COVERAGE. The post names anchor scale as the suspect for Khartoum's
   missed proposals but does not test it. detectron2's default FPN anchor
   generator uses sizes [32, 64, 128, 256, 512] across P2-P6 with aspect ratios
   [0.5, 1.0, 2.0]. If Khartoum's footprints sit below the smallest anchor, the
   proposal stage cannot generate a box at the right scale and the failure is
   structural rather than learned.

   CRITICAL DETAIL: the comparison must be done in NETWORK INPUT space, not tile
   space. Tiles are 650 px native and MIN_SIZE_TEST is 800, so every box is
   scaled by 800/650 = 1.231 before the anchors ever see it. Comparing raw tile
   pixels against anchor sizes would understate coverage by 23% and manufacture
   a finding.

2. ACQUISITION METADATA. Off-nadir angle and ground sample distance vary between
   WorldView-3 collects and both drive boundary blur in dense low-rise scenes.
   The post analyses hue, brightness, boundary, shadow, size and density without
   establishing whether look angle was controlled across cities. Report what the
   rasters actually carry rather than asserting either way.
"""
import json
import os
import sys

import numpy as np
import rasterio

CITIES = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]
ROOT = "data/spacenet2"
COCO = os.path.join(ROOT, "coco")

ANCHOR_SIZES = [32.0, 64.0, 128.0, 256.0, 512.0]
ASPECTS = [0.5, 1.0, 2.0]
NATIVE = 650.0
TEST_SIZE = 800.0
SCALE = TEST_SIZE / NATIVE


def best_anchor_iou(w, h):
    """Max IoU between a GT box and any anchor, both centred at the origin.

    Anchors and GT are axis-aligned and centre-aligned here, so IoU depends only
    on the two shapes. This is the standard way to ask whether the anchor set can
    represent an object at all, independent of where it sits in the image.
    """
    best = 0.0
    ga = w * h
    for s in ANCHOR_SIZES:
        for ar in ASPECTS:
            # detectron2: anchor area = s^2, aspect = h/w
            aw = s / np.sqrt(ar)
            ah = s * np.sqrt(ar)
            inter = min(w, aw) * min(h, ah)
            iou = inter / (ga + aw * ah - inter)
            best = max(best, iou)
    return best


print("=" * 78)
print("1. ANCHOR COVERAGE  (boxes scaled by %.3f into network input space)" % SCALE)
print("=" * 78)
print("%-16s %8s %8s %8s %8s %8s %8s" %
      ("city", "n", "p10", "median", "p90", "IoU<0.3", "IoU<0.5"))

out = {}
for city in CITIES:
    path = os.path.join(COCO, "%s_train.json" % city)
    if not os.path.isfile(path):
        continue
    with open(path) as f:
        d = json.load(f)
    ws, hs, ious = [], [], []
    for a in d["annotations"]:
        w, h = a["bbox"][2] * SCALE, a["bbox"][3] * SCALE
        if w <= 1 or h <= 1:
            continue
        ws.append(w)
        hs.append(h)
        ious.append(best_anchor_iou(w, h))
    ws, hs, ious = np.array(ws), np.array(hs), np.array(ious)
    side = np.sqrt(ws * hs)
    ar = hs / ws
    out[city] = {
        "n": int(side.size),
        "side_p10": float(np.percentile(side, 10)),
        "side_median": float(np.median(side)),
        "side_p90": float(np.percentile(side, 90)),
        "frac_iou_lt_0.3": float((ious < 0.3).mean()),
        "frac_iou_lt_0.5": float((ious < 0.5).mean()),
        "ar_p10": float(np.percentile(ar, 10)),
        "ar_median": float(np.median(ar)),
        "ar_p90": float(np.percentile(ar, 90)),
        "frac_below_smallest_anchor": float((side < ANCHOR_SIZES[0]).mean()),
    }
    print("%-16s %8d %8.1f %8.1f %8.1f %7.1f%% %7.1f%%" %
          (city, side.size, out[city]["side_p10"], out[city]["side_median"],
           out[city]["side_p90"], 100 * out[city]["frac_iou_lt_0.3"],
           100 * out[city]["frac_iou_lt_0.5"]))

print()
print("%-16s %10s %10s %10s   %s" %
      ("city", "AR p10", "AR med", "AR p90", "% smaller than the 32px anchor"))
for city in CITIES:
    if city not in out:
        continue
    r = out[city]
    print("%-16s %10.2f %10.2f %10.2f   %25.1f%%" %
          (city, r["ar_p10"], r["ar_median"], r["ar_p90"],
           100 * r["frac_below_smallest_anchor"]))

print()
print("anchor sizes %s, aspect ratios %s" % (ANCHOR_SIZES, ASPECTS))
print("IoU<0.3 is the fraction of footprints NO anchor can represent well.")
print("detectron2 RPN labels an anchor positive at IoU >= 0.7, negative below 0.3.")

print()
print("=" * 78)
print("2. ACQUISITION METADATA IN THE RASTERS")
print("=" * 78)
for city in CITIES:
    d = os.path.join(ROOT, city, "PS-RGB")
    if not os.path.isdir(d):
        continue
    name = sorted(os.listdir(d))[0]
    with rasterio.open(os.path.join(d, name)) as src:
        tags = src.tags()
        b = src.bounds
        # degrees -> metres, crude but adequate for a GSD sanity check
        lat = (b.bottom + b.top) / 2.0
        gsd_x = src.res[0] * 111320.0 * np.cos(np.deg2rad(lat))
        gsd_y = src.res[1] * 110540.0
        print("%-16s size %sx%s  GSD ~%.2f x %.2f m  lat %.2f" %
              (city, src.width, src.height, gsd_x, gsd_y, lat))
        interesting = {k: v for k, v in tags.items()
                       if any(t in k.upper() for t in
                              ("NADIR", "ANGLE", "AZIMUTH", "ELEV", "SUN",
                               "ACQ", "DATE", "SAT", "CATALOG", "TARGET"))}
        print("%-16s tags: %s" % ("", interesting if interesting else
                                  "none carrying acquisition geometry"))
        if src.tags(ns="IMAGE_STRUCTURE"):
            print("%-16s image_structure: %s" % ("", src.tags(ns="IMAGE_STRUCTURE")))

os.makedirs("outputs/city_analysis", exist_ok=True)
with open("outputs/city_analysis/anchor_coverage.json", "w") as f:
    json.dump(out, f, indent=1)
print()
print("wrote outputs/city_analysis/anchor_coverage.json")
