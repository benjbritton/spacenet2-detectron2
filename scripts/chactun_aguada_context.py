#!/usr/bin/env python
"""What does the terrain around an aguada actually look like?

Copy-paste augmentation needs a placement rule, and I asserted one from first
principles: aguadas are water reservoirs, so they sit in depressions, so they
should show LOW sky-view factor. That is a claim about the data and it should be
measured before it is built into an augmentation, because if it is wrong the
constraint will place reservoirs in exactly the wrong terrain and teach the model
something false.

Measures, over every aguada in the dataset:
  - band statistics INSIDE the aguada mask against the tile as a whole
  - the same for building and platform, so aguada can be told apart from
    "any annotated thing looks different from average"
  - whether a percentile threshold on sky-view factor would actually select
    aguada-like terrain, which is what the placement rule would use
"""
import json
import os
from collections import defaultdict

import numpy as np
import rasterio
from pycocotools import mask as maskutil

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)   # repo root, whatever it is called

COCO = ROOT + "/data/chactun/coco/chactun_cc.json"
LIDAR = ROOT + "/data/chactun/lidar"
BANDS = ["sky-view factor", "positive openness", "slope"]
CLASSES = ["building", "platform", "aguada"]


def main():
    d = json.load(open(COCO))
    names = {c["id"]: c["name"] for c in d["categories"]}
    tile_of = {im["id"]: im["tile"] for im in d["images"]}
    size_of = {im["id"]: (im["height"], im["width"]) for im in d["images"]}

    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)

    inside = {c: [[] for _ in range(3)] for c in CLASSES}
    outside = [[] for _ in range(3)]
    # for the placement rule: what SVF percentile does an aguada sit at,
    # relative to its own tile?
    aguada_pctile = []

    imgs = sorted(by_img)
    for n, img_id in enumerate(imgs):
        t = tile_of[img_id]
        p = os.path.join(LIDAR, "tile_%d_lidar.tif" % t)
        with rasterio.open(p) as src:
            arr = src.read().astype(np.float32)
        h, w = size_of[img_id]

        anymask = np.zeros((h, w), bool)
        for a in by_img[img_id]:
            rle = maskutil.merge(maskutil.frPyObjects(a["segmentation"], h, w))
            m = maskutil.decode(rle).astype(bool)
            if not m.any():
                continue
            anymask |= m
            cname = names[a["category_id"]]
            for b in range(3):
                inside[cname][b].append(float(arr[b][m].mean()))
            if cname == "aguada":
                svf = arr[0]
                aguada_pctile.append(
                    float((svf < svf[m].mean()).mean() * 100))
        for b in range(3):
            if (~anymask).any():
                outside[b].append(float(arr[b][~anymask].mean()))
        if (n + 1) % 500 == 0:
            print("  %d/%d tiles" % (n + 1, len(imgs)))

    print()
    print("Mean band value INSIDE each class, against unannotated terrain")
    print()
    print("%-22s %16s %18s %10s" % ("band", "unannotated", "class", "mean"))
    for b in range(3):
        base = np.mean(outside[b])
        print("%-22s %16.2f" % (BANDS[b], base))
        for c in CLASSES:
            v = np.mean(inside[c][b])
            print("%-22s %16s %18s %10.2f  (%+.2f)"
                  % ("", "", c, v, v - base))
        print()

    print("=== the placement rule ===")
    ap = np.array(aguada_pctile)
    print("  An aguada sits at the %.1fth percentile of its own tile's sky-view"
          % np.mean(ap))
    print("  factor on average (median %.1f, p10 %.1f, p90 %.1f)."
          % (np.median(ap), np.percentile(ap, 10), np.percentile(ap, 90)))
    print()
    if np.mean(ap) < 40:
        print("  CONFIRMED: aguadas occupy locally DARK, low sky-view terrain,")
        print("  which is what a depression looks like. Constraining pastes to")
        print("  low-SVF regions is justified by the data.")
    elif np.mean(ap) > 60:
        print("  REFUTED, and in the opposite direction: aguadas sit in locally")
        print("  BRIGHT terrain. A low-SVF constraint would place them exactly")
        print("  wrong.")
    else:
        print("  NOT SUPPORTED: aguadas sit near the middle of their tile's SVF")
        print("  distribution, so sky-view factor does not identify aguada")
        print("  terrain and a constraint built on it would be arbitrary.")
        print("  Uniform placement is then the honest default, and the")
        print("  topographic-plausibility argument should be dropped rather")
        print("  than implemented on a hunch.")


if __name__ == "__main__":
    main()
