#!/usr/bin/env python
"""Why does our conversion find 661 tiles with no annotation when the paper says
every one of the 2094 records contains at least one object?

Three candidates, and they are distinguishable:
  - the raw masks really are empty for those tiles, and the paper's claim does
    not hold for them
  - foreground exists but every component falls under the min_area filter
  - foreground exists at a size the filter keeps, and the converter is losing it
"""
import json
import os
from collections import Counter

import numpy as np
import rasterio
from scipy import ndimage

ROOT = "/w/data/chactun"
CLASSES = ["building", "platform", "aguada"]
MIN_AREA = 9

coco = json.load(open(os.path.join(ROOT, "coco", "chactun_cc.json")))
have = {a["image_id"] for a in coco["annotations"]}
tile_of = {im["id"]: im["tile"] for im in coco["images"]}
empty = sorted(tile_of[i] for i in tile_of if i not in have)
print("tiles with no annotation in our COCO: %d" % len(empty))
print()

stat = Counter()
sizes_dropped = []
for t in empty:
    any_fg = False
    any_kept = False
    for c in CLASSES:
        p = os.path.join(ROOT, "masks", "tile_%d_mask_%s.tif" % (t, c))
        if not os.path.isfile(p):
            stat["missing mask file"] += 1
            continue
        with rasterio.open(p) as s:
            v = s.read(1)
        fg = v == 0                      # object = 0, background = 255
        if not fg.any():
            continue
        any_fg = True
        lab, n = ndimage.label(fg, structure=np.ones((3, 3), bool))
        for k in range(1, n + 1):
            a = int((lab == k).sum())
            if a >= MIN_AREA:
                any_kept = True
            else:
                sizes_dropped.append(a)
    if not any_fg:
        stat["genuinely empty in all three masks"] += 1
    elif not any_kept:
        stat["foreground present, all components below min_area"] += 1
    else:
        stat["foreground present and large enough -- CONVERTER BUG"] += 1

print("%-52s %s" % ("explanation", "tiles"))
for k, v in stat.most_common():
    print("%-52s %d" % (k, v))

if sizes_dropped:
    a = np.array(sizes_dropped)
    print()
    print("components dropped by min_area=%d on these tiles: %d" % (MIN_AREA, len(a)))
    print("   sizes: min %d, median %d, max %d" % (a.min(), np.median(a), a.max()))

print()
print("=== also: what does the FULL dataset look like without the filter? ===")
tot = Counter()
tot_f = Counter()
tiles = sorted(int(n.split("_")[1]) for n in os.listdir(os.path.join(ROOT, "lidar"))
               if n.endswith("_lidar.tif"))
for t in tiles:
    for c in CLASSES:
        p = os.path.join(ROOT, "masks", "tile_%d_mask_%s.tif" % (t, c))
        if not os.path.isfile(p):
            continue
        with rasterio.open(p) as s:
            v = s.read(1)
        fg = v == 0
        if not fg.any():
            continue
        lab, n = ndimage.label(fg, structure=np.ones((3, 3), bool))
        tot[c] += n
        for k in range(1, n + 1):
            if int((lab == k).sum()) >= MIN_AREA:
                tot_f[c] += 1

print("%-10s %14s %14s %14s" % ("class", "no filter", "min_area=9", "paper Table 6"))
paper = {"building": 8275, "platform": 1996, "aguada": 51}
for c in CLASSES:
    print("%-10s %14d %14d %14d" % (c, tot[c], tot_f[c], paper[c]))
