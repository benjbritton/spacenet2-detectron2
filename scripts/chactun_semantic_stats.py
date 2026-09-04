#!/usr/bin/env python
"""Measure what a semantic formulation of Chactun has to represent.

TWO QUESTIONS THIS SETTLES BEFORE ANY MODEL IS BUILT

1. Are the three classes mutually exclusive? If they are, the task is ordinary
   multi-class segmentation with a softmax. If they are not, a softmax is the
   wrong head: it would force a choice where the ground truth asserts both, and
   the resulting model could not reproduce the ground truth even in principle.

2. How rare is each class by PIXEL? Instance counts say aguada is rare; a
   segmentation loss sees pixels, and a class that is rare in pixels needs its
   loss weighting stated rather than defaulted.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

CLASSES = ["building", "platform", "aguada"]
MASKS = "data/chactun/masks"


def gt(tile, cls):
    return np.array(Image.open(os.path.join(
        MASKS, "tile_%d_mask_%s.tif" % (tile, cls)))) == 0


def main():
    split = json.load(open("data/chactun/splits/canonical_challenge.json"))
    # folds[0] is train (tiles 0-1764), folds[1] is the challenge test set
    tiles = sorted(split["folds"][0])
    print("train tiles: %d" % len(tiles))

    px = {c: 0 for c in CLASSES}
    inter = {"building&platform": 0, "building&aguada": 0, "platform&aguada": 0}
    total = 0
    ntiles_with = {c: 0 for c in CLASSES}

    for i, t in enumerate(tiles):
        m = {c: gt(t, c) for c in CLASSES}
        h, w = m["building"].shape
        total += h * w
        for c in CLASSES:
            s = int(m[c].sum())
            px[c] += s
            if s:
                ntiles_with[c] += 1
        inter["building&platform"] += int((m["building"] & m["platform"]).sum())
        inter["building&aguada"] += int((m["building"] & m["aguada"]).sum())
        inter["platform&aguada"] += int((m["platform"] & m["aguada"]).sum())
        if (i + 1) % 400 == 0:
            print("  ...%d" % (i + 1))

    print()
    print("PIXEL FREQUENCY (of %.1f M pixels)" % (total / 1e6))
    for c in CLASSES:
        print("  %-9s %10d  %6.3f%% of pixels   present in %4d tiles"
              % (c, px[c], 100.0 * px[c] / total, ntiles_with[c]))
        if px[c]:
            print("             BCE pos_weight if balanced: %.1f"
                  % ((total - px[c]) / px[c]))

    print()
    print("CLASS OVERLAP")
    for k, v in inter.items():
        a, b = k.split("&")
        print("  %-20s %10d px = %5.1f%% of %s, %5.1f%% of %s"
              % (k, v, 100.0 * v / max(px[a], 1), a,
                 100.0 * v / max(px[b], 1), b))

    excl = inter["building&platform"] + inter["building&aguada"] + inter["platform&aguada"]
    print()
    if excl > 0.01 * sum(px.values()):
        print("VERDICT: the classes are NOT mutually exclusive. A softmax head")
        print("cannot represent this ground truth. Use three independent sigmoid")
        print("channels -- which is also the form the challenge scored, one")
        print("binary raster per class per tile.")
    else:
        print("VERDICT: overlap is negligible; a softmax head is defensible.")


if __name__ == "__main__":
    main()
