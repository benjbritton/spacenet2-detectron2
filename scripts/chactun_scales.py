#!/usr/bin/env python
"""Where do Chactun structures sit relative to FPN anchors, and at what input size?

Milestone B established that this comparison has to be made in NETWORK INPUT
space, not native pixels: detectron2 rescales the short edge before the anchors
ever see the image, so a structure that looks well-covered at native resolution
can fall under the smallest anchor once the real scale factor is applied.

Chactun tiles are 480x480. The default detectron2 training augmentation resizes
the short edge to 640-800, so the operative question is which input size puts
the bulk of the structures on an anchor at all.
"""
import json

import numpy as np
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)   # repo root, whatever it is called

COCO = ROOT + "/data/chactun/coco/chactun_cc.json"
ANCHORS = [32, 64, 128, 256, 512]          # detectron2 FPN default, P2..P6
PX_M2 = 0.25
NATIVE = 480


def main():
    d = json.load(open(COCO))
    byid = {c["id"]: c["name"] for c in d["categories"]}
    per = {}
    for a in d["annotations"]:
        per.setdefault(byid[a["category_id"]], []).append(a["area"])

    print("Structure size in NATIVE pixels (tiles are 480x480 at 0.5 m)")
    print()
    print("%-10s %7s %10s %10s %10s %10s %10s"
          % ("class", "n", "p10", "median", "p90", "med m2", "med px"))
    med_px = {}
    for c in ["building", "platform", "aguada"]:
        a = np.array(per[c], float)
        s = np.sqrt(a)                      # side of an equal-area square
        med_px[c] = np.median(s)
        print("%-10s %7d %10.1f %10.1f %10.1f %10.0f %10.1f"
              % (c, len(a), np.percentile(s, 10), np.median(s),
                 np.percentile(s, 90), np.median(a) * PX_M2, np.median(s)))

    print()
    print("Fraction of structures BELOW the smallest FPN anchor (%d px)"
          % ANCHORS[0])
    print("as a function of the input size the short edge is resized to:")
    print()
    print("%-10s %10s %10s %10s %10s"
          % ("input", "scale", "building", "platform", "aguada"))
    for size in [480, 640, 800, 1024, 1333]:
        sc = size / NATIVE
        row = []
        for c in ["building", "platform", "aguada"]:
            s = np.sqrt(np.array(per[c], float)) * sc
            row.append(100.0 * (s < ANCHORS[0]).mean())
        print("%-10d %10.3f %9.1f%% %9.1f%% %9.1f%%"
              % (size, sc, row[0], row[1], row[2]))

    print()
    print("Median structure in input space, against the anchor ladder %s:"
          % ANCHORS)
    print()
    print("%-10s %10s %12s %12s %12s"
          % ("input", "scale", "building", "platform", "aguada"))
    for size in [480, 640, 800, 1024]:
        sc = size / NATIVE
        print("%-10d %10.3f %12.1f %12.1f %12.1f"
              % (size, sc, med_px["building"] * sc,
                 med_px["platform"] * sc, med_px["aguada"] * sc))


if __name__ == "__main__":
    main()
