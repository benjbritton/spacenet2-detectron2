#!/usr/bin/env python
"""Can Chactun measure high-IoU localisation quality at all?

Cascade R-CNN buys its gain at the strict end of the IoU range -- AP75 and up.
That gain is only observable if the ground truth is itself accurate to better
than the IoU threshold being tested. Chactun's ground truth is not hand-digitised
vector: it is raster masks, traced to polygons, so every boundary carries at
least half a pixel of quantisation, and the objects are small.

The consequence is arithmetic. For a shape of area A and perimeter L, moving the
whole boundary out by one pixel adds about L to the area, so the IoU between the
true shape and the one-pixel-off shape is roughly A / (A + L). A 25 px square --
the MEDIAN Chactun building -- gives 625 / 725 = 0.86. So a single pixel of
boundary disagreement already puts that object below an AP90 match, and not far
above an AP75 one.

This script computes that ceiling from the real polygons rather than a square
approximation, and reports what fraction of each class is already unmeasurable
at each IoU threshold before the model makes a single error.
"""
import json

import numpy as np

COCO = "/w/data/chactun/coco/chactun_cc.json"
THRESHOLDS = [0.5, 0.6, 0.75, 0.9]


def poly_area_perim(seg):
    """Shoelace area and perimeter over all rings of one instance."""
    A = 0.0
    L = 0.0
    for ring in seg:
        p = np.array(ring, float).reshape(-1, 2)
        if len(p) < 3:
            continue
        x, y = p[:, 0], p[:, 1]
        A += 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        d = np.diff(np.vstack([p, p[:1]]), axis=0)
        L += np.hypot(d[:, 0], d[:, 1]).sum()
    return A, L


def main():
    d = json.load(open(COCO))
    byid = {c["id"]: c["name"] for c in d["categories"]}

    per = {}
    for a in d["annotations"]:
        if isinstance(a["segmentation"], list):
            A, L = poly_area_perim(a["segmentation"])
            if A > 0 and L > 0:
                per.setdefault(byid[a["category_id"]], []).append((A, L))

    print("IoU ceiling from ONE pixel of boundary uncertainty")
    print("ceiling = A / (A + L), computed per instance from its own polygon")
    print()
    print("%-10s %8s %10s %10s %10s"
          % ("class", "n", "p10", "median", "p90"))
    ceilings = {}
    for c in ["building", "platform", "aguada"]:
        v = np.array([A / (A + L) for A, L in per[c]])
        ceilings[c] = v
        print("%-10s %8d %10.3f %10.3f %10.3f"
              % (c, len(v), np.percentile(v, 10), np.median(v),
                 np.percentile(v, 90)))

    print()
    print("Percentage of instances whose 1-pixel ceiling is ALREADY below the")
    print("IoU threshold -- i.e. a perfect model scores them as misses:")
    print()
    print("%-10s %12s %12s %12s %12s"
          % ("class", "IoU 0.50", "IoU 0.60", "IoU 0.75", "IoU 0.90"))
    for c in ["building", "platform", "aguada"]:
        v = ceilings[c]
        print("%-10s %11.1f%% %11.1f%% %11.1f%% %11.1f%%"
              % (c, 100 * (v < 0.50).mean(), 100 * (v < 0.60).mean(),
                 100 * (v < 0.75).mean(), 100 * (v < 0.90).mean()))

    # instance-weighted, since buildings dominate
    allv = np.concatenate([ceilings[c] for c in ceilings])
    print()
    print("%-10s %11.1f%% %11.1f%% %11.1f%% %11.1f%%"
          % ("ALL", 100 * (allv < 0.50).mean(), 100 * (allv < 0.60).mean(),
             100 * (allv < 0.75).mean(), 100 * (allv < 0.90).mean()))

    print()
    print("=== reading ===")
    b75 = 100 * (ceilings["building"] < 0.75).mean()
    b50 = 100 * (ceilings["building"] < 0.50).mean()
    print("  At IoU 0.50, %.1f%% of buildings are unmeasurable. At IoU 0.75,"
          % b50)
    print("  %.1f%%. Cascade R-CNN improves exactly the strict-IoU regime that" % b75)
    print("  this ground truth cannot resolve, so its headline advantage would")
    print("  land on the part of the metric that is measuring raster")
    print("  quantisation rather than the model.")


if __name__ == "__main__":
    main()
