#!/usr/bin/env python
"""Are coincident cyan/yellow boxes duplicates WITHIN a run, or one from each run?

The comparison raster draws two runs together, so a building box and a platform
box on the same feature may be one from each rather than the model emitting both.
Those have different causes and different fixes, so this measures it instead of
guessing.

Also counts detections landing in nodata, which the review flagged and which is
a defect in the inference harness rather than in the model.
"""
import json
import sys

import numpy as np
import rasterio


def load(path):
    d = json.load(open(path))
    out = []
    for f in d["features"]:
        c = f["geometry"]["coordinates"][0]
        xs = [p[0] for p in c]
        ys = [p[1] for p in c]
        out.append({"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
                    "cls": f["properties"]["class"]})
    return out


def iou(a, b):
    ix = max(0.0, min(a["x1"], b["x1"]) - max(a["x0"], b["x0"]))
    iy = max(0.0, min(a["y1"], b["y1"]) - max(a["y0"], b["y0"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = ((a["x1"] - a["x0"]) * (a["y1"] - a["y0"])
          + (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) - inter)
    return inter / ua if ua > 0 else 0.0


def within(dets, name):
    n = 0
    pairs = {}
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if dets[i]["cls"] == dets[j]["cls"]:
                continue
            v = iou(dets[i], dets[j])
            if v > 0.3:
                n += 1
                k = tuple(sorted([dets[i]["cls"], dets[j]["cls"]]))
                pairs[k] = pairs.get(k, 0) + 1
    print("%-28s %d detections, %d cross-class overlaps (IoU>0.3) %s"
          % (name, len(dets), n, pairs if pairs else ""))
    return n


def across(a, b):
    n = 0
    for da in a:
        for db in b:
            if iou(da, db) > 0.3 and da["cls"] != db["cls"]:
                n += 1
    print("cross-RUN, different class, IoU>0.3: %d pairs" % n)


def nodata_hits(dets, raster, name):
    with rasterio.open(raster) as s:
        bad = 0
        for d in dets:
            cx = (d["x0"] + d["x1"]) / 2.0
            cy = (d["y0"] + d["y1"]) / 2.0
            try:
                r, c = s.index(cx, cy)
                if not (0 <= r < s.height and 0 <= c < s.width):
                    continue
                px = s.read(1, window=((r, r + 1), (c, c + 1)))
                if px.size and px[0, 0] == 0:
                    bad += 1
            except Exception:
                pass
    print("%-28s %d of %d centred on nodata (%.0f%%)"
          % (name, bad, len(dets), 100.0 * bad / max(len(dets), 1)))


if __name__ == "__main__":
    three = load("/workspace/outputs/gliht/S395_3band_matched.geojson")
    g1 = load("/workspace/outputs/gliht/S395_matched.geojson")
    print()
    within(three, "matched 3-band run:")
    within(g1, "G1 blend run:")
    print()
    across(three, g1)
    print()
    r = "/workspace/outputs/gliht/S395_matched3band.tif"
    nodata_hits(three, r, "matched 3-band:")
    nodata_hits(g1, r, "G1 blend:")
    print()
    for cls in ("building", "platform", "aguada"):
        n3 = sum(1 for d in three if d["cls"] == cls)
        ng = sum(1 for d in g1 if d["cls"] == cls)
        print("  %-10s 3-band %3d   G1 %3d" % (cls, n3, ng))
