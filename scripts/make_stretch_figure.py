#!/usr/bin/env python
"""What the published stretch changed, on one patch of ground.

WHY THIS FIGURE
---------------
The transfer to G-LiHT was attempted three times on identical terrain, and the
only thing that differed was how the DEM was turned into three bytes:

  RVT, Table 3 stretch   the recipe published in the dataset paper
  G1 composite as-is     the survey's own delivered product
  RVT, mean/sd matched   band STATISTICS matched to Chactun, not the function

Counts over the whole strip are 471 / 235 / 38. A count is not a quality
judgement and this tile carries no annotations, so the figure exists to let a
reader judge the detections by eye rather than to argue from the counts.

Each panel shows its OWN input raster, not a shared basemap. That is the point:
the model was handed three different images of the same hillside.

NODATA IS CROPPED. The G-LiHT strip runs diagonally, so a square map window
around any group includes a large black wedge. The valid-data bounding box is
computed across all panels and applied to all of them, so the panels stay
directly comparable while the wasted space goes.

    ./scripts/run.sh python scripts/make_stretch_figure.py
    ./scripts/run.sh python scripts/make_stretch_figure.py --order g1,spec
"""
import argparse
import json
import os

import cv2
import numpy as np
import rasterio
from rasterio.windows import from_bounds

COLOR = {"building": (255, 255, 0),    # cyan
         "platform": (0, 215, 255),    # amber
         "aguada": (60, 60, 255)}      # red -- NOT pink: the mean/sd matched
                                       # raster renders pink and a pink box on
                                       # it is invisible
ORDER = ["aguada", "platform", "building"]

PANELS = {
    "spec": ("RVT, Table 3 stretch", "S395_spec_0p5m.tif",
             "S395_spec_fixed.geojson", "the recipe published in the dataset paper"),
    "g1": ("G1 composite, as delivered", "S395_G1_0p5m.tif",
           "S395_g1_fixed.geojson", "the survey's own product, unmodified"),
    "matched": ("RVT, mean/sd matched", "S395_matched_0p5m.tif",
                "S395_matched_fixed.geojson",
                "statistics matched, not the function"),
}


def crop(path, cx, cy, halfw, halfh, stretch=True):
    with rasterio.open(path) as src:
        win = from_bounds(cx - halfw, cy - halfh, cx + halfw, cy + halfh,
                          src.transform)
        arr = src.read(window=win, boundless=True, fill_value=0)
        tr = src.window_transform(win)
    rgb = (np.transpose(arr[:3], (1, 2, 0)) if arr.shape[0] >= 3
           else np.stack([arr[0]] * 3, -1)).astype(np.float32)
    if stretch:
        # DISPLAY ONLY, and deliberately not the training preprocessing: the
        # delivered composite renders flat without it. Nodata (all-zero) is
        # held out of the percentiles so the black wedge does not drag the
        # low end down and grey out the terrain.
        v = rgb.max(axis=2) > 8
        if v.any():
            lo = np.percentile(rgb[v], 2.0)
            hi = np.percentile(rgb[v], 98.0)
            out = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
            out[~v] = 0
            rgb = out
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR), tr


def draw(bgr, tr, gj, cx, cy, halfw, halfh):
    inv = ~tr
    n = 0
    feats = json.load(open(gj))["features"]
    for c in ORDER:
        for f in [x for x in feats if x["properties"].get("class") == c]:
            ring = f["geometry"]["coordinates"][0]
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            if max(xs) < cx - halfw or min(xs) > cx + halfw:
                continue
            if max(ys) < cy - halfh or min(ys) > cy + halfh:
                continue
            pts = np.array([inv * (x, y) for x, y in ring], np.float32)
            cv2.polylines(bgr, [np.round(pts).astype(np.int32)], True,
                          COLOR[c], 2, cv2.LINE_AA)
            n += 1
    return n


def valid_bbox(panels, thresh=8):
    """Rows/cols where ANY panel carries data. Shared, so panels stay aligned."""
    acc = None
    for p in panels:
        v = p.max(axis=2) > thresh
        acc = v if acc is None else (acc & v)
    rows = np.where(acc.any(axis=1))[0]
    cols = np.where(acc.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return 0, acc.shape[0], 0, acc.shape[1]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def annotate(panel, res_m, fs):
    """North arrow (imagery is north-up) and a scale bar, lower left."""
    h, w = panel.shape[:2]
    pad = int(18 * fs)

    # scale bar: a round number of metres that fits comfortably
    for metres in (200, 100, 50, 25):
        px = metres / res_m
        if px < w * 0.34:
            break
    px = int(metres / res_m)
    y = h - pad
    x0 = pad
    cv2.rectangle(panel, (x0 - 6, y - int(26 * fs)),
                  (x0 + px + 6, y + 8), (0, 0, 0), -1)
    cv2.line(panel, (x0, y), (x0 + px, y), (255, 255, 255), max(2, int(3 * fs)))
    for xt in (x0, x0 + px):
        cv2.line(panel, (xt, y - int(7 * fs)), (xt, y + int(7 * fs)),
                 (255, 255, 255), max(2, int(3 * fs)))
    cv2.putText(panel, "%d m" % metres, (x0, y - int(11 * fs)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs, (255, 255, 255),
                max(1, int(2 * fs)), cv2.LINE_AA)

    # north arrow, straight up
    ax = x0 + px + int(52 * fs)
    ay = y
    tip = ay - int(52 * fs)
    cv2.rectangle(panel, (ax - int(20 * fs), tip - int(22 * fs)),
                  (ax + int(20 * fs), ay + 8), (0, 0, 0), -1)
    cv2.arrowedLine(panel, (ax, ay), (ax, tip), (255, 255, 255),
                    max(2, int(3 * fs)), cv2.LINE_AA, tipLength=0.42)
    cv2.putText(panel, "N", (ax - int(9 * fs), tip - int(7 * fs)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68 * fs, (255, 255, 255),
                max(1, int(2 * fs)), cv2.LINE_AA)
    return panel


def caption(panel, title, sub, count, fs):
    w = panel.shape[1]
    bar = np.zeros((int(58 * fs), w, 3), np.uint8)
    cv2.putText(bar, title, (14, int(26 * fs)), cv2.FONT_HERSHEY_SIMPLEX,
                0.82 * fs, (255, 255, 255), max(1, int(2 * fs)), cv2.LINE_AA)
    cv2.putText(bar, "%s   -   %d detections" % (sub, count), (14, int(48 * fs)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56 * fs, (185, 185, 185),
                max(1, int(1 * fs)), cv2.LINE_AA)
    return np.vstack([bar, panel])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/gliht_spec")
    p.add_argument("--easting", type=float, default=751950.0)
    p.add_argument("--northing", type=float, default=2096550.0)
    p.add_argument("--extent-m", type=float, default=460.0,
                   help="window width in metres")
    p.add_argument("--height-m", type=float, default=None,
                   help="window height in metres; defaults to --extent-m")
    p.add_argument("--basemap", default="g1",
                   help="'own' = each panel on its own input raster; or a panel "
                        "key (g1/spec/matched) to put every detection set on one "
                        "shared basemap, which is what makes the SETS comparable")
    p.add_argument("--order", default="spec,g1,matched",
                   help="comma-separated panel keys, left to right")
    p.add_argument("--font-scale", type=float, default=1.0)
    p.add_argument("--out", default="posts/figures/gliht_stretch_comparison.png")
    a = p.parse_args()

    keys = [k.strip() for k in a.order.split(",") if k.strip()]
    halfw = a.extent_m / 2.0
    halfh = (a.height_m if a.height_m else a.extent_m) / 2.0

    raws, metas = [], []
    for k in keys:
        title, ras, gj, sub = PANELS[k]
        base = ras if a.basemap == "own" else PANELS[a.basemap][1]
        bgr, tr = crop(os.path.join(a.dir, base), a.easting, a.northing,
                       halfw, halfh)
        n = draw(bgr, tr, os.path.join(a.dir, gj), a.easting, a.northing,
                 halfw, halfh)
        annotate(bgr, abs(tr.a), a.font_scale)
        print("%-28s %3d detections in view" % (title, n))
        raws.append(bgr)
        metas.append((title, sub, n))

    r0, r1, c0, c1 = valid_bbox(raws)
    before = raws[0].shape
    raws = [x[r0:r1, c0:c1] for x in raws]
    print("cropped nodata: %dx%d -> %dx%d (%.0f%% of pixels kept)"
          % (before[1], before[0], raws[0].shape[1], raws[0].shape[0],
             100.0 * raws[0].size / max(1, before[0] * before[1] * 3)))

    fs = a.font_scale
    out = [caption(r, t, s, n, fs) for r, (t, s, n) in zip(raws, metas)]
    h = min(x.shape[0] for x in out)
    out = [x[:h] for x in out]
    gap = np.full((h, 12, 3), 40, np.uint8)
    fig = out[0]
    for x in out[1:]:
        fig = np.hstack([fig, gap, x])


    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, fig)
    print("wrote %s  (%d x %d)" % (a.out, fig.shape[1], fig.shape[0]))


if __name__ == "__main__":
    main()
