#!/usr/bin/env python
"""Draw the detections over the G-LiHT tile so a human can judge them.

Numbers cannot answer "would this help me find stuff". Someone who knows the
ground has to look at where the boxes landed. This renders an overview of the
whole tile plus close crops on the densest clusters, with both normalisation
modes drawn together so the difference between them is visible rather than
inferred from counts.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

COLOURS = {"building": "#00e5ff", "platform": "#ffd60a", "aguada": "#ff2d55"}


def load_geojson(path):
    if not os.path.isfile(path):
        return []
    d = json.load(open(path))
    out = []
    for f in d["features"]:
        c = f["geometry"]["coordinates"][0]
        xs = [p[0] for p in c]
        ys = [p[1] for p in c]
        out.append({"x0": min(xs), "x1": max(xs), "y0": min(ys), "y1": max(ys),
                    "cls": f["properties"]["class"],
                    "score": f["properties"]["score"]})
    return out


def to_px(src, x, y):
    r, c = src.index(x, y)
    return c, r


def draw(ax, src, dets, col_off, row_off, scale, lw=1.2, style="-"):
    for d in dets:
        c0, r0 = to_px(src, d["x0"], d["y1"])     # y1 is north = smaller row
        c1, r1 = to_px(src, d["x1"], d["y0"])
        x = (min(c0, c1) - col_off) * scale
        y = (min(r0, r1) - row_off) * scale
        w = abs(c1 - c0) * scale
        h = abs(r1 - r0) * scale
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False,
                                        edgecolor=COLOURS.get(d["cls"], "w"),
                                        linewidth=lw, linestyle=style))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raster", required=True)
    p.add_argument("--matched", required=True)
    p.add_argument("--fixed", default=None)
    p.add_argument("--out-dir", default="/workspace/figures")
    p.add_argument("--crops", type=int, default=4)
    a = p.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    src = rasterio.open(a.raster)
    m = load_geojson(a.matched)
    f = load_geojson(a.fixed) if a.fixed else []
    print("matched %d, fixed %d detections" % (len(m), len(f)))

    # ---------- overview ----------
    step = max(1, src.height // 4000)
    ov = src.read(1, out_shape=(src.height // step, src.width // step))
    scale = 1.0 / step
    fig_h = min(60, max(8, ov.shape[0] / 100))
    fig, ax = plt.subplots(figsize=(ov.shape[1] / 100 * 3, fig_h))
    ax.imshow(ov, cmap="gray", vmin=0, vmax=200)
    draw(ax, src, m, 0, 0, scale, lw=0.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("South GLAS 395, matched normalisation: %d detections over "
                 "4.5 km2" % len(m), fontsize=11)
    handles = [mpatches.Patch(color=v, label=k) for k, v in COLOURS.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(a.out_dir, "gliht_S395_overview.png")
    fig.savefig(p1, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p1)

    # ---------- crops on the densest clusters ----------
    if not m:
        return
    # grid the detections and take the busiest cells
    cell = 300.0                                   # metres
    buckets = {}
    for d in m:
        key = (int(d["x0"] // cell), int(d["y0"] // cell))
        buckets.setdefault(key, []).append(d)
    busiest = sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:a.crops]

    n = len(busiest)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.6))
    if n == 1:
        axes = [axes]
    for ax, ((gx, gy), ds) in zip(axes, busiest):
        cx = (gx + 0.5) * cell
        cy = (gy + 0.5) * cell
        col, row = to_px(src, cx, cy)
        half = int(round(200 / abs(src.transform.a)))   # 400 m across
        c0 = max(0, col - half); r0 = max(0, row - half)
        w = min(2 * half, src.width - c0); h = min(2 * half, src.height - r0)
        sub = src.read(1, window=Window(c0, r0, w, h))
        ax.imshow(sub, cmap="gray", vmin=0, vmax=200)
        draw(ax, src, m, c0, r0, 1.0, lw=1.6)
        if f:
            draw(ax, src, f, c0, r0, 1.0, lw=1.0, style=":")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("%d detections\n%.0f E, %.0f N" % (len(ds), cx, cy),
                     fontsize=9)
    fig.suptitle("Densest clusters, 400 m across. Solid = matched "
                 "normalisation, dotted = Chactun constants.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p2 = os.path.join(a.out_dir, "gliht_S395_clusters.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    print("wrote", p2)
    src.close()


if __name__ == "__main__":
    main()
