#!/usr/bin/env python
"""Show why aguadas are invisible and buildings are not.

The measurement says aguadas differ from unannotated terrain by about 2 counts
in every band while buildings differ by 45 to 60. That is a claim worth being
able to SEE, both as a check on the arithmetic and because it is the figure that
explains the whole aguada result.

Each row is one annotated instance, centred, with its outline drawn. Columns are
the three-band composite and then each band alone, so it is visible which
visualisation carries the object and which does not.
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Rectangle
from pycocotools import mask as maskutil

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)   # repo root, whatever it is called

COCO = ROOT + "/data/chactun/coco/chactun_cc.json"
LIDAR = ROOT + "/data/chactun/lidar"
OUT = ROOT + "/figures"
BANDS = ["sky-view factor", "positive openness", "slope"]
CROP = 200


def load(tile):
    with rasterio.open(os.path.join(LIDAR, "tile_%d_lidar.tif" % tile)) as s:
        return s.read().astype(np.float32)


def crop_around(arr, mask, cy, cx, half):
    h, w = mask.shape
    y0 = max(0, min(h - 2 * half, cy - half))
    x0 = max(0, min(w - 2 * half, cx - half))
    return (arr[:, y0:y0 + 2 * half, x0:x0 + 2 * half],
            mask[y0:y0 + 2 * half, x0:x0 + 2 * half])


def pick(anns, names, cls, n, by_img, size_of, tile_of, want_big=True):
    cand = [a for a in anns if names[a["category_id"]] == cls]
    cand.sort(key=lambda a: -a["area"] if want_big else a["area"])
    # spread across tiles rather than showing one cluster
    seen, out = set(), []
    for a in cand:
        if a["image_id"] in seen:
            continue
        seen.add(a["image_id"])
        out.append(a)
        if len(out) == n:
            break
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    d = json.load(open(COCO))
    names = {c["id"]: c["name"] for c in d["categories"]}
    tile_of = {im["id"]: im["tile"] for im in d["images"]}
    size_of = {im["id"]: (im["height"], im["width"]) for im in d["images"]}
    by_img = defaultdict(list)
    for a in d["annotations"]:
        by_img[a["image_id"]].append(a)

    rows = []
    for cls, n in (("aguada", 3), ("building", 3)):
        rows += [(cls, a) for a in pick(d["annotations"], names, cls, n,
                                        by_img, size_of, tile_of)]

    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.1 * len(rows)))
    for r, (cls, a) in enumerate(rows):
        img_id = a["image_id"]
        h, w = size_of[img_id]
        arr = load(tile_of[img_id])
        rle = maskutil.merge(maskutil.frPyObjects(a["segmentation"], h, w))
        m = maskutil.decode(rle).astype(bool)
        ys, xs = np.nonzero(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        half = max(CROP // 2, int(max(ys.ptp(), xs.ptp()) * 0.8))
        half = min(half, min(h, w) // 2)
        sub, submask = crop_around(arr, m, cy, cx, half)

        # contrast of THIS instance against its own tile
        inside = [arr[b][m].mean() for b in range(3)]
        outside = [arr[b][~m].mean() for b in range(3)]

        panels = [("3-band composite", None)] + [(BANDS[b], b) for b in range(3)]
        for c, (title, b) in enumerate(panels):
            ax = axes[r, c]
            if b is None:
                comp = np.stack([sub[i] for i in range(3)], -1) / 255.0
                ax.imshow(np.clip(comp, 0, 1))
            else:
                ax.imshow(sub[b], cmap="gray", vmin=0, vmax=255)
            ax.contour(submask, levels=[0.5], colors=["#ff2d55"], linewidths=1.6)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel("%s\ntile %d" % (cls, tile_of[img_id]),
                              fontsize=10)
            if b is not None:
                ax.set_xlabel("in %.0f  vs  out %.0f   (%+.0f)"
                              % (inside[b], outside[b], inside[b] - outside[b]),
                              fontsize=8)

    fig.suptitle("Why aguadas are hard: they have almost no relief signature\n"
                 "outline = annotation. Buildings differ from surrounding "
                 "terrain by 45-60 counts; aguadas by about 2.",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    p = os.path.join(OUT, "chactun_aguada_vs_building.png")
    fig.savefig(p, dpi=130)
    print("wrote", p)

    # second figure: the distributions behind the claim
    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 3.6))
    vals = {c: [[] for _ in range(3)] for c in ("building", "platform", "aguada")}
    bg = [[] for _ in range(3)]
    imgs = sorted(by_img)
    for img_id in imgs[:600]:
        h, w = size_of[img_id]
        arr = load(tile_of[img_id])
        anymask = np.zeros((h, w), bool)
        for a in by_img[img_id]:
            rle = maskutil.merge(maskutil.frPyObjects(a["segmentation"], h, w))
            mm = maskutil.decode(rle).astype(bool)
            if not mm.any():
                continue
            anymask |= mm
            for b in range(3):
                vals[names[a["category_id"]]][b].append(arr[b][mm].mean())
        for b in range(3):
            if (~anymask).any():
                bg[b].append(arr[b][~anymask].mean())

    for b in range(3):
        ax = axes2[b]
        ax.hist(bg[b], bins=40, alpha=0.45, label="unannotated", color="#888888",
                density=True)
        for c, col in (("building", "#0a84ff"), ("aguada", "#ff2d55")):
            if vals[c][b]:
                ax.hist(vals[c][b], bins=30, alpha=0.55, label=c, color=col,
                        density=True)
        ax.set_title(BANDS[b], fontsize=10)
        ax.set_yticks([])
        if b == 0:
            ax.legend(fontsize=8)
    fig2.suptitle("Aguada overlaps the background distribution; building does not",
                  fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.9])
    p2 = os.path.join(OUT, "chactun_band_separability.png")
    fig2.savefig(p2, dpi=130)
    print("wrote", p2)


if __name__ == "__main__":
    main()
