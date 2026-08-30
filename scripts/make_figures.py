#!/usr/bin/env python
"""Figures for the blog post. Three, each answering a question prose cannot.

fig1_vegas_vs_khartoum.png
    Matched crops, Vegas above Khartoum, footprints outlined. Makes the soft
    boundary and the absent hue separation visible rather than asserted.

fig2_khartoum_missed.png
    Ground-truth footprints with NO overlapping prediction at IoU 0.10 -- the
    32% that nothing was proposed on. Anchors the "upstream proposal failure"
    conclusion in the actual imagery.

fig3_anchor_coverage.png
    Footprint size distribution per city in NETWORK INPUT space against the
    detectron2 default anchor ladder, plus best-achievable anchor IoU. Shows why
    anchor scale looked like the culprit and why the data does not support it.

Display stretch here is per-image 2-98 percentile: for eyes, deliberately
independent of the training preprocessing. A rendering choice must not quietly
become a claim.
"""
import json
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.patches import Rectangle

from detlab.spacenet_f1 import _gt_rles, match_greedy_pairs
import pycocotools.mask as mask_util

OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)
ROOT = "data/spacenet2"
SCALE = 800.0 / 650.0
ANCHORS = [32.0, 64.0, 128.0, 256.0, 512.0]
THR = 0.544

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def load_rgb(path):
    with rasterio.open(path) as src:
        a = src.read([1, 2, 3]).astype(np.float32)
    out = np.empty_like(a)
    for c in range(3):
        v = a[c][a[c] > 0]
        lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0, 1)
        out[c] = np.clip((a[c] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out.transpose(1, 2, 0)


def nodata_fraction(path):
    """Tiles that overrun the imaged strip carry a black nodata border. They are
    legitimate data but make a poor illustration, so figures skip them."""
    with rasterio.open(path) as src:
        a = src.read([1, 2, 3])
    return float((a == 0).all(axis=0).mean())


def polys(anns):
    for a in anns:
        for seg in a["segmentation"]:
            yield np.array(seg, float).reshape(-1, 2)


# ----------------------------------------------------------------- figure 1
def fig1(dsets):
    from detectron2.data import DatasetCatalog
    pairs = [("AOI_2_Vegas", "#1f6feb"), ("AOI_5_Khartoum", "#d1481f")]
    ncol = 4
    fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.4))
    for row, (city, colour) in enumerate(pairs):
        cand = [r for r in DatasetCatalog.get("spacenet2_val_%s" % city)
                if 10 <= len(r["annotations"]) <= 45]
        cand = sorted(cand, key=lambda r: r["image_id"])
        picks = []
        for r in cand:
            if len(picks) == ncol:
                break
            if nodata_fraction(r["file_name"]) < 0.02:
                picks.append(r)
        for col, rec in enumerate(picks):
            ax = axes[row, col]
            img = load_rgb(rec["file_name"])
            # centre crop, so roofs and surrounding ground are both in frame
            c = 300
            y0 = (img.shape[0] - c) // 2
            x0 = (img.shape[1] - c) // 2
            ax.imshow(img[y0:y0 + c, x0:x0 + c])
            for p in polys(rec["annotations"]):
                q = p - [x0, y0]
                m = (q[:, 0] > -20) & (q[:, 0] < c + 20) & (q[:, 1] > -20) & (q[:, 1] < c + 20)
                if m.sum() > 2:
                    ax.plot(np.append(q[:, 0], q[0, 0]), np.append(q[:, 1], q[0, 1]),
                            color=colour, lw=1.1)
            ax.set_xlim(0, c); ax.set_ylim(c, 0)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(city.split("_", 2)[-1].replace("_", " "),
                              fontsize=12, color=colour, labelpad=8)
    axes[0, 0].set_title("Vegas — pitched roofs, cast shadow, vegetation:\n"
                         "hue separation 29.4°, boundary contrast 0.435",
                         loc="left", color=pairs[0][1])
    axes[1, 0].set_title("Khartoum — flat roofs on flat ground, no shadow:\n"
                         "hue separation 2.3°, boundary contrast 0.315",
                         loc="left", color=pairs[1][1])
    fig.suptitle("Same model, same weights.  F1 0.895 above, 0.625 below.",
                 fontsize=12, y=0.995)
    fig.savefig(os.path.join(OUT, "fig1_vegas_vs_khartoum.png"))
    plt.close(fig)
    print("wrote fig1")


# ----------------------------------------------------------------- figure 2
def fig2():
    from detectron2.data import DatasetCatalog
    name = "spacenet2_val_AOI_5_Khartoum"
    recs = {r["image_id"]: r for r in DatasetCatalog.get(name)}
    preds = torch.load("outputs/spacenet2_r50fpn/inference/%s/"
                       "instances_predictions.pth" % name, weights_only=False)

    found = []
    for entry in preds:
        rec = recs.get(entry["image_id"])
        if not rec or not rec["annotations"]:
            continue
        gt = _gt_rles(rec)
        rles, scores = [], []
        for i in entry.get("instances", []):
            s = i.get("segmentation")
            if s is None or i["score"] < THR:
                continue
            if isinstance(s.get("counts"), str):
                s = dict(s, counts=s["counts"].encode())
            rles.append(s); scores.append(i["score"])
        # IoU 0.10: barely more than "was anything put roughly here"
        # match_greedy_pairs returns the matched GT index PER PREDICTION,
        # -1 where the prediction hit nothing. Not (pred, gt) pairs.
        matched = {g for g in match_greedy_pairs(rles, scores, gt, 0.10) if g >= 0}
        missed = [k for k in range(len(gt)) if k not in matched]
        if len(gt) >= 12 and len(missed) >= 3:
            found.append((len(missed), len(gt), rec, missed, gt))

    # Choose tiles whose miss rate is CLOSE TO THE CITY AVERAGE, not the worst.
    # Khartoum misses about 32% of its footprints at IoU 0.10; showing the tail
    # would illustrate a failure four times worse than the one being reported.
    CITY_MISS_RATE = 0.321
    found = [f for f in found if nodata_fraction(f[2]["file_name"]) < 0.02]
    found.sort(key=lambda f: abs((f[0] / f[1]) - CITY_MISS_RATE))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 11.2))
    axes = axes.ravel()
    for ax, (nm, ngt, rec, missed, gt) in zip(axes, found[:4]):
        ax.imshow(load_rgb(rec["file_name"]))
        for k, r in enumerate(gt):
            m = mask_util.decode(r)
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            miss = k in missed
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor="#e5484d" if miss else "#2f9e44",
                                   lw=1.6 if miss else 0.8,
                                   linestyle="-" if miss else ":"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("%d of %d footprints missed  (%.0f%%)"
                     % (nm, ngt, 100.0 * nm / ngt), fontsize=10)
    fig.suptitle("Khartoum: ground truth with NO prediction overlapping it, even at IoU 0.10\n"
                 "red solid = nothing was proposed there at all     "
                 "green dotted = detected\n"
                 "tiles chosen at the city-wide miss rate of 32%, not the worst cases",
                 fontsize=11.5, y=0.965)
    fig.savefig(os.path.join(OUT, "fig2_khartoum_missed.png"))
    plt.close(fig)
    print("wrote fig2  (%d tiles had >=4 misses)" % len(found))


# ----------------------------------------------------------------- figure 3
def fig3():
    cities = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]
    cols = {"AOI_2_Vegas": "#1f6feb", "AOI_3_Paris": "#2f9e44",
            "AOI_4_Shanghai": "#9a6700", "AOI_5_Khartoum": "#d1481f"}
    f1 = {"AOI_2_Vegas": 0.895, "AOI_3_Paris": 0.779,
          "AOI_4_Shanghai": 0.685, "AOI_5_Khartoum": 0.625}

    def best_iou(w, h):
        best, ga = 0.0, w * h
        for s in ANCHORS:
            for ar in (0.5, 1.0, 2.0):
                aw, ah = s / np.sqrt(ar), s * np.sqrt(ar)
                inter = np.minimum(w, aw) * np.minimum(h, ah)
                best = np.maximum(best, inter / (ga + aw * ah - inter))
        return best

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for c in cities:
        with open(os.path.join(ROOT, "coco", "%s_train.json" % c)) as f:
            d = json.load(f)
        w = np.array([a["bbox"][2] for a in d["annotations"]]) * SCALE
        h = np.array([a["bbox"][3] for a in d["annotations"]]) * SCALE
        ok = (w > 1) & (h > 1)
        w, h = w[ok], h[ok]
        side = np.sqrt(w * h)
        lbl = "%s  (F1 %.3f)" % (c.split("_", 2)[-1].replace("_", " "), f1[c])
        axL.hist(side, bins=np.linspace(0, 200, 90), histtype="step",
                 density=True, color=cols[c], lw=1.6, label=lbl)
        iou = np.sort(best_iou(w, h))
        axR.plot(iou, np.arange(iou.size) / iou.size, color=cols[c], lw=1.8, label=lbl)

    for a in ANCHORS[:3]:
        axL.axvline(a, color="0.35", ls="--", lw=0.9)
        axL.text(a, axL.get_ylim()[1] * 0.96, " %d px" % int(a),
                 fontsize=8, color="0.35", va="top")
    axL.set_xlabel("footprint size in network input space, sqrt(w·h) px")
    axL.set_ylabel("density")
    axL.set_title("Footprints vs the default anchor ladder", loc="left")
    axL.legend(fontsize=8, frameon=False)

    axR.axvline(0.3, color="0.35", ls="--", lw=0.9)
    axR.text(0.305, 0.80, "anchors below IoU 0.3\nare labelled negative\nby the RPN",
             fontsize=8, color="0.35", va="top")
    axR.set_xlabel("best achievable IoU with any anchor")
    axR.set_ylabel("cumulative fraction of footprints")
    axR.set_title("Vegas is nearly as poorly covered as Khartoum —\n"
                  "and scores 0.895. Anchor scale is not the explanation.",
                  loc="left")
    axR.legend(fontsize=8, frameon=False, loc="lower right")
    axR.set_xlim(0, 1)

    fig.savefig(os.path.join(OUT, "fig3_anchor_coverage.png"))
    plt.close(fig)
    print("wrote fig3")


if __name__ == "__main__":
    from detlab.datasets import spacenet
    spacenet.register_pooled(root=ROOT)
    spacenet.register_val_per_aoi(root=ROOT)
    fig1(None)
    fig2()
    fig3()
    print("\nfigures in", OUT)
