#!/usr/bin/env python
"""Apportion Khartoum's difficulty between hue, boundary contrast, shadow and size.

WHY THIS IS NOT DONE AT CITY LEVEL
----------------------------------
The obvious analysis -- correlate each factor against per-city F1 -- cannot work.
Four cities, three or four candidate predictors, and the predictors are strongly
correlated with each other because Vegas simply has more of everything. Zero
residual degrees of freedom: any set of coefficients fits perfectly and none of
them means anything.

The constraint is an artefact of aggregating. Those four cities are 2118
validation tiles, each of which has its own hue separation, its own boundary
contrast, its own shadow signature, its own building sizes, and its own measured
recall. At tile level the same question has thousands of observations.

WHAT IS REGRESSED
-----------------
Outcome: per-tile recall at IoU 0.5 and score >= 0.544 -- the reporting operating
point -- weighted by the tile's ground-truth count, so a tile with 40 buildings
counts forty times a tile with one. That makes the fit an instance-level
statement rather than a tile-level one, and stops sparse tiles dominating through
sheer noise.

Predictors, all measured on the image the network sees except hue, which is
measured on raw DN because the per-tile stretch is a per-channel white balance
that manufactures hue separation (measured: it inflates Shanghai 5 deg -> 64 deg):

  hue_sep      circular roof-vs-ground hue distance, saturation-gated
  boundary     |roof ring - ground ring| brightness across the footprint edge
  shadow       dark-pixel fraction in a 2-6 px band outside footprints, over the
               same fraction tile-wide
  log_area     log median footprint area, the size confound
  density      buildings per tile

THE ATTRIBUTION
---------------
Two complementary readings, because neither alone is honest:

1. Standardised coefficients and their partial contributions to R-squared, which
   say how much each factor moves recall per standard deviation.

2. The residual city gap. Fit with the factors, then ask how much of the
   Vegas-minus-Khartoum recall difference the factors have absorbed and how much
   survives as an unexplained city term. THAT is the percentage asked for. A
   factor set that explains the gap leaves nothing behind; one that explains
   nothing leaves it all.

Collinearity is reported, not hidden: if hue and boundary move together the
individual coefficients are unstable even though their joint contribution is not,
and the variance inflation factors say when to distrust an individual number.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
import rasterio
import torch

from detlab.spacenet_f1 import _gt_rles, match_greedy

CITIES = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]
DN_SCALE = 2047.0
PRED = ["hue_sep", "boundary", "shadow", "log_area", "density"]


def circ_stats(deg):
    if deg.size == 0:
        return float("nan")
    r = np.deg2rad(deg.astype(np.float64))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360.0)


def circ_dist(a, b):
    if np.isnan(a) or np.isnan(b):
        return np.nan
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def poly_area(segm):
    total = 0.0
    for seg in segm:
        xy = np.array(seg, np.float64).reshape(-1, 2)
        x, y = xy[:, 0], xy[:, 1]
        total += abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0
    return total


def footprint_mask(anns, h, w):
    m = np.zeros((h, w), np.uint8)
    for a in anns:
        for seg in a["segmentation"]:
            pts = np.array(seg, np.float64).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
    return m


def tile_features(path, anns, k3):
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3])
    h, w = arr.shape[1], arr.shape[2]
    m = footprint_mask(anns, h, w)
    if m.sum() < 50:
        return None
    a = arr.astype(np.float32)
    valid = (arr > 0).all(axis=0)

    # hue on raw DN
    rgb = np.clip(a / DN_SCALE, 0, 1)
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb.transpose(1, 2, 0)), cv2.COLOR_RGB2HSV)
    H, S = hsv[:, :, 0], hsv[:, :, 1]

    # brightness on the stretched image the network sees
    g = np.empty_like(a)
    for c in range(3):
        band = a[c]
        v = band[band > 0]
        lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0.0, 1.0)
        g[c] = np.clip((band - lo) / max(hi - lo, 1.0), 0, 1)
    gray = g.mean(axis=0)

    inside = (m > 0) & valid
    outside = (m == 0) & valid
    if inside.sum() < 50 or outside.sum() < 50:
        return None

    sat_ok = S > 0.05
    hi_px, ho_px = H[inside & sat_ok], H[outside & sat_ok]
    hue_sep = circ_dist(circ_stats(hi_px), circ_stats(ho_px)) \
        if hi_px.size >= 30 and ho_px.size >= 30 else np.nan

    er = cv2.erode(m, k3, iterations=2)
    di = cv2.dilate(m, k3, iterations=2)
    iring = ((m - er) > 0) & valid
    oring = ((di - m) > 0) & outside
    sd = gray[valid].std() or 1.0
    boundary = abs(gray[iring].mean() - gray[oring].mean()) / sd \
        if iring.sum() > 30 and oring.sum() > 30 else np.nan

    di6 = cv2.dilate(m, k3, iterations=6)
    band = ((di6 - di) > 0) & outside
    thr = gray[valid].mean() - gray[valid].std()
    nonb = outside & (~band)
    shadow = ((gray[band] < thr).mean() / max((gray[nonb] < thr).mean(), 1e-6)) \
        if band.sum() > 30 and nonb.sum() > 30 else np.nan

    # detectron2 load_coco_json does not carry the COCO "area" field through, so
    # it is recomputed from the polygon by shoelace. Same quantity the converter
    # wrote and the same one COCOeval buckets on.
    areas = [poly_area(x["segmentation"]) for x in anns]
    areas = [a for a in areas if a > 0] or [1.0]
    return {
        "hue_sep": hue_sep,
        "boundary": boundary,
        "shadow": shadow,
        "log_area": float(np.log(np.median(areas))),
        "density": float(len(anns)),
    }


def wls(X, y, w):
    """Weighted least squares. Returns beta, R2."""
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    pred = X @ beta
    ybar = np.average(y, weights=w)
    ss_res = np.sum(w * (y - pred) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    return beta, 1.0 - ss_res / ss_tot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", default="outputs/spacenet2_r50fpn")
    p.add_argument("--data-root", default="data/spacenet2")
    p.add_argument("--threshold", type=float, default=0.544)
    p.add_argument("--out", default="outputs/city_analysis/attribution.json")
    args = p.parse_args()

    from detectron2.data import DatasetCatalog
    from detlab.datasets import spacenet

    spacenet.register_pooled(root=args.data_root)
    spacenet.register_val_per_aoi(root=args.data_root)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    rows = []
    for ci, city in enumerate(CITIES):
        name = "spacenet2_val_%s" % city
        recs = {r["image_id"]: r for r in DatasetCatalog.get(name)}
        preds = torch.load(os.path.join(args.run_dir, "inference", name,
                                        "instances_predictions.pth"),
                           weights_only=False)
        for entry in preds:
            rec = recs.get(entry["image_id"])
            if rec is None or not rec["annotations"]:
                continue
            gt = _gt_rles(rec)
            if not gt:
                continue
            rles, scores = [], []
            for i in entry.get("instances", []):
                segm = i.get("segmentation")
                if segm is None or i["score"] < args.threshold:
                    continue
                if isinstance(segm.get("counts"), str):
                    segm = dict(segm, counts=segm["counts"].encode("utf-8"))
                rles.append(segm)
                scores.append(i["score"])
            tp = sum(1 for _, ok in match_greedy(rles, scores, gt, 0.5) if ok)
            feat = tile_features(rec["file_name"], rec["annotations"], k3)
            if feat is None or any(np.isnan(feat[k]) for k in PRED):
                continue
            feat["recall"] = tp / float(len(gt))
            feat["n_gt"] = len(gt)
            feat["city"] = ci
            rows.append(feat)
        print("  %s: %d tiles usable" % (city, sum(1 for r in rows if r["city"] == ci)),
              flush=True)

    y = np.array([r["recall"] for r in rows])
    w = np.array([r["n_gt"] for r in rows], dtype=np.float64)
    Xraw = np.array([[r[k] for k in PRED] for r in rows])
    city = np.array([r["city"] for r in rows])

    mu = np.average(Xraw, axis=0, weights=w)
    sd = np.sqrt(np.average((Xraw - mu) ** 2, axis=0, weights=w))
    Z = (Xraw - mu) / sd
    ones = np.ones((len(y), 1))

    print()
    print("n tiles %d, n instances %d" % (len(y), int(w.sum())))
    print()
    print("=" * 66)
    print("STANDARDISED COEFFICIENTS  (recall per 1 sd of predictor)")
    print("=" * 66)
    beta, r2 = wls(np.hstack([ones, Z]), y, w)
    for i, k in enumerate(PRED):
        # partial: R2 lost by dropping this predictor
        keep = [j for j in range(len(PRED)) if j != i]
        _, r2_drop = wls(np.hstack([ones, Z[:, keep]]), y, w)
        print("  %-10s beta %+7.4f    partial R2 %+.4f" % (k, beta[i + 1], r2 - r2_drop))
    print("  full model R2 = %.4f" % r2)

    # variance inflation
    print()
    print("collinearity (VIF > 5 means the individual beta is unreliable)")
    for i, k in enumerate(PRED):
        keep = [j for j in range(len(PRED)) if j != i]
        _, rr = wls(np.hstack([ones, Z[:, keep]]), Z[:, i], w)
        print("  %-10s VIF %5.2f" % (k, 1.0 / max(1e-9, 1.0 - rr)))

    print()
    print("=" * 66)
    print("THE ATTRIBUTION: how much of the Vegas-Khartoum recall gap survives")
    print("=" * 66)
    veg, kha = (city == 0), (city == 3)
    raw_gap = (np.average(y[veg], weights=w[veg])
               - np.average(y[kha], weights=w[kha]))
    print("  raw gap in weighted recall            %+.4f" % raw_gap)

    resid = y - (np.hstack([ones, Z]) @ beta)
    adj_gap = (np.average(resid[veg], weights=w[veg])
               - np.average(resid[kha], weights=w[kha]))
    print("  gap remaining after all factors       %+.4f   (%.1f%% explained)"
          % (adj_gap, 100 * (1 - adj_gap / raw_gap)))
    print()
    for i, k in enumerate(PRED):
        b1, _ = wls(np.hstack([ones, Z[:, [i]]]), y, w)
        r1 = y - (np.hstack([ones, Z[:, [i]]]) @ b1)
        g1 = (np.average(r1[veg], weights=w[veg])
              - np.average(r1[kha], weights=w[kha]))
        print("  %-10s alone explains %5.1f%% of the gap" % (k, 100 * (1 - g1 / raw_gap)))

    out = {"n_tiles": len(y), "n_instances": int(w.sum()), "r2": r2,
           "raw_gap": raw_gap, "adjusted_gap": adj_gap,
           "beta": {k: beta[i + 1] for i, k in enumerate(PRED)}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
