#!/usr/bin/env python
"""Why is Khartoum hard? Measure roof-vs-ground separability per city.

THE QUESTION
------------
Per-city F1 spans 0.895 (Vegas) to 0.627 (Khartoum) at ~200 sigma, so the
ordering is structural and unexplained. Visual inspection of the overlays
suggests two candidate mechanisms, both of which predict low recall:

  contrast   Khartoum roofs and bare ground look alike, so there may be little
             radiometric signal marking a building at all.
  relief     Khartoum roofs are flat. A pitched roof gives two faces at
             different brightness plus a cast shadow -- boundary cues that
             survive even when albedo does not.

Two confounds produce the same symptom and are measured in the same pass, so
they can be ruled in or out rather than argued about:

  size       small objects score worse everywhere; the pooled 37/59/4 split is
             not broken out per city anywhere in the notebook.
  crowding   footprints that abut share a boundary, so even a found building
             may not clear IoU 0.5.

WHAT IS MEASURED
----------------
All statistics are computed on the uint8 image the NETWORK sees -- per_image
2-98 percentile stretch -- not on raw DN. The question is what the model can
discriminate, and a stretch applied before the model changes that.

  cohens_d          global inside-footprint vs outside-footprint brightness,
                    in pooled standard deviations. Confounded by roads and
                    vegetation, so it is the weaker of the two contrast numbers.
  boundary_contrast the same difference across a 2 px ring either side of the
                    footprint edge, normalised by tile sd. This is closer to
                    what an edge-sensitive backbone actually keys on, and it is
                    not confounded by whatever else is in the tile. Pixels
                    belonging to any OTHER building are excluded from the
                    outside ring, or dense blocks would measure roof against
                    roof.
  shadow_ratio      fraction of dark pixels (< mean - 1 sd) in a 2-6 px band
                    outside footprints, divided by the same fraction over all
                    non-building pixels. > 1 means darkness concentrates around
                    buildings, which is what a cast shadow looks like. A proxy,
                    not a shadow detector: it cannot tell a shadow from a dark
                    courtyard.

Sampling is a fixed stride through the tile list rather than a random draw, so
the answer is reproducible without carrying yet another seed.

    ./scripts/run.sh python scripts/city_separability.py --tiles-per-city 200
"""
import argparse
import json
import os

import cv2
import numpy as np
import rasterio

CITIES = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]
# Reported F1, 3-seed mean, from the 2026-08-27 notebook entry. Carried here
# only so the printed table can be read against it in one glance.
F1 = {"AOI_2_Vegas": 0.8952, "AOI_3_Paris": 0.7791,
      "AOI_4_Shanghai": 0.6877, "AOI_5_Khartoum": 0.6272}


def stretched_gray(path, low=2.0, high=98.0):
    """Tile -> float32 grayscale of the uint8 image the model is fed."""
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32)
    lo = np.percentile(arr, low, axis=(1, 2), keepdims=True)
    hi = np.percentile(arr, high, axis=(1, 2), keepdims=True)
    rgb = np.clip((arr - lo) / np.maximum(hi - lo, 1e-6), 0, 1) * 255.0
    # Rec.601 luma, matching cv2's RGB2GRAY.
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).astype(np.float32)


def footprint_mask(anns, h, w):
    m = np.zeros((h, w), dtype=np.uint8)
    for a in anns:
        for seg in a["segmentation"]:
            pts = np.asarray(seg, dtype=np.float64).reshape(-1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/spacenet2")
    p.add_argument("--coco-dir", default="data/spacenet2/coco")
    p.add_argument("--tiles-per-city", type=int, default=200)
    p.add_argument("--out", default="outputs/city_analysis/separability.json")
    args = p.parse_args()

    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    results = {}

    for city in CITIES:
        with open(os.path.join(args.coco_dir, "%s_train.json" % city)) as f:
            d = json.load(f)
        by_img = {}
        for a in d["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        # Tiles with buildings only: an empty tile has no inside pixels.
        ids = sorted(by_img)
        stride = max(1, len(ids) // args.tiles_per_city)
        picks = ids[::stride][:args.tiles_per_city]
        info = {im["id"]: im for im in d["images"]}
        imgdir = os.path.join(args.root, city, "PS-RGB")

        acc = {k: [] for k in ("in_mean", "out_mean", "in_var", "out_var",
                               "n_in", "n_out", "bring", "iring",
                               "dark_band", "dark_all", "cover")}
        areas, per_tile, shared = [], [], []

        for image_id in picks:
            im = info[image_id]
            g = stretched_gray(os.path.join(imgdir, im["file_name"]))
            h, w = g.shape
            anns = by_img[image_id]
            m = footprint_mask(anns, h, w)
            if m.sum() == 0 or m.sum() == m.size:
                continue

            inside, outside = m.astype(bool), ~m.astype(bool)
            acc["in_mean"].append(g[inside].mean())
            acc["out_mean"].append(g[outside].mean())
            acc["in_var"].append(g[inside].var())
            acc["out_var"].append(g[outside].var())
            acc["n_in"].append(int(inside.sum()))
            acc["n_out"].append(int(outside.sum()))
            acc["cover"].append(float(m.mean()))

            # 2 px either side of the footprint edge; other buildings excluded
            # from the outside ring so dense blocks do not measure roof v roof.
            er = cv2.erode(m, k3, iterations=2)
            di = cv2.dilate(m, k3, iterations=2)
            iring = (m - er).astype(bool)
            oring = ((di - m) > 0) & outside
            if iring.sum() and oring.sum():
                acc["iring"].append(g[iring].mean())
                acc["bring"].append(g[oring].mean())

            # Shadow proxy: darkness in a 2-6 px band outside footprints.
            di6 = cv2.dilate(m, k3, iterations=6)
            band = ((di6 - di) > 0) & outside
            thr = g.mean() - g.std()
            if band.sum():
                acc["dark_band"].append(float((g[band] < thr).mean()))
            acc["dark_all"].append(float((g[outside] < thr).mean()))

            areas.extend(float(a["area"]) for a in anns)
            per_tile.append(len(anns))
            # Crowding: what fraction of the ring just outside the footprints
            # is itself another footprint. High means buildings abut, so a
            # detection's boundary error spills straight onto a neighbour.
            ring1 = (cv2.dilate(m, k3, iterations=1) - er).astype(bool)
            if ring1.sum():
                touch = np.zeros_like(m, dtype=bool)
                for ann in anns:
                    one = footprint_mask([ann], h, w)
                    grown = cv2.dilate(one, k3, iterations=1).astype(bool)
                    touch |= grown & (m.astype(bool) & ~one.astype(bool))
                shared.append(float(touch.sum()) / float(m.sum()))

        a = np.asarray(areas)
        in_m, out_m = np.mean(acc["in_mean"]), np.mean(acc["out_mean"])
        pooled_sd = np.sqrt((np.mean(acc["in_var"]) + np.mean(acc["out_var"])) / 2.0)
        tile_sd = pooled_sd
        results[city] = {
            "tiles": len(acc["in_mean"]),
            "f1": F1[city],
            "inside_mean": round(float(in_m), 2),
            "outside_mean": round(float(out_m), 2),
            "cohens_d": round(float((in_m - out_m) / max(pooled_sd, 1e-6)), 3),
            "boundary_contrast": round(
                float((np.mean(acc["iring"]) - np.mean(acc["bring"]))
                      / max(tile_sd, 1e-6)), 3),
            "shadow_ratio": round(
                float(np.mean(acc["dark_band"]) / max(np.mean(acc["dark_all"]), 1e-9)), 3),
            "median_area_px": round(float(np.median(a)), 1),
            "pct_small": round(float((a < 1024).mean() * 100), 1),
            "pct_medium": round(float(((a >= 1024) & (a <= 9216)).mean() * 100), 1),
            "pct_large": round(float((a > 9216).mean() * 100), 1),
            "buildings_per_tile": round(float(np.mean(per_tile)), 1),
            "footprint_cover_pct": round(float(np.mean(acc["cover"]) * 100), 1),
            "abut_frac": round(float(np.mean(shared)) if shared else 0.0, 4),
            "instances": len(a),
        }
        print("done", city)

    hdr = ("city", "F1", "cohd", "bdry", "shad", "medpx", "%sm", "b/tile",
           "cov%", "abut")
    print("\n%-16s %6s %6s %6s %6s %8s %6s %7s %6s %6s" % hdr)
    for city in CITIES:
        r = results[city]
        print("%-16s %6.3f %6.3f %6.3f %6.3f %8.0f %6.1f %7.1f %6.1f %6.3f"
              % (city, r["f1"], r["cohens_d"], r["boundary_contrast"],
                 r["shadow_ratio"], r["median_area_px"], r["pct_small"],
                 r["buildings_per_tile"], r["footprint_cover_pct"],
                 r["abut_frac"]))

    # Rank correlation against F1: does any single measure reproduce the
    # difficulty ordering? With four cities this is suggestive, never proof.
    f1v = np.array([results[c]["f1"] for c in CITIES])
    print("\nSpearman rho vs F1 (n=4, indicative only):")
    for key in ("cohens_d", "boundary_contrast", "shadow_ratio",
                "median_area_px", "pct_small", "buildings_per_tile",
                "abut_frac"):
        v = np.array([results[c][key] for c in CITIES])
        rf = np.argsort(np.argsort(f1v)).astype(float)
        rv = np.argsort(np.argsort(v)).astype(float)
        rho = np.corrcoef(rf, rv)[0, 1]
        print("  %-20s %+.2f" % (key, rho))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
