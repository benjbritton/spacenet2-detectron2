#!/usr/bin/env python
"""Is HUE a factor in Khartoum being hard, independent of brightness?

THE HYPOTHESIS
--------------
`city_separability.py` measured brightness and found Khartoum has the HIGHEST
global roof-vs-ground separation of the four cities (d 1.575) but a soft
boundary. That was all value. Ben's proposal is that Khartoum roofs and terrain
share a hue even where their value differs, so the chromatic channel carries no
signal marking a building.

Hue is a genuinely separate axis from value, so it can be tested separately.

WHY THIS IS NOT JUST cohens_d WITH DIFFERENT PIXELS
---------------------------------------------------
Two things make hue its own problem:

1. **It is circular.** 359 degrees and 1 degree are two degrees apart, not 358.
   Every mean and every distance here uses circular statistics -- the mean is
   atan2(mean sin, mean cos), and the distance wraps. A linear mean over hue
   angles is simply wrong and would put the mean of red pixels in the cyans.

2. **It is undefined at low saturation.** A neutral grey pixel has no hue; what
   the arithmetic returns for it is amplified sensor noise. Desert ground and
   concrete roofs are both near-neutral, which is exactly the regime where hue
   statistics look meaningful and are not. Saturation is therefore reported
   beside every hue number, and the circular spread is reported so that a large
   mean difference over an enormous spread is visible as the non-result it is.

MEASURED IN TWO COLOUR SPACES, AND THE DIFFERENCE MATTERS
---------------------------------------------------------
  raw        HSV from the 11-bit sensor values, scaled by a FIXED divisor so the
             ratios between channels -- which is what hue is -- survive. This is
             the physical hue of the scene.

  stretched  HSV after the per_image 2-98 percentile stretch, which is what the
             network is actually fed.

They are not the same, and the difference is a finding rather than bookkeeping:
the stretch computes an INDEPENDENT low and high per channel, which is a per-tile
white balance. It moves hue. If chromatic signal exists in the sensor data and
the preprocessing destroys it, that is actionable in a way that a fact about
Khartoum geology is not.

8-bit OpenCV HSV quantises hue to 2 degrees per step. Everything here is float32,
H in [0, 360).

SAMPLING
--------
Identical to city_separability.py: 250 tiles per city at a fixed stride, and the
same masks -- inside footprint, and a 2 px outside ring with pixels belonging to
any other building excluded, so dense blocks do not measure roof against roof.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np
import rasterio

CITIES = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]
F1 = {"AOI_2_Vegas": 0.895, "AOI_3_Paris": 0.779,
      "AOI_4_Shanghai": 0.688, "AOI_5_Khartoum": 0.627}
DN_SCALE = 2047.0          # 11-bit ceiling, measured across all 10592 tiles


def hsv_from(arr, mode):
    """(3,H,W) uint16 -> H[0,360), S[0,1], V[0,1] as float32, plus a valid mask."""
    a = arr.astype(np.float32)
    valid = (arr > 0).all(axis=0)
    if mode == "raw":
        rgb = np.clip(a / DN_SCALE, 0, 1)
    else:
        rgb = np.empty_like(a)
        for c in range(3):
            band = a[c]
            v = band[band > 0]
            lo, hi = (np.percentile(v, 2), np.percentile(v, 98)) if v.size else (0, 1)
            rgb[c] = np.clip((band - lo) / max(hi - lo, 1.0), 0, 1)
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb.transpose(1, 2, 0)),
                       cv2.COLOR_RGB2HSV)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2], valid


def circ_stats(deg):
    """Circular mean (deg) and resultant length R in [0,1]."""
    if deg.size == 0:
        return float("nan"), 0.0
    r = np.deg2rad(deg.astype(np.float64))
    c, s = np.cos(r).mean(), np.sin(r).mean()
    R = float(np.hypot(c, s))
    return float((np.degrees(np.arctan2(s, c))) % 360.0), R


def circ_dist(a, b):
    """Wraparound-safe angular distance in degrees, 0..180."""
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def circ_sd(R):
    """Circular standard deviation in degrees from resultant length."""
    R = min(max(R, 1e-9), 1 - 1e-12)
    return float(np.degrees(np.sqrt(-2.0 * np.log(R))))


def footprint_mask(anns, h, w):
    m = np.zeros((h, w), np.uint8)
    for a in anns:
        for seg in a["segmentation"]:
            pts = np.array(seg, np.float64).reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/spacenet2")
    p.add_argument("--tiles-per-city", type=int, default=250)
    p.add_argument("--out", default="outputs/city_analysis/hue.json")
    args = p.parse_args()

    from detectron2.data import DatasetCatalog
    from detlab.datasets import spacenet

    spacenet.register_pooled(root=args.data_root)
    spacenet.register_val_per_aoi(root=args.data_root)
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    results = {}
    for city in CITIES:
        recs = DatasetCatalog.get("spacenet2_val_%s" % city)
        recs = [r for r in recs if r["annotations"]]
        stride = max(1, len(recs) // args.tiles_per_city)
        picks = recs[::stride][:args.tiles_per_city]

        acc = {k: [] for k in ("h_in_raw", "h_out_raw", "h_in_str", "h_out_str",
                               "h_iring_raw", "h_oring_raw",
                               "s_in", "s_out", "v_in", "v_out")}
        for rec in picks:
            with rasterio.open(rec["file_name"]) as src:
                arr = src.read([1, 2, 3])
            h, w = arr.shape[1], arr.shape[2]
            m = footprint_mask(rec["annotations"], h, w)
            if m.sum() == 0:
                continue

            Hr, Sr, Vr, valid = hsv_from(arr, "raw")
            Hs, Ss, Vs, _ = hsv_from(arr, "stretched")

            inside = (m > 0) & valid
            outside = (m == 0) & valid
            if inside.sum() < 50 or outside.sum() < 50:
                continue

            # Hue is meaningless where there is no colour; require a little
            # saturation before a pixel is allowed to vote on hue.
            sat_ok = Sr > 0.05
            hi_px = Hr[inside & sat_ok]
            ho_px = Hr[outside & sat_ok]
            if hi_px.size < 30 or ho_px.size < 30:
                continue
            acc["h_in_raw"].append(circ_stats(hi_px))
            acc["h_out_raw"].append(circ_stats(ho_px))
            acc["h_in_str"].append(circ_stats(Hs[inside & sat_ok]))
            acc["h_out_str"].append(circ_stats(Hs[outside & sat_ok]))

            acc["s_in"].append(float(Sr[inside].mean()))
            acc["s_out"].append(float(Sr[outside].mean()))
            acc["v_in"].append(float(Vr[inside].mean()))
            acc["v_out"].append(float(Vr[outside].mean()))

            # Boundary: same 2 px rings city_separability.py uses, other
            # buildings excluded from the outside ring.
            er = cv2.erode(m, k3, iterations=2)
            di = cv2.dilate(m, k3, iterations=2)
            iring = ((m - er) > 0) & valid & sat_ok
            oring = ((di - m) > 0) & outside & sat_ok
            if iring.sum() > 30 and oring.sum() > 30:
                acc["h_iring_raw"].append(circ_stats(Hr[iring]))
                acc["h_oring_raw"].append(circ_stats(Hr[oring]))

        def pooled(key):
            """Circular mean and sd over the per-tile circular means."""
            vals = np.array([v[0] for v in acc[key] if not np.isnan(v[0])])
            mu, R = circ_stats(vals)
            return mu, circ_sd(R), len(vals)

        hin, hin_sd, n = pooled("h_in_raw")
        hout, hout_sd, _ = pooled("h_out_raw")
        sin_, sout = float(np.mean(acc["s_in"])), float(np.mean(acc["s_out"]))
        hin_s, hin_s_sd, _ = pooled("h_in_str")
        hout_s, _, _ = pooled("h_out_str")
        hir, hir_sd, _ = pooled("h_iring_raw")
        hor, _, _ = pooled("h_oring_raw")

        sep_raw = circ_dist(hin, hout)
        pooled_sd = np.sqrt((hin_sd ** 2 + hout_sd ** 2) / 2.0)
        results[city] = {
            "n_tiles": n,
            "f1": F1[city],
            "hue_in_raw": hin, "hue_out_raw": hout,
            "hue_sep_raw_deg": sep_raw,
            "hue_sd_pooled_deg": float(pooled_sd),
            "hue_d_circ": float(sep_raw / pooled_sd) if pooled_sd else 0.0,
            "hue_sep_stretched_deg": circ_dist(hin_s, hout_s),
            "hue_boundary_sep_deg": circ_dist(hir, hor),
            "sat_in": sin_, "sat_out": sout,
            "sat_ratio": sin_ / sout if sout else 0.0,
            "val_in": float(np.mean(acc["v_in"])),
            "val_out": float(np.mean(acc["v_out"])),
        }

    print("%-16s %5s %8s %8s %8s %8s %8s %8s %7s %7s" %
          ("city", "F1", "hue_in", "hue_out", "sep_deg", "hue_sd", "d_circ",
           "bnd_deg", "sat_in", "sat_out"))
    for c in CITIES:
        r = results[c]
        print("%-16s %5.3f %8.1f %8.1f %8.2f %8.1f %8.3f %8.2f %7.3f %7.3f" %
              (c, r["f1"], r["hue_in_raw"], r["hue_out_raw"],
               r["hue_sep_raw_deg"], r["hue_sd_pooled_deg"], r["hue_d_circ"],
               r["hue_boundary_sep_deg"], r["sat_in"], r["sat_out"]))

    print()
    print("%-16s %14s %14s %10s" %
          ("city", "sep raw (deg)", "sep after (deg)", "kept"))
    for c in CITIES:
        r = results[c]
        a, b = r["hue_sep_raw_deg"], r["hue_sep_stretched_deg"]
        print("%-16s %14.2f %14.2f %9.0f%%"
              % (c, a, b, 100 * b / a if a else 0))
    print()
    print("sep_deg   circular distance between mean roof hue and mean ground hue")
    print("hue_sd    pooled circular sd -- the spread the separation sits inside")
    print("d_circ    sep / hue_sd. Below ~0.3 the hue channel carries no usable")
    print("          building signal regardless of how large sep looks")
    print("sat_in    mean saturation on roofs; hue is unreliable below ~0.1")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("\\nwrote", args.out)


if __name__ == "__main__":
    main()
