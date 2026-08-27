#!/usr/bin/env python
"""Per-city radiometric statistics for the SpaceNet 2 PS-RGB tiles.

WHY
---
The imagery is UInt16 and the model wants 8-bit. Two stretches are on the table:

  per-image percentile  -- what SpaceNet's own baseline write-ups describe, and
                           the closest thing to a reference implementation now
                           that Solaris is unrunnable (tensorflow==1.13.1, and a
                           git:// dependency GitHub disabled in 2022).
  per-city constants    -- one stretch per AOI, so two tiles of the same city
                           stay radiometrically comparable to each other.

Per-image maximizes contrast tile by tile but destroys cross-tile consistency: a
dim tile and a bright tile receive different stretches, so absolute brightness
stops carrying information. Per-city keeps that signal at the cost of some
per-tile contrast. This script produces the constants the second mode needs.

METHOD
------
Exact percentiles over the sampled pixels, via a full 65536-bin histogram per
channel rather than np.percentile over a stored sample. Histogram accumulation is
O(1) in memory regardless of how many tiles are read, so this can run over every
tile without holding 26 GB of pixels anywhere.

Pixels are subsampled (default every 4th row and column, 6.25% of each tile)
because the file read dominates the cost and a percentile does not need every
pixel. Tiles are NOT subsampled by default: a percentile meant to characterize a
whole city should see the whole city.

Zeros are excluded. SpaceNet tiles carry nodata borders where a tile overruns the
imaged strip, and counting those collapses the low percentile onto 0.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import rasterio

CITIES = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]
NBINS = 65536


def percentiles_from_hist(hist, qs):
    """Exact percentile of the sampled population, read off the CDF."""
    total = hist.sum()
    if total == 0:
        return [0] * len(qs)
    cdf = np.cumsum(hist)
    return [int(np.searchsorted(cdf, q / 100.0 * total)) for q in qs]


def city_stats(image_dir, pixel_stride, tile_stride, low, high):
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".tif"))
    files = files[::tile_stride]
    hist = np.zeros((3, NBINS), dtype=np.int64)
    n_pixels = 0
    t0 = time.time()

    for i, name in enumerate(files):
        with rasterio.open(os.path.join(image_dir, name)) as src:
            arr = src.read([1, 2, 3])
        arr = arr[:, ::pixel_stride, ::pixel_stride]
        for c in range(3):
            band = arr[c].ravel()
            band = band[band > 0]          # drop nodata border
            if band.size:
                hist[c] += np.bincount(band, minlength=NBINS)[:NBINS]
        n_pixels += arr.shape[1] * arr.shape[2]
        if (i + 1) % 500 == 0:
            print("    %d/%d tiles  %.0fs" % (i + 1, len(files), time.time() - t0),
                  flush=True)

    out = {"tiles_read": len(files), "pixels_sampled_per_channel": n_pixels}
    for c, name in enumerate(["R", "G", "B"]):
        p_lo, p_hi = percentiles_from_hist(hist[c], [low, high])
        nz = hist[c].sum()
        vals = np.nonzero(hist[c])[0]
        out[name] = {
            "p%.1f" % low: p_lo,
            "p%.1f" % high: p_hi,
            "min_nonzero": int(vals[0]) if vals.size else 0,
            "max": int(vals[-1]) if vals.size else 0,
            "mean": float((np.arange(NBINS) * hist[c]).sum() / nz) if nz else 0.0,
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/spacenet2")
    p.add_argument("--out", default="configs/spacenet2_stretch.json")
    p.add_argument("--low", type=float, default=2.0)
    p.add_argument("--high", type=float, default=98.0)
    p.add_argument("--pixel-stride", type=int, default=4)
    p.add_argument("--tile-stride", type=int, default=1,
                   help="1 = every tile; raise only for a quick smoke run")
    args = p.parse_args()

    result = {
        "_note": ("Per-city percentile constants for 16-bit to 8-bit stretch. "
                  "Computed over nonzero pixels only; zeros are nodata border."),
        "percentiles": {"low": args.low, "high": args.high},
        "pixel_stride": args.pixel_stride,
        "tile_stride": args.tile_stride,
        "cities": {},
    }
    for city in CITIES:
        d = os.path.join(args.root, city, "PS-RGB")
        if not os.path.isdir(d):
            print("skip (missing):", d)
            continue
        print("===", city, flush=True)
        s = city_stats(d, args.pixel_stride, args.tile_stride, args.low, args.high)
        result["cities"][city] = s
        print("   R %5d-%5d   G %5d-%5d   B %5d-%5d   (%d tiles)" % (
            s["R"]["p%.1f" % args.low], s["R"]["p%.1f" % args.high],
            s["G"]["p%.1f" % args.low], s["G"]["p%.1f" % args.high],
            s["B"]["p%.1f" % args.low], s["B"]["p%.1f" % args.high],
            s["tiles_read"]), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
