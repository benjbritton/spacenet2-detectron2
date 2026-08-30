#!/usr/bin/env python
"""Per-band pixel mean and std for Chactun, to replace the COCO RGB constants.

detectron2 normalises with (image - PIXEL_MEAN) / PIXEL_STD, and the zoo configs
carry PIXEL_MEAN [103.53, 116.28, 123.675] with PIXEL_STD [1, 1, 1] -- ImageNet
BGR statistics with no variance scaling. Chactun's three bands are sky-view
factor, positive openness and slope, whose distributions sit nowhere near those
values, so the defaults would leave the input badly off-centre.

These constants are computed over all 2094 tiles rather than per fold. Six
scalars derived from the whole set is a token leak, and much more importantly it
is IDENTICAL across every arm of the comparison, so it cannot bias A against B
against C. Per-fold constants would vary the preprocessing between folds, which
is the worse trade.

Streaming accumulation, because the full stack is 2094 x 3 x 480 x 480.
"""
import os

import numpy as np
import rasterio

LIDAR = "/w/data/chactun/lidar"
BANDS = ["sky-view factor", "positive openness", "slope"]


def main():
    ids = sorted(int(n.split("_")[1]) for n in os.listdir(LIDAR)
                 if n.endswith("_lidar.tif"))

    n = 0
    s = np.zeros(3, np.float64)
    ss = np.zeros(3, np.float64)
    mn = np.full(3, np.inf)
    mx = np.full(3, -np.inf)

    for k, t in enumerate(ids):
        with rasterio.open(os.path.join(LIDAR, "tile_%d_lidar.tif" % t)) as src:
            a = src.read().astype(np.float64)      # (3, H, W)
        flat = a.reshape(3, -1)
        n += flat.shape[1]
        s += flat.sum(axis=1)
        ss += (flat ** 2).sum(axis=1)
        mn = np.minimum(mn, flat.min(axis=1))
        mx = np.maximum(mx, flat.max(axis=1))
        if (k + 1) % 500 == 0:
            print("  %d/%d" % (k + 1, len(ids)))

    mean = s / n
    var = ss / n - mean ** 2
    std = np.sqrt(np.maximum(var, 0))

    print()
    print("%d tiles, %d pixels per band" % (len(ids), n // 1))
    print()
    print("%-20s %10s %10s %8s %8s" % ("band", "mean", "std", "min", "max"))
    for i in range(3):
        print("%-20s %10.3f %10.3f %8.0f %8.0f"
              % (BANDS[i], mean[i], std[i], mn[i], mx[i]))

    print()
    print("For the config (band order as stored, INPUT.FORMAT must match):")
    print()
    print("  PIXEL_MEAN: [%.3f, %.3f, %.3f]" % tuple(mean))
    print("  PIXEL_STD:  [%.3f, %.3f, %.3f]" % tuple(std))
    print()
    print("Against the COCO defaults these replace:")
    print("  PIXEL_MEAN: [103.530, 116.280, 123.675]   PIXEL_STD: [1.0, 1.0, 1.0]")
    print()
    off = mean - np.array([103.53, 116.28, 123.675])
    print("  centring error if left unchanged: %+.1f %+.1f %+.1f counts"
          % tuple(off))
    print("  in units of each band's own sd  : %+.2f %+.2f %+.2f"
          % tuple(off / std))


if __name__ == "__main__":
    main()
