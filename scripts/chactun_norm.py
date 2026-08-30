#!/usr/bin/env python
"""Was the seam test valid, or did per-tile normalisation erase the evidence?

The all-pairs search found no adjacent tiles. That supports "the tiles are
spatially separated samples" -- but it has a competing explanation: if each tile
was stretched to 0-255 independently, then two genuinely adjacent tiles get
different scalings and their shared seam stops matching. The test would then be
measuring the normalisation, not the geography.

The two cases are distinguishable. Under per-tile normalisation nearly every
tile hits both 0 and 255 in every band, and its range is pinned to exactly
0-255. Under one global stretch the per-tile ranges vary with the terrain.
"""
import os

import numpy as np
import rasterio

LIDAR = "/w/data/chactun/lidar"
BANDS = ["sky-view factor", "positive openness", "slope"]


def main():
    ids = sorted(int(n.split("_")[1]) for n in os.listdir(LIDAR)
                 if n.endswith("_lidar.tif"))
    rng = np.random.default_rng(0)
    sample = rng.choice(ids, size=300, replace=False)

    mins = np.zeros((len(sample), 3), np.float64)
    maxs = np.zeros((len(sample), 3), np.float64)
    means = np.zeros((len(sample), 3), np.float64)

    for k, t in enumerate(sample):
        with rasterio.open(os.path.join(LIDAR, "tile_%d_lidar.tif" % t)) as s:
            a = s.read()
        for b in range(3):
            mins[k, b] = a[b].min()
            maxs[k, b] = a[b].max()
            means[k, b] = a[b].mean()

    print("300 random tiles, per-band statistics ACROSS tiles")
    print()
    print("%-20s %14s %14s %16s %16s"
          % ("band", "min of mins", "max of maxs", "frac hitting 0",
             "frac hitting 255"))
    for b in range(3):
        print("%-20s %14.0f %14.0f %16.3f %16.3f"
              % (BANDS[b], mins[:, b].min(), maxs[:, b].max(),
                 (mins[:, b] == 0).mean(), (maxs[:, b] == 255).mean()))
    print()
    print("%-20s %14s %14s %14s"
          % ("band", "mean of means", "sd of means", "sd of ranges"))
    for b in range(3):
        rngs = maxs[:, b] - mins[:, b]
        print("%-20s %14.1f %14.1f %14.1f"
              % (BANDS[b], means[:, b].mean(), means[:, b].std(), rngs.std()))
    print()

    pinned = np.mean([((mins[:, b] == 0) & (maxs[:, b] == 255)).mean()
                      for b in range(3)])
    print("=== verdict ===")
    print("  tiles pinned to exactly 0-255 in all three bands: %.1f%%"
          % (100 * pinned))
    if pinned > 0.9:
        print("  Per-tile normalisation. The seam test is INVALID -- adjacency")
        print("  cannot be ruled out this way, because the stretch would hide it.")
    elif pinned < 0.5:
        print("  Not per-tile normalised: tile ranges vary with the terrain, so a")
        print("  shared seam would have survived. The seam test stands, and the")
        print("  tiles really are not neighbours.")
    else:
        print("  Mixed evidence; inspect further before relying on either result.")


if __name__ == "__main__":
    main()
