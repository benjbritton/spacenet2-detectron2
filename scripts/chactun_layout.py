#!/usr/bin/env python
"""Can the Chactun tile layout be recovered when the rasters carry no geotransform?

The tiles have no CRS and no affine transform, so a spatially blocked split
cannot be built from coordinates. Two hypotheses, tested here against the pixels:

  H1  Tile IDs run in raster order across a grid, so id and id+1 are horizontal
      neighbours and id and id+W are vertical neighbours for some row width W.
  H2  The tiles are disjoint cuts of one continuous survey, so a true neighbour
      pair shares a seam: the right column of one tile sits physically beside
      the left column of the next, and on smooth terrain those two strips
      correlate far more than two unrelated strips do.

If both hold, the layout is recoverable and a blocked split is possible. If H2
holds but H1 does not, the layout is still recoverable, just not from the
numbering. If H2 fails there is no seam signal and no blocked split.
"""
import os
import sys

import numpy as np
import rasterio

LIDAR = "/w/data/chactun/lidar"
CACHE = "/s/chactun_edges.npz"


def tile_ids():
    ids = sorted(int(n.split("_")[1]) for n in os.listdir(LIDAR)
                 if n.endswith("_lidar.tif"))
    return ids


def build_edges(ids):
    """One pass over the tiles, keeping only the four border strips of each."""
    if os.path.isfile(CACHE):
        z = np.load(CACHE)
        print("loaded cached edges from %s" % CACHE)
        return {k: z[k] for k in ("ids", "left", "right", "top", "bottom")}

    left, right, top, bottom = [], [], [], []
    for i, t in enumerate(ids):
        with rasterio.open(os.path.join(LIDAR, "tile_%d_lidar.tif" % t)) as s:
            a = s.read().astype(np.float32)          # (bands, h, w)
        left.append(a[:, :, 0])
        right.append(a[:, :, -1])
        top.append(a[:, 0, :])
        bottom.append(a[:, -1, :])
        if (i + 1) % 500 == 0:
            print("  read %d/%d" % (i + 1, len(ids)))

    out = dict(ids=np.array(ids),
               left=np.stack(left), right=np.stack(right),
               top=np.stack(top), bottom=np.stack(bottom))
    np.savez_compressed(CACHE, **out)
    print("cached edges to %s" % CACHE)
    return out


def corr(a, b):
    """Correlation of two (bands, n) strips, flattened, per pair."""
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def main():
    ids = tile_ids()
    print("tiles: %d   id range: %d..%d   gaps: %d"
          % (len(ids), ids[0], ids[-1], ids[-1] - ids[0] + 1 - len(ids)))
    print()

    E = build_edges(ids)
    ids = list(E["ids"])
    pos = {t: i for i, t in enumerate(ids)}
    print()

    # ---- baseline: what does an UNRELATED pair of strips score? ------------
    rng = np.random.default_rng(0)
    n = len(ids)
    ra = rng.integers(0, n, 4000)
    rb = rng.integers(0, n, 4000)
    keep = ra != rb
    base = corr(E["right"][ra[keep]], E["left"][rb[keep]])
    print("=== H2 baseline: unrelated tile pairs ===")
    print("  right-vs-left correlation over %d random pairs: "
          "mean %.3f   p95 %.3f   p99 %.3f"
          % (keep.sum(), base.mean(), np.percentile(base, 95),
             np.percentile(base, 99)))
    thresh = np.percentile(base, 99)
    print("  a real seam must beat the 99th percentile of noise, %.3f" % thresh)
    print()

    # ---- H1: is id+1 the right-hand neighbour? -----------------------------
    pairs = [(pos[t], pos[t + 1]) for t in ids if (t + 1) in pos]
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    h = corr(E["right"][a], E["left"][b])
    print("=== H1 horizontal: does id+1 sit to the right of id? ===")
    print("  %d consecutive-id pairs   mean %.3f   median %.3f   "
          "frac above noise %.3f"
          % (len(h), h.mean(), np.median(h), (h > thresh).mean()))
    print()

    # ---- H1: sweep the row width W ----------------------------------------
    print("=== H1 vertical: sweep candidate row widths ===")
    print("  a real grid width shows a spike in bottom-vs-top correlation")
    rows = []
    for W in range(2, 260):
        pr = [(pos[t], pos[t + W]) for t in ids if (t + W) in pos]
        if len(pr) < 100:
            continue
        a = np.array([p[0] for p in pr])
        b = np.array([p[1] for p in pr])
        v = corr(E["bottom"][a], E["top"][b])
        rows.append((W, len(pr), v.mean(), (v > thresh).mean()))

    rows.sort(key=lambda r: -r[3])
    print("  %6s %8s %10s %14s" % ("width", "pairs", "mean corr", "frac>noise"))
    for W, npair, m, f in rows[:12]:
        print("  %6d %8d %10.3f %14.3f" % (W, npair, m, f))
    print()
    print("  worst 3 for contrast:")
    for W, npair, m, f in rows[-3:]:
        print("  %6d %8d %10.3f %14.3f" % (W, npair, m, f))


if __name__ == "__main__":
    main()
