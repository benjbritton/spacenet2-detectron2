#!/usr/bin/env python
"""Generate Chactun-style 3-band input from a DEM: SVF, positive openness, slope.

WHY
---
The Chactun model was trained on three channels carrying DIFFERENT information.
The G1 composite hands it one visualisation replicated three times, so the model
is reading a representation it has never seen. This builds the matched
representation from the DEM sitting in the same folder, so the two can be run on
identical ground and the difference attributed to input alone.

THE THREE VISUALISATIONS
------------------------
For N azimuths, look outward to radius R and find the maximum elevation angle
to the horizon, gamma.

  sky-view factor      mean over azimuths of (1 - sin(max(gamma, 0)))
                       the fraction of sky visible; low in hollows
  positive openness    mean over azimuths of (90deg - gamma)
                       Yokoyama's measure; high on convex ground
  slope                arctan of the gradient magnitude, in degrees

RADIUS IS MATCHED IN METRES, NOT PIXELS. Chactun is 0.5 m and this data is
0.33 m, so an identical pixel radius would search a smaller patch of ground and
produce a systematically different visualisation. Same reasoning as tiling by
ground extent rather than pixel count.

THE 8-BIT STRETCH
-----------------
RVT's exact byte mapping is not reproducible here, so rather than guess, each
band is linearly mapped so its valid pixels carry Chactun's own mean and
standard deviation. That puts the input where the model expects it by
construction, which is the property that matters, and it is stated here rather
than hidden.
"""
import argparse
import os

import numpy as np
import rasterio
from rasterio.windows import Window

CHACTUN_MEAN = [216.527, 198.453, 228.612]
CHACTUN_SD = [26.915, 16.698, 21.212]


def horizon_stats(dem, res, n_dir, R, valid):
    """Return (svf, positive openness in degrees) for one block."""
    h, w = dem.shape
    svf_sum = np.zeros((h, w), np.float32)
    op_sum = np.zeros((h, w), np.float32)

    pad = R
    dp = np.pad(dem, pad, mode="edge")

    for az in np.linspace(0.0, 2.0 * np.pi, n_dir, endpoint=False):
        dxu, dyu = np.cos(az), np.sin(az)
        best = np.full((h, w), -np.inf, np.float32)
        for d in range(1, R + 1):
            ox = int(round(dxu * d))
            oy = int(round(dyu * d))
            sl = dp[pad + oy: pad + oy + h, pad + ox: pad + ox + w]
            # tangent of the elevation angle to that neighbour
            np.maximum(best, (sl - dem) / (d * res), out=best)
        gamma = np.arctan(best)                      # radians
        svf_sum += 1.0 - np.sin(np.maximum(gamma, 0.0))
        op_sum += (np.pi / 2.0) - gamma
    return svf_sum / n_dir, np.degrees(op_sum / n_dir)


def slope_deg(dem, res):
    gy, gx = np.gradient(dem, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dem", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--directions", type=int, default=16)
    p.add_argument("--radius-m", type=float, default=5.0,
                   help="search radius in METRES, matched across resolutions")
    p.add_argument("--block", type=int, default=2500)
    a = p.parse_args()

    src = rasterio.open(a.dem)
    res = abs(src.transform.a)
    R = max(1, int(round(a.radius_m / res)))
    print("dem       : %s" % os.path.basename(a.dem))
    print("size      : %d x %d at %.3f m, crs %s" % (src.width, src.height, res, src.crs))
    print("radius    : %.1f m = %d px, %d directions" % (a.radius_m, R, a.directions))
    nod = src.nodata
    print("nodata    : %s" % nod)

    out = np.zeros((3, src.height, src.width), np.float32)
    validmask = np.zeros((src.height, src.width), bool)

    step = a.block
    for row in range(0, src.height, step):
        h = min(step, src.height - row)
        top = max(0, row - R)
        bot = min(src.height, row + h + R)
        dem = src.read(1, window=Window(0, top, src.width, bot - top)).astype(np.float32)

        v = np.isfinite(dem)
        if nod is not None:
            v &= dem != nod
        v &= dem > -9000
        if v.sum() > 0:
            dem = np.where(v, dem, np.nanmedian(dem[v]))
        else:
            dem = np.zeros_like(dem)

        svf, opos = horizon_stats(dem, res, a.directions, R, v)
        slp = slope_deg(dem, res)

        o0 = row - top
        out[0, row:row + h] = svf[o0:o0 + h]
        out[1, row:row + h] = opos[o0:o0 + h]
        out[2, row:row + h] = slp[o0:o0 + h]
        validmask[row:row + h] = v[o0:o0 + h]
        print("  rows %d-%d" % (row, row + h))

    print()
    print("Raw band ranges before the stretch:")
    names = ["sky-view factor", "positive openness", "slope (deg)"]
    for b in range(3):
        v = out[b][validmask]
        print("  %-20s min %8.3f  max %8.3f  mean %8.3f"
              % (names[b], v.min(), v.max(), v.mean()))

    rgb = np.zeros((3, src.height, src.width), np.uint8)
    for b in range(3):
        v = out[b][validmask]
        sd = v.std()
        if sd < 1e-6:
            sd = 1.0
        z = (out[b] - v.mean()) / sd
        scaled = z * CHACTUN_SD[b] + CHACTUN_MEAN[b]
        scaled[~validmask] = 0
        rgb[b] = np.clip(scaled, 0, 255).astype(np.uint8)
        vv = rgb[b][validmask]
        print("  band %d after stretch: mean %.1f sd %.1f (Chactun %.1f / %.1f)"
              % (b + 1, vv.mean(), vv.std(), CHACTUN_MEAN[b], CHACTUN_SD[b]))

    prof = {"driver": "GTiff", "height": src.height, "width": src.width,
            "count": 3, "dtype": "uint8", "crs": src.crs,
            "transform": src.transform, "compress": "deflate",
            "photometric": "RGB", "tiled": True}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with rasterio.open(a.out, "w", **prof) as dst:
        dst.write(rgb)
        dst.update_tags(bands="svf,positive_openness,slope",
                        radius_m=str(a.radius_m),
                        directions=str(a.directions),
                        stretch="linear to Chactun per-band mean/sd")
    print("wrote %s" % a.out)
    src.close()


if __name__ == "__main__":
    main()
