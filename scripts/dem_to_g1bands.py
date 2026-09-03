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

RADIUS IS MATCHED IN METRES, NOT PIXELS. Chactun is 0.5 m and other surveys are
not, so an identical pixel radius would search a different patch of ground and
produce a systematically different visualisation. Same reasoning as tiling by
ground extent rather than pixel count.

THE 8-BIT STRETCH: USE --stretch spec
-------------------------------------
An earlier version of this file asserted that RVT's byte mapping was not
reproducible and fell back to matching Chactun's per-band mean and standard
deviation. That assertion was wrong, and the fallback is what made the first
G-LiHT transfer attempt score BELOW the unmodified G1 composite.

The mapping is published. Table 3 of Kokalj et al., Scientific Data 10:558
(2023) -- the Chactun data descriptor itself -- gives, for general terrain:

  sky-view factor      5 m radius, 16 directions, linear 0.7 - 1.0
  positive openness    5 m radius, 16 directions, linear 68 - 93 degrees
  slope                INVERTED greyscale, linear 0 - 50 degrees

and its caption states that those three, at the general-terrain settings, are
the raster bands in the data records. The defaults of --radius-m and
--directions below are those settings.

Two independent checks that this is the right column of Table 3 (it also lists
a flat-terrain column):

  Inverting the stretch on Chactun's own bytes gives physically sensible values
  -- mean SVF 0.955, openness 87.5 deg, slope 5.1 deg. The flat-terrain column
  implies a mean slope of 1.5 deg, which this landscape is not.

  Applying it to a DIFFERENT survey's DEM (G-LiHT Yucatan, South GLAS l0s395
  at 0.5 m) lands on band means of 204.2 / 196.1 / 223.5 over the 4.47 km2 put
  through the model, against Chactun's 216.5 / 198.5 / 228.6 -- within half a
  standard deviation on all three bands, with no statistic matched anywhere in
  the process. A wrong recipe does not land there. Band means depend on the
  extent chosen, so this script prints its own beside Chactun's rather than
  asking anyone to take those figures on trust.

WHY --stretch matched IS STILL HERE
-----------------------------------
Matching band STATISTICS is not the same as applying the same stretch FUNCTION:
it assigns different byte values to the same physical quantity, so the model
receives a third representation rather than its training one. It is kept only so
that result stays reproducible, and it should not be used for new work.
"""
import argparse
import os

import numpy as np
import rasterio
from rasterio.windows import Window

CHACTUN_MEAN = [216.527, 198.453, 228.612]
CHACTUN_SD = [26.915, 16.698, 21.212]

# Kokalj et al. 2023, Table 3, general terrain column. (low, high, inverted)
# byte = 255 * (x - low) / (high - low), or its complement where inverted.
TABLE3 = [
    (0.7, 1.0, False),     # sky-view factor, dimensionless
    (68.0, 93.0, False),   # positive openness, degrees
    (0.0, 50.0, True),     # slope, degrees, inverted greyscale colour bar
]


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


def implied_physical(byte_value, band):
    """Invert the Table 3 stretch: what physical value does this byte mean?"""
    low, high, inv = TABLE3[band]
    frac = byte_value / 255.0
    if inv:
        frac = 1.0 - frac
    return low + frac * (high - low)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dem", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--directions", type=int, default=16,
                   help="Table 3 general terrain: 16")
    p.add_argument("--radius-m", type=float, default=5.0,
                   help="search radius in METRES, matched across resolutions; "
                        "Table 3 general terrain: 5")
    p.add_argument("--stretch", choices=("spec", "matched"), default="spec",
                   help="spec: the published Table 3 mapping (use this). "
                        "matched: rescale to Chactun's per-band mean/sd, kept "
                        "only to reproduce the earlier run")
    p.add_argument("--block", type=int, default=2500)
    a = p.parse_args()

    src = rasterio.open(a.dem)
    res = abs(src.transform.a)
    R = max(1, int(round(a.radius_m / res)))
    print("dem       : %s" % os.path.basename(a.dem))
    print("size      : %d x %d at %.3f m, crs %s" % (src.width, src.height, res, src.crs))
    print("radius    : %.1f m = %d px, %d directions" % (a.radius_m, R, a.directions))
    print("stretch   : %s" % a.stretch)
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
    print()
    for b in range(3):
        if a.stretch == "spec":
            low, high, inv = TABLE3[b]
            frac = (out[b] - low) / (high - low)
            if inv:
                frac = 1.0 - frac
            scaled = frac * 255.0
        else:
            v = out[b][validmask]
            sd = v.std()
            if sd < 1e-6:
                sd = 1.0
            scaled = (out[b] - v.mean()) / sd * CHACTUN_SD[b] + CHACTUN_MEAN[b]
        scaled[~validmask] = 0
        rgb[b] = np.clip(scaled, 0, 255).astype(np.uint8)

        vv = rgb[b][validmask]
        print("  band %d after stretch: mean %6.1f sd %5.1f  (Chactun %.1f / %.1f)"
              % (b + 1, vv.mean(), vv.std(), CHACTUN_MEAN[b], CHACTUN_SD[b]))
        if a.stretch == "spec":
            # The byte means should land near Chactun's WITHOUT having been made
            # to. Report the physical reading of both, since that is the
            # comparison that means something across two different surveys.
            print("      implied %-18s here %7.3f   Chactun bytes imply %7.3f"
                  % (names[b], implied_physical(vv.mean(), b),
                     implied_physical(CHACTUN_MEAN[b], b)))
            clipped = float((np.clip(scaled[validmask], 0, 255) != scaled[validmask]).mean())
            print("      clipped outside the Table 3 range: %.2f%%" % (100.0 * clipped))

    if a.stretch == "spec":
        tag = ("Kokalj et al. 2023 Table 3, general terrain: "
               "svf linear 0.7-1.0, opos linear 68-93 deg, "
               "slope inverted linear 0-50 deg")
    else:
        tag = "linear to Chactun per-band mean/sd (superseded, see docstring)"

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
                        stretch=tag)
    print()
    print("wrote %s" % a.out)
    src.close()


if __name__ == "__main__":
    main()
