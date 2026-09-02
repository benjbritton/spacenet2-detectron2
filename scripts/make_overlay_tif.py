#!/usr/bin/env python
"""Burn detections into a georeferenced RGB GeoTIFF over the G1 imagery.

Output is 1000 px wide with height set by the source aspect ratio, 3-band uint8
(24-bit colour), carrying the source CRS and a transform scaled to match the
downsample, so it lands in the right place when dropped on a map.

The imagery is drawn as grey and the boxes in colour, so the detections read
against the terrain rather than competing with it.
"""
import argparse
import json
import os

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling

# RGB
COLOURS = {
    "building": (0, 229, 255),
    "platform": (255, 214, 10),
    "aguada": (255, 45, 85),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raster", required=True)
    p.add_argument("--geojson", required=True, nargs="+",
                   help="one or more detection files; later ones drawn thinner")
    p.add_argument("--width", type=int, default=1000)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    src = rasterio.open(a.raster)
    w = a.width
    h = int(round(src.height * w / src.width))
    print("source : %d x %d at %.3f m, %s" % (src.width, src.height,
                                              abs(src.transform.a), src.crs))
    print("output : %d x %d, 3-band uint8" % (w, h))

    band = src.read(1, out_shape=(h, w), resampling=Resampling.average)
    # stretch the valid range so the terrain is legible at this size
    valid = band > 0
    if valid.sum() > 100:
        lo, hi = np.percentile(band[valid], [1, 99])
    else:
        lo, hi = 0, 255
    g = np.clip((band.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255)
    g[~valid] = 0
    rgb = np.dstack([g, g, g]).astype(np.uint8)

    # transform for the downsampled grid
    tr = src.transform * src.transform.scale(src.width / w, src.height / h)

    total = 0
    for i, gj_path in enumerate(a.geojson):
        if not os.path.isfile(gj_path):
            print("missing:", gj_path)
            continue
        d = json.load(open(gj_path))
        thick = 2 if i == 0 else 1
        n = 0
        for feat in d["features"]:
            coords = feat["geometry"]["coordinates"][0]
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            # map -> output pixel via the inverse of the scaled transform
            inv = ~tr
            c0, r0 = inv * (min(xs), max(ys))
            c1, r1 = inv * (max(xs), min(ys))
            x0, y0 = int(round(min(c0, c1))), int(round(min(r0, r1)))
            x1, y1 = int(round(max(c0, c1))), int(round(max(r0, r1)))
            # a structure can be under 2 px at this scale; keep boxes visible
            if x1 - x0 < 3:
                x0, x1 = x0 - 1, x1 + 2
            if y1 - y0 < 3:
                y0, y1 = y0 - 1, y1 + 2
            col = COLOURS.get(feat["properties"].get("class"), (255, 255, 255))
            cv2.rectangle(rgb, (x0, y0), (x1, y1), col, thick)
            n += 1
        print("drew %d from %s (line width %d)" % (n, os.path.basename(gj_path), thick))
        total += n

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": 3,
        "dtype": "uint8", "crs": src.crs, "transform": tr,
        "compress": "deflate", "photometric": "RGB",
    }
    with rasterio.open(a.out, "w", **profile) as dst:
        for b in range(3):
            dst.write(rgb[:, :, b], b + 1)
        dst.update_tags(detections=str(total),
                        source=os.path.basename(a.raster))
    print("wrote %s  (%d boxes)" % (a.out, total))
    src.close()


if __name__ == "__main__":
    main()
