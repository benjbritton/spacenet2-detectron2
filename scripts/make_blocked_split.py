#!/usr/bin/env python
"""Spatially blocked 80/20 split, as a harder alternative to the random one.

THE PROBLEM IT MEASURES
-----------------------
SpaceNet chips do not overlap, but adjacent chips share a street grid, roof
materials, sun angle and acquisition. A random tile split therefore puts near
neighbours on both sides: for most val tiles, the model has trained on the block
next door. Scores are optimistic against genuinely unseen ground by an unknown
amount, which is the loudest open caveat on the XD_XD comparison.

Holding out whole contiguous BLOCKS instead of scattered tiles removes most of
that adjacency. Comparing F1 under the two splits turns "unquantified caveat"
into a number.

BLOCK SIZE
----------
Tiles are 650 px at about 0.3 m, so roughly 195 m across, which in these
EPSG:4326 rasters is about 0.00176 degrees. Blocks are BLOCK_TILES tiles on a
side, default 5, so about 1 km. That is comfortably past the range over which
one neighbourhood looks like the next, while still leaving enough blocks per city
that an 80/20 split lands close to target.

Blocks are assigned by a seeded shuffle rather than by taking one contiguous
chunk of each city. A single chunk would hold out one neighbourhood type -- one
suburb, or one industrial strip -- and measure that rather than generalisation.

VERIFICATION
------------
The point is adjacency, so adjacency is what gets measured: the fraction of val
tiles having at least one immediate (8-neighbour) grid neighbour in train. Under
a random split that is near 1.0. If blocking does not drive it far down, the
block size is too small and the split is not doing its job.
"""
import argparse
import json
import os
import random
from collections import defaultdict

import rasterio

CITIES = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]


def tile_centroids(image_dir):
    """{img_key: (lon, lat)} from each raster geotransform. Headers only."""
    out = {}
    for name in sorted(os.listdir(image_dir)):
        if not name.endswith(".tif"):
            continue
        key = name.rsplit("_", 1)[-1].replace(".tif", "")
        with rasterio.open(os.path.join(image_dir, name)) as src:
            b = src.bounds
        out[key] = ((b.left + b.right) / 2.0, (b.bottom + b.top) / 2.0)
    return out


def grid_index(centroids, cell):
    """{img_key: (col, row)} on a fixed grid, and the tile pitch actually seen."""
    lons = sorted(c[0] for c in centroids.values())
    lats = sorted(c[1] for c in centroids.values())
    lon0, lat0 = lons[0], lats[0]
    return {k: (int((v[0] - lon0) // cell), int((v[1] - lat0) // cell))
            for k, v in centroids.items()}, (lon0, lat0)


def tile_grid(centroids, pitch, origin):
    """{img_key: (col, row)} at TILE resolution, for the adjacency check."""
    lon0, lat0 = origin
    return {k: (int(round((v[0] - lon0) / pitch)), int(round((v[1] - lat0) / pitch)))
            for k, v in centroids.items()}


def adjacency_rate(tile_xy, val_keys, train_keys):
    """Fraction of val tiles with an 8-neighbour in train."""
    train_cells = {tile_xy[k] for k in train_keys}
    touching = 0
    for k in val_keys:
        x, y = tile_xy[k]
        if any((x + dx, y + dy) in train_cells
               for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               if not (dx == 0 and dy == 0)):
            touching += 1
    return touching / float(len(val_keys)) if val_keys else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/spacenet2")
    p.add_argument("--out", default="configs/spacenet2_split_blocked.json")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--block-tiles", type=int, default=5)
    p.add_argument("--split-seed", type=int, default=20260827)
    p.add_argument("--random-split", default="configs/spacenet2_split.json",
                   help="for the adjacency comparison")
    p.add_argument("--coco-out", default="data/spacenet2/coco_blocked",
                   help="where the pooled COCO files go. A separate directory "
                        "rather than a filename prefix, so registration picks "
                        "them up via coco_dir with no other change")
    args = p.parse_args()

    with open(args.random_split) as f:
        rnd = json.load(f)

    record = {
        "_note": ("Spatially blocked 80/20. Whole blocks of about %d tiles on a "
                  "side are assigned to train or val, so val tiles mostly do not "
                  "border training tiles. Harder and fairer than the random "
                  "split in configs/spacenet2_split.json."
                  % args.block_tiles),
        "block_tiles": args.block_tiles,
        "split_seed": args.split_seed,
        "val_frac": args.val_frac,
        "cities": {},
    }

    print("%-16s %6s %6s %6s   %8s   %s" %
          ("city", "tiles", "train", "val", "blocks", "val tiles touching train"))
    for city in CITIES:
        image_dir = os.path.join(args.root, city, "PS-RGB")
        if not os.path.isdir(image_dir):
            continue
        cent = tile_centroids(image_dir)

        # Tile pitch, from the smallest nonzero gap between distinct longitudes.
        lons = sorted({round(v[0], 9) for v in cent.values()})
        gaps = [b - a for a, b in zip(lons, lons[1:]) if b - a > 1e-9]
        pitch = min(gaps) if gaps else 0.001755
        cell = pitch * args.block_tiles

        blocks, origin = grid_index(cent, cell)
        by_block = defaultdict(list)
        for k, b in blocks.items():
            by_block[b].append(k)

        order = sorted(by_block)
        random.Random(args.split_seed).shuffle(order)
        target = int(round(len(cent) * args.val_frac))
        val_keys, n = [], 0
        for b in order:
            if n >= target:
                break
            val_keys.extend(by_block[b])
            n += len(by_block[b])
        val_set = set(val_keys)
        train_keys = [k for k in sorted(cent) if k not in val_set]

        tile_xy = tile_grid(cent, pitch, origin)
        blocked_rate = adjacency_rate(tile_xy, val_keys, train_keys)
        rnd_val = set(rnd["cities"][city]["val"])
        rnd_train = [k for k in cent if k not in rnd_val]
        random_rate = adjacency_rate(tile_xy, [k for k in rnd_val if k in tile_xy],
                                     rnd_train)

        record["cities"][city] = {
            "n_images": len(cent),
            "n_train": len(train_keys),
            "n_val": len(val_keys),
            "n_blocks": len(by_block),
            "adjacency_blocked": blocked_rate,
            "adjacency_random": random_rate,
            "train": sorted(train_keys),
            "val": sorted(val_keys),
        }
        print("%-16s %6d %6d %6d   %8d   blocked %.3f  vs random %.3f"
              % (city, len(cent), len(train_keys), len(val_keys),
                 len(by_block), blocked_rate, random_rate))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1)
    print("wrote", args.out)

    write_pooled(args.root, record, args.coco_out)


def write_pooled(root, record, coco_out):
    """Pooled COCO files for the blocked split, rebuilt from the per-AOI COCO.

    Identical in form to the random-split pooled files: file_name relative to
    root so one image_root spans four directories, and aoi carried per record so
    the per-city breakdown works unchanged.
    """
    src_dir = os.path.join(root, "coco")
    os.makedirs(coco_out, exist_ok=True)
    pooled = {s: {"images": [], "annotations": [], "categories": None}
              for s in ("train", "val")}
    nid = {"train": 1, "val": 1}
    aid = {"train": 1, "val": 1}

    for city in CITIES:
        path = os.path.join(src_dir, "%s_train.json" % city)
        if city not in record["cities"] or not os.path.isfile(path):
            continue
        with open(path) as f:
            d = json.load(f)
        member = {}
        for k in record["cities"][city]["train"]:
            member[k] = "train"
        for k in record["cities"][city]["val"]:
            member[k] = "val"

        by_image = {}
        for a in d["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)

        for im in d["images"]:
            key = im["file_name"].rsplit("_", 1)[-1].replace(".tif", "")
            split = member.get(key)
            if split is None:
                continue
            new_id = nid[split]
            nid[split] += 1
            pooled[split]["images"].append({
                "id": new_id,
                "file_name": os.path.join(city, "PS-RGB", im["file_name"]),
                "width": im["width"],
                "height": im["height"],
                "aoi": city,
            })
            for a in by_image.get(im["id"], []):
                a = dict(a)
                a["id"] = aid[split]
                aid[split] += 1
                a["image_id"] = new_id
                pooled[split]["annotations"].append(a)
        if pooled["train"]["categories"] is None:
            pooled["train"]["categories"] = d["categories"]
            pooled["val"]["categories"] = d["categories"]

    for split in ("train", "val"):
        pooled[split]["info"] = {
            "description": "SpaceNet 2 pooled %s, SPATIALLY BLOCKED split" % split,
            "block_tiles": record["block_tiles"],
            "split_seed": record["split_seed"],
            "file_name_root": root,
        }
        out = os.path.join(coco_out, "pooled_%s.json" % split)
        with open(out, "w") as f:
            json.dump(pooled[split], f)
        print("%-5s : %6d images  %7d annotations  -> %s"
              % (split, len(pooled[split]["images"]),
                 len(pooled[split]["annotations"]), out))


if __name__ == "__main__":
    main()
