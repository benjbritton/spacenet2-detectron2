#!/usr/bin/env python
"""Build the SpaceNet 2 train/val split: 80/20 stratified within each AOI.

WHY STRATIFIED RATHER THAN A GLOBAL SHUFFLE
-------------------------------------------
Splitting 80/20 inside every city, then pooling, gives both readings from one
split: per-city AP from the city subsets, and a pooled number that is simply
their union. A global shuffle would leave each city's val share to chance --
Khartoum has 1012 tiles against Shanghai's 4582, so the small AOIs are exactly
the ones a global shuffle would under-represent, and they are the ones whose AP
is already least stable.

WHY THE SPLIT IS WRITTEN TO A FILE RATHER THAN REGENERATED FROM A SEED
----------------------------------------------------------------------
A seed reproduces a shuffle only for as long as the RNG behaves identically.
Python's random is stable across versions, but the guarantee is not one worth
resting a semester's comparability on: if the split silently moves, every run
before the move is measuring against a different val set and the numbers stop
being comparable, with nothing to indicate it happened. The seed is recorded for
provenance; the membership lists are the authority.

THIS SEED IS NOT THE TRAINING SEED. cfg.SEED varies deliberately between runs to
measure variance. The split must stay fixed across all of them, or seed variance
and split variance are added together with no way to separate them afterwards.

KNOWN LIMITATION -- SPATIAL AUTOCORRELATION
-------------------------------------------
SpaceNet chips do not overlap, but adjacent chips share a street grid, roof
materials, sun angle and acquisition. A random tile split therefore puts near
neighbours on both sides, and val scores are optimistic relative to genuinely
unseen territory. The published baselines split the same way, so this is the
comparable choice, but it should be stated in any write-up rather than implied
away. A spatially blocked split would quantify the gap.

OUTPUT
------
configs/spacenet2_split.json   versioned membership record (the authority)
data/spacenet2/coco/pooled_{train,val}.json
                               COCO files for detectron2, gitignored, derived.
                               file_name is rewritten relative to data/spacenet2
                               so one image_root can span all four AOIs.
"""
import argparse
import json
import os
import random

CITIES = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]


def key_of(file_name):
    """img1234 from SN2_buildings_train_AOI_3_Paris_PS-RGB_img1234.tif."""
    return file_name.rsplit("_", 1)[-1].replace(".tif", "")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/spacenet2")
    p.add_argument("--split-out", default="configs/spacenet2_split.json")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--split-seed", type=int, default=20260827,
                   help="provenance only; the written lists are the authority")
    args = p.parse_args()

    coco_dir = os.path.join(args.root, "coco")
    record = {
        "_note": ("80/20 stratified within each AOI. This is NOT the training "
                  "seed -- cfg.SEED varies per run, this split must not."),
        "split_seed": args.split_seed,
        "val_frac": args.val_frac,
        "cities": {},
    }

    pooled = {
        "train": {"images": [], "annotations": [], "categories": None},
        "val": {"images": [], "annotations": [], "categories": None},
    }
    next_img_id = {"train": 1, "val": 1}
    next_ann_id = {"train": 1, "val": 1}

    for city in CITIES:
        path = os.path.join(coco_dir, "%s_train.json" % city)
        if not os.path.isfile(path):
            print("skip (missing):", path)
            continue
        with open(path) as f:
            d = json.load(f)

        # Sort first so the shuffle starts from a defined order regardless of
        # how the source file happened to be written.
        imgs = sorted(d["images"], key=lambda im: int(key_of(im["file_name"])[3:]))
        rng = random.Random(args.split_seed)
        order = list(range(len(imgs)))
        rng.shuffle(order)
        n_val = int(round(len(imgs) * args.val_frac))
        val_idx = set(order[:n_val])

        by_image = {}
        for a in d["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)

        members = {"train": [], "val": []}
        for i, im in enumerate(imgs):
            split = "val" if i in val_idx else "train"
            members[split].append(key_of(im["file_name"]))

            new_id = next_img_id[split]
            next_img_id[split] += 1
            pooled[split]["images"].append({
                "id": new_id,
                # Relative to --root so a single image_root spans all four AOIs.
                "file_name": os.path.join(city, "PS-RGB", im["file_name"]),
                "width": im["width"],
                "height": im["height"],
                "aoi": city,
            })
            for a in by_image.get(im["id"], []):
                a = dict(a)
                a["id"] = next_ann_id[split]
                next_ann_id[split] += 1
                a["image_id"] = new_id
                pooled[split]["annotations"].append(a)

        if pooled["train"]["categories"] is None:
            pooled["train"]["categories"] = d["categories"]
            pooled["val"]["categories"] = d["categories"]

        record["cities"][city] = {
            "n_images": len(imgs),
            "n_train": len(members["train"]),
            "n_val": len(members["val"]),
            "train": members["train"],
            "val": members["val"],
        }
        print("%-16s %5d images -> %5d train / %4d val"
              % (city, len(imgs), len(members["train"]), len(members["val"])))

    os.makedirs(os.path.dirname(os.path.abspath(args.split_out)), exist_ok=True)
    with open(args.split_out, "w") as f:
        json.dump(record, f, indent=1)

    for split in ("train", "val"):
        out = os.path.join(coco_dir, "pooled_%s.json" % split)
        pooled[split]["info"] = {
            "description": "SpaceNet 2 pooled %s, 80/20 stratified per AOI" % split,
            "split_seed": args.split_seed,
            "file_name_root": args.root,
        }
        with open(out, "w") as f:
            json.dump(pooled[split], f)
        print("%-5s : %6d images  %7d annotations  -> %s"
              % (split, len(pooled[split]["images"]),
                 len(pooled[split]["annotations"]), out))
    print("wrote", args.split_out)


if __name__ == "__main__":
    main()
