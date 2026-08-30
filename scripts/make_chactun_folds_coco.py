#!/usr/bin/env python
"""Expand the Chactun fold assignment into per-fold COCO files.

register_coco_instances takes one JSON per split, which is the pattern the
SpaceNet side already uses, so the folds are materialised as files rather than
filtered at registration time. Each fold k produces:

  fold{k}_train.json          the other four folds
  fold{k}_val.json            fold k
  fold{k}_val_noedge.json     fold k, minus every instance touching a tile edge

The noedge variant exists because 35% of Chactun instances are cut by tile
boundaries and are therefore partial structures. Whether they help or hurt is an
empirical question, and answering it should not require reconverting anything --
so both ground truths are written up front and the same predictions can be
scored against either.

Note what the noedge variant does NOT do: it drops the annotations but keeps the
images. A partial structure still appears in the imagery, so a model that
detects it is not wrong, merely unrewarded. Scores on this variant are therefore
a lower bound on precision, and that is the honest way to read them.
"""
import argparse
import copy
import json
import os


def write(path, base, images, annotations):
    d = copy.deepcopy({k: v for k, v in base.items()
                       if k not in ("images", "annotations")})
    d["images"] = images
    d["annotations"] = annotations
    with open(path, "w") as f:
        json.dump(d, f)
    return len(images), len(annotations)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco", default="/w/data/chactun/coco/chactun_cc.json")
    p.add_argument("--folds", default="/w/data/chactun/splits/folds5.json")
    p.add_argument("--out-dir", default="/w/data/chactun/coco")
    a = p.parse_args()

    base = json.load(open(a.coco))
    folds = json.load(open(a.folds))["folds"]
    n = len(folds)

    by_tile = {im["tile"]: im for im in base["images"]}
    anns_by_image = {}
    for ann in base["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(a.out_dir, exist_ok=True)
    print("%d folds, %d images, %d annotations"
          % (n, len(base["images"]), len(base["annotations"])))
    print()
    print("%-24s %8s %14s" % ("file", "images", "annotations"))

    for k in range(n):
        val_tiles = set(folds[k])
        train_tiles = set(t for j in range(n) if j != k for t in folds[j])
        assert not (val_tiles & train_tiles), "fold %d overlaps train" % k

        for name, tiles in (("train", train_tiles), ("val", val_tiles)):
            imgs = [by_tile[t] for t in sorted(tiles)]
            anns = [x for im in imgs for x in anns_by_image.get(im["id"], [])]
            f = os.path.join(a.out_dir, "fold%d_%s.json" % (k, name))
            ni, na = write(f, base, imgs, anns)
            print("%-24s %8d %14d" % (os.path.basename(f), ni, na))

        # edge-free ground truth for the same val images
        imgs = [by_tile[t] for t in sorted(val_tiles)]
        anns = [x for im in imgs for x in anns_by_image.get(im["id"], [])
                if not x.get("edge_touching")]
        f = os.path.join(a.out_dir, "fold%d_val_noedge.json" % k)
        ni, na = write(f, base, imgs, anns)
        print("%-24s %8d %14d" % (os.path.basename(f), ni, na))

    print()
    print("wrote %d files to %s" % (3 * n, a.out_dir))


if __name__ == "__main__":
    main()
