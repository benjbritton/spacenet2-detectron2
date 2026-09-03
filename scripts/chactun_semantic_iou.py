#!/usr/bin/env python
"""Score instance predictions as semantic masks, the way the challenge did.

WHY THIS EXISTS
---------------
Our evaluation is instance AP. The ECML PKDD 2021 leaderboard (Kocev et al.)
is semantic IoU on tiles 1765-2093. To place our result beside those published
numbers the same predictions have to be scored their way, so the instance masks
are unioned per class per tile and compared against the released mask rasters.

WHAT THE PUBLISHED METRIC LEAVES OPEN
-------------------------------------
The overall column IS resolved: it is the unweighted mean of the three class
IoUs. Checked against every leaderboard row -- Aksell 0.9844/0.7651/0.7530
averages to 0.8342 against 0.8341 reported, and the other rows agree exactly.

What is NOT stated is whether a class IoU is pooled over all pixels or averaged
over tiles, and if averaged, what an empty prediction on an empty tile scores.
That matters enormously for aguadas, which appear in 13 of the 329 test tiles:
under per-tile averaging with empty-empty counted as a hit, ~96% of the score
is agreement about absence. So all three conventions are computed and reported
side by side rather than one being assumed.

GROUND TRUTH is read from the released mask rasters, not from our COCO file,
because that is what the leaderboard was scored against. Polarity is 0 = object,
255 = background, as stated by the organisers.

The score threshold is swept, since a competition submission is a binary mask
and every team tuned that choice against the leaderboard.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

CLASSES = ["building", "platform", "aguada"]


def gt_mask(masks_dir, tile, cls):
    p = os.path.join(masks_dir, "tile_%d_mask_%s.tif" % (tile, cls))
    return np.array(Image.open(p)) == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True,
                   help="instances_predictions.pth from the detectron2 evaluator")
    p.add_argument("--coco", required=True, help="fold9_val.json, for image_id -> tile")
    p.add_argument("--masks-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    a = p.parse_args()

    coco = json.load(open(a.coco))
    id2tile = {im["id"]: im.get("tile", im["id"]) for im in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    shape = (coco["images"][0]["height"], coco["images"][0]["width"])

    # detectron2 writes the same predictions twice with DIFFERENT category
    # ids: instances_predictions.pth keeps the contiguous 0-based ids the
    # model emits, while coco_instances_results.json has the reverse mapping
    # to dataset ids applied. Scoring the .pth as if it were the .json
    # silently shifts every class by one, which reads as a plausible-looking
    # score for one class and zero for the rest. Detect and normalise.
    if a.pred.endswith('.json'):
        flat = json.load(open(a.pred))
    else:
        raw = torch.load(a.pred, map_location='cpu', weights_only=False)
        flat = [i for r in raw for i in r.get('instances', [])]

    seen = sorted({d['category_id'] for d in flat})
    dataset_ids = sorted(cats)
    if seen and min(seen) == 0:
        remap = {i: dataset_ids[i] for i in range(len(dataset_ids))}
        for d in flat:
            d['category_id'] = remap[d['category_id']]
        print('category ids were contiguous %s -> remapped to %s'
              % (seen, dataset_ids))
    else:
        print('category ids already dataset ids: %s' % seen)

    by_img = {}
    for d in flat:
        by_img.setdefault(d['image_id'], []).append(d)
    print('predictions for %d images; %d val images'
          % (len(by_img), len(coco['images'])))
    ths = [float(x) for x in a.thresholds.split(",")]
    results = {}

    for th in ths:
        # accumulators
        pool_i = {c: 0 for c in CLASSES}
        pool_u = {c: 0 for c in CLASSES}
        tile_iou = {c: [] for c in CLASSES}          # empty-empty counted as 1.0
        tile_iou_nz = {c: [] for c in CLASSES}       # empty-empty excluded

        for im in coco["images"]:
            tile = id2tile[im["id"]]
            insts = by_img.get(im["id"], [])
            per_cls = {c: np.zeros(shape, bool) for c in CLASSES}
            for d in insts:
                if d.get("score", 0.0) < th:
                    continue
                name = cats.get(d["category_id"])
                if name not in per_cls:
                    continue
                seg = d["segmentation"]
                m = mask_util.decode(seg).astype(bool)
                if m.shape != shape:
                    continue
                per_cls[name] |= m

            for c in CLASSES:
                g = gt_mask(a.masks_dir, tile, c)
                pr = per_cls[c]
                inter = int(np.logical_and(g, pr).sum())
                union = int(np.logical_or(g, pr).sum())
                pool_i[c] += inter
                pool_u[c] += union
                if union == 0:
                    tile_iou[c].append(1.0)           # both empty, agreed
                else:
                    v = inter / union
                    tile_iou[c].append(v)
                    tile_iou_nz[c].append(v)

        row = {}
        for label, getter in (
                ("pooled", lambda c: pool_i[c] / pool_u[c] if pool_u[c] else float("nan")),
                ("per_tile_emptyhit", lambda c: float(np.mean(tile_iou[c]))),
                ("per_tile_nonempty", lambda c: float(np.mean(tile_iou_nz[c]))
                    if tile_iou_nz[c] else float("nan")),
        ):
            vals = {c: getter(c) for c in CLASSES}
            vals["overall"] = float(np.mean([vals[c] for c in CLASSES]))
            row[label] = vals
        results["%.2f" % th] = row

    hdr = "%-6s %-20s %9s %9s %9s %9s" % ("thr", "convention", "buildings",
                                          "platforms", "aguadas", "overall")
    print()
    print(hdr)
    print("-" * len(hdr))
    for th in ths:
        r = results["%.2f" % th]
        for conv in ("pooled", "per_tile_emptyhit", "per_tile_nonempty"):
            v = r[conv]
            print("%-6.2f %-20s %9.4f %9.4f %9.4f %9.4f"
                  % (th, conv, v["building"], v["platform"], v["aguada"], v["overall"]))
    print()
    print("leaderboard for reference (Kocev et al. 2021, 25 teams):")
    print("  1 Aksell           0.7530    0.7651    0.9844    0.8341")
    print("  5 TheSentinels     0.7394    0.7300    0.9854    0.8183")
    print("  overall = unweighted mean of the three class IoUs")

    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
        print()
        print("wrote", a.out)


if __name__ == "__main__":
    main()
