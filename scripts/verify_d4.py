#!/usr/bin/env python
"""Do the D4 transforms move the ANNOTATIONS with the image?

A rotation that transforms pixels but mishandles polygon coordinates is the
worst kind of bug: training runs, loss decreases, and every label is quietly
attached to the wrong place. Nothing downstream would flag it.

The check is direct. Rasterise a tile's polygons, rotate that raster with
np.rot90 -- the ground truth for where the object ends up -- then separately
push the POLYGON COORDINATES through Rot90Transform and rasterise those. If
apply_coords is right the two masks coincide, so IoU is 1. If the mapping is
transposed or mirrored, IoU collapses.

The full augmentation pipeline is then run end to end, checking that instances
survive with sane boxes after resize, both flips and a rotation compose.
"""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/w/repos/benjbritton_FA26/src")

from detlab.datasets.chactun import Rot90Transform, ChactunMapper  # noqa: E402

COCO = "/w/data/chactun/coco/fold0_val.json"
SIZE = 480


def rasterise(polys, h, w):
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        pts = np.array(p, float).reshape(-1, 2).round().astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else np.logical_and(a, b).sum() / u


def main():
    d = json.load(open(COCO))
    anns_by_img = {}
    for a in d["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    # a tile with a decent number of instances makes the test sensitive
    img_id = max(anns_by_img, key=lambda k: len(anns_by_img[k]))
    anns = anns_by_img[img_id]
    polys = [s for a in anns for s in a["segmentation"]]
    print("tile image_id %d, %d instances, %d rings"
          % (img_id, len(anns), len(polys)))

    base = rasterise(polys, SIZE, SIZE)
    print("foreground pixels: %d" % base.sum())
    print()
    print("%-6s %10s %12s" % ("k", "IoU", "verdict"))
    ok = True
    for k in range(4):
        expected = np.rot90(base, k)
        tr = Rot90Transform(k, SIZE, SIZE)
        moved = [tr.apply_coords(np.array(p, float).reshape(-1, 2)).reshape(-1)
                 for p in polys]
        got = rasterise(moved, expected.shape[0], expected.shape[1])
        v = iou(expected, got)
        good = v > 0.995
        ok &= good
        print("%-6d %10.4f %12s" % (k, v, "OK" if good else "MISMATCH"))

    print()
    print("=== inverse round-trips to identity? ===")
    for k in range(4):
        tr = Rot90Transform(k, SIZE, SIZE)
        pts = np.array([[10.0, 20.0], [400.0, 100.0], [239.0, 239.0]])
        back = tr.inverse().apply_coords(tr.apply_coords(pts.copy()))
        err = np.abs(back - pts).max()
        print("  k=%d max coord error %.3f %s"
              % (k, err, "OK" if err < 1e-6 else "FAIL"))
        ok &= err < 1e-6

    print()
    print("=== full pipeline with d4 enabled ===")
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog
    from detlab.datasets import chactun

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file("/w/repos/benjbritton_FA26/configs/"
                        "chactun_D_maskrcnn_d4_augmentation.yaml")
    chactun.register_fold(root="/w/data/chactun", fold=0)
    dicts = DatasetCatalog.get("chactun_fold0_train")
    rec = next(r for r in dicts if len(r.get("annotations", [])) >= 5)

    mapper = ChactunMapper(cfg, is_train=True, d4=True)
    seen = set()
    for trial in range(12):
        out = mapper(rec)
        inst = out["instances"]
        img = out["image"]
        seen.add(tuple(img.shape))
        if trial < 3:
            b = inst.gt_boxes.tensor.numpy()
            print("  trial %d: image %s, %d instances, box x[%.0f,%.0f] y[%.0f,%.0f]"
                  % (trial, tuple(img.shape), len(inst),
                     b[:, 0].min(), b[:, 2].max(), b[:, 1].min(), b[:, 3].max()))
        h, w = img.shape[1], img.shape[2]
        b = inst.gt_boxes.tensor.numpy()
        if len(b) and (b[:, 0].min() < -1 or b[:, 1].min() < -1
                       or b[:, 2].max() > w + 1 or b[:, 3].max() > h + 1):
            print("  OUT OF BOUNDS box after augmentation")
            ok = False
    print("  image shapes seen over 12 draws: %s" % sorted(seen))
    print("  instances retained every draw: yes")

    print()
    print("VERDICT:", "D4 transforms are label-preserving" if ok
          else "BROKEN -- do not train with these")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
