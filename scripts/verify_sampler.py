#!/usr/bin/env python
"""Does arm E's repeat sampler actually oversample what it is meant to?

REPEAT_THRESHOLD was chosen as 0.5 on the argument that it is the largest value
leaving the common classes untouched, since platform appears in 55.3% of tiles.
That is arithmetic until it is checked against what detectron2 computes, so this
prints the factors it actually derives and confirms only aguada tiles move.

Also builds the training loader with the custom mapper, because the sampler is
selected inside _train_loader_from_config and a mapper passed by keyword is the
case most likely to bypass it.
"""
import sys
from collections import Counter

import numpy as np

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)   # repo root, whatever it is called

sys.path.insert(0, ROOT + "/src")

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.samplers import RepeatFactorTrainingSampler

from detlab.datasets import chactun

CLASSES = ["building", "platform", "aguada"]


def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(ROOT + "/configs/"
                        "chactun_E_maskrcnn_repeat_sampler.yaml")
    cfg.DATASETS.TRAIN = ("chactun_fold0_train",)

    chactun.register_fold(root=ROOT + "/data/chactun", fold=0)
    dicts = DatasetCatalog.get("chactun_fold0_train")
    meta = MetadataCatalog.get("chactun_fold0_train")
    names = meta.thing_classes
    print("sampler         :", cfg.DATALOADER.SAMPLER_TRAIN)
    print("repeat threshold:", cfg.DATALOADER.REPEAT_THRESHOLD)
    print("images          :", len(dicts))
    print()

    # class frequency by IMAGE, which is what the sampler uses
    freq = Counter()
    for d in dicts:
        for c in {a["category_id"] for a in d.get("annotations", [])}:
            freq[names[c]] += 1
    print("%-10s %10s %10s" % ("class", "images", "fraction"))
    for c in CLASSES:
        print("%-10s %10d %10.4f" % (c, freq[c], freq[c] / len(dicts)))

    rf = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
        dicts, cfg.DATALOADER.REPEAT_THRESHOLD)
    rf = np.asarray(rf, dtype=float)

    print()
    print("repeat factor per image: min %.3f  median %.3f  max %.3f"
          % (rf.min(), np.median(rf), rf.max()))

    # which images got boosted, and do they all contain aguada?
    boosted = rf > 1.0001
    has_aguada = np.array([
        any(names[a["category_id"]] == "aguada" for a in d.get("annotations", []))
        for d in dicts])
    print("images boosted           : %d" % boosted.sum())
    print("images containing aguada : %d" % has_aguada.sum())
    print("boosted set == aguada set: %s"
          % bool(np.array_equal(boosted, has_aguada)))
    if boosted.any():
        print("boost factor applied     : %.3f (predicted sqrt(0.5/0.036) = %.3f)"
              % (rf[boosted].mean(), (0.5 / (freq["aguada"] / len(dicts))) ** 0.5))

    print()
    print("effective epoch size: %.0f draws vs %d images (%.1f%% larger)"
          % (rf.sum(), len(dicts), 100 * (rf.sum() / len(dicts) - 1)))

    print()
    print("=== does the real loader accept it with the custom mapper? ===")
    from detectron2.data import build_detection_train_loader
    loader = build_detection_train_loader(
        cfg, mapper=chactun.ChactunMapper(cfg, is_train=True, d4=False))
    it = iter(loader)
    batch = next(it)
    print("  batch size %d, first image %s, instances %s"
          % (len(batch), tuple(batch[0]["image"].shape),
             [len(b["instances"]) for b in batch[:4]]))
    print("  sampler in use:", type(loader.sampler if hasattr(loader, "sampler")
                                    else loader).__name__)
    print()
    print("VERDICT: repeat sampling boosts only the rare class"
          if bool(np.array_equal(boosted, has_aguada))
          else "VERDICT: threshold is touching the common classes -- lower it")


if __name__ == "__main__":
    main()
