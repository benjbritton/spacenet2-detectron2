#!/usr/bin/env python
"""How much does resolution mismatch cost a Chactun-trained detector?

THE QUESTION
------------
The tool is meant to run on other people's LiDAR, which arrives at whatever
resolution the survey happened to use -- 1 m, 50 cm, 33 cm. Training was at
50 cm. Q2000 (trained at 1 m) did detect objects on 33 cm G-LiHT, so mismatch is
clearly survivable; what is unknown is what it COSTS. "It worked" and "it worked
as well as matched resolution" are different claims and only the second one
justifies skipping resampling discipline across a region.

WHAT IS SIMULATED, AND WHY BOTH HALVES MATTER
---------------------------------------------
Feeding a survey at a different resolution changes two things at once, and a
simulation that moves only one of them measures the wrong quantity.

  1. DETAIL. A 1 m survey of the same ground genuinely holds less information
     than a 50 cm one. Simulated by resampling the raster to 480*s and back up,
     so detail is capped at the coarser grid.

  2. SCALE IN NETWORK INPUT. Tiling code that emits fixed 480x480 pixel tiles
     covers FOUR times the ground per tile at 1 m, so objects occupy half as
     many pixels. ResizeShortestEdge would otherwise silently normalise this
     away, hiding exactly the effect being measured, so MIN_SIZE_TEST is scaled
     by s in step with the raster.

Net effect at scale s: objects appear at s times their baseline pixel size, with
detail capped at s times the baseline grid. s = 0.5 simulates a 1 m survey,
s = 1.5 a 33 cm survey, s = 3.0 the roughly threefold refinement Q2000 was
pushed through.

Predictions are scored against the ORIGINAL annotations. detectron2 rescales
detections to the height and width carried in the dataset dict, so keeping those
at the native 480 makes every scale directly comparable without touching the
ground truth.

Arms A and D are both run, because whether D4 augmentation buys scale robustness
as well as accuracy is worth knowing: rotation invariance is not scale
invariance, and it would be easy to assume the gain transfers.
"""
import copy
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/workspace/src")

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor  # noqa: F401  (kept for parity)
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger

from detlab.datasets import chactun

REPO = "/workspace"
ARMS = {
    "A": ("outputs/chactun_A_maskrcnn_default_anchors",
          "chactun_A_maskrcnn_default_anchors.yaml"),
    "D": ("outputs/chactun_D_maskrcnn_d4_augmentation",
          "chactun_D_maskrcnn_d4_augmentation.yaml"),
}
BASE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
SCALES = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
NATIVE_M = 0.5


class ScaledMapper:
    """Test mapper that resamples the tile, simulating a coarser or finer survey."""

    def __init__(self, cfg, scale):
        from detectron2.data import transforms as T

        self.scale = scale
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST, sample_style="choice")
        self._T = T

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = chactun.load_image(dataset_dict["file_name"])
        h, w = image.shape[:2]

        if abs(self.scale - 1.0) > 1e-6:
            nh, nw = max(16, int(round(h * self.scale))), \
                     max(16, int(round(w * self.scale)))
            interp = cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR
            image = cv2.resize(image, (nw, nh), interpolation=interp)

        aug_input = self._T.AugInput(image)
        self.aug(aug_input)
        image = aug_input.image
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))
        # height/width stay NATIVE so detections are rescaled back for scoring
        dataset_dict["height"] = h
        dataset_dict["width"] = w
        dataset_dict.pop("annotations", None)
        return dataset_dict


def build_cfg(arm, scale):
    root, repo_cfg = ARMS[arm]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE))
    cfg.merge_from_file(os.path.join(REPO, "configs", repo_cfg))
    # scale the test size in step with the raster, so the object size the
    # network sees actually changes rather than being normalised away
    cfg.INPUT.MIN_SIZE_TEST = int(round(cfg.INPUT.MIN_SIZE_TEST * scale))
    cfg.INPUT.MAX_SIZE_TEST = int(round(cfg.INPUT.MAX_SIZE_TEST * max(scale, 1.0)))
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def main():
    setup_logger()
    os.chdir(REPO)
    for f in range(5):
        chactun.register_fold(root=os.path.join(REPO, "data", "chactun"), fold=f)

    results = {}
    for arm in ("A", "D"):
        root = ARMS[arm][0]
        for scale in SCALES:
            aps, aps50, blds, plats, agus = [], [], [], [], []
            for fold in range(5):
                ckpt = os.path.join(REPO, root, "fold%d_seed0" % fold,
                                    "model_final.pth")
                if not os.path.isfile(ckpt):
                    continue
                cfg = build_cfg(arm, scale)
                model = build_model(cfg)
                DetectionCheckpointer(model).load(ckpt)
                model.eval()

                name = "chactun_fold%d_val" % fold
                out = os.path.join("/tmp", "scale_%s_%s_%d" % (arm, scale, fold))
                evaluator = COCOEvaluator(name, output_dir=out)
                loader = build_detection_test_loader(
                    cfg, name, mapper=ScaledMapper(cfg, scale))
                res = inference_on_dataset(model, loader, evaluator)
                s = res["segm"]
                aps.append(s["AP"]); aps50.append(s["AP50"])
                blds.append(s.get("AP-building", float("nan")))
                plats.append(s.get("AP-platform", float("nan")))
                agus.append(s.get("AP-aguada", float("nan")))
                del model
                torch.cuda.empty_cache()
            if aps:
                results[(arm, scale)] = dict(
                    AP=float(np.mean(aps)), AP50=float(np.mean(aps50)),
                    building=float(np.nanmean(blds)),
                    platform=float(np.nanmean(plats)),
                    aguada=float(np.nanmean(agus)),
                    sd=float(np.std(aps, ddof=1)) if len(aps) > 1 else 0.0)
                print("[scale] arm %s  s=%.2f  AP %.2f" %
                      (arm, scale, results[(arm, scale)]["AP"]))

    print()
    print("=== scale sensitivity: trained at 0.5 m, tested elsewhere ===")
    print("%-6s %7s %12s %9s %8s %10s %10s %8s"
          % ("arm", "scale", "simulated m", "segm AP", "AP50", "building",
             "platform", "aguada"))
    for arm in ("A", "D"):
        base = results.get((arm, 1.0), {}).get("AP")
        for scale in SCALES:
            r = results.get((arm, scale))
            if not r:
                continue
            delta = "" if base is None else "  (%+.2f)" % (r["AP"] - base)
            print("%-6s %7.2f %12.3f %9.2f %8.2f %10.2f %10.2f %8.2f%s"
                  % (arm, scale, NATIVE_M / scale, r["AP"], r["AP50"],
                     r["building"], r["platform"], r["aguada"], delta))
        print()

    with open("/workspace/outputs/scale_sensitivity.json", "w") as fh:
        json.dump({"%s@%.2f" % k: v for k, v in results.items()}, fh, indent=1)
    print("wrote outputs/scale_sensitivity.json")


if __name__ == "__main__":
    main()
