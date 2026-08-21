#!/usr/bin/env python
"""Fine-tune Mask R-CNN R50-FPN on the balloon dataset, tracked in W&B.

Run inside the m2/detectron2 container -- see scripts/run.sh.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode, Visualizer

from detlab.datasets import balloon
from detlab.trainer import LabTrainer

BASE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=None, help="override SOLVER.MAX_ITER")
    p.add_argument("--eval-period", type=int, default=None)
    p.add_argument("--data-root", default=os.path.join(REPO, "data"))
    p.add_argument("--output", default=None, help="override OUTPUT_DIR")
    p.add_argument("--project", default="fa26-independent-study")
    p.add_argument("--run-name", default=None)
    p.add_argument("--offline", action="store_true", help="WANDB_MODE=offline; sync later")
    p.add_argument("--no-wandb", action="store_true", help="disable W&B entirely")
    p.add_argument("--smoke", action="store_true", help="50 iters, no eval -- plumbing check")
    return p.parse_args()


def build_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE))
    cfg.merge_from_file(os.path.join(REPO, "configs", "balloon_mask_rcnn_R50_FPN.yaml"))
    # COCO-pretrained starting weights.
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(BASE)

    if args.smoke:
        cfg.SOLVER.MAX_ITER = 50
        cfg.SOLVER.STEPS = []
        cfg.TEST.EVAL_PERIOD = 0
        cfg.OUTPUT_DIR = os.path.join(REPO, "outputs", "smoke")
    if args.iters is not None:
        cfg.SOLVER.MAX_ITER = args.iters
        cfg.SOLVER.STEPS = [s for s in cfg.SOLVER.STEPS if s < args.iters]
    if args.eval_period is not None:
        cfg.TEST.EVAL_PERIOD = args.eval_period
    if args.output is not None:
        cfg.OUTPUT_DIR = args.output
    if not os.path.isabs(cfg.OUTPUT_DIR):
        cfg.OUTPUT_DIR = os.path.join(REPO, cfg.OUTPUT_DIR)
    return cfg


def save_sample_predictions(cfg, data_root, n=3):
    """Visual sanity check -- the numbers can look fine and the masks still be wrong."""
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    predictor = DefaultPredictor(cfg)
    meta = MetadataCatalog.get("balloon_val")
    dicts = balloon.get_balloon_dicts(os.path.join(data_root, "balloon", "val"))

    out_dir = os.path.join(cfg.OUTPUT_DIR, "samples")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for d in dicts[:n]:
        im = cv2.imread(d["file_name"])
        inst = predictor(im)["instances"].to("cpu")
        vis = Visualizer(im[:, :, ::-1], meta, scale=1.0, instance_mode=ColorMode.IMAGE_BW)
        path = os.path.join(out_dir, os.path.basename(d["file_name"]))
        cv2.imwrite(path, vis.draw_instance_predictions(inst).get_image()[:, :, ::-1])
        written.append((path, len(inst)))
    return written


def main():
    args = parse_args()
    setup_logger()

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    balloon.register(args.data_root)
    cfg = build_cfg(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project=args.project,
            name=args.run_name or f"balloon-maskrcnn-r50-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "base_config": BASE,
                "max_iter": cfg.SOLVER.MAX_ITER,
                "base_lr": cfg.SOLVER.BASE_LR,
                "ims_per_batch": cfg.SOLVER.IMS_PER_BATCH,
                "amp": cfg.SOLVER.AMP.ENABLED,
                "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
            },
        )
        print(f"W&B run: {run.url}")

    trainer = LabTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")

    if not args.smoke:
        results = LabTrainer.test(cfg, trainer.model)
        print("final eval:", results)
        if run is not None:
            run.summary.update({f"final/{k}": v for k, v in results.get("segm", {}).items()})
        for path, n in save_sample_predictions(cfg, args.data_root):
            print(f"wrote {path}  ({n} instances)")

    print(f"artifacts in: {cfg.OUTPUT_DIR}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
