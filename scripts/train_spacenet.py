#!/usr/bin/env python
"""Train Mask R-CNN R50-FPN on pooled SpaceNet 2, tracked in W&B.

Run inside the m2/detectron2 container -- see scripts/run.sh.

    ./scripts/run.sh python scripts/train_spacenet.py --smoke
    ./scripts/run.sh python scripts/train_spacenet.py
    ./scripts/run.sh python scripts/train_spacenet.py --stretch per_city --seed 1

THREE THINGS THIS DOES THAT train_balloon.py DOES NOT
-----------------------------------------------------
1. Loads imagery through SpaceNetMapper. The tiles are 11-bit values in UInt16
   containers; detectron2 stock read path would hand raw UInt16 to code that
   assumes 8-bit. No 8-bit files exist on disk -- the stretch happens at load.
2. Evaluates per AOI as well as pooled, from the same val tiles, so the
   per-city question is answered without training four models.
3. Seeds explicitly. cfg.SEED is inert in this codebase: detectron2 reads it only
   inside default_setup(), which these scripts do not call. Setting SEED in the
   YAML and nothing else would look correct and do nothing.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import inference_on_dataset
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger

from detlab.datasets import spacenet
from detlab.spacenet_f1 import SpaceNetF1Evaluator
from detlab.trainer import LabTrainer

# This experiment's identity. The W&B project is resolved from it via
# configs/wandb_projects.json, so a run cannot inherit another
# experiment's project from a stale environment variable.
from detlab.wandb_registry import resolve as resolve_project
EXPERIMENT_KEY = "spacenet2"


BASE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG = os.path.join(REPO, "configs", "spacenet2_mask_rcnn_R50_FPN.yaml")
STRETCH_JSON = os.path.join(REPO, "configs", "spacenet2_stretch.json")


class SpaceNetTrainer(LabTrainer):
    """LabTrainer with the 16-bit load path wired into both loaders.

    `stretch` is a class attribute because detectron2 calls build_train_loader
    and build_test_loader as classmethods, with no route for passing an instance
    through. Set it before constructing the trainer.
    """

    stretch = None

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """COCO AP and SpaceNet F1 together, from one inference pass.

        They measure different things and Milestone B needs both: COCO mAP is
        what detectron2 reports and what the balloon work is comparable to, while
        published SpaceNet numbers are F1 at IoU 0.5. Running both in one
        DatasetEvaluators costs one extra mask IoU per prediction rather than a
        second pass over 2118 tiles.
        """
        from detectron2.evaluation import DatasetEvaluators

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return DatasetEvaluators([
            LabTrainer.build_evaluator(cfg, dataset_name, output_folder),
            SpaceNetF1Evaluator(dataset_name),
        ])

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=spacenet.SpaceNetMapper(cfg, True, cls.stretch))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name,
            mapper=spacenet.SpaceNetMapper(cfg, False, cls.stretch))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stretch", choices=["per_image", "per_city"],
                   default="per_image",
                   help="per_image matches SpaceNet published practice and is "
                        "the baseline-comparable run; per_city is the experiment")
    p.add_argument("--seed", type=int, default=None,
                   help="training RNG; overrides SEED in the config. NOT the "
                        "split seed, which is fixed in spacenet2_split.json")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--eval-period", type=int, default=None)
    p.add_argument("--batch", type=int, default=None, help="SOLVER.IMS_PER_BATCH")
    p.add_argument("--lr", type=float, default=None, help="SOLVER.BASE_LR")
    p.add_argument("--data-root", default=os.path.join(REPO, "data", "spacenet2"))
    p.add_argument("--output", default=None)
    p.add_argument("--project",
                   default=None,
                   help="override the registry; normally leave unset")
    p.add_argument("--run-name", default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--skip-per-aoi", action="store_true",
                   help="pooled evaluation only")
    p.add_argument("--grayscale", action="store_true",
                   help="ablation: collapse chroma after the stretch and "
                        "replicate across 3 channels. Tests whether hue is "
                        "causal for the per-city gap rather than merely "
                        "correlated with it")
    p.add_argument("--split", choices=["random", "blocked"], default="random",
                   help="random is the baseline-comparable split; blocked holds "
                        "out contiguous ~2 km blocks so val tiles mostly do not "
                        "border training tiles, and measures how much the random "
                        "split flatters the result")
    p.add_argument("--eval-only", default=None, metavar="WEIGHTS",
                   help="skip training; load these weights and evaluate. Used to "
                        "score a finished model on a dataset it was not "
                        "evaluated on, e.g. the train split for threshold "
                        "selection")
    p.add_argument("--eval-dataset", default=None,
                   help="override DATASETS.TEST, e.g. spacenet2_train")
    p.add_argument("--no-eval", action="store_true",
                   help="skip evaluation entirely (DATASETS.TEST emptied). For "
                        "memory and throughput probes, where a 3 minute pass "
                        "over 2118 val tiles measures nothing of interest")
    p.add_argument("--smoke", action="store_true",
                   help="500 iters, no periodic eval -- plumbing check")
    return p.parse_args()


def build_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE))
    cfg.merge_from_file(CONFIG)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(BASE)

    if args.smoke:
        cfg.SOLVER.MAX_ITER = 500
        cfg.SOLVER.STEPS = []
        cfg.TEST.EVAL_PERIOD = 0
        cfg.SOLVER.CHECKPOINT_PERIOD = 100000   # nothing mid-run
        cfg.OUTPUT_DIR = os.path.join(REPO, "outputs", "spacenet2_smoke")
    if args.iters is not None:
        cfg.SOLVER.MAX_ITER = args.iters
        cfg.SOLVER.STEPS = [s for s in cfg.SOLVER.STEPS if s < args.iters]
    if args.eval_period is not None:
        cfg.TEST.EVAL_PERIOD = args.eval_period
    if args.batch is not None:
        cfg.SOLVER.IMS_PER_BATCH = args.batch
    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
    if args.seed is not None:
        cfg.SEED = args.seed
    if args.eval_dataset is not None:
        cfg.DATASETS.TEST = (args.eval_dataset,)
    if args.eval_only is not None:
        cfg.MODEL.WEIGHTS = args.eval_only
    if args.no_eval:
        cfg.DATASETS.TEST = ()
        cfg.TEST.EVAL_PERIOD = 0
    if args.output is not None:
        cfg.OUTPUT_DIR = args.output
    if not os.path.isabs(cfg.OUTPUT_DIR):
        cfg.OUTPUT_DIR = os.path.join(REPO, cfg.OUTPUT_DIR)
    return cfg


def evaluate_per_aoi(cfg, model, names):
    out = {}
    for name in names:
        folder = os.path.join(cfg.OUTPUT_DIR, "inference", name)
        evaluator = SpaceNetTrainer.build_evaluator(cfg, name, folder)
        loader = SpaceNetTrainer.build_test_loader(cfg, name)
        out[name] = inference_on_dataset(model, loader, evaluator)
    return out


def main():
    args = parse_args()
    setup_logger()

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    # Registration must precede cfg use: DATASETS.TRAIN names have to resolve.
    os.chdir(REPO)
    # The blocked split lives in its own coco dir under its own dataset prefix,
    # so both can be registered in one process without colliding and a run cannot
    # silently mix them.
    coco_dir = (os.path.join(args.data_root, "coco_blocked")
                if args.split == "blocked" else None)
    prefix = "spacenet2b" if args.split == "blocked" else "spacenet2"

    pooled = spacenet.register_pooled(root=args.data_root, coco_dir=coco_dir,
                                      prefix=prefix)
    # --no-eval must mean no evaluation at all. Emptying DATASETS.TEST silences
    # the pooled pass but not this one, and a memory probe that then spends five
    # minutes on per-AOI evaluation measures nothing anybody asked for.
    per_aoi = ([] if (args.skip_per_aoi or args.no_eval)
               else spacenet.register_val_per_aoi(root=args.data_root,
                                                  coco_dir=coco_dir,
                                                  prefix=prefix))
    print("registered pooled :", pooled)
    print("registered per-AOI:", per_aoi)

    cfg = build_cfg(args)
    if args.split == "blocked" and args.eval_dataset is None:
        cfg.DATASETS.TRAIN = ("%s_train" % prefix,)
        cfg.DATASETS.TEST = ("%s_val" % prefix,)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Explicit. cfg.SEED is read by default_setup() only, which is not called.
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED)
    print("seed          :", "random" if cfg.SEED < 0 else cfg.SEED)
    print("stretch       :", args.stretch)
    print("iterations    :", cfg.SOLVER.MAX_ITER,
          "| batch", cfg.SOLVER.IMS_PER_BATCH, "| lr", cfg.SOLVER.BASE_LR)
    print("filter empty  :", cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
          "(False keeps the 2069 no-building tiles)")

    SpaceNetTrainer.stretch = spacenet.Stretch.from_json(
        STRETCH_JSON, args.stretch, grayscale=args.grayscale)
    print("grayscale     :", args.grayscale)

    run = None
    if not args.no_wandb:
        import wandb

        with open(os.path.join(REPO, "configs", "spacenet2_split.json")) as f:
            split = json.load(f)
        run = wandb.init(
            project=resolve_project(EXPERIMENT_KEY, args.project),
            name=args.run_name or "spacenet2-r50fpn-%s-seed%d-%s" % (
                args.stretch, cfg.SEED, time.strftime("%Y%m%d-%H%M%S")),
            config={
                "base_config": BASE,
                "dataset": "SpaceNet2 pooled (4 AOIs)",
                "split": args.split,
                "stretch": args.stretch,
                "grayscale": args.grayscale,
                "seed": cfg.SEED,
                "split_seed": split["split_seed"],
                "val_frac": split["val_frac"],
                "max_iter": cfg.SOLVER.MAX_ITER,
                "base_lr": cfg.SOLVER.BASE_LR,
                "ims_per_batch": cfg.SOLVER.IMS_PER_BATCH,
                "roi_batch_per_image": cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE,
                "filter_empty": cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                "amp": cfg.SOLVER.AMP.ENABLED,
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
            },
        )
        print("W&B run:", run.url)

    trainer = SpaceNetTrainer(cfg)
    trainer.resume_or_load(resume=False)
    if args.eval_only:
        # No training. resume_or_load already put cfg.MODEL.WEIGHTS into the
        # model, so evaluate straight away rather than running a schedule.
        results = SpaceNetTrainer.test(cfg, trainer.model)
        print("eval-only on %s:" % (cfg.DATASETS.TEST,), results)
        if run is not None:
            run.finish()
        return
    trainer.train()

    # NOT torch.cuda.max_memory_allocated() as a training peak: LabTrainer
    # attaches TorchMemoryStats(period=100), which calls
    # reset_peak_memory_stats(), so this counter only covers whatever happened
    # since the last reset. The true training peak is the max_mem field in the
    # d2.utils.events lines of the log.
    print("VRAM allocated since last TorchMemoryStats reset: %.2f GiB"
          % (torch.cuda.max_memory_allocated() / 1024 ** 3))

    # EvalHook.after_train() already evaluated DATASETS.TEST once training
    # finished -- hooks.py:74, unconditional on a completed run. Calling test()
    # here as well ran the whole 2118 tile val set a second time for nothing,
    # about 3 minutes per run. DefaultTrainer stashes the hook result, and
    # detectron2 own train() reads the same attribute, so this is its intended
    # use rather than reaching into internals.
    results = getattr(trainer, "_last_eval_results", None)
    if results is None:
        results = SpaceNetTrainer.test(cfg, trainer.model)
    print("pooled eval:", results)
    if run is not None:
        run.summary.update({"final/%s" % k: v
                            for k, v in results.get("segm", {}).items()})
        run.summary.update({"final/f1_%s" % k: v
                            for k, v in results.get("spacenet", {}).items()})

    if per_aoi:
        print("=== per-AOI ===")
        for name, res in evaluate_per_aoi(cfg, trainer.model, per_aoi).items():
            segm = res.get("segm", {})
            sn = res.get("spacenet", {})
            # F1 at the FIXED report threshold, not the tuned ceiling. The
            # evaluator used to lead with the tuned value and it reached the
            # public README as a result; see SpaceNetF1Evaluator.
            print("  %-28s segm AP %6.2f  APs %6.2f  |  F1 %5.3f @ %.2f fixed  "
                  "(P %5.3f R %5.3f)   [tuned ceiling %5.3f]"
                  % (name, segm.get("AP", float("nan")),
                     segm.get("APs", float("nan")),
                     sn.get("f1", float("nan")),
                     sn.get("report_threshold", float("nan")),
                     sn.get("precision", float("nan")),
                     sn.get("recall", float("nan")),
                     sn.get("f1_tuned", float("nan"))))
            if run is not None:
                city = name.split("_val_")[-1]
                run.summary.update({"final/%s/%s" % (city, k): v
                                    for k, v in segm.items()})
                run.summary.update({"final/%s/f1_%s" % (city, k): v
                                    for k, v in sn.items()})

    print("artifacts in:", cfg.OUTPUT_DIR)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
