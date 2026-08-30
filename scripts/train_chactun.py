#!/usr/bin/env python
"""Train one arm of the Chactun comparison on one fold, tracked in W&B.

Run inside the m2/detectron2 container -- see scripts/run.sh.

    ./scripts/run.sh python scripts/train_chactun.py --arm A --fold 0 --smoke
    ./scripts/run.sh python scripts/train_chactun.py --arm A --fold 0
    ./scripts/run.sh python scripts/train_chactun.py --arm C --fold 3 --seed 2

THE EXPERIMENT
--------------
Three arms, each differing from its neighbour by exactly one thing:

    A  Mask R-CNN,    anchors [32,64,128,256,512]    detectron2 default
    B  Mask R-CNN,    anchors [16,32,64,128,256]     A vs B isolates anchor scale
    C  Cascade R-CNN, anchors [16,32,64,128,256]     B vs C isolates the head

Five folds each at seed 0, plus seeds 1 and 2 on fold 0, so every arm carries
both a fold spread (n=5) and a seed spread (n=3). A difference between arms has
to clear both to mean anything. Folds are cluster-blocked and fixed in
data/chactun/splits/folds5.json; they must not be regenerated between runs or
fold variance and seed variance stop being separable.

WHAT THIS DOES THAT train_spacenet.py DOES NOT
----------------------------------------------
1. Switches the zoo base per arm. A and B build on mask_rcnn_R_50_FPN_3x, C on
   Misc/cascade_mask_rcnn_R_50_FPN_3x. Those two configs differ in exactly one
   field -- ROI_HEADS.NAME -- which is what makes B vs C a controlled comparison
   rather than two unrelated recipes.
2. Loads through ChactunMapper. The files are 3-band GeoTIFFs whose bands are
   sky-view factor, positive openness and slope; detectron2's PIL read path
   mishandles them and nothing else pins the band order.
3. Evaluates the same predictions against two ground truths -- the ordinary val
   set and the edge-free one -- so the effect of tile-cut structures is measured
   rather than assumed. One inference pass, two scorings.
4. Seeds explicitly. cfg.SEED is inert here: detectron2 reads it only inside
   default_setup(), which these scripts never call. Setting SEED in YAML alone
   would look correct and do nothing.
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

from detlab.datasets import chactun
from detlab.trainer import LabTrainer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# arm -> (model zoo base, repo config). The zoo base differs ONLY for C.
ARMS = {
    "A": ("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
          "chactun_A_maskrcnn_default_anchors.yaml"),
    "B": ("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
          "chactun_B_maskrcnn_shifted_anchors.yaml"),
    "C": ("Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml",
          "chactun_C_cascade_shifted_anchors.yaml"),
}


class ChactunTrainer(LabTrainer):
    """LabTrainer with the 3-band GeoTIFF load path wired into both loaders."""

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=chactun.ChactunMapper(cfg, is_train=True))

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=chactun.ChactunMapper(cfg, is_train=False))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--eval-period", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--data-root", default=os.path.join(REPO, "data", "chactun"))
    p.add_argument("--output", default=None)
    p.add_argument("--project", default=os.environ.get("WANDB_PROJECT",
                                                       "benjbritton_FA26"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-eval", action="store_true")
    p.add_argument("--skip-noedge", action="store_true",
                   help="skip the edge-free scoring pass")
    p.add_argument("--eval-only", default=None, metavar="WEIGHTS")
    p.add_argument("--smoke", action="store_true",
                   help="500 iterations, no periodic eval, throwaway output dir")
    return p.parse_args()


def build_cfg(args):
    base, repo_cfg = ARMS[args.arm]
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base))
    cfg.merge_from_file(os.path.join(REPO, "configs", repo_cfg))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(base)

    cfg.DATASETS.TRAIN = ("chactun_fold%d_train" % args.fold,)
    cfg.DATASETS.TEST = ("chactun_fold%d_val" % args.fold,)

    if args.smoke:
        cfg.SOLVER.MAX_ITER = 500
        cfg.SOLVER.STEPS = []
        cfg.TEST.EVAL_PERIOD = 0
        cfg.SOLVER.CHECKPOINT_PERIOD = 100000
        cfg.OUTPUT_DIR = os.path.join(REPO, "outputs", "chactun_smoke_%s" % args.arm)
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
    if args.eval_only is not None:
        cfg.MODEL.WEIGHTS = args.eval_only
    if args.no_eval:
        cfg.DATASETS.TEST = ()
        cfg.TEST.EVAL_PERIOD = 0
    if args.output is not None:
        cfg.OUTPUT_DIR = args.output

    # keep each arm/fold/seed in its own directory, or run N silently overwrites
    # run N-1 and the spread being measured disappears
    if not args.smoke and args.output is None:
        cfg.OUTPUT_DIR = os.path.join(
            cfg.OUTPUT_DIR, "fold%d_seed%d" % (args.fold, cfg.SEED))
    if not os.path.isabs(cfg.OUTPUT_DIR):
        cfg.OUTPUT_DIR = os.path.join(REPO, cfg.OUTPUT_DIR)
    return cfg


def evaluate_noedge(cfg, model, fold):
    """Score the same model against the edge-free ground truth."""
    name = "chactun_fold%d_val_noedge" % fold
    folder = os.path.join(cfg.OUTPUT_DIR, "inference_noedge")
    evaluator = ChactunTrainer.build_evaluator(cfg, name, folder)
    loader = ChactunTrainer.build_test_loader(cfg, name)
    return inference_on_dataset(model, loader, evaluator)


def main():
    args = parse_args()
    setup_logger()

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"

    os.chdir(REPO)
    registered = chactun.register_fold(root=args.data_root, fold=args.fold)
    print("registered:", registered)

    cfg = build_cfg(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Explicit. cfg.SEED is read by default_setup() only, which is not called.
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED)

    base, repo_cfg = ARMS[args.arm]
    print("arm           :", args.arm, "|", repo_cfg)
    print("zoo base      :", base)
    print("roi heads     :", cfg.MODEL.ROI_HEADS.NAME)
    print("anchors       :", cfg.MODEL.ANCHOR_GENERATOR.SIZES)
    print("fold / seed   :", args.fold, "/", cfg.SEED)
    print("iterations    :", cfg.SOLVER.MAX_ITER,
          "| batch", cfg.SOLVER.IMS_PER_BATCH, "| lr", cfg.SOLVER.BASE_LR)
    print("pixel mean    :", cfg.MODEL.PIXEL_MEAN)
    print("pixel std     :", cfg.MODEL.PIXEL_STD)
    print("filter empty  :", cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
          "(False keeps the 661 tiles with no structures)")
    print("output        :", cfg.OUTPUT_DIR)

    run = None
    if not args.no_wandb:
        import wandb

        run = wandb.init(
            project=args.project,
            name=args.run_name or "chactun-%s-fold%d-seed%d-%s" % (
                args.arm, args.fold, cfg.SEED, time.strftime("%Y%m%d-%H%M%S")),
            config={
                "arm": args.arm,
                "base_config": base,
                "repo_config": repo_cfg,
                "dataset": "Chactun (Somrak et al. 2023)",
                "roi_heads": cfg.MODEL.ROI_HEADS.NAME,
                "anchor_sizes": str(cfg.MODEL.ANCHOR_GENERATOR.SIZES),
                "fold": args.fold,
                "n_folds": 5,
                "seed": cfg.SEED,
                "max_iter": cfg.SOLVER.MAX_ITER,
                "base_lr": cfg.SOLVER.BASE_LR,
                "ims_per_batch": cfg.SOLVER.IMS_PER_BATCH,
                "filter_empty": cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                "amp": cfg.SOLVER.AMP.ENABLED,
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
            },
        )
        print("W&B run:", run.url)

    trainer = ChactunTrainer(cfg)
    trainer.resume_or_load(resume=False)

    if args.eval_only:
        results = ChactunTrainer.test(cfg, trainer.model)
        print("eval-only on %s:" % (cfg.DATASETS.TEST,), results)
        if not args.skip_noedge:
            print("edge-free:", evaluate_noedge(cfg, trainer.model, args.fold))
        if run is not None:
            run.finish()
        return

    trainer.train()

    # Read what EvalHook already computed rather than re-running test(): the
    # final evaluation has happened by now, and repeating it would double the
    # inference cost for a number already in hand.
    final = getattr(trainer, "_last_eval_results", None)
    if final:
        print("final eval:", json.dumps(final, indent=1, default=str))

    if not args.skip_noedge and not args.no_eval:
        noedge = evaluate_noedge(cfg, trainer.model, args.fold)
        print("edge-free eval:", json.dumps(noedge, indent=1, default=str))
        if run is not None:
            import wandb

            flat = {"noedge/%s/%s" % (task, k): v
                    for task, d in noedge.items() if isinstance(d, dict)
                    for k, v in d.items()}
            wandb.log(flat)

    print("VRAM allocated since last TorchMemoryStats reset: %.2f GiB"
          % (torch.cuda.max_memory_allocated() / 1024 ** 3))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
