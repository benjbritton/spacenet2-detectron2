#!/usr/bin/env python
"""Measure the REALIZED positive/negative sampling ratio in the ROI head.

MODEL.ROI_HEADS.POSITIVE_FRACTION is a CEILING, not a quota. If too few
proposals overlap ground truth, the sampler backfills with background and the
realized ratio drifts far from the configured one. That distinction is the
mechanism behind a degenerate "predict nothing" minimum.

This uses detectron2's own instrumentation -- roi_head/num_fg_samples and
roi_head/num_bg_samples, written in roi_heads.py:296-298 -- so nothing is
patched or monkeyed with. JSONWriter flushes them to metrics.json.

Four short runs; D deliberately starves positives by demanding IoU 0.8.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

from detlab.datasets import balloon

BASE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ITERS = 300

RUNS = [
    ("A_baseline",  128, 0.25, [0.5], "our config"),
    ("B_bigsample", 512, 0.25, [0.5], "d2 default sample size"),
    ("C_highcap",   128, 0.50, [0.5], "raised ceiling"),
    ("D_scarcity",  128, 0.25, [0.8], "positives starved (MPR regime)"),
]


def build_cfg(name, batch_per_img, pos_frac, iou_thresh):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE))
    cfg.merge_from_file(os.path.join(REPO, "configs", "balloon_mask_rcnn_R50_FPN.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(BASE)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_per_img
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = pos_frac
    cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS = iou_thresh
    cfg.SOLVER.MAX_ITER = ITERS
    cfg.SOLVER.STEPS = []
    cfg.TEST.EVAL_PERIOD = 0          # measuring sampling, not accuracy
    cfg.OUTPUT_DIR = os.path.join(REPO, "outputs", "roi_experiment", name)
    return cfg


def read_metrics(output_dir):
    """metrics.json is JSON-lines, one object per writer flush."""
    path = os.path.join(output_dir, "metrics.json")
    iters, fg, bg = [], [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if "roi_head/num_fg_samples" in rec and "iteration" in rec:
                iters.append(rec["iteration"])
                fg.append(rec["roi_head/num_fg_samples"])
                bg.append(rec["roi_head/num_bg_samples"])
    return iters, fg, bg


def main():
    setup_logger()
    balloon.register(os.path.join(REPO, "data"))

    results = {}
    for name, bpi, pf, iou, desc in RUNS:
        print(f"\n{'=' * 70}\n{name}: sample={bpi} pos_frac={pf} iou={iou}  ({desc})\n{'=' * 70}")
        cfg = build_cfg(name, bpi, pf, iou)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        results[name] = read_metrics(cfg.OUTPUT_DIR) + (bpi, pf, iou, desc)

    out_dir = os.path.join(REPO, "outputs", "roi_experiment")

    # ---- summary table -------------------------------------------------
    print("\n" + "=" * 92)
    print(f"{'run':<14}{'sample':>7}{'cap':>7}{'IoU':>6}{'mean fg':>10}{'mean bg':>10}{'realized':>11}{'bg:fg':>12}")
    print("=" * 92)
    summary = {}
    for name, (_, fg, bg, bpi, pf, iou, desc) in results.items():
        mfg, mbg = sum(fg) / len(fg), sum(bg) / len(bg)
        realized = mfg / (mfg + mbg)
        ratio = mbg / mfg if mfg > 0 else float("inf")
        summary[name] = dict(sample=bpi, cap=pf, iou=iou[0], mean_fg=mfg,
                             mean_bg=mbg, realized_fraction=realized,
                             bg_to_fg=ratio, description=desc)
        print(f"{name:<14}{bpi:>7}{pf:>7}{iou[0]:>6}{mfg:>10.1f}{mbg:>10.1f}"
              f"{realized:>11.3f}{ratio:>11.1f}:1")
    print("=" * 92)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- chart ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for name, (it, fg, bg, bpi, pf, iou, desc) in results.items():
        frac = [f / (f + b) if (f + b) else 0 for f, b in zip(fg, bg)]
        ratio = [b / f if f else None for f, b in zip(fg, bg)]
        label = f"{name}  (sample={bpi}, cap={pf}, IoU={iou[0]})"
        ax1.plot(it, frac, label=label, linewidth=1.6)
        ax2.plot(it, ratio, label=label, linewidth=1.6)

    ax1.axhline(0.25, color="grey", linestyle="--", linewidth=1,
                label="configured cap 0.25")
    ax1.set_ylabel("realized positive fraction\nfg / (fg + bg)")
    ax1.set_title("ROI head sampling: configured ceiling vs what actually happens", fontsize=13)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    ax2.set_yscale("log")
    ax2.set_ylabel("background : foreground\n(log scale)")
    ax2.set_xlabel("iteration")
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    chart = os.path.join(out_dir, "roi_positive_fraction.png")
    fig.savefig(chart, dpi=140)
    print(f"\nchart:   {chart}")
    print(f"summary: {os.path.join(out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
