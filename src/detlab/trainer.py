"""DefaultTrainer subclass wiring in COCO evaluation and W&B logging.

Uses only documented extension points:
  - build_evaluator  -- consumed by detectron2's stock EvalHook
  - build_writers    -- detectron2/engine/defaults.py:502
  - build_hooks      -- detectron2/engine/defaults.py:452

All hooks used here are detectron2's own (detectron2/engine/hooks.py).
Nothing custom is written where an upstream hook already exists.
"""

import os

from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator
from detectron2.utils import comm

from .wandb_writer import WandbWriter


class LabTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)

    def build_writers(self):
        import wandb

        writers = super().build_writers()
        # No active run (--no-wandb, or wandb never initialised) means no
        # writer -- attaching one would raise on the first flush.
        if comm.is_main_process() and wandb.run is not None:
            writers.append(WandbWriter())
        return writers

    def build_hooks(self):
        hook_list = super().build_hooks()
        if not comm.is_main_process():
            return hook_list

        extra = [hooks.TorchMemoryStats(period=100)]

        # BestCheckpointer reads the metric EvalHook just wrote, so it only
        # makes sense when periodic evaluation is actually running.
        if self.cfg.TEST.EVAL_PERIOD > 0:
            extra.append(
                hooks.BestCheckpointer(
                    self.cfg.TEST.EVAL_PERIOD,
                    self.checkpointer,
                    "segm/AP",
                    mode="max",
                )
            )

        # PeriodicWriter must remain LAST -- it flushes whatever the hooks
        # above it put into storage during this iteration.
        return hook_list[:-1] + extra + hook_list[-1:]
