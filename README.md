# detection-lab

detectron2 training scaffolding for the FA26 Independent Study (Britton, UC Geography & GIS).

First experiment: Mask R-CNN R50-FPN fine-tuned on `balloon`, tracked in Weights & Biases.
The structure is intended to carry forward to the SpaceNet 2 baseline (Milestone B).

## Requirements

- WSL2 Ubuntu 24.04, Docker Engine, NVIDIA Container Toolkit
- Image `m2/detectron2:cu124-torch251` (built from `docker/Dockerfile.detectron2`)
- A W&B API key (https://wandb.ai/authorize)

Keep this repo, `data/` and `outputs/` on the WSL ext4 filesystem, **not** under
`/mnt/c`. Cross-filesystem I/O to the Windows drive is slow for the many-small-file
access pattern training uses.

## Reproduce

```bash
# 1. One-time: store the W&B credential (writes ~/.netrc, mounted into containers)
./scripts/run.sh wandb login

# 2. Plumbing check -- 50 iterations, no evaluation, ~1 minute
./scripts/run.sh python scripts/train_balloon.py --smoke

# 3. Full run -- 1500 iterations, periodic eval, ~7-10 min on an RTX 2080 Ti
./scripts/run.sh python scripts/train_balloon.py
```

No W&B account yet? `--offline` records locally and `wandb sync` uploads later.
`--no-wandb` disables tracking entirely.

## Layout

| Path | Purpose |
|---|---|
| `src/detlab/wandb_writer.py` | W&B EventWriter. detectron2 ships no W&B support |
| `src/detlab/trainer.py` | DefaultTrainer subclass: COCO eval + W&B + best-checkpointing |
| `src/detlab/datasets/balloon.py` | Download, VIA to detectron2 conversion, registration |
| `configs/` | Overrides layered on a model-zoo base config |
| `scripts/run.sh` | Version-controlled docker run invocation |
| `docker/` | Image definition |
| `LAB_NOTEBOOK.md` | Running record of what was done and why |

`data/`, `outputs/`, `wandb/` and checkpoints are gitignored.

## Notes

- **fp16, not bf16.** `SOLVER.AMP.ENABLED` uses fp16 because bf16 and TF32 are
  Ampere-only. A bf16 config runs on an A5000 and fails on the 2080 Ti (sm_75).
  fp16 works on both, so configs survive the planned GPU swap.
- **`numpy<2` is pinned in the image** via `PIP_CONSTRAINT`. detectron2 predates
  numpy 2; the breakage shows up in COCO evaluation, not at import, so it is easy
  to miss. The Dockerfile asserts the pin at build time.
- **The image is built for `sm_75;sm_86`**, covering both the current 2080 Ti and
  the incoming A5000 without a rebuild.
