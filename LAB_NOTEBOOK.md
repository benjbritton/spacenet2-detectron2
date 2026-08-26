# Lab Notebook

Running record for the FA26 Independent Study. Process and decisions, written as
work happens. Feeds the weekly research journal and end-of-semester report.

---

## 2026-08-21 - Environment build

**Goal.** Get detectron2 running. It cannot be built on Windows: it compiles C++
and CUDA extensions requiring MSVC and a matching CUDA Toolkit, neither present,
and upstream does not support Windows. Route chosen: WSL2 + Docker.

**Done.**

1. **WSL2 + Ubuntu 24.04.4 LTS.** WSL 2.7.12 was already installed; no distro. User
   `benja` (uid 1000), passwordless sudo - WSL is not a security boundary, since
   `wsl -u root` already grants root from the Windows session with no auth.
2. **Docker Engine 29.7.2** from Docker's official apt repo. Not Docker Desktop.
3. **NVIDIA Container Toolkit 1.20.0.** No NVIDIA driver installed inside the
   distro - the Windows driver passes through at `/usr/lib/wsl/lib`. Installing a
   Linux driver would break that and would need redoing on every GPU change.
4. **Verified GPU passthrough:** `docker run --gpus all nvidia/cuda:12.4.1-base
   nvidia-smi` gives RTX 2080 Ti, 11264 MiB, driver 591.86.

**Problem: every `docker pull` failed.** Registry CDN resolves to both IPv4 and
IPv6; WSL's default NAT network has no IPv6 route, and containerd kept selecting
the IPv6 address.

Disabling IPv6 inside the distro did **not** fix it. glibc complied - `getent` and
`curl` correctly returned IPv4 only - but **containerd uses Go's pure resolver**,
which bypasses glibc/NSS and gai.conf entirely and queries AAAA itself. Also note
Docker 29 uses the containerd image store, so `containerd.service` does the
fetching and must be restarted too, not just `docker`.

**Fix:** the Windows host has genuine IPv6 (global addresses via router
advertisement). Set `networkingMode=mirrored` in `.wslconfig`, giving the distro
the host's network stack. A proper fix rather than suppression, and it also gives
localhost interop, which the plan's FastAPI endpoint will want.

**Image `m2/detectron2:cu124-torch251` (22.1 GB).**
Base `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, matching the Windows `m2` conda
env so results are comparable. `devel` not `runtime` - nvcc is required.
detectron2 pinned to commit `a2f4a877` (2026-08-18); there has been no tagged
release since 0.6, so `main` is a moving target. Cloned rather than
`pip install git+...` so `projects/ViTDet` is on disk.

**Gotcha - the numpy pin silently failed.** The first build finished with
**numpy 2.4.6** despite an explicit early `pip install "numpy<2"`; a later
`pip install` layer re-resolved it upward. detectron2 still *imported* fine, so
this would have surfaced much later inside COCO evaluation, looking like a metrics
bug rather than an environment bug. Fixed with `PIP_CONSTRAINT`, which binds every
pip invocation in the image, plus a build-time assertion that fails the *build*.
Settled at numpy 1.26.4.

**Second gotcha:** Git Bash rewrites standalone absolute-path arguments into
Windows paths, so `wsl -u root -- bash /root/provision.sh` became
`C:/Program Files/Git/root/provision.sh`. Piping through `tr` made the pipeline
exit 0, so it looked successful while installing nothing. Wrap the command in
`bash -c` instead, and verify long installs by their stage markers, not exit code.

**Smoke test.** Pretrained Mask R-CNN R50-FPN on COCO val2017 `000000439715`
(people on horses): 15 instances - 1 horse at 1.000, 8 people, 3 umbrellas, 1
backpack. 0.51 GiB peak VRAM. No training; equipment check only.
Output: `~/m2/work/detectron2_verify.jpg`.

**GPU note.** Card present is an RTX 2080 Ti (11 GB, sm_75, Turing), not the A5000
24 GB the research plan assumes. Adequate for Weeks 1-2. Requirement stated: the
upgrade must be a hardware swap plus a driver update, nothing more. Accordingly the
image is compiled for `TORCH_CUDA_ARCH_LIST="7.5;8.6+PTX"` and configs use fp16
rather than Ampere-only bf16.

---

## 2026-08-21 - Repo scaffolding (~/m2/repos/benjbritton_FA26)

**Goal.** A version-controlled home for the first tracked experiment, structured to
carry into the SpaceNet 2 baseline rather than be thrown away.

**Notable finding: detectron2 ships no W&B integration.** `detectron2/utils/events.py`
provides only `JSONWriter`, `TensorboardXWriter`, `CommonMetricPrinter`. So
`src/detlab/wandb_writer.py` supplies the missing writer, modelled directly on
`TensorboardXWriter` (events.py:141) - same smoothing window, same `_last_write`
guard, same responsibility to drain accumulated images.

Everything else uses documented extension points: `build_writers`
(defaults.py:502), `build_evaluator`, `build_hooks` (defaults.py:452), and
detectron2's own stock hooks (`EvalHook`, `BestCheckpointer`, `TorchMemoryStats`).
Nothing reimplemented that already exists upstream.

`balloon` needed a loader because it is not in the detectron2 repo and ships VIA
polygon annotations rather than COCO.

**Status:** scaffolding complete. Training blocked only on a W&B API key.

**W&B account diagnosis.** `wandb.ai/home` and `/settings` both redirect to the
marketing page. GitHub Settings > Applications shows the Weights & Biases OAuth
app authorized but **"Never used"** - the grant exists, but no sign-in was ever
completed through it, so no W&B account/entity exists and there is no workspace to
route to. Fix: wandb.ai/login, sign in with GitHub, complete the username step.

---

## 2026-08-22 - First full training run (Milestone A complete)

**Goal.** Fine-tune Mask R-CNN R50-FPN on balloon, tracked live in W&B, as the
first reproducible experiment required by Milestone A (due 2026-09-06).

**Run:** `balloon-maskrcnn-r50-20260822-054456`
https://wandb.ai/benjbritton-geoai/benjbritton_FA26/runs/3m81zlqa

1500 iterations, batch 2, fp16 AMP, LR 0.00025 decaying x0.1 at 1000 and 1350,
evaluation every 250 iterations on the 13 held-out images.

| iter | segm/AP | bbox/AP | segm/APs |
|------|---------|---------|----------|
| 249  | 11.93   | 7.81    | 0.32     |
| 499  | 52.30   | 44.71   | 0.93     |
| 749  | 73.81   | 66.86   | 1.33     |
| 999  | 78.88   | 71.95   | 1.63     |
| 1249 | **81.55** | 77.01 | 13.81    |
| 1500 | 81.54   | 78.59   | 13.94    |

Final: segm AP 81.54, bbox AP 78.59, cls_accuracy 0.967, false-negative 0.074.
Peak VRAM 2.68 GiB of 11 GiB. Wall time roughly 6 minutes.

### Two findings worth more than the headline number

**Converged at ~1250, not 1500.** 81.55 -> 81.54 over the last 250 iterations.
`BestCheckpointer` correctly kept iteration 1249 as `model_best.pth` rather than
the final weights. 1500 was slightly more than this dataset needs.

**`segm/APs` jumped 1.63 -> 13.81 between iteration 999 and 1249** -- an 8x step
change while overall AP moved about 3 points. That is the LR decay at iteration
1000 landing almost entirely on *small* objects: large instances were already
learned, and the coarse learning rate had been preventing the fine-boundary
refinement small instances depend on.

Generalizable: the back half of the LR schedule does most of the small-object
work. Truncating training early costs small-object performance specifically
while the headline AP still looks healthy. Relevant to SpaceNet buildings and to
any small-target detection.

> **Revised 2026-08-26 -- do not cite this subsection as written.** Five runs
> confirm the *direction* but not the *magnitude*: the 8x figure is one draw from
> a metric whose final value ranges 7.05 to 26.19 across seeds, and `segm/APs` on
> this dataset is computed over **three** validation instances. See the
> 2026-08-26 entry below.

### Artifacts

```
outputs/balloon_r50fpn/
  model_best.pth     iteration 1249, best segm/AP
  model_final.pth    iteration 1500
  metrics.json       full metric history
  samples/           3 annotated validation images
```

### Infrastructure fixed along the way

**Containers were writing as root.** Everything written through the bind mount --
checkpoints, metrics.json, wandb run dirs -- landed on the host owned by
`root:root`, so outputs could not be deleted or edited without sudo and were
untouchable from Windows Explorer. `run.sh` now passes `-u $(id -u):$(id -g)`.
A non-root UID has no passwd entry so HOME is unset; it now points at a mounted
`.cache/`, which also persists model-zoo downloads between runs.

**Geospatial stack added to the image** for the SpaceNet converter: rasterio
1.4.4, shapely 2.1.2, pyproj 3.7.2 (PROJ 9.5.1). Placed *after* the detectron2
layer so the cached CUDA compile survived -- rebuild took 8 seconds instead of
~15 minutes. The numpy assertion now runs last and still passes at 1.26.4,
confirming the geo wheels did not drag numpy past 2.

### W&B account resolution

The redirect loop was **not** a provisioning bug and **not** the username rename.
W&B would not create an entity until the onboarding flow was completed by
clicking one of the three intro sections and pressing Continue. Navigating
directly to `/authorize` or `/settings` bypassed that screen, which is why every
direct attempt bounced to the marketing page.

Also: accounts created under an organization get **no personal entity**. There is
no `benjbritton` entity; only the `benjbritton-geoai` team. An `api.projects()`
check returns an empty list for a nonexistent entity as well as an empty one, so
it cannot distinguish them -- it is not a valid existence test. Renaming the team
later would carry its runs along, since a rename keeps the same entity.


## 2026-08-26 - GPU swap, seeding, and what five runs did to the small-object claim

### RTX 2080 Ti -> RTX A5000: no rebuild, no config change

The card was swapped and the Windows driver updated. Nothing else was touched.

| | before | after |
|---|---|---|
| card | RTX 2080 Ti, sm_75 | RTX A5000, sm_86 |
| VRAM | 11264 MiB | 24564 MiB |
| driver / CUDA | 591.86 / 13.1 | 596.86 / 13.2 |

Verified with `scripts/verify_gpu.py` (added this session; run it via
`./scripts/run.sh python scripts/verify_gpu.py`). The check that matters is not
`nvidia-smi` but whether detectron2's *compiled* CUDA kernels launch: `nms` and
`ROIAlign` run natively on sm_86. That is the payoff from building the image with
`TORCH_CUDA_ARCH_LIST="7.5;8.6+PTX"` back in August -- a 7.5-only build would
fail here with "no kernel image is available for execution on the device".

`torch.cuda.is_bf16_supported()` is now True. TF32 remains off at torch's
default, so nothing changed silently. The fp16 AMP config still runs unmodified;
bf16 is now an option rather than something the hardware refused.

Same-config rerun: 4:33 wall vs roughly 6:00 on the 2080 Ti, peak VRAM 2754 MB
vs 2745 MB. This model is far too small to be limited by either card. The 24 GB
matters for ViTDet + Cascade later, not for R50-FPN at batch 2.

### Seeding: setting `cfg.SEED` alone would have been a silent no-op

detectron2 reads `cfg.SEED` in exactly one place -- `seed_all_rng(...)` inside
`engine/defaults.py:244`, which lives in `default_setup()`. **This script never
calls `default_setup()`**; it calls `setup_logger()` directly. So a `--seed` flag
wired only to `cfg.SEED` would have appeared to work and changed nothing. The
flag calls `seed_all_rng()` explicitly, before `LabTrainer(cfg)` builds the model
and the loader.

Default remains `-1`, which detectron2 turns into
`os.getpid() + clock_microseconds + os.urandom(2)`. Both earlier balloon runs
were unseeded, so they differ in every weight initialization, batch order,
augmentation draw and ROI sample -- hardware was never the only variable between
them.

Seeding is verified at the level it can be verified. Building the model twice at
seed 0 gives a bit-identical state-dict hash (`7f264ada4f1433904753f900`), and
seed 1 gives a different one. **But training is still not bit-exact**: two
smoke runs at seed 0 diverged (total_loss 2.072 vs 2.12 by iteration 19).
That is nondeterministic CUDA kernel accumulation plus fp16 loss scaling, exactly
what the comment above `_C.SEED` in `config/defaults.py` warns about. Seeds buy
comparable runs, not identical ones; bit-exactness would additionally need
`cudnn.deterministic` and no autotuner, at a throughput cost not worth paying.

### Five runs: overall AP is solid, `APs` is not a measurement

Three seeded runs (0, 1, 2) plus the two unseeded ones. Identical config
throughout.

`segm/AP`:

| run | 250 | 500 | 750 | 1000 | 1250 | 1500 |
|---|---|---|---|---|---|---|
| 2080 Ti unseeded | 11.93 | 52.30 | 73.81 | 78.88 | 81.55 | 81.54 |
| A5000 unseeded | 16.35 | 61.36 | 75.37 | 79.22 | 81.40 | 81.61 |
| A5000 seed 0 | 4.23 | 42.81 | 67.79 | 77.45 | 81.28 | 81.62 |
| A5000 seed 1 | 11.10 | 45.78 | 62.20 | 77.86 | 80.69 | 81.27 |
| A5000 seed 2 | 2.77 | 46.26 | 70.20 | 76.82 | 80.12 | 81.52 |

mean 81.51, sd 0.14, **CV 0.2%**.

`segm/APs`:

| run | 250 | 500 | 750 | 1000 | 1250 | 1500 |
|---|---|---|---|---|---|---|
| 2080 Ti unseeded | 0.32 | 0.93 | 1.33 | 1.63 | 13.81 | 13.94 |
| A5000 unseeded | 1.05 | 1.97 | 8.92 | 11.53 | 25.95 | 26.19 |
| A5000 seed 0 | 0.19 | 0.79 | 1.35 | 1.08 | 6.94 | 7.05 |
| A5000 seed 1 | 0.00 | 0.64 | 0.97 | 4.73 | 6.54 | 7.78 |
| A5000 seed 2 | 0.40 | 1.69 | 2.34 | 0.94 | 7.01 | 10.36 |

mean 13.06, sd 7.82, **CV 59.8%** -- about 300x the relative variability of
overall AP. Note also that early trajectories differ wildly (AP at iteration 250
spans 2.77 to 16.35) while all five converge to the same place. Early-iteration
comparisons between runs are worthless here.

### Revising the 2026-08-22 small-object claim

**The direction holds, 5 runs out of 5.** Every run shows a large step change in
`segm/APs` between iteration 999 and 1249, which is where `STEPS: (1000, 1350)`
drops the learning rate. Across that interval overall AP gains 2-4 points (~3%
relative) while `APs` gains 140% to 750% relative. The decay's benefit really is
concentrated on small objects, across seeds and across two GPU generations.

**The magnitude does not hold.** The jump ratio ranges from 1.4x (seed 1) to 8.5x
(2080 Ti). The original "8x" was one draw, and the 13.94 endpoint was a
mid-range sample from a metric spanning 7.05 to 26.19.

Stated properly: *LR decay disproportionately benefits small objects; the effect
is directionally robust but its size is not estimable from this dataset.*

### Why `APs` cannot be estimated here: three instances

Instance-size census of the balloon annotations, by COCO thresholds
(small < 32^2 = 1024 px, large > 96^2 = 9216 px):

```
VAL   - 13 images,  50 instances
   small  (< 1024 px):   3 ( 6.0%)     areas: 84, 166, 1021
   medium (1024-9216):  17 (34.0%)
   large  (> 9216 px):  30 (60.0%)
TRAIN - 61 images, 255 instances
   small  (< 1024 px):  10 ( 3.9%)
```

`segm/APs` is average precision over **three** instances. Two are ~9x9 and ~13x13
pixels; the third is 1021 px, three pixels under the small/medium boundary, so a
marginally different annotation would move it out of the bucket entirely. AP over
three instances moves in enormous discrete steps -- detecting two rather than one
swings it by tens of points. That is the whole explanation for both the low value
and the 60% CV. No architectural account is needed: the model is Mask R-CNN
R50-FPN, a CNN with an FPN, and nothing about it is being tested by this metric.

Lesson to carry forward: check the per-bucket instance count before reading a
per-bucket metric. SpaceNet 2's validation sets are large enough that this exact
failure will not recur, but the habit should.

### Artifacts

```
outputs/balloon_seed0/  seed 0    segm AP 81.62   APs  7.05
outputs/balloon_seed1/  seed 1    segm AP 81.27   APs  7.78
outputs/balloon_seed2/  seed 2    segm AP 81.52   APs 10.36
outputs/balloon_r50fpn_a5000/     segm AP 81.61   APs 26.19   (unseeded)
outputs/balloon_r50fpn/           segm AP 81.54   APs 13.94   (unseeded, 2080 Ti)
```

W&B runs `balloon-a5000-seed0/1/2` and `balloon-a5000` in project
`benjbritton_FA26`.

### Housekeeping

- W&B project renamed `fa26-independent-study` -> `benjbritton_FA26`, entity
  unchanged, both existing runs carried across with their ids intact. The rename
  returned a UI error page while succeeding server-side; the post-rename redirect
  targets the old URL.
- The account has no personal entity and cannot get one by renaming: signup
  provisioned an org and a team, `wandb.init(entity="benjbritton")` fails with
  `CommError: entity benjbritton not found during upsertBucket` ("not found", not
  "forbidden"), and the team cannot take the name `benjbritton` because the user
  account holds it. Published URLs are `wandb.ai/benjbritton-geoai/...` unless
  W&B support provisions one on request.
- Repo published (private) at `github.com/benjbritton/benjbritton_FA26`. All
  commits were rewritten from `brittobj@mail.uc.edu` to
  `benjaminbritton@yahoo.com` before the first push, so they attribute to the
  GitHub account; git identity is now set globally to match.
