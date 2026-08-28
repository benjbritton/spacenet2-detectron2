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

> **Superseded 2026-08-27.** Even that guarded statement does not survive
> SpaceNet. On 8474 training tiles the LR decay moved `segm/APs` by 0.62 points,
> 2.4% relative, against 8.5x on balloon. The effect was an artefact of a 61
> image training set and a 13 image val set, not a property of LR schedules.
> See the 2026-08-27 entry. Do not carry this finding into a blog post.

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


## 2026-08-27 - SpaceNet 2 baseline (Milestone B)

### Data

26 GB, 10592 tiles, all four AOIs, PS-RGB imagery plus building footprints.
Deliberately NOT downloaded:

- `test_public/` -- imagery only, no `geojson_buildings` and no solutions csv.
  The competition test labels were never released, so it cannot be scored
  locally. 19 GiB for nothing.
- `MS/ PAN/ PS-MS/` -- 8-band multispectral and panchromatic, which a 3-channel
  detectron2 baseline cannot consume. Another 32 GiB.

The bucket is requester-pays, so this billed to the AWS account: about 2.30 USD.
Downloading everything would have been roughly 7.20 USD and 76 GiB for no gain.

### The imagery is 11-bit, and that decides the load path

Across all 10592 tiles, every channel in every AOI tops out below 2047, and
AOI_2_Vegas green reaches exactly 2047 = 2^11 - 1. This is 11-bit WorldView-3
radiometry stored in a 16-bit container, occupying 3.1% of the nominal range.

Consequence, measured not estimated: a naive divide-by-256 produces a tile whose
maximum value is **6 out of 255**. It would train, converge, and quietly feed the
network a near-black image. The p2-p98 windows also differ about 3x between
cities (Paris R 126-464, Vegas R 151-1038), so one global constant would
misexpose entire AOIs.

**No 8-bit files are written.** The stretch happens in the mapper at load time,
so the georeferenced UInt16 GeoTIFFs stay the only copy on disk. `data_time` is
0.057 s against a 0.839 s step at batch 16 -- under 7% -- so writing derivatives
would have bought nothing and created a second copy to drift out of sync.

Two modes exist because there is no single canonical answer: `per_image`
percentile (what the SpaceNet write-ups describe, the baseline-comparable run)
and `per_city` constants from `scripts/spacenet_stats.py` (the experiment, which
keeps absolute brightness comparable across tiles of one city). This run used
`per_image`.

### Solaris cannot be run

The plan names Solaris as the reference implementation to reproduce. It is not
installable. Last commit to `main` is 2021-04-29; CosmiQ Works folded into IQT
Labs in March 2021. `requirements.txt` pins `tensorflow==1.13.1` (Python <= 3.7,
against our 3.11), `pyyaml==5.2` (will not build against modern Cython), and a
`git://` dependency, a protocol GitHub permanently disabled in March 2022 -- so
`pip install -r requirements.txt` fails before reaching anything else.

**Milestone B is therefore reproducing the published method and numbers, not the
published code.** Worth raising with the advisor rather than explaining in
December.

### Converter, audited against real data

`geojson_to_coco.py` was written in August against a format description. Running
it against real files found one crash and three gaps:

- **Latent crash.** SpaceNet footprints carry a Z ordinate, so
  `for x, y in part.exterior.coords` raised `ValueError` on the first real tile.
- **One-directional orphan reporting.** Only images-without-labels were reported,
  but the case that occurs is the reverse: AOI_2_Vegas ships 3851 labels against
  3850 images, and `img1000` was being dropped in silence.
- **Split polygons became separate buildings.** Clipping at the tile edge and
  `make_valid` both split a polygon, and each piece was emitted as its own
  annotation -- 1630 phantom instances across the four AOIs. Now grouped: one
  footprint, one annotation, bbox spanning all pieces, area summed.
- Overlay renders confirmed the geo-to-pixel transform, which counts cannot: a
  wrong transform produces the right number of correctly-shaped polygons in the
  wrong place.

Result: 10592 images, 218681 instances, 2069 empty tiles kept, all EPSG:4326.
Retention 99.48-99.83%, the remainder sub-pixel edge fragments.

**Area buckets: 37.4% small, 58.6% medium, 4.0% large** -- the inverse of balloon
(6% / 34% / 60%). Small-object behaviour dominates headline AP here.

### Split

80/20 stratified within each AOI, then pooled. A global shuffle would leave each
city val share to chance, and the small AOIs are both the likeliest to be
under-represented and the least stable.

| AOI | train | val | val instances |
|---|---|---|---|
| Vegas | 3080 | 770 | 22250 |
| Paris | 918 | 230 | 3048 |
| Shanghai | 3666 | 916 | 13388 |
| Khartoum | 810 | 202 | 4802 |
| pooled | 8474 | 2118 | 43488 |

Membership is written to `configs/spacenet2_split.json` and **that file, not the
seed, is the authority**. The split seed is not the training seed: `cfg.SEED`
varies between runs to measure variance, the split must not.

**Known limitation.** Adjacent chips share street grid, roof materials, sun angle
and acquisition, so a random tile split puts near neighbours on both sides and
val scores are optimistic against genuinely unseen ground. The published
baselines split the same way, so this is the comparable choice, but it must be
stated. A spatially blocked split would quantify the gap.

### Batch size: measured, and the answer was not the obvious one

The A5000 has 24 GB and batch 8 used 3.7 GB, which looked like an invitation to
scale up. It was not.

| batch | s/iter | images/sec | data_time |
|---|---|---|---|
| 8 | 0.449 | 17.8 | 0.027 |
| **16** | 0.839 | **19.1** | 0.057 |
| 32 | 1.729 | 18.5 | 0.176 |
| 48 | 2.747 | 17.5 | 0.302 |

**Throughput peaks at 16 and falls away on both sides.** Batch 48 is slower per
image than batch 8 and reserves 24.2 of 24.6 GB. The cause is `data_time`, up 11x
from batch 8 to 48: the 8 loader workers cannot feed the 16-bit read and
percentile stretch fast enough, so the GPU starves. **Memory was never the
binding constraint; the input pipeline was.** "Only 15% of the card is used" was
about memory, not compute -- the SMs were already near saturation at batch 8.

16 is also the detectron2 COCO recipe, so the run stays comparable rather than
being a configuration nobody has published.

### What the smoke run caught

- `EvalHook.after_train()` evaluates unconditionally once training completes
  (`hooks.py:74`), so the explicit `test()` call after `train()` ran all 2118 val
  tiles a **second** time -- about 3 minutes per run, on every run.
- `--no-eval` emptied `DATASETS.TEST` but did not stop the per-AOI block, so a
  memory probe spent five minutes evaluating city subsets.
- The peak-VRAM report was meaningless: `LabTrainer` attaches
  `TorchMemoryStats(period=100)`, which calls `reset_peak_memory_stats()`, so
  `max_memory_allocated()` only covers iterations since the last reset.

None of these would have failed loudly. All three were found by running 500
iterations before committing to 105 minutes.

### Results

Mask R-CNN R50-FPN, COCO-pretrained, batch 16, LR 0.02, 6000 iterations
(11.3 epochs), fp16, seed 0, `per_image` stretch. **1:45:02** wall, 17:14 of it
in evaluation. Peak ~7 GB.

| iter | segm/AP | segm/APs | segm/APm | segm/APl | bbox/AP |
|---|---|---|---|---|---|
| 999 | 44.28 | 21.23 | 56.02 | 50.87 | 46.57 |
| 1999 | 46.14 | 22.80 | 57.83 | 52.37 | 49.65 |
| 2999 | 48.17 | 25.47 | 59.81 | 55.69 | 51.05 |
| 3999 | 48.88 | 25.82 | 60.53 | 56.66 | 50.67 |
| 4999 | 49.39 | 26.38 | 60.93 | 58.22 | 52.71 |
| 6000 | **49.44** | 26.44 | 60.91 | 58.45 | **52.76** |

Converged near 5000 of 6000 -- the same slightly-too-long schedule balloon had.

### SpaceNet F1 versus the published reference

COCO mAP and the SpaceNet score are not convertible, and the published numbers
are F1 at IoU 0.5. `src/detlab/spacenet_f1.py` implements it: greedy
score-ordered matching, each ground truth claimable once, micro-averaged.

The operating point is the real difference from AP. Competitors submitted fixed
polygon sets with no confidence scores, so each published F1 is one point a team
tuned for itself. Reporting the best F1 over all thresholds would mean reporting a
value at a threshold chosen using the set being scored -- a tuned hyperparameter
presented as a result -- so **the threshold is selected on other data.**

Two sources were computed, because each has a different flaw:

- **train** -- unbiased with respect to val, but the model has memorised those
  tiles. Train F1 is 0.8154 against 0.7930 on val, so its predictions there
  really are more confident, and the optimal threshold is offset from what suits
  unseen ground. Selected 0.544.
- **val half** -- split val by image id, select on half A, report on half B.
  Unbiased AND distribution-matched, at the cost of halving the reporting set.
  Needs no extra inference. Selected 0.539.

**They agree.** On held-out val half B those thresholds give F1 0.7939 and 0.7941,
two ten-thousandths apart. The memorisation bias is real in the threshold and does
not propagate to the score. Worth checking; it changed nothing, and that is now a
measured statement rather than an assumption.

Reported at the train-selected threshold of 0.544:

| AOI | F1 | precision | recall | XD_XD (2017 winner) |
|---|---|---|---|---|
| Vegas | 0.8941 | 0.9264 | 0.8640 | 0.885 |
| Paris | 0.7762 | 0.8223 | 0.7349 | 0.745 |
| Shanghai | 0.6828 | 0.7360 | 0.6368 | 0.597 |
| Khartoum | 0.6257 | 0.6757 | 0.5827 | 0.544 |
| **macro** | **0.7447** | | | **0.693** |

Pooled val F1 0.7930, against 0.7935 for the best-over-sweep that is not
reportable. Full rigour cost 0.0015 of F1.

**This is not a claim of beating the 2017 winner.** Three differences all favour
us: XD_XD was scored on the withheld competition test set while this is a val
split from the training data; the random tile split is spatially autocorrelated
and therefore optimistic; and IoU here is computed on rasterised masks rather
than georeferenced polygons.

Note also that the pooled micro-averaged F1 of 0.7935 is NOT the number to
compare -- Vegas alone is 51% of val instances and is the easiest city, so micro
averaging weights the easy case. The macro average of 0.7462 is the comparable
figure.

**What is a real signal:** the city difficulty ordering reproduces XD_XD exactly,
Vegas >> Paris > Shanghai > Khartoum, with similar gaps. The pipeline recovers
the known difficulty structure of this dataset even if the absolute level is
optimistic.

Milestone B stated honestly: the plan asks for mAP@[0.5:0.95] within 20% of a
published reference, and **no published mAP for SN2 exists**. Substituting F1,
the metric the reference actually reports, the target is >= 0.554 against 0.7462
achieved.

### The balloon LR-decay finding does not generalize

The 2026-08-22 entry claimed the back half of the LR schedule does most of the
small-object work, and generalized it in as many words to "SpaceNet buildings and
any small-target detection". Tested here, it fails.

LR decays at iteration 4000. Nothing here resembles the 8.5x jump balloon showed;
most of the small-object gain happens early, before the decay.

> **Corrected 2026-08-28 -- the first version of this paragraph quoted "0.62
> points, 2.4% relative" for the gain across the decay. That was a single run.**
> Across three seeds the gain is **1.87 +/- 1.20** points, because seed 0 simply
> happened to be further along at iteration 4000 than the other two. Quoting an
> n=1 delta as a result, in the very entry arguing against doing so, is the same
> mistake one level up. See "Three seeds" below.

The balloon effect was an artefact of 61 training images and a 13-image val set
where `APs` rested on three instances. Three rounds of scrutiny have now cost that
finding its magnitude, then its generality, and finally forced a correction to its
own refutation. The residue worth keeping: **do not extrapolate from a toy
dataset**, check the per-bucket instance count before reading a per-bucket metric,
and **do not quote a delta from one run**, including a delta that refutes
something.

### Three seeds

Seeds 0, 1 and 2. Everything else held fixed, including the split, which is read
from `configs/spacenet2_split.json` rather than regenerated -- otherwise seed
variance and split variance are summed with no way to separate them. Sequential,
one GPU. About 1:52 each, up from 1:45 now that the F1 evaluator runs alongside
COCO at every eval point.

**Final values are highly reproducible.**

| metric | seed 0 | seed 1 | seed 2 | mean | sd | CV |
|---|---|---|---|---|---|---|
| segm AP | 49.44 | 49.60 | 49.47 | 49.504 | 0.088 | 0.18% |
| segm APs | 26.44 | 26.63 | 26.69 | 26.587 | 0.135 | 0.51% |
| pooled F1 | 0.7935 | 0.7951 | 0.7950 | 0.7945 | 0.0009 | 0.11% |

`APs` had **CV 59.8% on balloon against 0.51% here** -- a hundredfold reduction,
which is what the three-instances-versus-81862-instances diagnosis predicted. The
instability was sample size. That is now measured rather than argued.

**The per-city differences are structural, not noise.**

| AOI | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| Vegas | 0.8947 | 0.895 | 0.896 | 0.8952 | 0.0007 |
| Paris | 0.7773 | 0.782 | 0.778 | 0.7791 | 0.0025 |
| Shanghai | 0.6862 | 0.689 | 0.688 | 0.6877 | 0.0014 |
| Khartoum | 0.6267 | 0.628 | 0.627 | 0.6272 | 0.0007 |

The Vegas-Khartoum gap is 0.268 against a seed sd near 0.001, roughly 200 sigma.
This also fixes the resolution of the `per_image` vs `per_city` stretch
comparison still to come: anything above about 0.003 in F1 will be real.

**But mid-training trajectories are NOT reproducible**, and this is the finding
that matters most:

```
segm/APs   iter1000  iter2000  iter3000  iter4000  iter5000  iter6000
seed 0        21.23     22.80     25.47     25.82     26.38     26.44
seed 1        20.95     21.65     24.19     23.63     26.44     26.63
seed 2        20.95     24.09     24.15     24.69     26.61     26.69

gain across the LR decay at 4000:   +0.62     +3.00     +2.00
```

At iteration 4000 the three runs span 1.5 AP, and the gain attributable to the
decay ranges from +0.62 to +3.00 -- a factor of five. The endpoints agree to
0.26.

Generalisable, and it is the sharper version of everything above: **final
performance can be highly reproducible while mid-training trajectories are not.**
Any claim about *when* something happens during training needs multiple seeds even
when the endpoint clearly does not. Every trajectory-shaped claim in this notebook
before today was made from a single run.

### Artifacts

```
outputs/spacenet2_r50fpn/          seed 0
outputs/spacenet2_r50fpn_seed1/    seed 1
outputs/spacenet2_r50fpn_seed2/    seed 2
outputs/thresh_select/             train-split predictions, threshold selection
  model_best.pth, model_final.pth
  metrics.json
  inference/instances_predictions.pth          pooled val predictions
  inference/spacenet2_val_AOI_*/               per-AOI predictions
configs/spacenet2_split.json                   split membership (authority)
configs/spacenet2_stretch.json                 per-city percentile constants
```

`scripts/score_f1.py` scores F1 from saved predictions, so a finished run can be
re-scored without another inference pass.

### Open

- ~~Threshold selection~~ -- done, selected on train at 0.544, cross-checked
  against a val-half selection that agreed to 0.005.
- ~~Seed variance~~ -- done, three seeds. segm AP 49.504 +/- 0.088.
- **`per_city` stretch comparison**, paired by seed against `per_image`. Now
  interpretable: the resolution is about 0.003 F1.
- ~~Spatially blocked split~~ -- done, see the 2026-08-28 entry. Pooled
  inflation is about 0.4%. Per-city is unresolved and needs replicates.
- **Khartoum at 0.627 against Vegas 0.895**, now known to be structural at
  ~200 sigma. Worth understanding rather than reporting.
- **Milestone B remainder:** public repo and first blog post. The modelling is
  done.


## 2026-08-28 - Spatially blocked split: the caveat, quantified

### The question

SpaceNet chips do not overlap, but adjacent chips share a street grid, roof
materials, sun angle and acquisition. Under the random split almost every val
tile has a training tile next door, so val scores flatter the model against
genuinely unseen ground by an unmeasured amount. That was the loudest remaining
caveat on the comparison with the published numbers.

Holding out whole contiguous blocks instead of scattered tiles removes most of
that adjacency. The difference between the two splits is the size of the
inflation.

### Block size, chosen by measurement

The metric that matters is the fraction of val tiles having an 8-neighbour in
train, so that is what was measured rather than assumed:

| block size | adjacency | blocks in smallest AOI | val fraction |
|---|---|---|---|
| random (none) | 0.995 | -- | 20.0% |
| 5 tiles (~1 km) | 0.485 | 96 | 20.1% |
| **10 tiles (~2 km)** | **0.277** | **29** | **20.7%** |
| 16 tiles (~3 km) | 0.159 | 11 | 23.5% |

10 tiles is the knee: adjacency falls 73% while the val fraction holds near 20.
At 16 tiles Paris has only 11 blocks and val overshoots to 23.5%, so val stops
sampling the city and becomes a handful of neighbourhoods. The residual 0.277 is
block perimeter and cannot be removed without giving up that granularity, so this
**bounds** the inflation rather than eliminating it.

Blocks are assigned by seeded shuffle, not by taking one contiguous chunk per
city: a single chunk would hold out one neighbourhood type and measure that
instead of generalisation.

Split: 8427 train / 2165 val, against 8474 / 2118 for the random split.

### Pooled result: real, and small

| | random (3-seed mean) | blocked | delta | in sigma |
|---|---|---|---|---|
| segm AP | 49.504 +/- 0.088 | 49.179 | -0.325 | 3.7 |
| pooled F1 | 0.7945 +/- 0.0009 | 0.7911 | -0.0034 | 3.8 |

Statistically real -- and the only reason that judgement can be made is that the
three-seed run established sigma first. **But it is 0.4% relative.** Spatial
autocorrelation was not materially inflating the headline number.

### Per-city result: it moved the wrong way, and that is the finding

| AOI | random | blocked | delta |
|---|---|---|---|
| Vegas | 0.8952 | 0.891 | -0.004 |
| Paris | 0.7791 | 0.795 | **+0.016** |
| Shanghai | 0.6877 | 0.679 | -0.009 |
| Khartoum | 0.6272 | 0.668 | **+0.041** |
| macro | 0.7462 | 0.7583 | **+0.012** |

The macro average went **up** under the harder split, driven by Khartoum gaining
0.041 -- forty times the seed noise.

The cause is that a blocked split does not only remove adjacency, it changes
**which regions** are held out. Khartoum has 37 blocks, so a 20% val sample is
seven or eight of them, and which particular neighbourhoods those are is itself
high-variance. That variance is roughly +/- 0.04 per city, **an order of magnitude
larger than the ~0.003 effect it was built to measure.**

So for the small AOIs the measurement introduces more noise than the quantity
being measured. The pooled micro figure is the trustworthy one: it aggregates
44363 instances across four cities and averages the region-sampling variance
away.

Settling the per-city question properly needs three or four blocked splits with
different block-assignment seeds, about 8 hours of GPU. The pooled answer is
already in hand and is the one the published comparison needed.

### What it changes

"The random split flatters us by an unknown amount" becomes "by about 0.4% on the
pooled metric". That is a defensible sentence. It does not rescue the comparison
entirely -- evaluation is still on a split carved from training data rather than
the competition withheld test set, and that remains the larger gap -- but the
specific worry about neighbouring tiles is now bounded and small.

Note also that Khartoum, the hardest city, scored **better** under the harder
split. A caution against reading much into any single held-out sample of a small
AOI: the same lesson the balloon work taught, arriving from the opposite
direction.
