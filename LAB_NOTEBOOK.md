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
  commits were rewritten from the university address to the permanent personal
  one before the first push, so they attribute to the GitHub account; git
  identity is now set globally to match. One contact address is correct for this
  work and every other address is not, so the superseded one is not spelled out
  here -- what matters to the record is that the identity was unified, not what
  it was unified away from.

  > **Superseded 2026-08-28.** That repository was deleted and recreated, and
  > the identity rewritten again. Current home:
  > `github.com/benjbritton/spacenet2-detectron2`, public. See the identity
  > entry below.


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

> **Amended 2026-08-28.** The conclusion is right, the framing was wrong.
> Solaris has no published SN2 results to reproduce in the first place -- it
> postdates the challenge by about two years, and its CosmiQ baselines are SN4,
> SN6 and SN7. The installability problem was never what stood between this
> project and a comparison. The reference is the SpaceNet dataset paper
> (arXiv 1807.01232), which publishes per-city F1 for the top three competitors
> *and* two baselines (YOLT 0.60, modified MNC 0.57). See the last
> 2026-08-28 entry.

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
averaging weights the easy case. The macro average is the comparable figure.

> **Corrected 2026-08-29.** This said 0.7462, and so did every later quotation of
> it, including the public README. **0.7462 is the mean of per-city
> BEST-threshold F1** -- a threshold tuned per city on the set being scored,
> which is the practice this very entry argues against two paragraphs above. The
> table at the top of this section had the right number (0.7447) all along; the
> prose drifted to the flattering one and everything downstream inherited it.
>
> The reportable figure, every city at the fixed train-selected 0.544, recomputed
> across all three seeds: **macro 0.7459 +/- 0.0012** (Vegas 0.8948, Paris
> 0.7787, Shanghai 0.6848, Khartoum 0.6254).
>
> The correction is worth 0.0003. That is the part to keep: the honest number and
> the tuned one were indistinguishable, so nothing was ever gained by the slip --
> which is exactly why it survived six months of quotation unnoticed. A number
> being unimportant is what lets it go unchecked.

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
- ~~**Khartoum at 0.627 against Vegas 0.895**~~ -- partly answered, see the
  second 2026-08-28 entry. Not composition (size explains 18%), not albedo
  (Khartoum leads on it), not crowding. Tracks boundary contrast and absence of
  cast shadow. The failure is recall, not precision.
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


## 2026-08-28 - Why Khartoum is hard: not composition, and not albedo

The per-city spread has been measured (0.895 Vegas to 0.627 Khartoum) and shown
to be structural at ~200 sigma, but never explained. This entry explains as much
of it as the data supports and rules out two candidate explanations, one of which
was mine and wrong.

### Getting eyes on the data first

Everything below started from looking at tiles, which nothing in this project had
done systematically since the converter QA. Two review artefacts were built:

**`scripts/export_predictions_geojson.py`** -- every val prediction as
georeferenced GeoJSON, EPSG:4326, one file per AOI, with `score`, `tile`,
`area_px` and bbox dimensions as attributes. Ground truth likewise with `--gt`.
Vectors rather than a rendered picture on purpose: a burned-in overlay fixes the
score threshold at render time, whereas a definition query on `score` makes the
threshold a slider and lets precision and recall be watched trading against each
other on real geography. Everything down to score 0.05 is exported so the slider
has range below the 0.544 reporting threshold. No re-inference -- it reads
`instances_predictions.pth`.

Free correctness check: the exported ground-truth counts are 22250 / 3048 /
13388 / 4802, matching the val instance counts in the 2026-08-27 split table
exactly. The whole pixel-to-lon/lat path is therefore consistent with what was
scored.

**`scripts/overlay_geotiff.py`** -- ground truth and predictions burned onto
tiles and written back out **as GeoTIFFs carrying the source CRS and transform**,
so a figure lands in place on a map instead of floating. `--png` writes a plain
copy alongside, because Windows renders GeoTIFF unreliably and the source tiles
are 16-bit and display black. 4 tiles per city, chosen at even intervals through
each city's building-count ranking rather than at random: that samples the range
(densest, two middling, and -- since 2069 tiles are empty -- always one empty
tile, which doubles as a false-positive check on bare ground).

The display stretch in both scripts is deliberately independent of the training
preprocessing in `detlab.datasets.spacenet`. It is for eyes, not for the network,
and conflating the two is how a rendering choice quietly becomes a claim.

### Two hypotheses from looking at Khartoum

1. **Albedo.** Khartoum roofs and bare ground look alike, so perhaps there is
   little radiometric signal marking a building at all.
2. **Relief.** Khartoum roofs are flat. A pitched roof gives two faces at
   different brightness plus a cast shadow, boundary cues that survive even where
   albedo does not.

Both predict low recall specifically, which is consistent with Khartoum's
reported precision 0.676 against recall 0.583.

Note that (2) cuts both ways and is not obviously right a priori: pitched and
tall buildings suffer parallax, so the visible roof is displaced from the ground
footprint the annotation traces, while Khartoum's low single-storey compounds
have almost none of that. Worth measuring rather than asserting.

### Measured: `scripts/city_separability.py`

250 tiles per city, fixed stride through the tile list rather than a random draw
so the answer is reproducible without carrying another seed. All statistics
computed on the uint8 image the **network** sees (`per_image` 2-98 stretch), not
on raw DN -- the question is what the model can discriminate, and a stretch
applied before the model changes that. Numbers were stable from 40 tiles, so they
are not a sampling artefact.

| city | F1 | cohen d | boundary | shadow | med px | % small | abut |
|---|---|---|---|---|---|---|---|
| Vegas | 0.895 | 1.136 | **0.435** | **1.524** | 2327 | 29.0 | 0.000 |
| Paris | 0.779 | 0.972 | 0.314 | 0.974 | 1403 | 36.4 | 0.000 |
| Shanghai | 0.688 | 0.724 | 0.092 | 0.555 | 1087 | 48.0 | 0.001 |
| Khartoum | 0.627 | **1.575** | 0.315 | 0.634 | 1182 | 47.7 | 0.000 |

- **cohen d** -- inside-footprint vs outside-footprint brightness in pooled sd.
- **boundary** -- the same difference across a 2 px ring either side of the
  footprint edge, normalised by tile sd. Pixels belonging to any *other*
  building are excluded from the outside ring, or dense blocks measure roof
  against roof.
- **shadow** -- dark-pixel fraction (< mean - 1 sd) in a 2-6 px band outside
  footprints, over the same fraction across all non-building pixels. Above 1
  means darkness concentrates around buildings. A proxy, not a shadow detector:
  it cannot distinguish a shadow from a dark courtyard.
- **abut** -- fraction of footprint area whose 1 px dilation lands on a
  different footprint.

**The albedo hypothesis is wrong, and instructively so.** Khartoum has the
*highest* roof-vs-ground separation of the four, 1.575 pooled sd, 39% above
Vegas. Its buildings stand out radiometrically more than anyone's.

But its **boundary** contrast is 0.315 against Vegas's 0.435. Khartoum roofs
differ from the ground on average while the edge between them is soft. Detection
lives on edges, not on means, and the two measurements separate exactly there.
What reads by eye as "the roofs look like the dirt" is the absence of a crisp
boundary, not an absence of tonal difference. The global statistic and the
boundary statistic disagreeing is the useful part; either alone would have
misled.

**The relief hypothesis survives.** Shadow ratio orders with F1 (Spearman +0.80):
Vegas 1.52, Paris 0.97, Shanghai 0.56, Khartoum 0.63. Below 1.0 means darkness is
*less* common near buildings than elsewhere in the tile -- Shanghai and Khartoum
have no cast-shadow signal at all, while Vegas has a strong one.

**Crowding is ruled out.** Abutment is ~0 everywhere.

> The printed Spearman rho for `abut_frac` is -0.80, which is noise: the
> underlying values are 0.000, 0.000, 0.001, 0.000, so the ranking is decided by
> rounding. A rank correlation over four cities on a near-constant column means
> nothing, and it is recorded here only so the table is not read as evidence.
> The same n=4 caution applies, less severely, to every rho in that block.

### The size confound, and settling it: `scripts/f1_by_size.py`

Khartoum is 47.7% small instances against Vegas's 29.0%, median footprint 1182 px
against 2327. Small objects score worse everywhere, so composition alone could in
principle produce the entire ordering. That has to be excluded before any claim
about contrast or relief means anything.

The test is to score **within** a size bucket. Ground truth is bucketed by its own
COCO area, as pycocotools does. A false positive has no ground truth to inherit
an area from, so it is bucketed by its own predicted mask area -- COCOeval's
convention, stated here because it is a choice, not a law. Matching is the same
single greedy pass at IoU 0.5 the headline F1 used; only the bookkeeping differs.
That required `match_greedy` to expose *which* ground truth was hit, so
`src/detlab/spacenet_f1.py` now has `match_greedy_pairs` returning indices, with
`match_greedy` reduced to a two-line wrapper over it. One implementation of the
matching, two views of it. The module self-test still passes.

F1 within bucket, at the 0.544 reporting threshold:

| bucket | Vegas | Paris | Shanghai | Khartoum |
|---|---|---|---|---|
| small | 0.667 | 0.573 | 0.538 | **0.423** |
| medium | 0.980 | 0.879 | 0.792 | **0.760** |
| large | 0.932 | 0.807 | 0.804 | **0.746** |

**The city ordering survives inside every bucket**, and the gaps stay large:
Vegas's *medium* buildings score 0.980 against Khartoum's 0.760, a 0.22 gap
between objects of the same size class.

Re-weighting each city to Vegas's size mix:

| city | actual | size-standardised | delta |
|---|---|---|---|
| Vegas | 0.886 | 0.886 | +0.000 |
| Paris | 0.763 | 0.786 | +0.022 |
| Shanghai | 0.673 | 0.716 | +0.044 |
| Khartoum | 0.609 | 0.659 | +0.050 |

Composition closes 0.050 of the 0.277 Vegas-Khartoum gap, about **18%**. The
other 82% is genuine per-bucket difficulty. Size is a real term and a minor one.

Two things not to trip over. The standardisation is crude: it assumes bucket F1
is independent of the mix, which is not exactly true. And the "actual" column
reads 0.886 / 0.609 where the 2026-08-27 table reports 0.895 / 0.627 -- these are
bucket-weighted averages of bucket F1s, a different arithmetic, not a
disagreement between runs.

### What this establishes

- The Vegas-Khartoum gap is **not** composition (18%), **not** *global luminance*
  contrast (Khartoum leads on it), and **not** crowding (absent everywhere).
  "Albedo" was the word used here originally; it is imprecise now that chromatic
  contrast has been measured separately and separately refuted. See the hue entry
  below, which splits contrast into three readings and settles each.
- The failure mode is **missing buildings, not inventing them**: recall is below
  precision in all twelve city-bucket cells, most starkly Khartoum small,
  precision 0.535 against recall 0.350. Under half the small Khartoum buildings
  are found.
- The two measurements that track difficulty are **boundary contrast** and
  **shadow**. Both are flat-roof consequences: flat roofs on flat ground give a
  soft edge and no cast shadow.

  > **Corrected same day.** This bullet originally continued "...so the only
  > remaining cue is a tonal difference the model cannot localise precisely
  > enough to clear IoU 0.5." That mechanism is **refuted** -- see the Open list
  > below and `scripts/iou_sweep.py`. At IoU 0.10 Khartoum still misses 32% of
  > buildings, so they are not being found and lost on geometry; nothing is
  > proposed on them at all. Boundary contrast and shadow remain real
  > measurements that track difficulty. Neither has been shown to cause anything.
  >
  > Note what this leaves shadow as. The hue entry below ablated chroma and found
  > a 42.7% attribution collapse to nothing. **Shadow has had no equivalent test**
  > -- it is a correlate with an untested mechanism, exactly the position hue was
  > in before the ablation, and it should be read that way rather than as the
  > surviving explanation.
- **Vegas medium is 0.980, effectively saturated.** Whatever headroom the
  pipeline has left is in small objects generally and Khartoum specifically, not
  in the bulk of the easy city. That is worth knowing before choosing what to
  improve next.

Honest limits: n=4 cities, so every rank correlation here is suggestive and
nothing more. Size and relief are physically entangled -- small, single-storey,
flat-roofed buildings are one building type, not two independent variables -- and
four cities cannot separate them. The shadow proxy measures darkness near
buildings, which a shadow causes but does not uniquely cause. The tile-level
attribution in the hue entry below lifts the n=4 constraint for the factors it
covers, and its finding applies here in full: a factor can apportion a large
share of the gap and still not be a cue.

### Artifacts

```
scripts/export_predictions_geojson.py
scripts/overlay_geotiff.py
scripts/city_separability.py
scripts/f1_by_size.py
src/detlab/spacenet_f1.py           match_greedy_pairs added; match_greedy wraps it

outputs/vector_review/              8 GeoJSON: <aoi>_pred / <aoi>_gt, EPSG:4326
outputs/overlay_geotiff/            16 GeoTIFF + 16 PNG, 4 tiles per city
outputs/city_analysis/separability.json
outputs/city_analysis/f1_by_size.json
```

Windows note: the WSL filesystem has no drive letter -- it is an `ext4.vhdx`
exposed as the `\\wsl.localhost\Ubuntu-24.04` share while the distro is running.
Mapped to `Z:` (`net use Z: \\wsl.localhost\Ubuntu-24.04 /persistent:yes`) so
ArcGIS and Explorer can reach `outputs/` by path. The share dies when the distro
stops; `wsl -d Ubuntu-24.04 -- true` revives it.

### Open

- ~~**Boundary-quality decomposition.**~~ Done, and it came out against the
  expectation written here. Recall at IoU 0.5 conflates "not found" with "found
  but outlined too loosely"; `scripts/iou_sweep.py` separates them. Khartoum
  recovers 18.3% at IoU 0.25, *less* than Paris at 25.1%, and still misses 32% at
  IoU 0.10. **Nothing is proposed on those buildings at all**, so the failure is
  upstream of the mask head, not in it. This is what severs the boundary-contrast
  measurement from the failure mode -- see the correction in "What this
  establishes" above.
- **Per-city score thresholds.** 0.544 was selected once, pooled, on train and
  applied to all four cities. No per-city optimum has been computed, so the
  per-city table is at a threshold suboptimal for every city individually. The
  saved per-AOI predictions make this a sweep, not a run.
- Whether any of this is actionable. A finding that Khartoum lacks a cue is not
  yet a change to the model.


## 2026-08-28 - The published reference, located: it was never Solaris

### Solaris has no SN2 numbers, and could not have

The 2026-08-27 entry recorded that Solaris cannot be installed and concluded that
Milestone B is "reproducing the published method and numbers, not the published
code". That conclusion holds, but the reasoning was incomplete in a way worth
correcting: **Solaris has no published SpaceNet 2 results to reproduce.** It
postdates the challenge by roughly two years -- SN2 ran in 2017, Solaris appears
from 2019 -- and the CosmiQ baselines built on it are SN4, SN6 and SN7. Searching
turns up no SN2 F1 from Solaris at all.

So the installation failure was never the thing standing between this project and
a comparison. The comparison target is, and always was, **the SpaceNet dataset
paper itself** (arXiv 1807.01232).

### The published table

| method | Vegas | Paris | Shanghai | Khartoum | total |
|---|---|---|---|---|---|
| XD_XD (1st) | 0.885 | 0.745 | 0.597 | 0.544 | **0.69** |
| wleite (2nd) | 0.829 | 0.679 | 0.581 | 0.483 | 0.64 |
| nofto (3rd) | 0.787 | 0.584 | 0.520 | 0.424 | 0.58 |
| YOLT (baseline) | | | | | **0.60** |
| modified MNC (baseline) | | | | | **0.57** |

The two baseline rows are new to this notebook and matter more than the winners
for Milestone B: a *baseline* is what a baseline should be measured against.

Three details from the paper settle things this project had to assume:

1. **"Total Score" is the arithmetic mean of the per-city F1** -- an explicit
   macro average. The 2026-08-27 entry argued from first principles that macro
   (0.7462) rather than pooled micro (0.7935) is the comparable figure, because
   Vegas is 51% of val instances. That argument is now citable rather than merely
   defensible: micro was never what the competition reported.
2. **The metric is F1 at IoU >= 0.5 on polygons.** Confirms both the operating
   point `src/detlab/spacenet_f1.py` implements and the raster-vs-polygon caveat
   already recorded.
3. **Scores are on a withheld test set**, from a 60/20/20 train/test/validation
   split. Confirms the largest outstanding caveat: our val is carved from
   training data.

### Two headline numbers for one model, finally reconciled

This notebook has reported **segm AP 49.44** and **SpaceNet F1 0.7930** in
separate entries since 2026-08-27 without ever relating them. Same model, same
weights, same 2118 val tiles. A reader arriving cold sees two numbers sixty
points apart and draws the obvious inference -- that one metric says the model is
mediocre and the other says it is good -- and that inference is wrong. The
reconciling figures were in `metrics.json` the whole time and were never quoted.

Seed 0, pooled val, iteration 6000:

| | segm | bbox | what it requires |
|---|---|---|---|
| AP50 | **81.21** | 82.06 | 50% overlap |
| AP75 | 54.35 | 59.48 | 75% overlap |
| AP | 49.44 | 52.76 | mean of ten thresholds, 0.50 to 0.95 in steps of 0.05 |
| SpaceNet F1 | **0.7930** | | 50% overlap, at one score threshold |

Three differences, each accounting for part of the sixty points:

1. **Scale.** COCO reports AP x100 by convention. 49.44 is 0.4944.
2. **IoU strictness.** F1 is at IoU 0.5 only. AP averages ten thresholds up to
   0.95, where scores approach zero. Put both on the same overlap requirement and
   **AP50 0.8121 against F1 0.7930 -- about two points apart.**
3. **Confidence handling**, which is the residual two points. F1 commits to one
   score cutoff (0.544) and reports that operating point. AP integrates the whole
   precision-recall curve, including the low-score tail where precision collapses.

**So 49.44 is not a worse result than 0.793. It is the same result, averaged
across nine additional and progressively brutal overlap requirements.**

AP75 at 54.35 is the informative middle term: performance falls by a third when
the bar moves from 50% to 75% overlap, and the thresholds above 0.75 contribute
almost nothing. That is a localisation-quality statement, and it is what makes
COCO AP worth logging here even though nothing external can be compared to it --
it measures outline precision that F1 at IoU 0.5 is blind to by construction.

Two consequences already relied on elsewhere in this notebook and not previously
justified. The IoU sweep's premise -- that recall at IoU 0.5 conflates "not
found" with "found but outlined loosely" -- is exactly the AP50-to-AP75 fall made
per-city. And the reason SpaceNet F1 and COCO mAP are *not convertible*, asserted
on 2026-08-27, is this: they differ on two axes at once, and only one of them
(IoU) has a defined mapping.

### Lining the numbers up

No published mAP for SN2 exists, so COCO AP has nothing to compare against and
never will. F1 is the metric the reference reports, and it is already computed
alongside COCO at every evaluation point -- `SpaceNetF1Evaluator` has run in the
training loop since 2026-08-27, which is why the three-seed runs took 1:52
against 1:45. **No new work was needed to make the numbers line up; the right
metric was already being logged.** COCO AP stays in the notebook as the internal
diagnostic it is, useful for the size buckets and for comparing our own runs, and
carries no external claim.

| | ours (random split, 3 seeds) | XD_XD | YOLT | MNC |
|---|---|---|---|---|
| Vegas | 0.8948 | 0.885 | | |
| Paris | 0.7787 | 0.745 | | |
| Shanghai | 0.6848 | 0.597 | | |
| Khartoum | 0.6254 | 0.544 | | |
| **macro** | **0.7459** +/- 0.0012 | **0.693** | 0.60 | 0.57 |

Every city at the **fixed** train-selected threshold 0.544, 3-seed mean. The
figures originally in this table (0.8952 / 0.7791 / 0.6877 / 0.6272, macro
0.7462) were per-city *best*-threshold values; see the correction in the
2026-08-27 entry. The difference is 0.0003.

Blocked split gives macro 0.7583, pooled F1 0.7911.

**Still not a claim of beating the 2017 winner**, and the reasons are unchanged
and now individually documented: their scores are on the withheld test set while
ours are on a val split of the training data; our random tile split is spatially
autocorrelated, worth about 0.4% on the pooled metric (2026-08-28, first entry);
and our IoU is on rasterised masks rather than georeferenced polygons. The
defensible statement is that the pipeline lands in the neighbourhood of the
published results and reproduces their per-city difficulty ordering exactly,
under evaluation conditions that favour it by an amount partly quantified and
partly not.

Milestone B's stated target -- F1 >= 0.554, within 20% of the reference -- is met
against 0.7462, and is also clear of both published baselines.

### The paper states a cause for Khartoum. Half of it is wrong.

The paper explains the city ordering in one sentence each, with no measurement
behind either:

> Khartoum, hardest, "partly due to the high variance in building size and low
> contrast between building and background".
>
> Vegas, easiest, "partly due to the many well separated residential buildings
> with low variance in size".

All three claims are now measured (second 2026-08-28 entry above):

| claim | verdict |
|---|---|
| Khartoum: building size | **supported, and small.** Khartoum is 47.7% small against Vegas's 29.0%, but the city ordering survives inside every size bucket and standardising to Vegas's mix closes only 18% of the 0.277 gap. |
| Khartoum: low contrast | **contradicted.** Khartoum has the *highest* roof-vs-ground brightness separation of the four, Cohen d 1.575 against Vegas's 1.136. |
| Vegas: well separated | **does not distinguish.** Abutment is ~0 in all four cities, so separation cannot be what sets Vegas apart. |

The contrast claim fails on the obvious reading and survives on a narrower one.
What is low in Khartoum is **boundary** contrast -- 0.315 against Vegas's 0.435 --
not global contrast. Its roofs differ from the ground on average while the edge
between them is soft, and detection lives on edges rather than on means. A single
global statistic would have agreed with the paper if it had been the only one
computed; the two statistics disagreeing is the finding.

Worth stating plainly, because it is the most defensible original result this
project has produced so far: **the dataset paper's stated explanation for its own
difficulty ordering is, on the natural reading, not supported by the data.** It
was an aside rather than a claim under test, and this is not a criticism of the
paper -- but it is measurable, it was not measured, and it is now.

The size half of their explanation also deserves credit where due: it is real,
just far smaller than the sentence implies.

### Open

- The blog post now has a spine: a reproduction that lands near published
  numbers, and a measured correction to the published explanation of why one
  city is hard.
- ~~Whether the boundary-contrast finding survives at IoU 0.25~~ -- run, and the
  prediction made here was **wrong**. This entry predicted that if Khartoum
  recall jumped sharply at a looser IoU, the soft-edge measurement would be tied
  to the failure mode. `scripts/iou_sweep.py` says the opposite: Khartoum
  recovers 18.3% at IoU 0.25, *less* than Paris at 25.1%, and at IoU 0.10 it
  still misses 32% of buildings.

  Loosening the geometric bar almost to nothing does not find them, so they were
  never proposed. **The soft-edge finding survives as a measurement and loses its
  causal link to the misses.** Boundary contrast is genuinely low in Khartoum
  (0.315 against Vegas's 0.435) and that is genuinely not what is producing the
  failure. The failure is upstream of localisation entirely -- backbone or RPN,
  not the mask head -- and the entry above should be read with that correction:
  it establishes what Khartoum's imagery lacks, not yet why the detector misses.

  Worth keeping as a method note. The boundary measurement and the recall
  failure were both real and both about Khartoum, which made the causal story
  between them feel settled without being tested. It took a threshold sweep
  costing no GPU to break the link. Three separate claims in this notebook have
  now died the same way, and the pattern is always a plausible mechanism
  connecting two true measurements.

  A fuller synthesis with the hue and grayscale results belongs in the
  process window's write-up rather than being pre-empted here.


## 2026-08-28 - Hue: measured, apportioned 42.7% of the gap, and then ablated to nothing

Ben's hypothesis, from looking at the overlays: Khartoum roofs and terrain share a
hue even where their brightness differs, so the chromatic channel carries no
signal marking a building. Everything measured so far had been luminance, so the
question was open and separate.

It ran to a clean negative, by way of a large positive that did not survive being
tested. The sequence is the point of this entry.

### Step 1 -- hue separation is real, large, and orders with difficulty

`scripts/city_hue.py`. Hue is circular, so every mean is `atan2(mean sin, mean
cos)` and every distance wraps; a linear mean over hue angles puts the mean of red
pixels in the cyans. Hue is also undefined at low saturation -- exactly the regime
bare desert and concrete roofs occupy -- so saturation is reported beside every
number and pixels below 0.05 do not vote. All float32: **8-bit OpenCV HSV
quantises hue to 2 degrees per step, and Khartoum's entire roof-to-ground
separation is 2.3 degrees.** Measured in 8-bit this finding does not exist.

Raw sensor hue, roof against ground:

| city | F1 | roof | ground | separation | d_circ | saturation (roof) |
|---|---|---|---|---|---|---|
| Vegas | 0.895 | 109.7 | 139.1 | **29.4** | 1.91 | 0.316 |
| Paris | 0.779 | 132.6 | 156.3 | **23.8** | 2.11 | 0.257 |
| Shanghai | 0.688 | 136.9 | 141.9 | 5.0 | 0.48 | 0.316 |
| Khartoum | 0.627 | 81.6 | 83.9 | **2.3** | 0.35 | 0.326 |

Khartoum roof and ground are 2.3 degrees apart against Vegas at 29.4, a factor of
thirteen, and raw hue splits the cities exactly along the F1 split as a binary
separation rather than a gradient. It is **not** a low-saturation artefact, which
was the obvious way this measurement could have lied: Khartoum has the *highest*
roof saturation of the four, well clear of the noise floor. There is real colour
there. It is the same colour on both sides of the wall.

Note the dissociation with the previous entry: Khartoum leads on *brightness*
separation (d 1.575) and comes last on *hue* separation. Those two come apart,
and the one that ordered with F1 was hue.

**A prediction of mine that failed, recorded because it was wrong.** The
per-channel percentile stretch was expected to destroy chromatic signal before the
network sees it. It does the opposite -- being an independent per-channel
normalisation it acts as a per-tile white balance and **amplifies** hue
separation, lifting Shanghai from 5.0 to 63.7 degrees. Khartoum is the only city
it cannot rescue, reaching 19.7 against roughly 65 for the other three.

### Step 2 -- apportioning it, which city-level analysis cannot do

`scripts/factor_attribution.py`. Correlating factors against per-city F1 is
impossible on its face: four cities, four correlated predictors, zero residual
degrees of freedom. Any coefficients fit perfectly and none mean anything.

The constraint is an artefact of aggregating. Those four cities are 1696 usable
validation tiles carrying 43478 instances, each with its own hue separation,
boundary contrast, shadow signature, size and measured recall. Weighted least
squares at tile level, weighted by ground-truth count so the fit is an
instance-level statement.

Of the Vegas-Khartoum recall gap of 0.2813, each factor alone:

| factor | explains alone | partial R2 (tile-to-tile) |
|---|---|---|
| **hue separation** | **42.7%** | 0.025 |
| shadow | 35.6% | 0.036 |
| size | 26.1% | **0.118** |
| boundary contrast | 12.7% | 0.017 |
| density | none | 0.000 |

Those sum past 100 because the factors overlap. Fitted jointly they explain 55.7%
and **44.3% of the gap survives unexplained.** Collinearity was low, all VIF under
1.5, so the individual coefficients were not mush. Full model R2 0.487.

Hue came out the largest single contributor to the city gap.

### Step 3 -- the ablation, and it kills the finding

Correlation on observational data cannot establish mechanism, and Vegas has more
of every favourable property at once, so association is what this analysis would
show whether hue mattered or not. The test is to remove hue and retrain.

Grayscale mode collapses chroma **after** the stretch and replicates the single
channel three times, so architecture, input shape and the COCO-pretrained stem
stay byte-identical and colour is the only variable. A single-channel input would
change the first convolution too and confound the two. Verified before launch: the
three output channels are byte-identical and equal the mean of the colour
channels.

**The interpretation of each outcome was written into commit `cf36ccc` before the
run finished**, so it could not be reasoned backwards afterwards. It predicted
that if hue were causal, Vegas (29.4 deg) and Paris (23.8) would lose most while
Khartoum (2.3) had almost nothing to lose, and the gap would close by something
like 42.7%.

Seed 0, everything else identical:

| AOI | colour | grayscale | delta |
|---|---|---|---|
| Vegas | 0.8947 | 0.893 | -0.002 |
| Paris | 0.7773 | 0.777 | -0.000 |
| Shanghai | 0.6862 | 0.678 | -0.008 |
| Khartoum | 0.6267 | 0.626 | -0.001 |
| pooled F1 | 0.7935 | 0.7895 | -0.004 |
| segm AP | 49.44 | 49.11 | -0.33 |
| **Vegas-Khartoum gap** | **0.2680** | **0.2670** | **-0.001** |

**The gap did not move.** Predicted to close by 42.7% if hue were causal; it
closed by 0.4%, which is nothing. Seed noise is ~0.001 to 0.0025 per city, so
only the pooled drop (-0.004, about 4 sigma) and Shanghai (-0.008) are real at
all, and both are tiny.

**Colour contributes almost nothing to this task. The model retains 99.5% of its
performance on grayscale.**

One coherent micro-result inside the negative: Shanghai lost the most, and
Shanghai is the city whose chroma the stretch amplified most (5.0 -> 63.7
degrees). Small, but it points the right way.

### What actually happened, and why the 42.7% was not a lie

Hue separation genuinely predicts difficulty. The detector genuinely does not use
it. Both are true because hue separation is a **proxy for scene complexity**
rather than a cue: Vegas has vegetation, pools, pitched roofs and cast shadows,
which produce chromatic variety *and* the structural cues the model actually keys
on. The correlation runs entirely through the confound.

This is the fourth claim in this notebook to die the same way, and the pattern the
previous entry named holds exactly: a plausible mechanism connecting two true
measurements. What is new here is that an R-squared apportionment -- a more
formal-looking instrument than the earlier eyeball inferences -- produced a
confident 42.7% that an ablation reduced to zero. **Attribution analysis on
observational data cannot distinguish a cue from a correlate, however good the
diagnostics look.** VIF under 1.5 said the coefficients were stable; stability is
not causality.

### Adjudicating the dataset paper's own claim

arXiv 1807.01232 explains Khartoum in one unmeasured sentence: *"low contrast
between building and background."* That sentence has three readings, and this
project can now settle all three:

| reading | verdict |
|---|---|
| global luminance contrast | **contradicted** -- Khartoum is highest, d 1.575 vs Vegas 1.136 |
| boundary luminance contrast | **supported** as a measurement -- 0.315 vs 0.435 -- but the IoU sweep severed it from the failure mode |
| chromatic contrast | **supported as a correlate, refuted as a cause** -- 2.3 vs 29.4 degrees, and grayscale changes nothing |

So the published explanation fails on its natural reading, and the two readings
that survive as measurements do not survive as mechanisms. What is established
is negative and worth stating plainly: **Khartoum's buildings are not being missed
for want of contrast, luminance or chromatic.** At IoU 0.10 a third of them are
still missed, so nothing is proposed on them at all, and the cause remains
unidentified.

### Consequences worth acting on

- **A hue-weighted objective on SpaceNet 2 would be building on sand.** If the
  network extracts nothing from chroma here, a loss that weights chromatic
  agreement has nothing to weight. This dataset is the wrong testbed for that
  method, and one 2-hour run establishing it beats a semester discovering it.
- **The asymmetry that makes chromatic detection work elsewhere is now explicit.**
  A method gating on hue needs the target class to be *chromatically defined* -- a
  known centroid to gate against. Buildings have none: Khartoum roofs sit at 81.6
  degrees, Vegas at 109.7, Paris at 132.6. There is no building hue. Hue
  separation from background is not the same property as chromatic definition, and
  this run is the empirical demonstration that the first without the second buys
  nothing.
- **For the community, the useful finding is the negative one.** Colour is worth
  0.4% on SN2 building detection. Anyone reaching for chromatic preprocessing,
  false-colour composites or multispectral bands on this benchmark should know
  the RGB chroma is already almost inert.

### Limits

- One seed for the grayscale run. The colour baseline has three (sd 0.088 AP), and
  the observed differences sit at or under that scale, so the *direction* of small
  per-city deltas is not established -- only that nothing large happened.
- Grayscale removes chroma; it does not test whether a *different* colour
  representation would help. HSV as network input, rather than RGB, remains
  untested and is a different question.
- The stretch amplifies hue before the network sees it, so the ablation removes
  amplified chroma, which is the correct thing to remove but worth stating.

### Artifacts

```
scripts/city_hue.py                 per-city circular hue statistics
scripts/factor_attribution.py       tile-level weighted least squares
scripts/iou_sweep.py                IoU rescore, detection vs geometry
outputs/city_analysis/hue.json
outputs/city_analysis/attribution.json
outputs/spacenet2_r50fpn_gray/      grayscale ablation run
```

W&B run `spacenet2-r50fpn-seed0-GRAYSCALE`.

### Open

- **What is actually causing the 32% of Khartoum buildings nothing is proposed
  on.** Contrast is ruled out on both axes, composition is 18%, crowding is
  absent. The remaining candidates are texture, scale relative to the anchor set,
  and annotation quality -- none measured.
- **Anchor sizes.** Never examined. Khartoum median footprint is 1182 px against
  Vegas 2327, and the FPN anchor set is the COCO default. A proposal stage that
  cannot generate boxes at the right scale would produce exactly this failure.
- **HSV as network input** rather than RGB, which grayscale does not address.


## 2026-08-28 - Identity, licensing, and publication

Housekeeping rather than research, recorded because two of these are the kind of
thing that is invisible until it is expensive.

### Identity unified across all history

Commit authorship was `Ben Britton <permanent personal address>`. Two problems.
The display name was one of four variants in circulation, and ORCID plus both
published works say **Benjamin Britton** -- those are the records that cannot be
edited casually, so everything else conforms to them rather than the reverse.
And the personal address in public commit metadata is scraped at scale through
the GitHub API.

All 38 commits rewritten to
`Benjamin Britton <317455538+benjbritton@users.noreply.github.com>`. The numeric
prefix is the account id, which is what makes the noreply form attribute
correctly; verified after publication that 38 of 38 commits resolve to the
account. Global git config set to match, and repo-local overrides removed so
nothing silently reintroduces the old identity.

**The GitHub repository was deleted and recreated rather than force-pushed.** A
force-push leaves the superseded commits unreachable but not gone: GitHub retains
dangling objects and they stay retrievable by direct SHA more or less
indefinitely. Deleting the repository destroys them with it. The URL is
unchanged in form because a GitHub URL is just `owner/name` and the name was
reclaimed on recreation, but the repository is a different object with a new id
and an empty object store.

The same lesson applied locally. The August rewrite had left
`refs/original/refs/heads/master` in place -- filter-branch's automatic backup,
holding thirteen commits under the superseded university address, unreachable
from any branch but alive in the object store and exposed to `push --mirror` or a
directory copy. Deleted, reflog expired, `gc --prune=now`: 230 loose objects to 0.
**A history rewrite is not finished when the branch looks right.**

### Renamed: `benjbritton_FA26` -> `spacenet2-detectron2`

The original name reads as a course folder. The new one says what the repository
contains and is what someone would search for.

Note the separation this forced. Eleven files referenced the old string, but most
were the **W&B project name**, an unrelated system where a rename would orphan
existing run URLs -- and the 2026-08-26 entry already records what that rename
cost. Only three references were repository identity. The W&B project stays
`benjbritton_FA26`.

### Licensing

`LICENSE`: MIT, scoped explicitly in the file to `src/ scripts/ configs/ docker/`
and the written record. It states what it does **not** cover, because the
interesting obligation is the one that survives it: SpaceNet 2 is **CC BY-SA
4.0**, and ShareAlike attaches to material derived from the dataset regardless of
any MIT grant on the code.

The repository had carried no attribution at all. README now records the licence,
the requested citation (Van Etten et al. 2018, arXiv:1807.01232 -- the same paper
the results are compared against), the access date, and a table of what is and is
not derivative. Nothing derivative is tracked, so the published repository
contains no dataset material. **Blog figures will**: the overlay rasters are
derived from the imagery and carry attribution and ShareAlike with them.

Also added: a third-party components table -- detectron2 Apache-2.0 at the pinned
commit, the COCO model-zoo weights, PyTorch BSD-3, pycocotools BSD-2. None
vendored, all fetched at build time, listed so the obligations are visible rather
than implicit. `wandb_writer.py` is described as mirroring detectron2's
`TensorboardXWriter` structure while being independently written against the
public `EventWriter` interface -- "modelled on" and "derived from" carry different
obligations and the distinction was checked rather than assumed.

Two questions left open rather than answered: whether trained weights are a
derivative work of training data is unsettled and is not asserted either way, and
the balloon dataset ships through an MIT-licensed repository that states no
separate terms for the images themselves. It contributes to no reported result.

### Reproducibility artefacts

`REPRODUCE.md`: every result and the literal command that produced it, in order,
each with its **expected value** so a rerun can be checked rather than merely
completed.

`docker/environment.lock.txt`: the resolved package set, plus image and base
digests. A **record, not a specification** -- the Dockerfile installs unpinned
names except `numpy<2`, so a rebuild resolves whatever is current.

One defect worth recording because it would have gone unnoticed. The first
capture used `pip freeze`, which emitted `file:///home/conda/...` build-artifact
paths for the 74 packages the conda-based base image installed. Unusable for
reinstall and misleading as a record, while looking entirely normal.
`pip list --format=freeze` gives real versions.

What does **not** reproduce is stated too: the data download was interactive and
unscripted, training is not bit-exact at fixed seed, and the 2080 Ti is gone.

### `.gitignore` before publication

Already sound -- no secrets, nothing untracked-and-unignored, nothing tracked
that should not have been. Hardened defensively for a public repository:
credentials, the remaining weight formats, editor and OS noise, and `*.tif`,
which matters most -- a stray GeoTIFF would be both large and CC BY-SA material
requiring attribution.

Verified nothing tracked was affected before and after.

### A measurement artefact worth knowing

Running `git status` against the repository through the Windows `\\wsl.localhost`
share reported seventeen modified files. From inside WSL the tree was clean. The
difference is file-mode reporting across the SMB boundary, not content. **Check
git state from inside the distro**, or a pre-publication audit reports changes
that do not exist.

### Two windows, one working tree

Both Claude sessions on this project operate on the same directory, not on
separate clones. This was not understood in the query window, which issued
`git fetch && git reset --hard origin/master` as sync instructions after the
rewrite. On a shared tree that discards whatever the other session has
uncommitted; it was harmless here only because the tree happened to be clean.
The process window declined to run it and said why.

The tell had already appeared and been misread: a `git pull` reporting "Already
up to date" while the other session's commits were plainly present. Recorded
because the failure mode is silent and the correct mental model -- two writers on
one directory, not two repositories -- changes what instructions are safe to give.
