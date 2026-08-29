# Reproducing the results

Every number in `LAB_NOTEBOOK.md` and the command that produced it. Commands are
literal: run them from the repository root, in this order.

`scripts/run.sh` wraps `docker run` with the GPU, the bind mount, the host
UID/GID and the W&B credential already wired up, so every command below executes
inside the pinned image. Nothing is installed on the host.

---

## 0. Environment

| | |
|---|---|
| host | WSL2 Ubuntu 24.04.4, Docker Engine 29.7.2, NVIDIA Container Toolkit 1.20.0 |
| GPU | RTX A5000 24 GB (sm_86). Results through 2026-08-26 were produced on an RTX 2080 Ti 11 GB (sm_75) |
| image | `m2/detectron2:cu124-torch251`, `sha256:d1da631a388f5856d06bf39f5a0b46d29e0219a57ed6c0da51afb92c7367068e` |
| base | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, `sha256:14611869895df612b7b07227d5925f30ec3cd6673bad58ce3d84ed107950e014` |
| detectron2 | commit `a2f4a8771ab77e8411c26b27f24f9489a28a2453` |
| resolved packages | `docker/environment.lock.txt` (188 entries, `pip list --format=freeze` from the built image) |

```bash
docker build -t m2/detectron2:cu124-torch251 -f docker/Dockerfile.detectron2 docker/
./scripts/run.sh python scripts/verify_gpu.py
```

`verify_gpu.py` checks the thing that matters, which is not `nvidia-smi` but
whether detectron2's *compiled* CUDA kernels (`nms`, `ROIAlign`) launch on the
present architecture.

### From a bare Windows machine

The table above assumes WSL2, Docker and the container toolkit are already
present. From nothing, on Windows 11 with an NVIDIA GPU:

```powershell
wsl --install -d Ubuntu-24.04          # then set a username; reboot if asked
```

```bash
# Docker ENGINE from Docker's apt repo -- not Docker Desktop.
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER          # log out and back in

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

**Do not install an NVIDIA driver inside the distro.** The Windows driver passes
through at `/usr/lib/wsl/lib`; a Linux driver breaks that and has to be redone
after every GPU change.

**Set `networkingMode=mirrored`** in `%USERPROFILE%\.wslconfig`, then
`wsl --shutdown`:

```ini
[wsl2]
networkingMode=mirrored
```

This is not cosmetic. Without it, `docker pull` fails on hosts with IPv6:
the registry CDN resolves to both families, WSL's default NAT network has no
IPv6 route, and containerd selects the IPv6 address anyway. Disabling IPv6
inside the distro does **not** fix it -- containerd uses Go's pure resolver,
which bypasses glibc and `gai.conf` and queries AAAA itself. Mirrored mode gives
the distro the host's stack, which has a working route. It also makes a listener
inside WSL reachable from the host without `netsh portproxy`, which matters for
remote access. See the 2026-08-21 notebook entry.

Then clone the repo and build the image as above. Expect ~15 minutes for the
CUDA extension compile and about 25 GB of disk for the image.

### On a second machine with less VRAM

The image is compiled `TORCH_CUDA_ARCH_LIST="7.5;8.6+PTX"`, so it runs unmodified
on Turing (sm_75) and Ampere (sm_86) without a rebuild.

VRAM is the real constraint, and it was measured rather than estimated: batch 8
uses 3.7 GB, batch 16 peaks near 7 GB. **On an 8 GB card, use `--batch 8`.**
Batch 16 may fit but leaves no headroom for an evaluation spike.

Results at batch 8 are **not comparable** to the numbers in this repository.
Batch 16 was chosen from a measured throughput peak and matches detectron2's COCO
recipe; changing it changes the effective learning-rate schedule. A second
machine at batch 8 is for development and smoke tests, not for producing
reportable figures.

**Three notes on how far this actually reproduces.**

The `FROM` line names a mutable tag, not the digest above. Pinning the digest
would make the build reproducible against future retags, at the cost of
invalidating the layer cache once (~15 minutes). The digest is recorded here so
the identity of what was used is not lost either way.

`docker/environment.lock.txt` is a **record, not a specification**. The
Dockerfile installs unpinned package names (except `numpy<2` via
`PIP_CONSTRAINT`), so a rebuild today resolves whatever is current. The lock
file says what was actually resolved for the results in the notebook. To rebuild
exactly, install from it instead.

Training is **not** bit-exact even at a fixed seed: nondeterministic CUDA kernel
accumulation plus fp16 loss scaling. Two seed-0 smoke runs diverged by iteration
19 (total_loss 2.072 vs 2.12). Seeds buy comparable runs, not identical ones.
Bit-exactness would additionally need `cudnn.deterministic` and no autotuner, at
a throughput cost not worth paying. See the 2026-08-26 notebook entry.

---

## 1. Data

**Not scripted, and that is a gap.** The SpaceNet 2 download was performed
interactively from the requester-pays AWS bucket on 2026-08-27 and no script
records the exact invocation. What is recorded: PS-RGB imagery and
`geojson_buildings` for all four AOIs, 26 GB, 10592 tiles, about USD 2.30.
Deliberately not downloaded: `test_public/` (imagery only, labels never
released, 19 GiB) and `MS/ PAN/ PS-MS/` (8-band and panchromatic, unusable by a
3-channel model, 32 GiB).

Expected layout:

```
data/spacenet2/AOI_2_Vegas/PS-RGB/*.tif
data/spacenet2/AOI_2_Vegas/geojson_buildings/*.geojson
data/spacenet2/AOI_3_Paris/...  AOI_4_Shanghai/...  AOI_5_Khartoum/...
```

Licensing and attribution: see README.

### 1a. GeoJSON to COCO, per AOI

```bash
for AOI in AOI_2_Vegas AOI_3_Paris AOI_4_Shanghai AOI_5_Khartoum; do
  ./scripts/run.sh python src/detlab/datasets/geojson_to_coco.py \
      --images data/spacenet2/$AOI/PS-RGB \
      --labels data/spacenet2/$AOI/geojson_buildings \
      --out    data/spacenet2/coco/${AOI}_train.json
done
```

Expected: 10592 images, 218681 instances, 2069 empty tiles kept, retention
99.48-99.83%.

### 1b. Splits and stretch constants

```bash
./scripts/run.sh python scripts/make_split.py                    # 8474 / 2118
./scripts/run.sh python scripts/make_blocked_split.py --block-tiles 10
./scripts/run.sh python scripts/spacenet_stats.py                # per-city percentiles
```

`--split-seed` defaults to 20260827 in both split scripts. **The split seed is
not the training seed.** `configs/spacenet2_split.json` is the authority on
membership; regenerating it while varying `--seed` would sum split variance and
seed variance with no way to separate them.

`make_blocked_split.py` defaults to `--block-tiles 5`; the notebook run used
**10**, chosen as the measured knee (adjacency 0.277, 29 blocks in the smallest
AOI, val fraction 20.7%).

### 1c. Transform QA

```bash
./scripts/run.sh python scripts/overlay_qa.py     # -> outputs/spacenet_qa/
```

Counts cannot catch a wrong geo-to-pixel transform: a consistent offset produces
exactly the right number of correctly-shaped polygons in the wrong place.

---

## 2. Training

### Baseline, three seeds (2026-08-27)

```bash
./scripts/run.sh python scripts/train_spacenet.py --seed 0 \
    --output outputs/spacenet2_r50fpn       --run-name spacenet2-seed0
./scripts/run.sh python scripts/train_spacenet.py --seed 1 \
    --output outputs/spacenet2_r50fpn_seed1 --run-name spacenet2-seed1
./scripts/run.sh python scripts/train_spacenet.py --seed 2 \
    --output outputs/spacenet2_r50fpn_seed2 --run-name spacenet2-seed2
```

About 1:52 each. Everything else comes from
`configs/spacenet2_mask_rcnn_R50_FPN.yaml`: batch 16, LR 0.02, 6000 iterations,
`STEPS: (4000, 5500)`, fp16, `per_image` stretch, eval every 1000.

Expected: segm AP 49.504 +/- 0.088, segm APs 26.587 +/- 0.135, pooled F1
0.7945 +/- 0.0009. **Those standard deviations are what makes every later
comparison readable** -- without them a delta of 0.003 cannot be called real.

`--seed` is wired to `seed_all_rng()` explicitly, not to `cfg.SEED`. This script
never calls `default_setup()`, which is the only place detectron2 reads
`cfg.SEED`, so a flag wired only to the config value would have appeared to work
and changed nothing.

### Blocked split (2026-08-28)

```bash
./scripts/run.sh python scripts/train_spacenet.py --seed 0 --split blocked \
    --output outputs/spacenet2_r50fpn_blocked --run-name spacenet2-blocked
```

Expected: segm AP 49.179, pooled F1 0.7911, macro 0.7583.

**`--split blocked` is the only thing that selects the blocked data**, and
omitting it fails silently rather than loudly. `train_spacenet.py:196` uses it to
point `coco_dir` at `data/spacenet2/coco_blocked` and to register under the
`spacenet2b` prefix; without the flag the run trains the ordinary baseline and
reports about 49.50, with nothing to indicate the wrong split was used.
`--data-root` selects nothing — it names the parent directory in both cases.

### Balloon (Milestone A)

```bash
./scripts/run.sh python scripts/train_balloon.py --smoke     # 50 iters
./scripts/run.sh python scripts/train_balloon.py --seed 0
```

Expected: segm AP ~81.5. `segm/APs` on balloon is computed over **three**
validation instances and has CV 59.8% across seeds -- do not read it.

### Grayscale ablation (2026-08-28)

The run that refuted the hue finding, so it belongs here rather than only in the
notebook.

```bash
./scripts/run.sh python scripts/train_spacenet.py --grayscale --seed 0 \
    --output outputs/spacenet2_r50fpn_gray \
    --run-name spacenet2-r50fpn-seed0-GRAYSCALE
```

Expected: segm AP 49.11, pooled F1 0.7895, per-AOI 0.893 / 0.777 / 0.678 / 0.626,
Vegas-Khartoum gap 0.267 against 0.268 for colour.

`--grayscale` collapses chroma *after* the stretch and replicates the single
channel three times, so input shape and the COCO-pretrained stem stay identical
and colour is the only variable.

One seed against a three-seed colour baseline: only the pooled delta (-0.004,
about 4 sigma) is established. Small per-city deltas are not.

---

## 3. Scoring

Everything here reads `inference/instances_predictions.pth` from a finished run,
so nothing below needs a second inference pass or a GPU.

```bash
# Pooled F1 at the reported operating point
./scripts/run.sh python scripts/score_f1.py \
    --predictions outputs/spacenet2_r50fpn/inference/instances_predictions.pth \
    --dataset spacenet2_val --threshold 0.544

# Per-AOI table + threshold selection on train, cross-checked on a val half
./scripts/run.sh python scripts/f1_report.py --run-dir outputs/spacenet2_r50fpn
```

**The threshold is selected on other data, deliberately.** Reporting the best F1
over a sweep of the set being scored is a tuned hyperparameter presented as a
result. Selected 0.544 on train, 0.539 on a val half; on held-out val half B
they give 0.7939 and 0.7941. Full rigour cost 0.0015 of F1.

Expected per-AOI **at the fixed threshold 0.544**, 3-seed mean:

| AOI | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|
| Vegas | 0.8941 | 0.8950 | 0.8954 | 0.8948 | 0.0007 |
| Paris | 0.7762 | 0.7823 | 0.7778 | 0.7787 | 0.0032 |
| Shanghai | 0.6828 | 0.6859 | 0.6858 | 0.6848 | 0.0018 |
| Khartoum | 0.6257 | 0.6250 | 0.6254 | 0.6254 | 0.0004 |
| **macro** | 0.7447 | 0.7470 | 0.7461 | **0.7459** | 0.0012 |

Against XD_XD's 0.693. **Macro, not pooled micro** -- the SpaceNet paper defines
Total Score as the arithmetic mean of per-city F1, and Vegas alone is 51% of val
instances.

**Do not confuse these with the per-city best-threshold figures** (Vegas 0.8947,
Paris 0.7773, Shanghai 0.6862, Khartoum 0.6267, macro 0.7462), which also appear
in the notebook. Those tune a threshold per city on the set being scored, which
is the practice the paragraph above argues against; they are a diagnostic, not a
reportable result. The difference is 0.0003 on the macro -- full rigour is
nearly free here, which is the point.

---

## 4. Analysis

```bash
# Roof-vs-ground separability, boundary contrast, shadow proxy, size, crowding
./scripts/run.sh python scripts/city_separability.py --tiles-per-city 250

# Per-city F1 within each COCO size bucket
./scripts/run.sh python scripts/f1_by_size.py

# Recovery across IoU thresholds: "not found" vs "found but outlined loosely"
./scripts/run.sh python scripts/iou_sweep.py --run-dir outputs/spacenet2_r50fpn

# Hue separation per city, and tile-level factor attribution
./scripts/run.sh python scripts/city_hue.py --tiles-per-city 250
./scripts/run.sh python scripts/factor_attribution.py --run-dir outputs/spacenet2_r50fpn
```

Sampling in `city_separability.py` and `city_hue.py` is a fixed stride through
the tile list, not a random draw, so results are reproducible without another
seed. All statistics are computed on the uint8 image the network sees
(`per_image` 2-98 stretch), not on raw DN.

---

## 5. Review artefacts

```bash
# Predictions and ground truth as EPSG:4326 vectors, score carried as attribute
./scripts/run.sh python scripts/export_predictions_geojson.py --gt

# Overlays burned onto tiles, written back with the source CRS and transform
./scripts/run.sh python scripts/overlay_geotiff.py --per-city 4 --png
```

Vectors rather than fixed pictures: a burned-in overlay commits to a score
threshold at render time, whereas a definition query on `score` in ArcGIS/QGIS
keeps it adjustable. `--min-score` defaults to 0.05 so the slider has range below
the 0.544 reporting threshold.

Windows access: the WSL filesystem has no drive letter -- it is an `ext4.vhdx`
exposed as `\\wsl.localhost\Ubuntu-24.04` while the distro runs. Map it with
`net use Z: \\wsl.localhost\Ubuntu-24.04 /persistent:yes`.

---

## What is not reproducible from this repo

Stated so a reader does not discover it the hard way.

- **The data download.** Interactive, unscripted, requester-pays. See section 1.
- **Bit-exact training.** By design; see section 0.
- **The 2080 Ti results.** That card is no longer in the machine. The 2026-08-26
  entry compares the two GPUs on the same config; the A5000 numbers are the
  current ones.
- **W&B run URLs.** Under the `benjbritton-geoai` entity, private.
- **Outputs.** `data/` and `outputs/` are gitignored. Every artefact is
  regenerable from the commands above; the numbers themselves are transcribed
  into `LAB_NOTEBOOK.md` so the findings survive without them.
