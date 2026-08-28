# spacenet2-detectron2

Multi-city building detection on SpaceNet 2, with detectron2. FA26 Independent
Study, University of Cincinnati, Geography & GIS.

Mask R-CNN R50-FPN, COCO-pretrained, trained across all four SpaceNet 2 AOIs
(Las Vegas, Paris, Shanghai, Khartoum) and scored with the SpaceNet F1 metric
against the published competition results.

| | macro F1 | Vegas | Paris | Shanghai | Khartoum |
|---|---|---|---|---|---|
| **this work** (3 seeds) | **0.7462** | 0.8952 | 0.7791 | 0.6877 | 0.6272 |
| XD_XD, 2017 winner | 0.6930 | 0.885 | 0.745 | 0.597 | 0.544 |
| YOLT baseline | 0.6000 | | | | |
| modified MNC baseline | 0.5700 | | | | |

**This is not a claim of beating the 2017 winner.** Those scores are on the
competition's withheld test set; these are on a validation split carved from the
training data, with a random tile split that is spatially autocorrelated (worth
about 0.4% on the pooled metric, measured) and IoU computed on rasterised masks
rather than georeferenced polygons. What the comparison does establish is that
the pipeline lands in the neighbourhood of published results and reproduces the
per-city difficulty ordering exactly.

Also reported: segm AP 49.504 +/- 0.088 across three seeds. On the same IoU
requirement as F1, AP50 is 81.21 against F1 0.7930 -- the two headline numbers
are the same result under different overlap requirements, reconciled in the
notebook.

## Read this first

| | |
|---|---|
| [`LAB_NOTEBOOK.md`](LAB_NOTEBOOK.md) | The actual record. What was done, what it cost, what turned out to be wrong. Written as work happened |
| [`REPRODUCE.md`](REPRODUCE.md) | Every result and the literal command that produced it, with expected values |

The notebook is the substance of this project. Four claims made in it were later
refuted by further measurement, and the refutations are kept in place rather
than edited out -- including one that had to refute its own correction.

## Requirements

- WSL2 Ubuntu 24.04, Docker Engine, NVIDIA Container Toolkit
- Image `m2/detectron2:cu124-torch251`, built from `docker/Dockerfile.detectron2`
- ~26 GB for SpaceNet 2 (PS-RGB plus building footprints, all four AOIs)
- An NVIDIA GPU. Developed on an RTX 2080 Ti (11 GB), current results on an
  RTX A5000 (24 GB). A full run is about 1:52
- A W&B API key, optional -- `--offline` or `--no-wandb` work without one

Keep this repo, `data/` and `outputs/` on the WSL ext4 filesystem, **not** under
`/mnt/c`. Cross-filesystem I/O to the Windows drive is slow for the many-small-file
access pattern training uses.

## Quick start

```bash
docker build -t m2/detectron2:cu124-torch251 -f docker/Dockerfile.detectron2 docker/
./scripts/run.sh python scripts/verify_gpu.py
./scripts/run.sh python scripts/train_spacenet.py --seed 0
```

`scripts/run.sh` wraps `docker run` with the GPU, bind mount, host UID/GID and
W&B credential wired up, so nothing is installed on the host. Full sequence,
including data preparation, in [`REPRODUCE.md`](REPRODUCE.md).

## Layout

| Path | Purpose |
|---|---|
| `src/detlab/datasets/spacenet.py` | SN2 registration and the 16-bit to 8-bit load path |
| `src/detlab/datasets/geojson_to_coco.py` | Footprint GeoJSON to COCO, with the geometry fixes real data forced |
| `src/detlab/spacenet_f1.py` | SpaceNet F1 at IoU 0.5, greedy matching, score-threshold sweep |
| `src/detlab/trainer.py` | DefaultTrainer subclass: COCO eval + SpaceNet F1 + W&B + best-checkpointing |
| `src/detlab/wandb_writer.py` | W&B EventWriter. detectron2 ships no W&B support |
| `scripts/train_spacenet.py` | The training entry point |
| `scripts/score_f1.py`, `f1_report.py` | Score a finished run without re-running inference |
| `scripts/city_separability.py`, `city_hue.py`, `factor_attribution.py`, `f1_by_size.py`, `iou_sweep.py` | The per-city difficulty analysis |
| `scripts/export_predictions_geojson.py`, `overlay_geotiff.py` | Predictions as GIS-ready vectors and georeferenced overlays |
| `configs/` | Overrides layered on a model-zoo base config, plus the split files |
| `docker/` | Image definition, and `environment.lock.txt` -- the resolved package set the results were produced with |

`data/`, `outputs/`, `wandb/` and checkpoints are gitignored. Every artefact is
regenerable from `REPRODUCE.md`; the numbers are transcribed into the notebook.

## Notes

- **The split file, not the seed, is the authority.** `configs/spacenet2_split.json`
  fixes train/val membership. `cfg.SEED` varies between runs to measure variance;
  the split must not, or seed variance and split variance become inseparable.
- **`--seed` is wired to `seed_all_rng()`, not `cfg.SEED`.** detectron2 reads
  `cfg.SEED` only inside `default_setup()`, which this script never calls, so a
  flag wired to the config value would have looked correct and done nothing.
- **No 8-bit files are written.** SN2 tiles are 11-bit data in a 16-bit
  container, occupying about 3% of the nominal range; a naive divide-by-256
  yields a tile whose maximum value is 6 of 255. The stretch happens in the
  mapper at load time, so the georeferenced UInt16 GeoTIFFs stay the only copy.
- **`FILTER_EMPTY_ANNOTATIONS: False` is required.** 2069 of 10592 tiles contain
  no buildings and detectron2 drops such images by default -- 20% of the dataset,
  and 45% of Paris, would vanish silently.
- **fp16, not bf16.** bf16 and TF32 are Ampere-only; fp16 runs on both cards the
  project has used, so configs survived the GPU swap unchanged.
- **`numpy<2` is pinned in the image** via `PIP_CONSTRAINT`. detectron2 predates
  numpy 2 and the breakage surfaces in COCO evaluation, not at import.

## Data attribution and licensing

**SpaceNet 2 (Building Detection v2)** -- imagery and building footprint labels.

> The SpaceNet Dataset by SpaceNet Partners is licensed under a
> [Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

Cite as:

> Van Etten, A., Lindenbaum, D., & Bacastow, T.M. (2018). SpaceNet: A Remote
> Sensing Dataset and Challenge Series. *arXiv:1807.01232*.

Accessed from the [SpaceNet AWS Open Data registry][aws] on 2026-08-27
(requester-pays bucket; PS-RGB and `geojson_buildings` for all four AOIs).

**What ShareAlike means here.** CC BY-SA obligations attach to material *derived
from the dataset*, not to independently written code. In this repo:

| | derivative? |
|---|---|
| `data/spacenet2/coco/*.json` (converted footprints) | yes -- a reformatting of the labels |
| exported prediction vectors, overlay rasters, figures | yes -- derived from the imagery |
| `configs/spacenet2_split*.json` (filename lists) | membership only, no dataset content |
| `scripts/`, `src/`, `docker/` | no -- independent code |

None of the derivative material is tracked in git (`data/` and `outputs/` are
ignored), so the repository as published contains no SpaceNet-derived content.
**Anything derived that does get published -- overlay figures in a blog post, a
released set of predicted footprints -- carries the attribution above and the
ShareAlike term with it.**

Whether trained model weights are a derivative work of the training data is
unsettled and not asserted either way here.

## Third-party components

None are vendored into this repository; all are fetched at build or run time.
Listed so the obligations are visible rather than implicit.

| component | licence | how it is used |
|---|---|---|
| [detectron2](https://github.com/facebookresearch/detectron2) (Meta), commit `a2f4a877` | Apache-2.0 | cloned into the image; `src/detlab/` extends its documented APIs (`EventWriter`, `DefaultTrainer`, `DatasetEvaluator`) |
| detectron2 model zoo, `mask_rcnn_R_50_FPN_3x` COCO weights | Apache-2.0 | initialisation for every run |
| [PyTorch](https://github.com/pytorch/pytorch) 2.5.1 and the `pytorch/pytorch` CUDA base image | BSD-3-Clause | base image |
| [pycocotools](https://github.com/ppwwyyxx/cocoapi) | BSD-2-Clause | COCO evaluation and RLE mask handling |
| rasterio, shapely, pyproj, OpenCV, numpy | BSD / MIT / Apache-2.0 | geospatial and array stack; see `docker/environment.lock.txt` |
| [balloon dataset](https://github.com/matterport/Mask_RCNN/releases) | see note | Milestone A only, not used for any SpaceNet result |

`src/detlab/wandb_writer.py` deliberately mirrors the structure of detectron2's
`TensorboardXWriter` (`events.py:141`) so the two read alike, but it is
independently written against the public `EventWriter` interface rather than
copied.

Note on the balloon dataset: it is distributed through the releases of
matterport/Mask_RCNN, an MIT-licensed repository, but that project does not state
separate terms for the images themselves. It was used for the Milestone A
plumbing test and contributes to no result reported here.

## Licence

Code and written record: [MIT](LICENSE), (c) 2026 Benjamin Britton.

The MIT grant does not extend to the SpaceNet 2 dataset or anything derived from
it, which remains CC BY-SA 4.0 -- see above.

[cc-by-sa]: https://creativecommons.org/licenses/by-sa/4.0/
[aws]: https://registry.opendata.aws/spacenet/
