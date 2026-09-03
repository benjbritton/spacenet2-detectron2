# A multiclass model of Chactún, Mexico: Pipeline design, external evaluation, and lessons learned

*Building a three-class detector for ancient Maya structures on Chactún, testing six mechanisms across 36 training runs, then taking it to a different LiDAR survey to find out what portability actually requires.*

---

Milestone C of this independent study asked for experience building a multiclass object detector from a provided dataset. The dataset was Chactún — 2,094 tiles of airborne laser scanning over 120.6 km² of the central Yucatán Peninsula, annotated for three classes of ancient Maya feature. The work divides cleanly in two.

The first half is what was assigned: build the detector, then establish what improves it. The detector recovers **92% of annotated structures**at a survey-appropriate operating point. Six candidate improvements were then tested against it under five-fold cross-validation — 36 training runs, paired comparisons, a measured seed-noise floor and multiple-comparison correction. One produced a replicated gain of**+4.16 AP** across every class and every fold, and it came from the data pipeline rather than the model, at no additional compute. The four model-side candidates came in within noise, which locates the remaining headroom: not in the architecture.

The second half was not assigned. The detector was taken to a different LiDAR survey — different sensor, processing chain and resolution — and evaluated there. It did not transfer, and the diagnosis is the substantive result: the constraint is not architectural but lies in what the source dataset makes available. Establishing that required measuring resolution sensitivity, decomposing it into scale and detail, and reconstructing the input representation from a second data source. The requirements a portable tool must satisfy fall directly out of that measurement, and they are the most transferable output of the milestone.

---

## The dataset, and three properties the conversion has to handle

Chactún (Kokalj et al., *Scientific Data*10:558, 2023, CC BY 4.0) ships 480×480 tiles at 0.5 m, three bands — sky-view factor, positive openness, slope — with annotations as**semantic masks**, one binary raster per class per tile.

Converting semantic masks into instance annotations is the step that determines what the model can learn, and three properties of this dataset govern the result.

**The masks are inverted.** Object pixels are 0; background is 255. Reading them the obvious way, mask > 0, produces one tile-sized "instance" per class per tile. Training proceeds and the loss decreases while the model learns nothing, since no error is raised. The inverted polarity was identified by inspecting the converted output rather than by any failure signal.

**Semantic masks are not instance masks.** Adjacent structures that touch fuse into one connected component. Plain connected components recover 7,442 buildings against the 8,275 the dataset paper reports as present in these tiles — a 10% undercount, entirely from merging. (The often-quoted 9,303 is the count for the whole 130 km² annotated section, not for the 2,094 records, and comparing against it overstates the loss.) A distance-transform watershed was tried to separate them and does not work at any setting:

| mode | buildings | vs components | median footprint |
|---|---|---|---|
| connected components | 1,679 | — | 157 m² |
| watershed d=8 | 2,345 | ×1.40 | 114 m² |
| watershed d=12 | 1,242 | ×0.74 | 193 m² |
| watershed d=16 | 1,279 | ×0.76 | 157 m² |
| watershed d=22 | 1,586 | ×0.94 | 156 m² |
| watershed d=30 | 1,678 | ×1.00 | 157 m² |

Recovering the merges needs roughly ×1.25 *with the footprint intact*. Small radii over-split single structures and halve their size; large radii lose components to peak suppression and converge back to plain components. The undercount stands, documented rather than hidden.

**Tile boundaries cut structures.**3,429 of 9,853 instances touch an edge. Platforms therefore *over*count — 2,335 against the 1,996 present in the records, +17% — in the same conversion where buildings undercount. Aguadas diverge further still at 76 against 51, +49%, because they are the largest class and cross tile boundaries most often. Opposite errors on different classes, because platforms are large enough to cross tiles while buildings are small enough to fuse with neighbors. Edge instances are kept, since a half-visible structure is a real detection target, and every annotation carries a flag so an evaluation can exclude them without reconverting.

---

## Why a spatially blocked split is not possible here

Milestone B established spatially blocked splits as the appropriate way to hold out geographic data. Chactún does not support one, and establishing that took two independent tests. The rasters carry no CRS and no affine transform, and the layout is not recoverable from the pixels either.

**The numbering carries no layout.** Edge correlation across all 2,093 consecutive-ID pairs is 0.291, against a random-pair baseline of 0.291. A sweep of every candidate row width from 2 to 259 is flat at ~0.283, with no spike anywhere.

**No seams exist at all.** An all-pairs search over 4.38 million ordered pairs, both axes, finds zero pairs that are both reciprocal and z > 8. Best-match z-scores top out at 6.2 and reciprocal matches occur for 4–6% of tiles, which is chance.

That negative could have been manufactured by per-tile contrast stretching, which would hide a real seam — so that was ruled out separately. Only 34% of tiles are pinned to exactly 0–255 across all bands, and per-tile ranges vary with the terrain, so a shared seam would have survived. The tiles genuinely are not neighbors.

This is almost certainly deliberate geomasking. Publishing precise coordinates for thousands of undocumented Maya structures is a looting risk, and withholding georeferencing is standard practice for unexcavated sites. It is a protective measure implemented by Kokalj et al., not an oversight. Because standard geographic re-stitching was impossible, evaluating spatial data leakage required ruling out edge seams entirely and forcing an alternative appearance-based partition.

The substitute partitions on **appearance**: cluster the tiles, then assign whole clusters to one side. Its effect was measured against a random control rather than assumed. Mean cross-split similarity moves from 0.743 to 0.761, p95 from 0.915 to 0.907, and the maximum from 0.978 to 0.954 — the tail tightens, the centre holds.

That measurement is itself the finding. Chactún tiles resemble one another closely enough that every validation tile has a near-twin in training under any partition, so the residual similarity is a property of the dataset rather than of the partitioning strategy. **Validation scores here carry that optimism whatever the split**, which is why it is stated in the results rather than left to a footnote.

---

## The experiment

Five cluster-blocked folds (data partitions for cross-validation), chosen over a single held-out split for a specific reason: aguada has 76 instances in the entire dataset, so one split would evaluate about fifteen of them — too small a validation sample to support a stable estimate for that class. Across five folds every instance is evaluated exactly once.

Six arms, each differing from its neighbor by one thing:

| arm | change |
|---|---|
| A | Mask R-CNN R50-FPN, stock anchors — the control |
| B | anchor ladder shifted down one octave |
| C | Cascade Mask R-CNN head |
| D | full D4 augmentation — flips plus 90° rotations |
| E | repeat-factor oversampling of rare-class tiles |
| F | 960 px input, double the native resolution |

Comparisons are **paired by fold**, because folds differ from each other far more than arms differ from each other — arm A alone spans 36.30 to 42.61 across folds. Every arm ran the same five folds, so the difference is taken per fold and tested on those differences, which removes fold difficulty entirely.

The seed noise floor was measured rather than assumed: fold 0 was run at three seeds per arm for arms A, B and C, giving 0.80–1.39 sd on segmentation AP.

Two predictions were written into the configs *before* the runs, so they could not be reasoned backwards afterwards.

---

## Results across the six arms

| arm | segm AP | vs control | verdict |
|---|---|---|---|
| **D — D4 augmentation**|**42.87 ± 2.80**|**+4.16**| real, survives correction |
| F — 960 px input | 38.81 ± 2.77 | +0.10 | null |
| A — control | 38.71 ± 2.45 | — | — |
| C — cascade head | 38.58 ± 2.71 | −0.13 | null |
| E — repeat sampling | 38.65 ± 2.31 | −0.06 | null |
| B — shifted anchors | 37.91 ± 2.67 | −0.80 | null |

**Anchor scale is ruled out.** At the standard 800 px input resolution, nearly a third (31.8%) of annotated buildings are smaller than 32 pixels — the smallest of the reference boxes (anchors) the detector matches candidates against during training. This is a common problem with a standard solution: shrink the reference anchors down an octave to catch smaller targets. Yet doing so yielded a slight performance drop (−0.80 ± 1.07 AP), well within random training noise, with every metric turning negative. It is the fifth mechanism in this project to align neatly with a theoretical symptom only to fail under empirical testing — a pattern worth noting in its own right.

**Cascade confirmed a predicted pattern at an irrelevant magnitude.** The prediction on record was AP50 within noise, AP75 favoring cascade, AP(0.5:0.95) favoring it partly spuriously. Observed: AP50 −0.36, AP75 +2.23, segm AP +0.68. The pattern held; nothing reached significance; it costs 27% more compute per run.

**Resolution is ruled out as well, and the effect runs counter to the prediction.**Arm B ruled out giving small objects* anchors*; arm F tested giving them *features*, which is a different mechanism — at stride 4, a 25 px building covers about 6 feature cells natively and about 12 at double scale. Two competing predictions were recorded in the config beforehand, +1.5 to +3.0 against +0 to +1.5. The result was **+0.10**, and small-object AP went *down* by 0.71. Both predictions agreed that a genuine resolution effect had to appear in AP75 and small-object AP rather than AP50; it appeared in neither. The effect runs opposite to the prediction, so the mechanism is ruled out on direction as well as magnitude.

That closes the scale family. Object scale can bind through the anchors that propose regions or the features that characterize them. Neither does.

**D4 data augmentation produced the one measurable gain**, and it improved every class. D4 is the set of eight symmetries of a square — horizontal and vertical flips, and rotations of 90°, 180° and 270° — so each training tile is presented in all eight orientations rather than one: +4.16 segm AP (t = 7.94), AP75 +5.89, building +4.91, platform +4.77, small objects +2.95. All five folds positive. Across roughly 42 tests in this milestone a Bonferroni threshold sits near 0.0012, and arm D's core results survive it while every marginal finding does not.

Its validity rests on a property of the bands. Sky-view factor, positive openness and slope are computed **isotropically**, so rotating them is label-preserving. Rotating a **hillshade**would not be — a fixed illumination azimuth is baked into the pixels, and the rotated image depicts terrain lit from an angle it never was. The field's most common visualization would have blocked the only intervention that worked here. D4 rather than arbitrary rotation also preserves the cardinal alignment common in Maya architecture, and on square tiles np.rot90 is exact where an affine warp would blur 25 px buildings.

The transforms were verified label-preserving rather than assumed: rasterize a tile's polygons, rotate that raster, and compare against the same polygons pushed through the coordinate transform. IoU 1.0000 for all four rotations. A rotation that moves pixels but not coordinates trains silently with every label detached from its object, and nothing downstream flags it.

**The trajectories refine the explanation.**Regularization would predict the baseline peaks early and decays. It does decay — but only 1.18 on building, against a 4.91 gain, so it accounts for at most a quarter. The trajectories show the arms identical through iteration 1500, after which the baseline stops improving and arm D does not. The baseline* exhausts* what 1,669 tiles can teach it; D4 keeps finding new information because each epoch presents genuinely different views. That also implies arm D was still climbing when training stopped, so +4.16 is a floor rather than the effect size.

---

## What the model actually does

Every number above is COCO AP, which integrates over all score thresholds and rewards precision at high confidence. How well this reflects utility depends on the deployment scenario.

For desktop preliminary survey and candidate generation, an analyst operates at lower thresholds where missing a feature is the primary error mode and false positives are quickly discarded on-screen. For on-the-ground field verification, however, false positives carry high physical and logistical costs, making precision at higher confidence paramount.

The table below reports both ends, so an operating point can be chosen against the cost structure of the work rather than inherited from the metric.

Pooled over all 2,094 tiles (120.6 km², every structure scored exactly once):

| score | arm A recall | A FP/km² | arm D recall | D FP/km² |
|---|---|---|---|---|
| 0.05 | 84.7% | 100 | **92.0%**| 188 |
| 0.10 | 82.4% | 75 | 89.9% | 128 |
| 0.20 | 79.3% | 55 | 86.2% | 81 |
| 0.50 | 72.8% | 32 | 75.0% | 32 |
| 0.70 | 67.7% | 23 | 65.1% | 16 |

**AP understates this tool for the first of those uses.** Arm D at AP 42.87 recalls 92% of real structures at score 0.05, and its advantage is largest exactly where a desktop survey would operate: +7.3 points of recall at 0.05, +2.2 at 0.50. At a matched budget of 100 false positives per km² the like-for-like gain is about +3.0 points rather than +7.3, since arm D buys some of its recall by emitting more detections.

At the high-confidence end the two arms suit the second scenario differently. At score 0.70 arm D recalls 2.6 points less than the control but produces 16 false positives per km² against 23 — fewer wasted journeys per real structure found, which is the trade that matters when each detection has to be reached on foot.

![Arm D detections on Chactun across five tiles spanning best to worst per-tile F1](figures/chactun_detections_on_chactun.png)

---

---

## Aguadas: a class the input bands do not carry

Aguadas — Maya water reservoirs — score 26–30 pooled, far below building and platform, and nothing moved them significantly. The reason is not scarcity, though there are only 76.

Measured against unannotated terrain, mean band values inside each class:

| class | sky-view factor | positive openness | slope |
|---|---|---|---|
| building | −45.4 | −9.0 | −60.1 |
| platform | −35.5 | −9.0 | −41.0 |
| **aguada**|**−1.8**|**−0.5**|**−2.0**|

**Aguadas are, to these three bands, indistinguishable from ordinary background terrain.** Buildings differ from their surroundings by 45 and 60 counts; aguadas by two. They sit at the 40.7th percentile of their own tile's sky-view factor — dead average.

![Aguadas and buildings across all three bands with annotation outlines](figures/chactun_aguada_vs_building.png)

---

The cause is a property of the visualization choice. Sky-view factor, positive openness, and slope all emphasize raised features; positive openness in particular highlights convex forms. Buildings and platforms are raised structures, so they light up. An aguada is a depression, and the diagnostic visualization for concavity is negative openness — which the Chactún dataset does not ship, nor could it be generated, as the source DEM was withheld from the release.**

![Band value distributions: aguada overlaps the background, building does not](figures/chactun_band_separability.png)

---

So the class is not underlearned. It is unrepresented — the Chactún dataset omits any visualization band that explicitly highlights terrain concavity. That distinction matters, because more examples would not have fixed it and a different band would.

---

## Taking it outside

The obvious next question for anything built as a survey tool: hand it somebody else's LiDAR and see if it helps. The test used G-LiHT Yucatán, South GLAS tile l0s395 at 33 cm — a different survey, different sensor, different processing pipeline, different resolution.

Tiling was done by **ground extent** rather than pixel count, which matters more than it sounds. Measured on Chactún, naive fixed-pixel tiling accounts for about two-thirds of the resolution penalty: at 1.0 m resolution, a fixed 480×480 pixel tile covers twice the linear ground distance (480 m), causing target structures to shrink to a quarter of their native pixel area. Tiling by ground extent normalizes object scale rather than information content: cropping a fixed 240 m ground window — matching Chactún's native spatial coverage — and resizing it to 800 px reproduces the expected training geometry across survey resolutions, isolating the remaining penalty to the detail the coarser survey never recorded.

| survey resolution | arm A | arm D |
|---|---|---|
| 0.50 m (native) | 38.71 | 42.87 |
| 0.67 m | −8% | −5% |
| 1.00 m | −16% | **−10%**|
| 1.5 m | −35% | −29% |
| 2.0 m | −51% | −42% |

That leaves a usable envelope of roughly 0.33–1 m once tiling is handled. Aguada is the first casualty of coarse data — 28.1 at native, 19.3 at 1 m, and **0.77 at 2 m**, essentially gone. A weak-contrast class sets a hard resolution floor regardless of what the other classes tolerate.

Two runs were made on the G-LiHT tile: the G1 composite as delivered, and a reconstruction using RVT-generated sky-view factor, positive openness and slope stacked in Chactún's band order.

![Detections on the G-LiHT tile, densest clusters](figures/gliht_S395_clusters.png)

---

Neither produced useful detections on review. The composite "found buildings a little bit, missing many"; the platform class fired mostly on walls and straight edges; the aguada detections were not worthwhile. The reconstructed bands scored lower still — "triggering on unknown factors having nothing to do with human origin."

---

## What portability actually requires

The second run — sky-view factor, positive openness and slope regenerated with RVT and restacked into Chactún's channel order — scored *below* the unmodified G1 composite it was built to improve on. That inversion is what identifies the constraint, because the reconstruction was the attempt to give the model the representation it was trained on.

Band **statistics**were matched — each band rescaled so its valid pixels carried Chactún's mean and standard deviation — rather than the stretch**function**. Sky-view factor is physically bounded 0 to 1. If Chactún mapped that range to bytes by some fixed function, the correct move is to apply that identical function. Recentering the distribution instead assigns *different byte values to the same physical quantity*, so the model received a third representation rather than the training one.

And the stretch function cannot be recovered. Verified against the deposit's file list, Chactún provides ML-ready visualizations, masks, a canopy height model, and Sentinel-1/2 imagery, but no DEM, no DTM, and no point cloud data.

The pipeline therefore faces three distinct dataset limitations, none of them reversible from the released data:

- **No source elevation data.** Without the raw DEM or DTM, alternative spatial visualizations cannot be regenerated downstream.
- **Quantized 8-bit rendering.** The three visualization rasters were truncated from float precision at export time, so fine gradients within sky-view factor, positive openness and slope cannot be recovered.
- **Fixed visualization selection.** The deposit committed permanently to three specific bands, omitting hillshading, local relief models, and negative openness — the precise diagnostic channel required to resolve terrain concavity (aguadas).

In short, the deposit sits several lossy steps downstream from the original LiDAR point cloud. A model trained at the end of that processing chain inherits every upstream constraint, with no mechanism to recover the lost spatial signal.

**A detector is only as portable as the recipe for its input.** To run a Chactún-trained model on new terrain, that terrain has to be turned into the same kind of raster the model saw in training — same visualizations, same value range, same byte mapping. Chactún supplies none of what that requires: no elevation data to generate visualizations from, no float values, and no record of the function that mapped physical quantities onto 0–255. The recipe cannot be reconstructed, so the input cannot be reproduced, so the model cannot be applied. This is a property of the dataset rather than of Mask R-CNN, and it would hold for any architecture trained on it.

Nothing in-domain revealed it. Six arms, 36 training runs, cross-validated evaluation across every class — none of it said anything about portability. That only appeared on contact with somebody else's data.

---

## Future work — outside the scope of this milestone

What follows was not part of the assigned work, and nothing here was built or tested.

It is recorded as a design specification derived from the result above. Given what was measured, these are the properties a portable regional tool would need to satisfy, and the reasoning that produces each one.

The derivation runs: *the model did not transfer*→* because its input representation could not be reproduced*→* therefore portability requires either a reproducible input specification, or a model indifferent to the representation it is given.*

The first route is straightforward and constraining — publish the recipe and require every user to run it.

The second route is the direct analogue of the intervention that produced the measurable gain in this milestone, and the analogy is worth spelling out.

Arm D worked by augmenting over a property the model should not depend on. A mound is a mound whichever way the tile is turned, so showing the model every rotation taught it to ignore orientation rather than requiring every survey to be flown on the same heading.

Portability has the same shape with a different property. A mound is a mound whichever visualization recipe rendered it, so showing the model many renderings would teach it to ignore the recipe rather than requiring every user to run one specific pipeline.

Training on the same terrain rendered many different ways — different stretches, visualizations and blends — would teach a model to key on the shape of a relief signature rather than on one rendering's byte patterns.

Testing it would require holding out a **rendering**, not merely a set of tiles. Otherwise the experiment measures in-domain accuracy and says nothing about robustness.

It would also carry a real trade. Forced invariance across genuinely different visualizations may cause a model to learn only their intersection, buying robustness at some cost in accuracy. Worth measuring rather than assuming.

Such a tool would aim at high-throughput, out-of-domain candidate retrieval, with four properties:

- **Sensitivity and spatial transferability** across uncurated third-party LiDAR
- **Feature invariance** across heterogeneous sensor resolutions, point densities and canopy-removal artifacts
- **No site-specific tuning** required to run on a survey it has not seen
- **Output suited to downstream use**— candidate centroids and areas that populate a spatial database directly

None of these has been demonstrated here. They are the specification, and each one follows from a measurement reported above.

---

## What carries forward

Two milestones, and the same pattern in both. In Milestone B, four candidate mechanisms for inter-city difficulty were tested and ruled out, and the operative factor turned out to be in the imagery. In Milestone C, four model-side interventions came in within noise and the measurable gain came from the data pipeline.

What the milestone establishes, as distinct from what it reports:

- **A three-class detector at 92% recall**, with its error modes separated per class into detection failure versus classification failure, and its operating characteriztics measured as recall against false positives per km² rather than as AP alone.
- **A measurement ceiling for the dataset itself**— how much of COCO AP the annotation precision can actually support, which turns out to be about half the threshold range for the dominant class.
- **A resolution envelope of roughly 0.33–1 m**, decomposed into the part attributable to object scale and the part to information loss, with ground-extent tiling identified as recovering about two-thirds of the penalty.
- **A portability constraint with a diagnosed cause**, established by external evaluation rather than inferred.
- **A characterization of the tile population**: 652 of the 2,094 records carry no annotated structure. Roughly a third of the dataset is therefore negative examples, which is why the empty-tile handling described above changes what the model sees.

And the methodological findings:

- **Model-side changes did not move this problem; data-side changes did.** Anchors, cascade refinement, input resolution and rare-class oversampling: all within noise. Augmentation: +4.16.
- **Report at the operating point the tool will actually use.** AP made a 92%-recall detector look mediocre.
- **A class can be unrepresented rather than underlearned.** No amount of data fixes a band that does not carry the signal.
- **In-domain evaluation does not reveal portability limits.** It took contact with foreign data to find the constraint that mattered most.
- **Train as close to the source measurement as the data allows.** Every processing step between the LiDAR and the training raster is a restriction the model inherits permanently.

---

## Reproducing this

Everything is in the repository. src/detlab/datasets/masks_to_coco.py converts the semantic masks, with all three properties and the watershed sweep and its outcome documented in its docstring. scripts/make_chactun_split.py builds the folds and measures the leak. scripts/train_chactun.py runs any arm on any fold at any seed, and scripts/run_chactun_matrix.sh runs the full matrix. The evidence for the negative results has its own scripts — chactun_layout.py and chactun_seams.py for the missing geography, chactun_headroom.py for the measurement ceiling, chactun_scale_sensitivity.py for the resolution curve, chactun_operating_point.py for recall against false positives per km².

The lab notebook carries the full record, including the predictions the data refuted and the explanations it revised.
