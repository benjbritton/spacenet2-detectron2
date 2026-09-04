# A multiclass model of Chactún, Mexico: Pipeline design, external evaluation, and lessons learned

*Building a three-class detector for ancient Maya structures on Chactún, testing six mechanisms across 36 training runs, then taking it to a different LiDAR survey to find out what portability actually requires.*

---

Milestone C of this independent study asked for experience building a multiclass object detector from a provided dataset. The dataset was Chactún — 2,094 tiles of airborne laser scanning over 120.6 km² of the central Yucatán Peninsula, annotated for three classes of ancient Maya feature. The work divides cleanly in two.

The first half covers the original goals of the exercise: build the detector, then establish what improves it. The detector recovers **92% of annotated structures** at a survey-appropriate operating point. Six candidate improvements were then tested against it under five-fold cross-validation — 36 training runs, paired comparisons, a measured seed-noise floor and multiple-comparison correction. One produced a replicated gain of **+4.16 AP** across every class and every fold, and it came from the data pipeline rather than the model, at no additional compute. The four model-side candidates came in within noise, which locates the remaining headroom: not in the architecture.

That gain was then checked against the split the dataset's own challenge defines, which shares no design decision with the ones used here. It reproduced to within **0.01 AP**. Rescoring the same predictions under the challenge's own metric, however, shows the gain vanish — not because it is unreal, but because the published metric cannot see the thing the intervention improves. Both facts are reported below.

The second half includes further exploration of the dataset. The detector was taken to a different LiDAR survey — different sensor, processing chain and resolution — and evaluated there. The first two attempts produced nothing useful, and the diagnosis is the substantive result: **a detector is only as portable as the specification of its input**, and the specification is not something a model can compensate for. In this case the specification turned out to be published — in a table of the dataset paper rather than in the dataset — and applying it changed the model's behaviour sharply, though not in the direction a detection count alone would suggest. That sequence yields one requirement rather than a full specification, and it is the one the others rest on: the input rendering has to be reproducible on new ground. That is the most transferable output of the milestone.

---

## The dataset, and three properties the conversion has to handle

Chactún ([Kokalj et al., *Scientific Data* 10:558, 2023](https://doi.org/10.1038/s41597-023-02455-x), CC BY 4.0, [figshare](https://doi.org/10.6084/m9.figshare.22202395)) ships 480×480 tiles at 0.5 m, three bands — sky-view factor, positive openness, slope — with annotations as **semantic masks**, one binary raster per class per tile.

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

**Tile boundaries cut structures.** 3,429 of 9,853 instances touch an edge. Platforms therefore *over* count — 2,335 against the 1,996 present in the records, +17% — in the same conversion where buildings undercount. Aguadas diverge further still at 76 against 51, +49%, because they are the largest class and cross tile boundaries most often. Opposite errors on different classes, because platforms are large enough to cross tiles while buildings are small enough to fuse with neighbors. Edge instances are kept, since a half-visible structure is a real detection target, and every annotation carries a flag so an evaluation can exclude them without reconverting.

---

## Why a spatially blocked split is not possible here

Milestone B established spatially blocked splits as the appropriate way to hold out geographic data. Chactún does not support one, and establishing that took two independent tests. The rasters carry no CRS and no affine transform, and the layout is not recoverable from the pixels either.

**The numbering carries no layout.** Edge correlation across all 2,093 consecutive-ID pairs is 0.291, against a random-pair baseline of 0.291. A sweep of every candidate row width from 2 to 259 is flat at ~0.283, with no spike anywhere.

**No seams exist at all.** An all-pairs search over 4.38 million ordered pairs, both axes, finds zero pairs that are both reciprocal and z > 8. Best-match z-scores top out at 6.2 and reciprocal matches occur for 4–6% of tiles, which is chance.

That negative could have been manufactured by per-tile contrast stretching, which would hide a real seam — so that was ruled out separately. Only 34% of tiles are pinned to exactly 0–255 across all bands, and per-tile ranges vary with the terrain, so a shared seam would have survived. The tiles genuinely are not neighbors.

This is deliberate geomasking. Publishing precise coordinates for thousands of undocumented Maya structures is a looting risk, and withholding georeferencing is standard practice for unexcavated sites. It is a protective measure implemented by Kokalj et al., not an oversight. Because standard geographic re-stitching was impossible, evaluating spatial data leakage required ruling out edge seams entirely and forcing an alternative appearance-based partition.

The substitute partitions on **appearance**: cluster the tiles, then assign whole clusters to one side. Its effect was measured against a random control rather than assumed. Mean cross-split similarity moves from 0.743 to 0.761, p95 from 0.915 to 0.907, and the maximum from 0.978 to 0.954 — the tail tightens, the center holds.

That measurement is itself the finding. Chactún tiles resemble one another closely enough that every validation tile has a near-twin in training under any partition, so the residual similarity is a property of the dataset rather than of the partitioning strategy. **Validation scores here carry that optimism whatever the split**, which is why it is stated in the results rather than left to a footnote.

### The dataset's own authors used the split the release cannot express

The above was established from the pixels, before reading how the original team had handled it. They handled it geographically. [Somrak, Džeroski and Kokalj (*Remote Sensing* 12:2215, 2020)](https://doi.org/10.3390/rs12142215), classifying these same structures, split roughly 80/20 **by map polygon**: a sample joins the training set if it falls entirely inside the training-area boundary and the test set if it falls entirely inside the test-area boundary, with their reasoning stated plainly — buildings sit on platforms, so their image samples overlap and must not be separated across the split.

That split cannot be reconstructed from the public release. It is defined in map space, and the release has geolocation stripped and tile identifiers randomized, so no tile can be assigned to either area. **The people who built this dataset used a spatially blocked split, and the form they published it in cannot express one.**

That is a stronger statement than the two negative tests above, and it comes from the other direction. The appearance-based substitute is not a workaround for an oversight; it is the only partition the released artifact supports.

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

**Resolution is ruled out as well, and the effect runs counter to the prediction.** Arm B ruled out giving small objects *anchors*; arm F tested giving them *features*, which is a different mechanism — at stride 4, a 25 px building covers about 6 feature cells natively and about 12 at double scale. Two competing predictions were recorded in the config beforehand, +1.5 to +3.0 against +0 to +1.5. The result was **+0.10**, and small-object AP went *down* by 0.71. Both predictions agreed that a genuine resolution effect had to appear in AP75 and small-object AP rather than AP50; it appeared in neither. The effect runs opposite to the prediction, so the mechanism is ruled out on direction as well as magnitude.

That closes the scale family. Object scale can bind through the anchors that propose regions or the features that characterize them. Neither does.

**D4 data augmentation produced the one measurable gain**, and it improved every class. D4 is the set of eight symmetries of a square — horizontal and vertical flips, and rotations of 90°, 180° and 270° — so each training tile is presented in all eight orientations rather than one: +4.16 segm AP (t = 7.94), AP75 +5.89, building +4.91, platform +4.77, small objects +2.95. All five folds positive. Across roughly 42 tests in this milestone a Bonferroni threshold sits near 0.0012, and arm D's core results survive it while every marginal finding does not.

Its validity rests on a property of the bands. Sky-view factor, positive openness and slope are computed **isotropically**, so rotating them is label-preserving. Rotating a **hillshade** would not be — a fixed illumination azimuth is baked into the pixels, and the rotated image depicts terrain lit from an angle it never was. This is not incidental: the dataset paper states the point directly, noting that its primary interpretive visualization uses a directional light source and is "therefore not suitable for data augmentation techniques such as rotation and flipping," and that the three isotropic bands were chosen for the data records partly for that reason. The field's most common visualization would have blocked the only intervention that worked here. D4 rather than arbitrary rotation also preserves the cardinal alignment common in Maya architecture, and on square tiles np.rot90 is exact where an affine warp would blur 25 px buildings.

Somrak et al. reached both halves of this independently. They rotated their aguada samples by 90°, 180° and 270° — turning 51 real aguadas into 380 training samples — precisely because the class count was too low, which is the reasoning behind arm D applied to the same rare class. And they declined rotation and flip for their other classes on the grounds that hillshading would produce inconsistent relief shading, adjusting the sun azimuth when they did rotate. That is this section's argument arrived at from the opposite side: the gain here exists because these three bands are the ones where the objection does not apply.

The transforms were verified label-preserving rather than assumed: rasterize a tile's polygons, rotate that raster, and compare against the same polygons pushed through the coordinate transform. IoU 1.0000 for all four rotations. A rotation that moves pixels but not coordinates trains silently with every label detached from its object, and nothing downstream flags it.

**The trajectories refine the explanation.** Regularization would predict the baseline peaks early and decays. It does decay — but only 1.18 on building, against a 4.91 gain, so it accounts for at most a quarter. The trajectories show the arms identical through iteration 1500, after which the baseline stops improving and arm D does not. The baseline *exhausts* what 1,669 tiles can teach it; D4 keeps finding new information because each epoch presents genuinely different views. That also implies arm D was still climbing when training stopped, so +4.16 is a floor rather than the effect size.

---

## Checking the result against somebody else's split, and somebody else's metric

There is exactly one external anchor available, and establishing that took reading the two papers that work on this data.

Somrak et al. 2020 is not one, for three reasons beyond the unreconstructable split: their unit is a per-object cropped patch and there are 21,495 of them, not 480×480 tiles; their task is four-class patch classification with VGG-19 at 128×128 — aguada, building, platform, terrain — rather than detection or instance segmentation; and their metrics are overall accuracy and per-class recall, with no mAP, F1 or IoU anywhere in the paper. The data descriptor publishes no model at all.

So **no published Chactún detection benchmark with a reproducible split exists.** What remains is the challenge, and it works only because it was scored on tiles 1765–2093, which the release does contain.

Every number above rests on splits built here: five appearance-clustered folds, constructed and evaluated in this project. A result measured that way carries a specific risk — that it is a property of the partition rather than of the intervention.

The dataset has an external split available. The ECML PKDD 2021 discovery challenge built on these tiles (Kocev, Simidjievski, Kostovska, Dimitrovski and Kokalj, eds., *Discover the Mysteries of the Maya*, Jozef Stefan Institute, 2021) defines one: **train on tiles 0–1764, test on 1765–2093**. It shares no design decision with the clustered folds, and 25 published leaderboard entries were scored on it. Arms A and D were retrained on it, single seed, everything else unchanged.

| arm | five-fold CV (clustered) | canonical split | shift |
|---|---|---|---|
| A — control | 38.71 ± 2.45 | 40.46 | +1.75 |
| D — D4 augmentation | 42.87 ± 2.80 | 44.63 | +1.76 |
| **D − A** | **+4.16** | **+4.17** | — |

The effect reproduces to within 0.01 AP. Both arms score about 1.75 AP higher on the canonical split, and by the same amount — that difference is the optimism of a contiguous index split relative to appearance clustering, now measured rather than asserted. It moves the level and leaves the contrast alone.

**The published metric is a different one, and it does not see the effect at all.** The leaderboard scores semantic IoU on unioned per-class masks, not instance AP. Rescoring the same predictions that way requires resolving a question the challenge documentation leaves open: whether a class IoU is pooled over pixels or averaged over tiles, and if averaged, what an empty prediction on an empty tile scores. All three conventions were computed rather than one assumed; the per-tile convention counting empty-on-empty as agreement is the one that reproduces the leaderboard's scale. The overall column *is* determined — it is the unweighted mean of the three class IoUs, which checks out against every published row.

Score threshold was swept 0.05–0.95, since a submission is a binary mask and every team tuned that choice.

| | buildings | platforms | aguadas | overall |
|---|---|---|---|---|
| Leaderboard 1st | 0.7530 | 0.7651 | 0.9844 | 0.8341 |
| Leaderboard 8th | — | — | — | 0.8110 |
| A — control | 0.7006 | 0.7158 | 0.9740 | **0.7968** |
| D — D4 | 0.6991 | 0.7096 | 0.9726 | **0.7938** |
| predict nothing | 0.3495 | 0.4559 | 0.9574 | 0.5876 |

**D beats A by 4.17 AP and is level with it here** — 0.7938 against 0.7968, on a single seed per arm, which reads as no detectable difference rather than as D being lower.

That dissociation is the substantive point. D4 buys instance separation and ranking quality, and a unioned per-class mask cannot see either: merge two adjacent buildings into one blob and the semantic mask is unchanged while AP falls. **The headline result of this milestone is specific to the metric that measured it**, and it is reported that way rather than carried across.

It cuts in the other direction too. The leaderboard cannot distinguish a model that separates structures from one that paints the right pixels — so placement on it is not evidence of the property a survey tool actually needs, which is one detection per structure.

**The aguada column is almost entirely agreement about absence.** Aguadas appear in 13 of the 329 test tiles. Predicting no aguada anywhere scores 0.9574 under this convention; this model scores 0.9740, a gain of 0.017 over that null, and the field's best scores 0.9844, a gain of 0.027. All 25 teams sit inside a one-point band above the null. A 0.97 aguada IoU is therefore not in tension with the finding below that aguadas are band-limited — the metric is reporting that 316 of 329 tiles correctly contain nothing. The informative number for that class is instance AP: 32.0 for the control, 36.2 for arm D.

**Where this places.** 0.7968 against 0.8110 for 8th of 25. Platforms at 0.716 sit inside the published range of 0.708–0.765; buildings at 0.701 fall just under the 0.707 floor. The threshold was tuned on the test tiles here as it was for the leaderboard entries, which inflates every value in the table including these two.

### What the comparison can and cannot say

That number is worth stating carefully, because the two systems were not asked for the same thing.

**The deliverables differ.** A challenge submission is a per-tile binary mask, one raster per class — that is what the metric consumes and therefore what the entries were built to produce. This pipeline produces an instance inventory: every detection a separate record carrying a stable identifier, a class, a confidence score, a centroid and a footprint.

**One deliverable subsumes the other.** Union the inventory per class and the challenge mask falls out, which is exactly how the table above was produced. The reverse does not hold, and the cost is measurable in this dataset rather than a matter of opinion: extracting instances from a semantic mask by connected components recovers **7,442 buildings against the 8,275 present, a 10% undercount**, entirely from adjacent structures fusing into one component. That figure is measured on the ground-truth masks, so it is a property of this terrain's density, not of any model. The leading entries are semantic architectures — DeepLabV3+, UneXt50 with ResNet101, HRNet — so recovering an inventory from their output would pay that cost.

**The published metric cannot see the difference.** This is not an assertion; it is the dissociation reported above. Arm D improves instance separation and ranking by 4.17 AP and moves semantic IoU by 0.003. A metric computed on unioned masks is blind to whether two adjacent buildings were found as two objects or one blob, and telling them apart is the whole purpose of a survey inventory.

**The score gap decomposes into two measured causes.** A semantic segmentation baseline was trained to separate them: DeepLabV3 with a ResNet-50 backbone — the same backbone as the Mask R-CNN it is compared against, and the same family as the DeepLabV3+ in the leading entries — on the same split, with the same D4 augmentation, scored with the same convention, in 38.6 minutes against the instance arms' 43.9 and 43.5. Only the architecture family changes, and it uses slightly less compute rather than more.

| | buildings | platforms | aguadas | overall |
|---|---|---|---|---|
| Leaderboard 1st | 0.7530 | 0.7651 | 0.9844 | 0.8341 |
| Leaderboard 8th | — | — | — | 0.8110 |
| **semantic, DeepLabV3-R50** | **0.7167** | **0.7258** | **0.9852** | **0.8092** |
| instance, unioned (arm A) | 0.7006 | 0.7158 | 0.9740 | 0.7968 |

**Cause 1 — architecture formulation.** Swapping instance prediction for semantic prediction and holding everything else fixed moves the score from **0.7968 to 0.8092**, a gain of 0.0124 that lands 0.0018 short of the eighth entry — indistinguishable from it on a single run. Mask R-CNN predicts each mask in a small fixed region grid and upsamples it, so boundary precision is capped independently of how well the object was found; a semantic model predicts across the whole tile at stride 8 or better. The per-class pattern runs the way that mechanism predicts, with buildings — the small class — gaining most, +0.0161 against +0.0100 for platforms. **This cause accounts for essentially the whole distance between the instance pipeline and the competition's eighth place.**

**Cause 2 — ensembling, capacity and test-time augmentation.** What remains is the **0.0249** between a single semantic model at 0.8092 and first place at 0.8341. The leading entries are five-fold ensembles of DeepLabV3+, UneXt50 with ResNet101 and HRNet, several using separate models per mask, with pseudo-labeling and test-time augmentation. That is a difference in capacity and effort rather than in formulation, and it is the part a single model at a matched budget was never going to close.

The two causes act on different parts of the gap, and both are real.

**One design decision was forced rather than chosen**, and it constrains anyone building a semantic model on this data. The three classes are not mutually exclusive: **57.2% of building pixels are also platform pixels**, because buildings sit on platforms, and 45.6% of platform pixels are building. A softmax head asserts exclusivity and could not reproduce this ground truth even in principle, so the baseline uses three independent sigmoid channels — which is also the form the challenge scored, one binary raster per class per tile, and likely why several leading entries trained separate models per mask.

### The trade, stated plainly

Choosing Mask R-CNN over DeepLabV3+ on this dataset costs **1.24 IoU points** — 0.8092 against 0.7968 — and that is a deliberate exchange, not a shortfall. What it buys is structured archaeological data: every detection a separate record with an identifier, a class, a confidence score, a centroid and a footprint, ready to join to whatever else is known about that location. The semantic model scores higher on the semantic metric and produces nothing to join on.

The conversion asymmetry is what makes the exchange one-directional. Union the instance masks per class and the challenge deliverable falls out — that is exactly how the instance row above was produced. Going the other way costs the measured 10% building undercount from connected components, and the semantic arm cannot go that way at all.

**Bounds on these figures.** Each arm is a single seed, so the 1.24-point difference is one run against one run rather than a replicated measurement; its direction is consistent with the mechanism and its margin exceeds the seed spread observed on the instance arms, which is the most that can be claimed. The semantic arm's threshold of 0.50 was selected on the test tiles, as the leaderboard entries' thresholds were, which inflates every value in the table including this one — it sits on a plateau running 0.802 to 0.809 across thresholds 0.10 to 0.70, so the choice is not load-bearing. And no instance metric exists for the semantic arm, so nothing here says whether it would be better or worse at finding structures as objects.

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

The cause is a property of the visualization choice. Sky-view factor, positive openness, and slope all emphasize raised features; positive openness in particular highlights convex forms. Buildings and platforms are raised structures, so they light up.

**The dataset's own authors reached the same conclusion, and their solution names the missing band precisely.** Kokalj et al. record that a fourth visualization, *local dominance*, "served as an additional aid for human vision interpretation of outer boundaries of aguadas, which are usually very faintly raised above the surrounding flat terrain," and Table 3 of that paper lists the settings they used for it. Local dominance is not among the three bands in the data records.

That is independent corroboration from the people who built the dataset, and it sharpens the mechanism. The diagnostic feature of an aguada is not simply that it is a depression — it is a very low-relief raised rim around one, and low relief against flat ground is exactly what local dominance is constructed to bring out and what a 0°–50° slope stretch flattens to nothing. The human annotators needed a band the model was never given.

![Band value distributions: aguada overlaps the background, building does not](figures/chactun_band_separability.png)

---

So the class is not underlearned. It is unrepresented — the data records omit the visualization the annotators themselves relied on to delineate this class. That distinction matters, because more examples would not have fixed it and a different band would.

---

## Taking it outside

The obvious next question for anything built as a survey tool: hand it somebody else's LiDAR and see if it helps. The test used G-LiHT Yucatán, South GLAS tile l0s395 — a different survey, different sensor, different processing pipeline, different resolution.

Tiling was done by **ground extent** rather than pixel count, which matters more than it sounds. Measured on Chactún, naive fixed-pixel tiling accounts for about two-thirds of the resolution penalty: at 1.0 m resolution, a fixed 480×480 pixel tile covers twice the linear ground distance (480 m), causing target structures to shrink to a quarter of their native pixel area. Tiling by ground extent normalizes object scale rather than information content: cropping a fixed 240 m ground window — matching Chactún's native spatial coverage — and resizing it to 800 px reproduces the expected training geometry across survey resolutions, isolating the remaining penalty to the detail the coarser survey never recorded.

| survey resolution | arm A | arm D |
|---|---|---|
| 0.50 m (native) | 38.71 | 42.87 |
| 0.67 m | −8% | −5% |
| 1.00 m | −16% | **−10%**|
| 1.5 m | −35% | −29% |
| 2.0 m | −51% | −42% |

That leaves a usable envelope of roughly 0.33–1 m once tiling is handled. Aguada is the first casualty of coarse data — 28.1 at native, 19.3 at 1 m, and **0.77 at 2 m**, essentially gone. A weak-contrast class sets a hard resolution floor regardless of what the other classes tolerate.

Two runs were made first: the G1 composite as delivered, and a reconstruction using RVT-generated sky-view factor, positive openness and slope stacked in Chactún's band order, each band rescaled so its valid pixels carried Chactún's mean and standard deviation.

![Detections on the G-LiHT tile, densest clusters](figures/gliht_S395_clusters.png)

---

Neither produced useful detections on review. The composite "found buildings a little bit, missing many"; the platform class fired mostly on walls and straight edges; the aguada detections were not worthwhile. The reconstructed bands scored lower still — "triggering on unknown factors having nothing to do with human origin."

**The reconstruction scoring below the composite it was built to improve on is what identifies the constraint**, because it was the attempt to hand the model the representation it had been trained on.

Band **statistics** were matched rather than the stretch **function**. Sky-view factor is physically bounded 0 to 1. If Chactún mapped that range to bytes by some fixed function, the correct move is to apply that identical function. Recentering the distribution instead assigns *different byte values to the same physical quantity*, so the model received a third representation rather than the training one.

---

## The stretch function was published

The working assumption at that point was that the function could not be recovered — the deposit ships 8-bit visualizations, masks, a canopy height model and Sentinel-1/2 imagery, but no DEM, no DTM and no point cloud, so it cannot be inverted from the data.

It does not have to be. **It is in Table 3 of the dataset paper**, which specifies for general terrain: sky-view factor over a 5 m radius in 16 directions, stretched linearly over 0.7–1.0; positive openness over the same geometry, stretched linearly over 68°–93°; slope as an inverted greyscale, stretched linearly over 0.0°–50°. The caption states that these three, computed with the general-terrain settings, are the raster bands in the data records.

The reasoning about what to do had been right. The function was two paragraphs of citation away in a paper the post's opening already cites.

Table 3 also gives a second, flat-terrain column, so which one produced the released bands was checked against the released bytes rather than taken from the caption alone:

- **Per-tile normalization is excluded.** Only 56% / 27% / 29% of tiles contain byte 0 in the three bands. A per-tile min–max stretch would put every tile at both 0 and 255. A fixed global map was applied, which is the premise the rest of the check rests on.
- **Clipping accumulates where the general-terrain settings predict.** Sky-view factor has 2.85% of pixels at byte 255 against ~1.78% at neighboring values — a spike, not a tail — and slope has 2.58% at 255 in a distribution whose mode is 235. Both endpoints are physically reachable: SVF = 1.0 is unobstructed sky, slope = 0° is flat ground. Positive openness shows no pile-up at either end (0.05%), which is correct, since 93° is rarely exceeded.
- **The low end separates the two columns decisively.** Under the flat-terrain settings, byte 0 would mean SVF ≤ 0.9 and slope ≥ 15° — neither rare in karst terrain carrying 30 m hills, so both should pile up at 0. Measured: 0.059% and 0.026%. Under general-terrain settings byte 0 means SVF ≤ 0.7 and slope ≥ 50°, which genuinely are rare.
- **The implied physical values are plausible only for one column.** Inverting the general-terrain stretch gives mean SVF 0.955, openness 87.5°, slope 5.1°, all reasonable for this landscape. The flat-terrain reading implies a mean slope of 1.5°, which is not.

This is consistency evidence rather than inversion — the source DEM is not released, so the bands cannot be recomputed and compared directly. But the alternative is excluded rather than merely disfavored, and four independent signatures point the same way.

**Applying it to different terrain reproduces Chactún's statistics without matching them.** The G-LiHT DEM rendered through the Table 3 settings gives band means of 204.2 / 196.1 / 223.5, over the 4.47 km² put through the model, against Chactún's 216.5 / 198.5 / 228.6 — within half a standard deviation on all three bands, with no statistic matched at any point in the process. The implied physical medians agree too: 0.953 SVF, 87.6° openness, 4.16° slope on G-LiHT, against 0.955, 87.5°, 5.12° implied by Chactún's own bytes. A wrong recipe does not land there.

![Three encodings of the same ground at Pixoyal, each with its own detections on a shared basemap](figures/gliht_stretch_comparison.png)

*The Pixoyal group, 460 m across, north up. Left, the Table 3 recipe: 164 detections here, 471 over the full strip. Center, the delivered G1 composite: 66 and 235. Right, band statistics matched instead of the stretch function: 3 and 38.*

In the image of the Pixoyal group above, the counts and the quality run in opposite directions. All three sets are drawn on one shared basemap, so what differs between the columns is the detections rather than the rendering. The center column is the only one that reads as archaeology: discrete boxes on individual mounds and platform groups, at a density a surveyor could walk. The left column stacks overlapping boxes across the central complex and reaches large aguada and platform boxes over open terrain, and the right finds almost nothing. On review the center is the variant worth using and the other two are not — the opposite of what the detection counts say.

**And it changes the model's behaviour, sharply.** All four encodings run through the same model (arm D), the same tiling and the same 4.47 km² of ground at 0.5 m:

| encoding | detections | /km² | median size | median score | share > 0.3 |
|---|---|---|---|---|---|
| **RVT, Table 3 stretch**|**471**|**105**| 14.3 m | 0.204 | 38% |
| G1 composite as delivered | 235 | 53 | 11.9 m | 0.172 | 35% |
| G1 composite, matched | 182 | 41 | 11.1 m | 0.168 | 29% |
| RVT, mean/sd matched | 38 | 8 | 20.2 m | 0.101 | 18% |

Chactún buildings are about 12.5 m across. The specification arm responds far more strongly than any other — twice the detections of the delivered composite, higher confidence, and a median size closer to Chactún's own. The mean/sd-matched arm reproduces the earlier collapse and sits at the bottom of every column, with most of its detections pressed against the 0.05 score floor.

**A count is not a quality ranking, and here the two disagree.** Reviewed against archaeological definitions of building, platform and aguada, the delivered G1 composite produces the more plausible detections: discrete boxes on individual mounds, few obvious errors. The Table 3 arm covers 0.95 km² of ground against the composite's 0.58 km², and much of that extra reach is over terrain that carries no visible structure. It is not duplication — no two same-class detections overlap above IoU 0.5 in either arm, so non-maximum suppression is behaving — it is a model that has become much readier to call ground a structure.

That is a real effect of the encoding and it is the finding. Which encoding is *better* cannot be settled here, because the G-LiHT tile carries no annotations, and the honest summary is that the two available signals point in opposite directions: the counts favour the published recipe, and the imagery, on review, favours the delivered composite.

**What this does not establish.** That tile carries no annotations, so these are detection counts and confidence profiles, not precision and recall. Whether those 471 are right is not something this run can answer, and the visual review above suggests many are not. The comparison is between encodings under identical conditions, which is what the question required; what it establishes is that the encoding controls the model's behaviour, not which encoding produces the better survey.

---

## What portability actually requires

**A detector is only as portable as the specification of its input.** To run a Chactún-trained model on new terrain, that terrain has to be rendered into the same kind of raster the model saw — same visualizations, same physical ranges, same byte mapping. Get any of those wrong and the model is reading a representation it has never seen, which is what the first two attempts demonstrated and the third corrected.

The constraint is real but it is a *specification* problem, not an information-theoretic one. The distinction matters because the two have different remedies:

1. **What the deposit cannot supply is elevation data.** No DEM, DTM or point cloud means the released tiles cannot be re-rendered into any other visualization — negative openness, local dominance, a local relief model. That is irreversible for Chactún's own tiles, and it is why the aguada limitation above cannot be repaired from the release.
2. **What the deposit does not need to supply is the recipe**, because the paper carries it. The bands can be regenerated on *new* terrain from that specification, which is exactly what portability requires.
3. **The 8-bit quantization is permanent** but turned out not to be the binding constraint. Applying the published stretch to fresh float data and quantizing it the same way lands in the same place.

So the earlier framing — that the pipeline sits several lossy steps downstream of the point cloud and the model inherits every one — holds for what can be done with Chactún's tiles, and does not hold for what can be done with a Chactún-trained model. Those are different questions, and conflating them cost two runs.

Nothing in-domain revealed any of this. Six arms, 36 training runs, cross-validated evaluation across every class — none of it said anything about portability. That only appeared on contact with somebody else's data, and the diagnosis only completed on a second reading of the dataset paper.

---

## Future work — outside the scope of this milestone

What follows was not part of the assigned work, and nothing here was built or tested.

It is recorded as a design specification derived from the results above. Given what was measured, these are the properties a portable regional tool would need to satisfy, and the reasoning that produces each one.

The derivation runs: *the model did not transfer* → *because its input representation was not reproduced* → *therefore portability requires either a reproducible input specification, or a model indifferent to the representation it is given.*

The first route now has a demonstrated instance: publish the recipe, apply it to new terrain, and the detector's behavior improves markedly. Its cost is that every user must run that pipeline, and every dataset must document it as Table 3 does — which is a request to the field as much as a design choice.

The second route is the direct analogue of the intervention that produced the measurable gain in this milestone, and the analogy is worth spelling out.

Arm D (D4 dihedral augmentation) worked by augmenting over a property the model should not depend on. A mound is a mound whichever way the tile is turned, so showing the model every rotation taught it to ignore orientation rather than requiring every survey to be flown on the same heading.

Portability follows the exact same logic, swapping orientation for visualization style. A mound is a mound whichever rendering recipe created it.

Training on the same terrain rendered many different ways — different stretches, visualizations, and blends — would teach the model to key on the shape of the relief signature rather than on one rendering's byte patterns. That would let the detector operate across datasets produced by heterogeneous processing pipelines without retraining.

It is also now testable in a way it was not before. Chactún cannot be re-rendered, but the G-LiHT DEMs can, at three resolutions, and the existing single-class annotations are sufficient to ask whether rendering-augmentation buys robustness. Testing it would require holding out a **rendering**, not merely a set of tiles. Otherwise the experiment measures in-domain accuracy and says nothing about robustness.

It would also carry a real trade. Forced invariance across genuinely different visualizations may cause a model to learn only their intersection, buying robustness at some cost in accuracy. Worth measuring rather than assuming.

Such a tool would aim at high-throughput, out-of-domain candidate retrieval, with four properties:

- **Sensitivity and spatial transferability** across uncurated third-party LiDAR
- **Feature invariance** across heterogeneous sensor resolutions, point densities and canopy-removal artifacts
- **No site-specific tuning** required to run on a survey it has not seen
- **Output suited to downstream use** — candidate centroids and areas that populate a spatial database directly

None of these has been demonstrated here. They are the specification, and each one follows from a measurement reported above.

They are also worth stating as a specification rather than as a wish, because the target is a production tool rather than a benchmark result. The question this milestone was built around is not which architecture scores best on a curated dataset, but what a detector has to satisfy before it can be pointed at somebody else's survey and produce something an archaeologist can act on. Those are different objectives, and the second is the one that requires an input specification, an operating point chosen against the cost of a field visit, and an output that populates a database rather than a score.

---

## What carries forward

Two milestones, and the same pattern in both. In Milestone B, four candidate mechanisms for inter-city difficulty were tested and ruled out, and the operative factor turned out to be in the imagery. In Milestone C, four model-side interventions came in within noise and the measurable gain came from the data pipeline.

What the milestone establishes, as distinct from what it reports:

- **A three-class detector at 92% recall**, with its error modes separated per class into detection failure versus classification failure, and its operating characteristics measured as recall against false positives per km² rather than as AP alone.
- **A replicated augmentation effect**, +4.16 AP under cross-validation and +4.17 on the challenge's own split — and level under the challenge's own metric, which cannot see instance separation.
- **A decomposition of the distance to the published leaderboard**, into architecture formulation (0.0124, essentially the whole distance to eighth place) and ensembling with test-time augmentation (the 0.0249 beyond that to first), measured against a semantic baseline at matched compute rather than inferred.
- **A measurement ceiling for the dataset itself** — how much of COCO AP the annotation precision can actually support, which turns out to be about half the threshold range for the dominant class.
- **A resolution envelope of roughly 0.33–1 m**, decomposed into the part attributable to object scale and the part to information loss, with ground-extent tiling identified as recovering about two-thirds of the penalty.
- **A portability constraint with a diagnosed cause**, established by external evaluation rather than inferred — and a demonstrated remedy whose benefit is measured in the model's responsiveness rather than in verified accuracy, because the receiving survey carries no labels to verify against.
- **An enrichable detection inventory rather than a raster.** Every detection carries a stable identifier, class, score, centroid and footprint, exported as georeferenced vectors, so the threshold stays adjustable after the fact and each row can be joined to whatever else is known about that location. This is the deliverable the metric above cannot measure.
- **Three findings independently corroborated by the primary sources.** The empty-tile count: the challenge organizers state that 1,211 of their 1,765 tiles contain a structure, so 31.4% are empty against the 31.1% measured here — which means the data descriptor's "2,094 data records with an object in at least one of the segmentation masks" is contradicted by the organizers describing the same data, not only by this project's count. The mask polarity: stated in the challenge documentation as "0 corresponds to the target being present, 255 to not present", matching what was found by inspecting converted output. And the dominant confusion: Somrak et al. report platform misclassified as building as their most common error across most tested scenarios, which is the same direction as the platform-to-building confusions here.
- **A characterization of the tile population**: 652 of the 2,094 records carry no annotated structure. Roughly a third of the dataset is therefore negative examples, which is why the empty-tile handling described above changes what the model sees.

And the methodological findings:

- **Model-side changes did not move this problem; data-side changes did.** Anchors, cascade refinement, input resolution and rare-class oversampling: all within noise. Augmentation: +4.16.
- **A result is a result about a metric.** The same intervention that moves instance AP by 4.17 moves semantic IoU by 0.003. Neither number is wrong; they measure different properties, and only one of them is the property a survey tool needs.
- **Report at the operating point the tool will actually use.** AP made a 92%-recall detector look mediocre.
- **A class can be unrepresented rather than underlearned.** No amount of data fixes a band that does not carry the signal — and here the dataset's own authors documented reaching for the missing band by hand.
- **In-domain evaluation does not reveal portability limits.** It took contact with foreign data to find the constraint that mattered most.
- **Read the dataset paper as documentation, not as background.** The parameter that governed the whole transfer result was in a table, not in the deposit.
- **On unlabeled ground, a detection count is not a result.** The encoding that produced twice the detections produced the less plausible ones on review. Where there is nothing to score against, the domain reading is the measurement, and it has to be allowed to contradict the arithmetic.

---

## Reproducing this

Code, configurations and the full lab notebook are in the repository: **[benjbritton/spacenet2-detectron2](https://github.com/benjbritton/spacenet2-detectron2)**. All training runs, with their metrics and configurations, are logged at **[wandb.ai/benjbritton-geoai/benjbritton_FA26](https://wandb.ai/benjbritton-geoai/benjbritton_FA26)**.

The dataset is Chactún, [Kokalj et al. 2023](https://doi.org/10.1038/s41597-023-02455-x), CC BY 4.0, available from [figshare](https://doi.org/10.6084/m9.figshare.22202395). The visualization settings used throughout the portability section are Table 3 of that paper.

Within the repository: `src/detlab/datasets/masks_to_coco.py` converts the semantic masks, with all three dataset properties and the watershed sweep and its outcome documented in its docstring. `scripts/make_chactun_split.py` builds the folds and measures the leak. `scripts/train_chactun.py` runs any arm on any fold at any seed, and `scripts/run_chactun_matrix.sh` runs the full matrix.

Each negative result has its own script, so the evidence can be re-run rather than taken on trust: `chactun_layout.py` and `chactun_seams.py` for the missing geography, `chactun_headroom.py` for the measurement ceiling, `chactun_scale_sensitivity.py` for the resolution curve, and `chactun_operating_point.py` for recall against false positives per km². `scripts/chactun_semantic_iou.py` rescores instance predictions under all three semantic-IoU conventions, and `configs/chactun_canonical_split.json` is the challenge split.

`LAB_NOTEBOOK.md` carries the full record, including the predictions the data refuted and the explanations it revised.
