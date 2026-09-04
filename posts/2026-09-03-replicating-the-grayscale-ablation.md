# Replicating the grayscale ablation: chroma is small, real, and still not the mechanism

*Three seeds a side instead of three against one. The conclusion holds, one claim inside it retires, and the city that was supposed to be a footnote turns out to carry the only mechanism-shaped signal in the result.*

---

## Why run it again

The original grayscale ablation asked whether colour causes the difficulty ordering across SpaceNet 2's four cities. Roof-to-ground hue separation had been measured at 2.3° in Khartoum against 29.4° in Vegas, and that ordering matched detection difficulty exactly. If the relationship were causal, removing colour should collapse the gap by roughly 42.7%. It closed it by 0.4%, so hue was a correlate rather than a cause.

That result stood on **one training run against a three-seed colour baseline**. Every uncertainty quoted for it was therefore borrowed from the colour arm — the grayscale arm had no spread of its own, and a single run cannot supply one. The consequence was a specific, stated limitation: only the pooled delta was treated as established, and every per-city movement was left unresolved.

Two more grayscale runs at seeds 1 and 2, matching the colour baseline's seeds, close that gap. Nothing else changed: same data, same configuration, same fixed reporting threshold of 0.544 selected on training data.

## The predictions, registered first

Six predictions were written into the lab notebook and committed **before the runs started**, so the outcome could not be reasoned backwards into an expectation. They are reproduced in the scorecard below exactly as recorded.

This matters more than it sounds. The most useful outcome of a replication is the one that changes something, and a prediction written afterwards cannot be surprised.

## Results

Three seeds per arm, fixed threshold 0.544, IoU 0.5. Uncertainties are sample standard deviations across seeds; *p* is a two-sided Welch test, which does not assume the two arms share a variance.

| metric | colour | grayscale | delta | relative | *p* |
|---|---|---|---|---|---|
| Vegas | 0.8948 ± 0.0007 | 0.8927 ± 0.0003 | −0.0022 | −0.24% | 0.020 |
| Paris | 0.7788 ± 0.0032 | 0.7736 ± 0.0015 | −0.0051 | −0.66% | 0.089 |
| **Shanghai** | 0.6848 ± 0.0018 | 0.6743 ± 0.0009 | **−0.0106** | **−1.54%** | **0.003** |
| Khartoum | 0.6254 ± 0.0004 | 0.6213 ± 0.0033 | −0.0040 | −0.64% | 0.167 |
| macro F1 | 0.7460 ± 0.0012 | 0.7405 ± 0.0010 | −0.0055 | −0.73% | 0.004 |
| pooled F1 | 0.7942 ± 0.0010 | 0.7893 ± 0.0003 | −0.0049 | −0.61% | 0.011 |
| segm AP | 49.504 ± 0.088 | 49.125 ± 0.043 | −0.379 | −0.77% | 0.008 |

Degrees of freedom are between 2 and 4 throughout, so a large *t* does not by itself imply a small *p*. Khartoum's −0.0040 has a *t* of −2.10 and remains unresolved; Paris sits at 0.089 and is best described as marginal.

## Scorecard

**Held.**

- **The pooled delta landed in the called range**, −0.003 to −0.005, observed −0.0049.
- **The headline is unmoved.** The Vegas–Khartoum gap went from 0.2695 to 0.2713 — it *widened* by 0.7%, where a causal account of hue required it to close by about 42.7%. Three seeds do not bridge that, and the grayscale arm's own gap spread (± 0.0036) comfortably contains the movement.

**Did not hold, all in the same direction.**

- **Grayscale seed spread was called at 0.0005–0.0020 and came in at 0.00025**, four times tighter than colour's 0.0010. The stated reasoning — that an arm seeing strictly less input variation should not be noisier — pointed the right way; the floor was set too high.
- **Significance was called to fall to 2–4 sigma and instead rose**, to a Welch *t* of −7.88 on pooled F1. The cause is the previous point: an arm this reproducible shrinks the standard error of the difference, so the same delta separates from zero more cleanly, not less.
- **Shanghai was called to survive between −0.004 and −0.008** and came in at −0.0106, larger than the single-seed figure rather than regressing toward the mean.
- **Vegas, Paris and Khartoum were called to stay within ± 0.003 and stay unresolved.** Paris (−0.0051) and Khartoum (−0.0040) exceeded the band, and Vegas, though small at −0.0022, is now resolved at *p* = 0.020.

Two of six. The four that missed all lean the same way: the effect is more precisely measurable, and slightly larger, than the single-seed run could show.

## What this changes

**The conclusion stands, and stands on firmer ground.** Chroma is not the mechanism behind the city difficulty ordering. The gap did not close. That was the question the ablation existed to answer, and replication reinforces the answer rather than qualifying it.

**One claim inside it retires.** "Colour contributes almost nothing" was a fair reading of one run; it is too strong for three. Removing chroma costs a small, replicated, resolvable amount — 0.61% of pooled F1, 0.73% of macro, 0.77% of segm AP, each separated from zero. The model retains **99.4%** of its performance without colour, not 99.5%. The distinction is not about the number. It is that the residue is now measurable, where before it was inside the noise.

**Shanghai carries the only mechanism-shaped signal here.** It is the largest per-city loss by a factor of two, at 1.54% relative and *p* = 0.003 — and it is the city whose chroma the per-image stretch amplifies most, from 5.0° to 63.7° of roof-to-ground separation. A model that ignored colour entirely would have no reason to lose most where colour was most amplified. So the coherent reading is narrower and more specific than either "colour matters" or "colour is inert": **the network extracts a little from chroma, concentrated where the preprocessing pipeline amplified it, and that little is nowhere near enough to explain why Khartoum is hard.**

That also sharpens the earlier per-image versus per-city stretch question. The prior recorded then — that a model indifferent to colour should be indifferent to which colour normalisation it receives — is weakened by this result, because the arm is not indifferent, and the place it is least indifferent is exactly the place the stretch does the most work.

## Correction to the prediction entry

The prediction entry committed before these runs (`4bf4d09`) carried a baseline table citing pooled F1 0.7904 and a delta of −0.0038 as the grayscale single-seed figures. **Those are the blocked-split run's figures**, from a different experiment in the same notebook. Grayscale seed 0 at the fixed 0.544 threshold is 0.7893, and the corresponding delta against the colour baseline is −0.0049.

The predictions themselves were stated as ranges and are scored above against the correct figures. The misattributed anchor affects prediction 1's central value, which is why that prediction is scored on its spread claim rather than its point estimate. The table has been corrected in place with a note rather than silently edited.

## Limitations

- **Three seeds is still three.** Degrees of freedom between 2 and 4 mean these *p*-values are sensitive to a single unusual run. Khartoum is unresolved and Paris is marginal; neither should be reported as a finding.
- **The threshold is fixed at 0.544, selected on training data**, not tuned per arm. This is the correct choice for comparability, and it means neither arm is shown at its own optimum.
- **Grayscale removes chroma as presented to the network.** It does not test whether a different colour representation — HSV as direct input, for instance — would carry signal. The claim is about chroma as this pipeline delivers it, not about colour in principle.
- **The backbone is COCO-RGB-pretrained**, and the grayscale input replicates one channel three times to keep the stem byte-identical. So the finding is that chroma is not *necessary* for this task, not that the network never uses it.
- **Shanghai's amplification link is correlational.** One city, one stretch measurement. It is consistent with the mechanism and does not establish it; testing that would mean varying the stretch and watching whether the loss tracks it.

## Reproducing this

```bash
./scripts/run.sh python scripts/train_spacenet.py --grayscale --seed 1 \
    --output outputs/spacenet2_r50fpn_gray_seed1 \
    --run-name spacenet2-r50fpn-seed1-GRAYSCALE

./scripts/run.sh python scripts/train_spacenet.py --grayscale --seed 2 \
    --output outputs/spacenet2_r50fpn_gray_seed2 \
    --run-name spacenet2-r50fpn-seed2-GRAYSCALE
```

Scoring, which needs no GPU and reads the finished predictions:

```bash
./scripts/run.sh python scripts/score_f1.py \
    --predictions outputs/spacenet2_r50fpn_gray_seed1/inference/instances_predictions.pth \
    --dataset spacenet2_val --threshold 0.544
```

Per-city figures come from the same command with `--dataset spacenet2_val_AOI_4_Shanghai` and its three siblings. Run times were 2:35:38 and 1:54:48 on the RTX A5000; the first absorbed a period of GPU contention and is not representative.

The full record, including the predictions as registered and the entry they revise, is in `LAB_NOTEBOOK.md`.
