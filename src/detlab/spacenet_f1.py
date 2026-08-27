"""The SpaceNet building-detection F1 metric, as a detectron2 DatasetEvaluator.

WHY THIS EXISTS
---------------
COCO mAP@[0.5:0.95] and the SpaceNet score are not convertible, and the FA26
Milestone B target ("within 20% of published reference") is stated against
published SpaceNet numbers. Those are F1 at IoU 0.5, not mAP. Reporting a COCO
number against an F1 reference would be comparing two different quantities and
calling the difference a result.

THE METRIC
----------
Per tile, each predicted footprint is matched to at most one ground-truth
footprint at IoU >= 0.5. Matched predictions are true positives, unmatched
predictions are false positives, unmatched ground truth are false negatives.
Counts are summed across every tile (micro-average, so dense tiles carry more
weight than sparse ones) and

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    F1        = 2 * precision * recall / (precision + recall)

THE OPERATING-POINT PROBLEM
---------------------------
This is the substantive difference from AP, and it has to be handled explicitly
rather than silently. Competitors submitted a fixed set of polygons with no
confidence scores, so the published F1 is one operating point that each team
tuned for itself. A detector emits scored instances, and its F1 depends entirely
on where the score threshold is put.

Reporting F1 at an arbitrary threshold would understate the model against
competitors who tuned theirs. Reporting only the best F1 over all thresholds
overstates it, because that threshold was chosen on the very set being scored.
Both are therefore reported, plus the threshold at which the best was found:

    f1_at_best / precision_at_best / recall_at_best / best_threshold
    f1_at_0.5                                        (fixed reference point)

For a defensible headline number, pick the threshold on train or on a held-out
slice and report F1 at that fixed threshold on val. best_f1 is a diagnostic
ceiling, not a result.

IoU IS COMPUTED ON MASKS, NOT POLYGONS
--------------------------------------
The original metric intersects georeferenced polygons. Here both sides are
rasterised to the tile grid and IoU is exact-computed by pycocotools. The reason
is that our predictions are masks: turning them into polygons first would insert
a vectorisation step whose parameters would themselves move the score, which is
a worse distortion than rasterisation. Ground truth rasterises exactly from the
same polygons the COCO file holds. The difference is small but real and belongs
in any write-up.
"""

import itertools
from collections import OrderedDict, defaultdict

import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.data import DatasetCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

IOU_THRESHOLD = 0.5


def _gt_rles(record):
    """Ground-truth instances of one tile as RLEs on the tile grid."""
    h, w = record["height"], record["width"]
    out = []
    for ann in record.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        segm = ann.get("segmentation")
        if not segm:
            continue
        # A single annotation may hold several polygons: the converter groups the
        # pieces of one footprint that clipping or make_valid split apart, so
        # they must be merged into ONE mask rather than counted as separate
        # buildings.
        rles = mask_util.frPyObjects(segm, h, w)
        out.append(mask_util.merge(rles))
    return out


def match_greedy(pred_rles, scores, gt_rles, thr=IOU_THRESHOLD):
    """Greedy score-ordered matching. Returns (score, is_tp) per prediction.

    Matching is done over ALL predictions regardless of score threshold, so that
    a threshold sweep afterwards is consistent: at threshold t, TP is the matched
    predictions scoring >= t. Re-matching per threshold could otherwise let a
    low-scoring prediction claim a ground truth that a higher-scoring one would
    have taken.
    """
    if not pred_rles:
        return []
    if not gt_rles:
        return [(s, False) for s in scores]

    # iou[d, g], iscrowd all zero -> plain intersection over union.
    ious = mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles))
    ious = np.asarray(ious).reshape(len(pred_rles), len(gt_rles))

    order = np.argsort(-np.asarray(scores))
    taken = np.zeros(len(gt_rles), dtype=bool)
    result = [None] * len(pred_rles)
    for d in order:
        row = ious[d].copy()
        row[taken] = -1.0
        g = int(np.argmax(row)) if row.size else -1
        if g >= 0 and row[g] >= thr:
            taken[g] = True
            result[d] = (scores[d], True)
        else:
            result[d] = (scores[d], False)
    return result


def sweep(records, n_gt):
    """F1 across every score threshold, from per-prediction (score, is_tp).

    Sorting by descending score and accumulating gives the same TP/FP counts the
    metric would produce if run separately at each threshold, in one pass.
    """
    if n_gt == 0:
        return {"f1_at_best": 0.0, "best_threshold": 0.0,
                "precision_at_best": 0.0, "recall_at_best": 0.0, "f1_at_0.5": 0.0}
    if not records:
        return {"f1_at_best": 0.0, "best_threshold": 1.0,
                "precision_at_best": 0.0, "recall_at_best": 0.0, "f1_at_0.5": 0.0}

    records = sorted(records, key=lambda r: -r[0])
    tp = fp = 0
    best = (-1.0, 0.0, 0.0, 0.0)          # f1, thr, prec, rec
    f1_at_half = 0.0
    for i, (score, is_tp) in enumerate(records):
        tp += 1 if is_tp else 0
        fp += 0 if is_tp else 1
        # Only evaluate at the end of a run of equal scores; a threshold cannot
        # split predictions that share a score.
        if i + 1 < len(records) and records[i + 1][0] == score:
            continue
        prec = tp / float(tp + fp)
        rec = tp / float(n_gt)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        if f1 > best[0]:
            best = (f1, score, prec, rec)
        if score >= 0.5:
            f1_at_half = f1
    return {
        "f1_at_best": best[0],
        "best_threshold": best[1],
        "precision_at_best": best[2],
        "recall_at_best": best[3],
        "f1_at_0.5": f1_at_half,
    }


class SpaceNetF1Evaluator(DatasetEvaluator):
    """Micro-averaged SpaceNet F1 at IoU 0.5, with a score-threshold sweep."""

    def __init__(self, dataset_name, iou_threshold=IOU_THRESHOLD):
        self._dataset_name = dataset_name
        self._iou = iou_threshold
        # Ground truth comes from the DatasetCatalog rather than the batched
        # inputs: the test mapper drops "annotations", by design, so the inputs
        # reaching an evaluator carry no labels.
        self._gt = {r["image_id"]: r for r in DatasetCatalog.get(dataset_name)}
        self.reset()

    def reset(self):
        self._records = []
        self._n_gt = 0
        self._n_pred = 0
        self._seen = set()

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            image_id = inp["image_id"]
            if image_id in self._seen:
                continue
            self._seen.add(image_id)

            record = self._gt.get(image_id)
            gt = _gt_rles(record) if record else []
            self._n_gt += len(gt)

            inst = out["instances"].to("cpu")
            if len(inst) == 0 or not inst.has("pred_masks"):
                continue
            masks = inst.pred_masks.numpy()
            scores = inst.scores.numpy().tolist()
            pred = [mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
                    for m in masks]
            self._n_pred += len(pred)
            self._records.extend(match_greedy(pred, scores, gt, self._iou))

    def evaluate(self):
        # Gather across processes so this behaves under distributed launch even
        # though the project currently runs single-GPU.
        recs = comm.gather(self._records, dst=0)
        n_gt = comm.gather(self._n_gt, dst=0)
        n_pred = comm.gather(self._n_pred, dst=0)
        if not comm.is_main_process():
            return {}
        recs = list(itertools.chain(*recs))
        n_gt = sum(n_gt)
        n_pred = sum(n_pred)

        res = sweep(recs, n_gt)
        res["n_gt"] = n_gt
        res["n_pred"] = n_pred
        return OrderedDict([("spacenet", res)])


# ---------------------------------------------------------------------------
# self-test: the matching and sweep logic, on geometry with known answers
# ---------------------------------------------------------------------------

def _box_rle(x0, y0, x1, y1, h=100, w=100):
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 1
    return mask_util.encode(np.asfortranarray(m))


def _selftest():
    # Two ground truths; three predictions: one exact hit, one near-miss below
    # IoU 0.5, one exact hit on the second GT.
    gt = [_box_rle(0, 0, 10, 10), _box_rle(50, 50, 60, 60)]
    preds = [_box_rle(0, 0, 10, 10),      # IoU 1.0 with gt0
             _box_rle(0, 0, 6, 10),       # IoU 0.6 with gt0, but gt0 is taken
             _box_rle(50, 50, 60, 60)]    # IoU 1.0 with gt1
    scores = [0.9, 0.8, 0.7]
    matched = match_greedy(preds, scores, gt)
    assert [m[1] for m in matched] == [True, False, True], matched

    r = sweep(matched, n_gt=2)
    # At threshold 0.7 all three are kept: TP 2, FP 1 -> P 0.667, R 1.0, F1 0.8
    # At threshold 0.9 only the first: TP 1, FP 0 -> P 1.0, R 0.5, F1 0.667
    assert abs(r["f1_at_best"] - 0.8) < 1e-6, r
    assert abs(r["best_threshold"] - 0.7) < 1e-6, r

    # A prediction that overlaps nothing is a false positive.
    assert [m[1] for m in match_greedy([_box_rle(80, 80, 90, 90)], [0.5], gt)] == [False]
    # No ground truth at all: every prediction is a false positive, F1 zero.
    assert sweep(match_greedy(preds, scores, []), n_gt=0)["f1_at_best"] == 0.0
    # Perfect detection scores 1.0.
    perfect = match_greedy(gt, [0.9, 0.9], gt)
    assert abs(sweep(perfect, n_gt=2)["f1_at_best"] - 1.0) < 1e-6

    # Grouped multi-polygon ground truth must merge into ONE instance.
    rec = {"height": 100, "width": 100, "annotations": [
        {"segmentation": [[0, 0, 10, 0, 10, 10, 0, 10],
                          [50, 50, 60, 50, 60, 60, 50, 60]], "iscrowd": 0}]}
    assert len(_gt_rles(rec)) == 1

    print("SpaceNetF1Evaluator self-test: all assertions passed")


if __name__ == "__main__":
    _selftest()
