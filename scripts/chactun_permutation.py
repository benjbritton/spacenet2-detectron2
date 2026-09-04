#!/usr/bin/env python
"""How much of the reported significance rests on the normality assumption?

The paired t-test on five folds gives arm D a p of 0.0014. A t-test on n=5
borrows most of its power from the assumption that the paired differences are
normally distributed, which five observations cannot check.

The assumption-free alternative on paired data is the exact sign-flip
permutation test: under the null the sign of each paired difference is
arbitrary, so enumerate all 2^n assignments and count how many give a mean at
least as extreme as the observed one. With n=5 there are only 32, which puts a
hard FLOOR on the p-value that no effect size can beat.

This computes that floor, and the exact permutation p for every arm.
"""
import itertools
import json
import math
import os

ARMS = [("B", "chactun_B_maskrcnn_shifted_anchors"),
        ("C", "chactun_C_cascade_shifted_anchors"),
        ("D", "chactun_D_maskrcnn_d4_augmentation"),
        ("E", "chactun_E_maskrcnn_repeat_sampler"),
        ("F", "chactun_F_maskrcnn_hires960")]
CONTROL = "chactun_A_maskrcnn_default_anchors"
FOLDS = [0, 1, 2, 3, 4]
METRIC = "segm/AP"


def final(run):
    p = os.path.join("outputs", run, "metrics.json")
    last = None
    if not os.path.exists(p):
        return None
    for line in open(p):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if METRIC in d:
            last = d[METRIC]
    return last


def perm_p(diffs):
    """Exact two-sided sign-flip permutation p."""
    n = len(diffs)
    obs = abs(sum(diffs) / n)
    hits = 0
    for signs in itertools.product((1, -1), repeat=n):
        m = sum(s * d for s, d in zip(signs, diffs)) / n
        if abs(m) >= obs - 1e-12:
            hits += 1
    return hits / float(2 ** n)


base = [final(os.path.join(CONTROL, "fold%d_seed0" % f)) for f in FOLDS]
n = len(FOLDS)
print("paired observations per comparison: n = %d" % n)
print("distinct sign assignments: 2^%d = %d" % (n, 2 ** n))
print("SMALLEST two-sided permutation p attainable: 2/%d = %.4f"
      % (2 ** n, 2.0 / 2 ** n))
print("  (one-sided: 1/%d = %.4f)" % (2 ** n, 1.0 / 2 ** n))
print()
print("%-4s %8s %10s %12s" % ("arm", "mean d", "t-test p", "permutation p"))
print("-" * 38)

T_P = {"B": 0.16900, "C": 0.82901, "D": 0.00136, "E": 0.92202, "F": 0.84868}
for key, d in ARMS:
    row = [final(os.path.join(d, "fold%d_seed0" % f)) for f in FOLDS]
    if any(v is None for v in row):
        continue
    diffs = [row[i] - base[i] for i in range(n)]
    print("%-4s %+8.2f %10.5f %12.4f"
          % (key, sum(diffs) / n, T_P[key], perm_p(diffs)))

print()
print("Reading: the t-test p for arm D is far below the smallest value the")
print("assumption-free test can return. That gap is the normality assumption")
print("doing the work, and five observations cannot verify it. The permutation")
print("result -- the most extreme of 32 possible arrangements -- is the honest")
print("distribution-free statement, and 5 folds is its ceiling by construction.")
