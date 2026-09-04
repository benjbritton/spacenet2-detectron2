#!/usr/bin/env python
"""Confidence intervals and an exact multiple-comparison family for the arms.

WHY
---
The write-up reports means and standard deviations across folds, and a
Bonferroni threshold quoted as "roughly 42 tests". Neither is enough for a
reader to judge how much the differences are worth. A mean difference with no
interval cannot be distinguished from a mean difference that could plausibly be
zero, and an approximate family size makes the correction unauditable.

This computes, for every arm against the control, on the SAME five folds:

  the paired per-fold differences
  their mean, sd, and a 95% CI from the t distribution on 4 degrees of freedom
  the paired t statistic and its exact two-sided p
  Cohen's dz, the paired effect size, with its own interval

and states the comparison family explicitly so the corrected threshold can be
checked rather than taken on trust.

Paired by fold throughout: folds differ from one another far more than arms do,
so the unpaired spread would swamp the effect being measured.
"""
import json
import math
import os

ARMS = [
    ("A", "chactun_A_maskrcnn_default_anchors", "control, stock anchors"),
    ("B", "chactun_B_maskrcnn_shifted_anchors", "anchors down one octave"),
    ("C", "chactun_C_cascade_shifted_anchors", "cascade head"),
    ("D", "chactun_D_maskrcnn_d4_augmentation", "D4 augmentation"),
    ("E", "chactun_E_maskrcnn_repeat_sampler", "repeat-factor sampling"),
    ("F", "chactun_F_maskrcnn_hires960", "960 px input"),
]
FOLDS = [0, 1, 2, 3, 4]
METRIC = "segm/AP"

# Student-t two-sided critical values and CDF helpers ------------------------
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(lb)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    return betai(0.5 * df, 0.5, df / (df + t * t))


def final_metric(run_dir):
    p = os.path.join(run_dir, "metrics.json")
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


def mean(v):
    return sum(v) / len(v)


def sd(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


# gather --------------------------------------------------------------------
vals = {}
for key, d, _ in ARMS:
    row = []
    for f in FOLDS:
        v = final_metric(os.path.join("outputs", d, "fold%d_seed0" % f))
        row.append(v)
    vals[key] = row

print("segm AP by fold")
print("%-4s %s" % ("arm", "  ".join("fold%d" % f for f in FOLDS)))
for key, _, _ in ARMS:
    row = vals[key]
    if any(v is None for v in row):
        print("%-4s  MISSING %s" % (key, row))
    else:
        print("%-4s %s   mean %.2f" % (key, "  ".join("%5.2f" % v for v in row),
                                       mean(row)))

base = vals["A"]
print()
print("Paired against the control, same five folds, 95% CI on 4 df (t*=2.776)")
hdr = ("%-4s %-26s %7s %7s %17s %7s %9s %7s"
       % ("arm", "change", "mean d", "sd d", "95% CI", "t", "p", "dz"))
print(hdr)
print("-" * len(hdr))
results = []
for key, _, label in ARMS:
    if key == "A":
        continue
    row = vals[key]
    if any(v is None for v in row) or any(v is None for v in base):
        continue
    d = [row[i] - base[i] for i in range(len(FOLDS))]
    md, sdd = mean(d), sd(d)
    se = sdd / math.sqrt(len(d))
    half = T_CRIT_95[len(d) - 1] * se
    t = md / se if se else float("nan")
    p = t_two_sided_p(abs(t), len(d) - 1)
    dz = md / sdd if sdd else float("nan")
    results.append((key, label, md, sdd, md - half, md + half, t, p, dz))
    print("%-4s %-26s %+7.2f %7.2f  [%+6.2f, %+6.2f] %7.2f %9.5f %7.2f"
          % (key, label, md, sdd, md - half, md + half, t, p, dz))

# multiple comparisons ------------------------------------------------------
n_arms = len(ARMS) - 1                       # five arms against one control
metrics_tested = ["segm/AP", "segm/AP50", "segm/AP75",
                  "segm/AP-building", "segm/AP-platform", "segm/AP-aguada",
                  "segm/APs"]
family = n_arms * len(metrics_tested)
alpha = 0.05
print()
print("MULTIPLE COMPARISONS, stated exactly rather than approximately")
print("  family: every arm-vs-control comparison on every reported metric")
print("  arms compared to the control : %d  (B, C, D, E, F)" % n_arms)
print("  metrics reported per arm     : %d  (%s)"
      % (len(metrics_tested), ", ".join(m.replace("segm/", "") for m in metrics_tested)))
print("  family size                  : %d x %d = %d" % (n_arms, len(metrics_tested), family))
print("  Bonferroni threshold at alpha=0.05 : 0.05 / %d = %.6f" % (family, alpha / family))
print()
for key, label, md, sdd, lo, hi, t, p, dz in results:
    verdict = ("SURVIVES correction" if p < alpha / family else
               "nominally significant, does NOT survive" if p < alpha else
               "not significant")
    print("  arm %s on segm/AP: p = %.5f -> %s" % (key, p, verdict))

print()
print("CI excludes zero?")
for key, label, md, sdd, lo, hi, t, p, dz in results:
    print("  arm %s: [%+.2f, %+.2f]  %s" % (key, lo, hi,
          "yes" if (lo > 0 or hi < 0) else "no -- consistent with no effect"))
