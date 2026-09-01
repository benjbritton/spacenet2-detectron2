#!/usr/bin/env python
"""Is arm D winning by REGULARISATION, or by something else?

The proposed explanation for D4's +4.16 AP is that the baseline overfits: 1669
training tiles is small, and the arm A pilot showed building AP peaking at
iteration 999 and then declining, which is what memorisation looks like. D4
supplies eight orientations of every tile and should suppress it.

That explanation makes a specific, checkable prediction about the SHAPE of the
training curve, not just its endpoint. If the baseline overfits and D4 fixes it:

  - arm A should peak before its final iteration and fall back
  - arm D should still be climbing, or flat, at the end

If instead both curves rise monotonically and D simply sits higher, the gain is
not regularisation and the overfitting story is wrong -- D would be learning
something better rather than forgetting less.

Read from metrics.json already on disk. No GPU, no retraining.
"""
import json
import os

import numpy as np

REPO = "/w/repos/benjbritton_FA26"
ARMS = {
    "A": "outputs/chactun_A_maskrcnn_default_anchors",
    "D": "outputs/chactun_D_maskrcnn_d4_augmentation",
}
KEY = "segm/AP"
BLD = "segm/AP-building"


def traj(arm, fold):
    p = os.path.join(REPO, ARMS[arm], "fold%d_seed0" % fold, "metrics.json")
    if not os.path.isfile(p):
        return None
    rows = [json.loads(l) for l in open(p)]
    ev = [r for r in rows if KEY in r]
    return [(r["iteration"], r[KEY], r.get(BLD, float("nan"))) for r in ev]


def main():
    print("Evaluation trajectory, segm AP by iteration, seed 0")
    print()
    for arm in ("A", "D"):
        print("--- arm %s ---" % arm)
        header = None
        peaks, finals, bpeaks, bfinals = [], [], [], []
        for f in range(5):
            t = traj(arm, f)
            if not t:
                continue
            its = [x[0] for x in t]
            aps = [x[1] for x in t]
            blds = [x[2] for x in t]
            if header is None:
                header = its
                print("  %-6s %s" % ("fold", " ".join("%7d" % i for i in its)))
            print("  %-6d %s" % (f, " ".join("%7.2f" % a for a in aps)))
            peaks.append(max(aps))
            finals.append(aps[-1])
            bpeaks.append(max(blds))
            bfinals.append(blds[-1])
        if not peaks:
            continue
        print()
        print("  mean peak  %.2f   mean final  %.2f   drop from peak  %.2f"
              % (np.mean(peaks), np.mean(finals),
                 np.mean(peaks) - np.mean(finals)))
        print("  building:  peak %.2f, final %.2f, drop %.2f"
              % (np.mean(bpeaks), np.mean(bfinals),
                 np.mean(bpeaks) - np.mean(bfinals)))
        # where does the peak sit -- early (overfit) or at the end (still learning)?
        pos = []
        for f in range(5):
            t = traj(arm, f)
            if not t:
                continue
            aps = [x[1] for x in t]
            pos.append(int(np.argmax(aps)) / (len(aps) - 1))
        print("  peak position in schedule: mean %.2f (1.00 = final eval)"
              % np.mean(pos))
        print()

    print("=== reading ===")
    print("  A peaking early and dropping, D peaking late and holding, means")
    print("  the baseline overfits and D4 regularises it.")
    print("  Both peaking at the end means D is not fixing overfitting, and the")
    print("  regularisation explanation should be dropped.")


if __name__ == "__main__":
    main()
