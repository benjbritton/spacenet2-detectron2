"""Audit v16 for re-explanation of the three recurring findings.

Distinguishes a REFERENCE (fine, conclusions should recur) from an
EXPLANATION (should appear once). A mention is flagged as explanatory when it
carries the mechanism words, not just the result.
"""
import io
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "posts/Chactun_Multiclass_M2_blog_090226v16.md"

THEMES = {
    "D4": {
        "mention": r"\bD4\b|\+4\.1[67]|arm D\b",
        "explanation": [r"eight symmetries", r"isotropic", r"label-preserving",
                        r"rot90", r"flips.*rotations", r"presented in all eight",
                        r"augmenting over a property", r"nuisance variable"],
    },
    "AP vs IoU": {
        "mention": r"semantic IoU|instance AP|unioned",
        "explanation": [r"unable to distinguish", r"cannot distinguish", r"one blob", r"blind to",
                        r"merge two adjacent", r"cannot see"],
    },
    "portability/rendering": {
        "mention": r"portab|rendering|input specification|stretch function",
        "explanation": [r"only as portable as", r"has to be rendered into",
                        r"reading a representation it has never seen",
                        r"same visualizations, same physical ranges",
                        r"reproducible on new ground",
                        r"specification problem"],
    },
}

s = io.open(path, encoding="utf-8").read()
paras = [p.strip() for p in s.split("\n\n") if p.strip()]

for theme, spec in THEMES.items():
    print("=" * 74)
    print(theme)
    print("=" * 74)
    n_expl = 0
    for i, p in enumerate(paras):
        if p.startswith("|") or p.startswith("!["):
            continue
        if not re.search(spec["mention"], p, re.I):
            continue
        hits = [e for e in spec["explanation"] if re.search(e, p, re.I)]
        kind = "EXPLAINS" if hits else "refers   "
        if hits:
            n_expl += 1
        first = p.split(". ")[0][:112]
        print("  [%s] para %-4d %s" % (kind, i, first))
        if hits:
            print("             mechanism words: %s" % ", ".join(hits))
    print("  --> explanatory paragraphs: %d" % n_expl)
    print()
