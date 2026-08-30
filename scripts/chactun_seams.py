#!/usr/bin/env python
"""All-pairs seam search: is ANY Chactun tile physically adjacent to any other?

The numbering carries no layout, but that only rules out seams where the
numbering guessed they would be. This tests every ordered pair. If the tiles are
disjoint cuts of one continuous survey, each interior tile has a true right-hand
neighbour, and that neighbour should stand out sharply from the other 2092
candidates -- and the match should be RECIPROCAL, i.e. if j is the best right
match for i, then i is the best left match for j.

Reciprocity is the part that matters. A best-of-2093 match is easy to produce by
chance; a best match that agrees in both directions is not.
"""
import numpy as np

CACHE = "/s/chactun_edges.npz"


def unit(strips):
    """(n, bands, len) -> (n, bands*len), zero-mean and unit-norm per row."""
    x = strips.reshape(strips.shape[0], -1).astype(np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(nrm, 1e-12)


def analyse(name, a_strips, b_strips):
    """a = trailing edge of tile i, b = leading edge of tile j."""
    A, B = unit(a_strips), unit(b_strips)
    C = A @ B.T                                  # C[i, j] = corr
    np.fill_diagonal(C, -np.inf)                 # a tile is not its own neighbour

    best_j = C.argmax(axis=1)
    best = C[np.arange(len(C)), best_j]

    # reverse direction: for each j, which i is its best predecessor?
    best_i = C.argmax(axis=0)
    reciprocal = best_i[best_j] == np.arange(len(C))

    finite = C[np.isfinite(C)]
    mu, sd = finite.mean(), finite.std()
    z = (best - mu) / sd

    print("=== %s ===" % name)
    print("  background over all %d pairs : mean %.3f  sd %.3f  max %.3f"
          % (finite.size, mu, sd, finite.max()))
    print("  best match per tile          : mean %.3f  median %.3f  max %.3f"
          % (best.mean(), np.median(best), best.max()))
    print("  best match z-score           : mean %.1f  max %.1f" % (z.mean(), z.max()))
    print("  reciprocal best matches      : %d of %d (%.1f%%)"
          % (reciprocal.sum(), len(C), 100.0 * reciprocal.mean()))
    strong = reciprocal & (z > 8)
    print("  reciprocal AND z > 8         : %d (%.1f%%)"
          % (strong.sum(), 100.0 * strong.mean()))
    if strong.any():
        idx = np.argsort(-best * strong)[:5]
        print("  strongest reciprocal pairs   : %s"
              % ", ".join("%d->%d r=%.3f z=%.1f" % (i, best_j[i], best[i], z[i])
                          for i in idx if strong[i]))
    print()
    return strong.sum()


def main():
    z = np.load(CACHE)
    ids = z["ids"]
    print("tiles: %d\n" % len(ids))

    n_h = analyse("horizontal: right edge of i vs left edge of j",
                  z["right"], z["left"])
    n_v = analyse("vertical: bottom edge of i vs top edge of j",
                  z["bottom"], z["top"])

    print("=== verdict ===")
    total = n_h + n_v
    if total < 20:
        print("  %d confident adjacencies across %d tiles." % (total, len(ids)))
        print("  There is no seam structure to recover. The tiles do not form a")
        print("  reconstructable mosaic, so a spatially blocked split cannot be")
        print("  built from the pixels either.")
    else:
        print("  %d confident adjacencies -- a partial mosaic exists and could")
        print("  seed a blocked split." % total)


if __name__ == "__main__":
    main()
