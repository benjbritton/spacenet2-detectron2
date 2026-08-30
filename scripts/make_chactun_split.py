#!/usr/bin/env python
"""Build a train/val split for Chactun that blocks on appearance, not position.

WHY NOT A SPATIALLY BLOCKED SPLIT
---------------------------------
There is no position to block on. The Chactun rasters carry no CRS and no affine
transform -- rasterio returns the identity matrix for every tile -- and the
layout is not recoverable from the pixels either. Two tests, both negative:

  Numbering.  If IDs ran in raster order, id and id+1 would share a vertical
              seam. Measured over all 2093 consecutive pairs the edge
              correlation is 0.291, against a random-pair baseline of 0.291.
              A sweep of every candidate row width from 2 to 259 is flat at
              ~0.283 with no spike at any width.

  All pairs.  Every ordered pair of tiles, both axes. Best-match z-scores top
              out at 6.2, and reciprocal best matches occur for 4-6% of tiles,
              which is chance. Zero pairs are both reciprocal and z > 8.

The second test could have been defeated by per-tile contrast stretching, which
would hide a real seam. It was not: only 34% of tiles are pinned to exactly
0-255 across all three bands, and per-tile ranges vary with the terrain
(sd 17.7 / 34.9 / 45.8 by band), so a shared seam would have survived. The tiles
are genuinely not neighbours -- consistent with georeferencing deliberately
withheld, which is normal for unexcavated archaeological sites.

WHAT THIS DOES INSTEAD
----------------------
A spatial block exists to stop near-duplicate content sitting on both sides of
the split. Adjacency is ruled out by the tests above, so the remaining leak is
tiles that merely LOOK alike -- same site, same terrain, same survey conditions.
That is blockable without coordinates: cluster the tiles on appearance and
assign whole clusters to one side.

This is a similarity block, not a spatial block, and it is named that way in the
output. It is weaker than a true spatial block, and it does not license any
claim about geographic generalisation.

AND IT BARELY WORKS -- measured, not assumed. Against a random control the
blocked split moves cross-split similarity from mean 0.743 to 0.761 (the WRONG
way), p95 0.915 to 0.907, max 0.978 to 0.954. Only the tail improves and only
slightly. The reading is that Chactun tiles are homogeneous enough that every
val tile has a near-twin in train under ANY partition, so the near-duplicate
leak is a property of the dataset rather than of the split. Val scores on this
data will be optimistic however it is cut, and that belongs in the results
rather than in a footnote. The split is kept because its tail is no worse than
random and its class balance is deliberate; not because blocking solved
anything.

One caveat on the metric itself: similarity is cosine distance on a PCA of
32x32 block-mean thumbnails, which is coarse and weights overall brightness
heavily. It would under-report structural near-duplicates. A learned embedding
would measure this better and has not been tried.

The rare class governs the split. Aguadas total 76 instances across 2094 tiles,
so cluster-blocking can easily starve one side; the search below optimises
explicitly for per-class balance and reports what it achieved.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import rasterio

CLASSES = ["building", "platform", "aguada"]
THUMB = 32


def load_thumbs(lidar_dir, ids, cache):
    """Downsampled appearance vector per tile."""
    if cache and os.path.isfile(cache):
        z = np.load(cache)
        if len(z["ids"]) == len(ids):
            print("loaded thumbnails from %s" % cache)
            return z["thumbs"]

    step = None
    out = np.zeros((len(ids), 3 * THUMB * THUMB), np.float32)
    for k, t in enumerate(ids):
        with rasterio.open(os.path.join(lidar_dir, "tile_%d_lidar.tif" % t)) as s:
            a = s.read().astype(np.float32)
        if step is None:
            step = a.shape[1] // THUMB
        # block mean, so the thumbnail is a real average not a subsample
        b = a[:, :step * THUMB, :step * THUMB]
        b = b.reshape(3, THUMB, step, THUMB, step).mean(axis=(2, 4))
        out[k] = b.reshape(-1)
        if (k + 1) % 500 == 0:
            print("  thumbnailed %d/%d" % (k + 1, len(ids)))
    if cache:
        np.savez_compressed(cache, ids=np.array(ids), thumbs=out)
    return out


def cluster(thumbs, k, seed):
    """PCA to 32 dims, then k-means. No sklearn dependency."""
    x = thumbs - thumbs.mean(axis=0, keepdims=True)
    # economy SVD on the 2094 x 3072 matrix is cheap
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    z = x @ vt[:32].T

    rng = np.random.default_rng(seed)
    # k-means++ seeding
    centres = [z[rng.integers(len(z))]]
    d2 = ((z - centres[0]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        p = d2 / max(d2.sum(), 1e-12)
        centres.append(z[rng.choice(len(z), p=p)])
        d2 = np.minimum(d2, ((z - centres[-1]) ** 2).sum(axis=1))
    C = np.stack(centres)

    for _ in range(60):
        lab = ((z[:, None, :] - C[None]) ** 2).sum(axis=2).argmin(axis=1)
        newC = np.stack([z[lab == j].mean(axis=0) if (lab == j).any() else C[j]
                         for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab, z


def per_tile_counts(coco_path, ids):
    d = json.load(open(coco_path))
    byid = {c["id"]: c["name"] for c in d["categories"]}
    tile_of_image = {im["id"]: im["tile"] for im in d["images"]}
    counts = {t: Counter() for t in ids}
    for a in d["annotations"]:
        counts[tile_of_image[a["image_id"]]][byid[a["category_id"]]] += 1
    return counts


def search_split(lab, counts, ids, val_frac, restarts, seed):
    """Assign whole clusters to val, optimising per-class balance."""
    k = lab.max() + 1
    cl_tiles = [[ids[i] for i in np.nonzero(lab == j)[0]] for j in range(k)]
    total = {c: sum(counts[t][c] for t in ids) for c in CLASSES}

    best, best_obj = None, np.inf
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        order = rng.permutation(k)
        val, n = [], 0
        for j in order:
            if n >= val_frac * len(ids):
                break
            val.append(j)
            n += len(cl_tiles[j])
        vset = set(val)
        vt = [t for j in vset for t in cl_tiles[j]]
        obj = max(abs(sum(counts[t][c] for t in vt) / max(total[c], 1) - val_frac)
                  for c in CLASSES)
        obj += abs(len(vt) / len(ids) - val_frac)
        if obj < best_obj:
            best_obj, best = obj, (sorted(vset), sorted(vt))
    return best, total, cl_tiles



def make_folds(lab, counts, ids, n_folds, restarts, seed):
    """Partition whole clusters into n_folds groups, balanced on every class.

    Cross-validation is used here rather than one held-out split because the
    val set is small enough that WHICH tiles are held out matters more than
    which seed is used. It also has a specific payoff for aguadas: with 76
    instances in the whole dataset a single split evaluates about 15 of them,
    which is not a measurement. Over the full set of folds every instance is
    evaluated exactly once.

    Clusters, not tiles, are the unit assigned, so appearance blocking is
    preserved inside every fold.
    """
    k = lab.max() + 1
    cl_tiles = [[ids[i] for i in np.nonzero(lab == j)[0]] for j in range(k)]
    total = {c: sum(counts[t][c] for t in ids) for c in CLASSES}
    target = 1.0 / n_folds

    order_by_size = sorted(range(k), key=lambda j: -len(cl_tiles[j]))
    best, best_obj = None, np.inf
    rng = np.random.default_rng(seed)

    for r in range(restarts):
        # largest-first placement, with the order jittered per restart so the
        # search is not stuck with one greedy answer
        order = list(order_by_size)
        if r:
            jit = rng.permutation(k)
            order.sort(key=lambda j: (-len(cl_tiles[j]), jit[j]))

        folds = [[] for _ in range(n_folds)]
        fc = [Counter() for _ in range(n_folds)]
        ft = [0] * n_folds
        def dev(cnt, ntile):
            d = abs(ntile / len(ids) - target)
            for c in CLASSES:
                d += abs(cnt[c] / max(total[c], 1) - target)
            return d

        for j in order:
            add = {c: sum(counts[t][c] for t in cl_tiles[j]) for c in CLASSES}
            size = len(cl_tiles[j])
            # cost is the DELTA to the global imbalance, so an empty fold reads
            # as a large improvement rather than a large absolute deviation
            costs = []
            for f in range(n_folds):
                before = dev(fc[f], ft[f])
                after = dev({c: fc[f][c] + add[c] for c in CLASSES}, ft[f] + size)
                costs.append(after - before)
            f = int(np.argmin(costs))
            folds[f].append(j)
            ft[f] += size
            for c in CLASSES:
                fc[f][c] += add[c]

        obj = max(max(abs(fc[f][c] / max(total[c], 1) - target) for c in CLASSES)
                  for f in range(n_folds))
        obj += max(abs(ft[f] / len(ids) - target) for f in range(n_folds))
        if obj < best_obj:
            best_obj, best = obj, (folds, fc, ft)

    folds, fc, ft = best
    fold_tiles = [sorted(t for j in folds[f] for t in cl_tiles[j])
                  for f in range(n_folds)]
    return fold_tiles, folds, total


def report_folds(fold_tiles, counts, total, ids, z):
    print()
    print("=== %d folds ===" % len(fold_tiles))
    print("  %-6s %8s %10s %10s %10s" % ("fold", "tiles", "building",
                                         "platform", "aguada"))
    for f, tiles in enumerate(fold_tiles):
        row = [sum(counts[t][c] for t in tiles) for c in CLASSES]
        print("  %-6d %8d %10d %10d %10d" % (f, len(tiles), row[0], row[1], row[2]))
    print("  %-6s %8d %10d %10d %10d"
          % ("all", sum(len(x) for x in fold_tiles),
             total["building"], total["platform"], total["aguada"]))
    print()
    print("  share of each class per fold, target %.1f%%:"
          % (100.0 / len(fold_tiles)))
    print("  %-6s %10s %10s %10s" % ("fold", "building", "platform", "aguada"))
    for f, tiles in enumerate(fold_tiles):
        print("  %-6d %9.1f%% %9.1f%% %9.1f%%"
              % (f, 100.0 * sum(counts[t]["building"] for t in tiles) / total["building"],
                 100.0 * sum(counts[t]["platform"] for t in tiles) / total["platform"],
                 100.0 * sum(counts[t]["aguada"] for t in tiles) / total["aguada"]))
    print()
    print("  leak check per fold (val = that fold, train = the rest):")
    leaks = []
    for f, tiles in enumerate(fold_tiles):
        leaks.append(cross_split_similarity(z, ids, tiles, "fold %d" % f))
    return leaks


def cross_split_similarity(z, ids, val_ids, name):
    """Worst-case leak: how similar is the closest train tile to a val tile?"""
    idx = {t: i for i, t in enumerate(ids)}
    v = np.array([idx[t] for t in val_ids], dtype=int)
    tr = np.array([idx[t] for t in ids if t not in set(val_ids)], dtype=int)
    if len(v) == 0 or len(tr) == 0:
        print("  %-18s EMPTY -- split is degenerate" % name)
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan")}
    zv, zt = z[v], z[tr]
    zv = zv / np.maximum(np.linalg.norm(zv, axis=1, keepdims=True), 1e-12)
    zt = zt / np.maximum(np.linalg.norm(zt, axis=1, keepdims=True), 1e-12)
    S = zv @ zt.T
    nearest = S.max(axis=1)
    print("  %-18s nearest-train similarity: mean %.3f  p95 %.3f  max %.3f"
          % (name, nearest.mean(), np.percentile(nearest, 95), nearest.max()))
    return {"mean": float(nearest.mean()),
            "p95": float(np.percentile(nearest, 95)),
            "max": float(nearest.max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/chactun")
    p.add_argument("--coco", default="data/chactun/coco/chactun_cc.json")
    p.add_argument("--out", default="data/chactun/splits/similarity_blocked.json")
    p.add_argument("--clusters", type=int, default=60)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--restarts", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default=None)
    p.add_argument("--folds", type=int, default=0,
                   help="emit N cross-validation folds instead of one split")
    a = p.parse_args()

    lidar = os.path.join(a.root, "lidar")
    ids = sorted(int(n.split("_")[1]) for n in os.listdir(lidar)
                 if n.endswith("_lidar.tif"))
    print("tiles: %d" % len(ids))

    thumbs = load_thumbs(lidar, ids, a.cache)
    lab, z = cluster(thumbs, a.clusters, a.seed)
    sizes = np.bincount(lab, minlength=a.clusters)
    print("clusters: %d   sizes min %d  median %d  max %d"
          % (a.clusters, sizes.min(), int(np.median(sizes)), sizes.max()))

    counts = per_tile_counts(a.coco, ids)

    if a.folds:
        fold_tiles, fold_clusters, total = make_folds(
            lab, counts, ids, a.folds, a.restarts, a.seed)
        leaks = report_folds(fold_tiles, counts, total, ids, z)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({
            "method": "similarity-blocked %d-fold cross-validation" % a.folds,
            "why": "no CRS or geotransform, layout not recoverable from pixels; "
                   "and with 76 aguadas a single split evaluates ~15 of them, "
                   "which is not a measurement. Over all folds every instance "
                   "is evaluated exactly once.",
            "limitation": "blocks near-duplicate appearance, not geography",
            "clusters": a.clusters,
            "n_folds": a.folds,
            "seed": a.seed,
            "leak_check_per_fold": leaks,
            "fold_clusters": [[int(c) for c in f] for f in fold_clusters],
            "folds": fold_tiles,
        }, open(a.out, "w"), indent=1)
        print()
        print("wrote %s" % a.out)
        return

    (val_clusters, val_ids), total, _ = search_split(
        lab, counts, ids, a.val_frac, a.restarts, a.seed)
    val_set = set(val_ids)
    train_ids = [t for t in ids if t not in val_set]

    print()
    print("=== split ===")
    print("  train %d tiles   val %d tiles (%.1f%%)   val clusters %d of %d"
          % (len(train_ids), len(val_ids),
             100.0 * len(val_ids) / len(ids), len(val_clusters), a.clusters))
    print()
    print("  %-10s %10s %10s %10s %9s" % ("class", "total", "train", "val", "val %"))
    for c in CLASSES:
        v = sum(counts[t][c] for t in val_ids)
        print("  %-10s %10d %10d %10d %8.1f%%"
              % (c, total[c], total[c] - v, v, 100.0 * v / max(total[c], 1)))

    print()
    print("=== leak check: does blocking actually reduce similarity? ===")
    leak_blocked = cross_split_similarity(z, ids, val_ids, "similarity-blocked")
    rng = np.random.default_rng(a.seed)
    rand_val = sorted(rng.choice(ids, size=len(val_ids), replace=False).tolist())
    leak_random = cross_split_similarity(z, ids, rand_val, "random (control)")
    if leak_blocked["max"] >= leak_random["max"]:
        print()
        print("  NOTE: blocking did not reduce the worst-case leak. Treat val")
        print("  scores as optimistic and say so when reporting them.")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({
        "method": "similarity-blocked, NOT spatially blocked",
        "why": "rasters carry no CRS or geotransform and the layout is not "
               "recoverable from pixels; see module docstring for the two "
               "negative tests",
        "limitation": "blocks near-duplicate appearance, not geography; licenses "
                      "no claim about geographic generalisation",
        "leak_check": {
            "metric": "cosine similarity of PCA-32 of 32x32 block-mean "
                      "thumbnails; coarse, brightness-weighted, would "
                      "under-report structural near-duplicates",
            "similarity_blocked": leak_blocked,
            "random_control": leak_random,
            "finding": "blocking barely changes cross-split similarity. Every "
                       "val tile has a near-twin in train under any partition, "
                       "so the leak is intrinsic to the dataset. Val scores are "
                       "optimistic and must be reported as such.",
        },
        "clusters": a.clusters,
        "val_clusters": [int(c) for c in val_clusters],
        "seed": a.seed,
        "train": train_ids,
        "val": val_ids,
    }, open(a.out, "w"), indent=1)
    print()
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
