#!/usr/bin/env python
"""Semantic segmentation baseline on the Chactun canonical challenge split.

WHY THIS EXISTS
---------------
The instance pipeline scores 0.7968 semantic IoU against 0.8110 for the eighth
of twenty-five leaderboard entries. Two explanations are available -- the
architecture family, and capacity/effort -- and the instance runs cannot
separate them. This holds the backbone, the data, the split, the augmentation
and roughly the compute budget fixed, and changes only the architecture family.
Whatever it scores, it produces no instances, which is the asymmetry the
comparison is about.

THREE SIGMOID CHANNELS, NOT A SOFTMAX
-------------------------------------
Measured by scripts/chactun_semantic_stats.py rather than assumed: 57.2% of
building pixels are ALSO platform pixels, because buildings sit on platforms. A
softmax asserts the classes are mutually exclusive and could not reproduce this
ground truth even in principle. Independent binary channels are also the form
the challenge scored -- one binary raster per class per tile.

MASK POLARITY is 0 = object, 255 = background, as the organisers state and as
the instance converter already handles. Reading it the obvious way trains a
model to predict the background.

    ./scripts/run.sh python scripts/train_chactun_semantic.py \
        --out outputs/chactun_S_deeplabv3_r50
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASSES = ["building", "platform", "aguada"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class ChactunSeg(Dataset):
    def __init__(self, tiles, root, train):
        self.tiles, self.root, self.train = tiles, root, train

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        t = self.tiles[i]
        img = np.array(Image.open(os.path.join(
            self.root, "lidar", "tile_%d_lidar.tif" % t)))
        if img.ndim == 2:
            img = np.stack([img] * 3, -1)
        img = img[:, :, :3].astype(np.float32) / 255.0
        m = np.stack([
            (np.array(Image.open(os.path.join(
                self.root, "masks", "tile_%d_mask_%s.tif" % (t, c)))) == 0)
            for c in CLASSES], 0).astype(np.float32)

        if self.train:
            # D4: the same dihedral group as arm D, valid for the same reason --
            # these three bands are isotropic, so a rotation is label-preserving.
            k = np.random.randint(4)
            if k:
                img = np.rot90(img, k, (0, 1))
                m = np.rot90(m, k, (1, 2))
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
                m = m[:, :, ::-1]
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(m)))


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    dims = (0, 2, 3)
    num = 2.0 * (p * target).sum(dims) + eps
    den = p.sum(dims) + target.sum(dims) + eps
    return (1.0 - num / den).mean()


def evaluate(model, tiles, root, device, thresholds):
    """Semantic IoU under the three conventions, matching chactun_semantic_iou.py."""
    model.eval()
    pool_i = {th: {c: 0 for c in CLASSES} for th in thresholds}
    pool_u = {th: {c: 0 for c in CLASSES} for th in thresholds}
    tile_iou = {th: {c: [] for c in CLASSES} for th in thresholds}
    tile_nz = {th: {c: [] for c in CLASSES} for th in thresholds}

    ds = ChactunSeg(tiles, root, train=False)
    dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4)
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(x)["out"]
            prob = torch.sigmoid(logit.float()).cpu().numpy()
            gt = y.numpy().astype(bool)
            for th in thresholds:
                pred = prob >= th
                for b in range(pred.shape[0]):
                    for ci, c in enumerate(CLASSES):
                        g, p = gt[b, ci], pred[b, ci]
                        inter = int(np.logical_and(g, p).sum())
                        union = int(np.logical_or(g, p).sum())
                        pool_i[th][c] += inter
                        pool_u[th][c] += union
                        if union == 0:
                            tile_iou[th][c].append(1.0)
                        else:
                            v = inter / union
                            tile_iou[th][c].append(v)
                            tile_nz[th][c].append(v)

    out = {}
    for th in thresholds:
        row = {}
        for label, get in (
                ("pooled", lambda c: (pool_i[th][c] / pool_u[th][c])
                    if pool_u[th][c] else float("nan")),
                ("per_tile_emptyhit", lambda c: float(np.mean(tile_iou[th][c]))),
                ("per_tile_nonempty", lambda c: float(np.mean(tile_nz[th][c]))
                    if tile_nz[th][c] else float("nan"))):
            v = {c: get(c) for c in CLASSES}
            v["overall"] = float(np.mean([v[c] for c in CLASSES]))
            row[label] = v
        out["%.2f" % th] = row
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/chactun")
    p.add_argument("--split", default="data/chactun/splits/canonical_challenge.json")
    p.add_argument("--out", default="outputs/chactun_S_deeplabv3_r50")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    device = "cuda"

    split = json.load(open(a.split))
    train_t, val_t = sorted(split["folds"][0]), sorted(split["folds"][1])
    print("train %d tiles, val %d tiles" % (len(train_t), len(val_t)))

    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(weights="DEFAULT", aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, 3, 1)
    model.aux_classifier[4] = nn.Conv2d(256, 3, 1)
    model = model.to(device)

    dl = DataLoader(ChactunSeg(train_t, a.root, True), batch_size=a.batch,
                    shuffle=True, num_workers=6, drop_last=True,
                    pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    bce = nn.BCEWithLogitsLoss()

    print("steps/epoch %d, total %d" % (len(dl), total))
    t0, step = time.time(), 0
    for ep in range(a.epochs):
        run = 0.0
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                o = model(x)
                loss = (bce(o["out"], y) + dice_loss(o["out"], y)
                        + 0.4 * (bce(o["aux"], y) + dice_loss(o["aux"], y)))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss)
            step += 1
        el = time.time() - t0
        print("epoch %2d/%d  loss %.4f  elapsed %.1f min  eta %.1f min"
              % (ep + 1, a.epochs, run / len(dl), el / 60,
                 (el / (ep + 1) * (a.epochs - ep - 1)) / 60), flush=True)

    torch.save(model.state_dict(), os.path.join(a.out, "model_final.pth"))
    print("training done in %.1f min" % ((time.time() - t0) / 60))

    ths = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
    res = evaluate(model, val_t, a.root, device, ths)
    with open(os.path.join(a.out, "semantic_iou_sweep.json"), "w") as f:
        json.dump(res, f, indent=1)

    hdr = "%-6s %-20s %9s %9s %9s %9s" % ("thr", "convention", "buildings",
                                          "platforms", "aguadas", "overall")
    print()
    print(hdr)
    print("-" * len(hdr))
    for th in ths:
        v = res["%.2f" % th]["per_tile_emptyhit"]
        print("%-6.2f %-20s %9.4f %9.4f %9.4f %9.4f"
              % (th, "per_tile_emptyhit", v["building"], v["platform"],
                 v["aguada"], v["overall"]))
    best = max(ths, key=lambda t: res["%.2f" % t]["per_tile_emptyhit"]["overall"])
    b = res["%.2f" % best]["per_tile_emptyhit"]
    print()
    print("BEST thr %.2f  overall %.4f  (arm A 0.7968, arm D 0.7938, 8th 0.8110)"
          % (best, b["overall"]))
    print("  buildings %.4f  platforms %.4f  aguadas %.4f"
          % (b["building"], b["platform"], b["aguada"]))


if __name__ == "__main__":
    main()
