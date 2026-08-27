"""Render footprints onto imagery to confirm the geo->pixel transform is right.

Counts cannot catch a wrong transform: a consistent offset or an axis flip
produces exactly the right number of correctly-shaped polygons in the wrong
place. One picture settles it.
"""
import json, os
import numpy as np
import rasterio
import cv2

COCO = "data/spacenet2/coco/AOI_2_Vegas_train.json"
IMGDIR = "data/spacenet2/AOI_2_Vegas/PS-RGB"
OUT = "outputs/spacenet_qa"
os.makedirs(OUT, exist_ok=True)

d = json.load(open(COCO))
by_img = {}
for a in d["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)

# Pick tiles with a healthy number of buildings, spread through the set.
cands = sorted(((len(v), k) for k, v in by_img.items()), reverse=True)
picks = [cands[0][1], cands[len(cands) // 2][1], cands[len(cands) // 4][1]]

info = {im["id"]: im for im in d["images"]}

for image_id in picks:
    im = info[image_id]
    path = os.path.join(IMGDIR, im["file_name"])
    with rasterio.open(path) as src:
        arr = src.read([1, 2, 3]).astype(np.float32)

    # Display stretch only -- NOT the training preprocessing decision.
    lo = np.percentile(arr, 2, axis=(1, 2), keepdims=True)
    hi = np.percentile(arr, 98, axis=(1, 2), keepdims=True)
    rgb = np.clip((arr - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    rgb = (rgb * 255).astype(np.uint8).transpose(1, 2, 0)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    anns = by_img[image_id]
    for a in anns:
        for seg in a["segmentation"]:
            pts = np.array(seg, dtype=np.float64).reshape(-1, 2).astype(np.int32)
            cv2.polylines(bgr, [pts], True, (0, 255, 255), 1, cv2.LINE_AA)
        x, y, w, h = a["bbox"]
        cv2.rectangle(bgr, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 255), 1)

    out = os.path.join(OUT, "overlay_%s_%d.png" % (im["file_name"].split("_img")[-1].replace(".tif", ""), len(anns)))
    cv2.imwrite(out, bgr)
    print("wrote", out, "  buildings:", len(anns), " size:", bgr.shape)
