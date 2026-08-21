"""Balloon dataset: fetch, parse, register.

Balloon is NOT part of the detectron2 repo -- it lives in the Mask_RCNN
release assets and ships in VIA (VGG Image Annotator) format, not COCO.
So it needs an explicit conversion into detectron2's standard dict format.

VIA stores each instance as polygon vertex lists (all_points_x/all_points_y).
detectron2 wants a bbox in an explicit BoxMode plus a flat [x1,y1,x2,y2,...]
polygon, so the conversion is mechanical but not skippable.
"""

import json
import os
import urllib.request
import zipfile

import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip"
CLASSES = ["balloon"]


def download(data_root: str) -> str:
    """Download + unzip if absent. Returns the balloon/ directory."""
    dest = os.path.join(data_root, "balloon")
    if os.path.isdir(dest):
        return dest
    os.makedirs(data_root, exist_ok=True)
    zip_path = os.path.join(data_root, "balloon_dataset.zip")
    if not os.path.exists(zip_path):
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zip_path)
    print(f"extracting to {data_root}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(data_root)
    return dest


def get_balloon_dicts(img_dir: str):
    """VIA annotations -> detectron2 standard dicts."""
    with open(os.path.join(img_dir, "via_region_data.json")) as f:
        via = json.load(f)

    records = []
    for idx, v in enumerate(via.values()):
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        # VIA writes "regions" as a dict in some exports and a list in others.
        regions = v["regions"]
        regions = regions.values() if isinstance(regions, dict) else regions

        objs = []
        for region in regions:
            shape = region["shape_attributes"]
            px, py = shape["all_points_x"], shape["all_points_y"]
            # +0.5 centres the vertex in its pixel, matching detectron2's
            # continuous coordinate convention.
            poly = [c for pair in zip(px, py) for c in (pair[0] + 0.5, pair[1] + 0.5)]
            objs.append(
                {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
            )

        records.append(
            {
                "file_name": filename,
                "image_id": idx,
                "height": height,
                "width": width,
                "annotations": objs,
            }
        )
    return records


def register(data_root: str) -> str:
    """Register balloon_train / balloon_val. Idempotent."""
    root = download(data_root)
    for split in ("train", "val"):
        name = f"balloon_{split}"
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        split_dir = os.path.join(root, split)
        DatasetCatalog.register(name, lambda d=split_dir: get_balloon_dicts(d))
        MetadataCatalog.get(name).set(thing_classes=CLASSES)
    return root
