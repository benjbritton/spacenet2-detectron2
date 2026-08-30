"""Chactun dataset registration and image loading for detectron2.

WHAT IS DIFFERENT FROM SPACENET
-------------------------------
Nothing needs stretching. Chactun ships 8-bit already, so there is no Stretch
equivalent here and no percentile logic -- the mapper exists only because the
files are 3-band GeoTIFFs that detectron2's PIL-based read path handles badly,
and because the band order has to be pinned.

BAND ORDER AND NORMALISATION
----------------------------
The three bands are sky-view factor, positive openness and slope, in that stored
order. They are NOT red, green and blue, and nothing about them is interchangeable
-- swapping them silently changes what the pretrained first-layer filters see.

The mapper therefore emits bands in stored order and never flips them, and
INPUT.FORMAT is set to "RGB" purely so that no other part of detectron2 decides
to reorder on its behalf. PIXEL_MEAN and PIXEL_STD below are in the same stored
order and must stay aligned with it.

Those constants are measured, not inherited: over all 2094 tiles the band means
are 216.5 / 198.5 / 228.6 against the COCO defaults of 103.5 / 116.3 / 123.7.
Leaving the defaults in place would present the network with input off-centre by
4.2 to 5.0 standard deviations, with PIXEL_STD of 1.0 applying no scaling at
all. They are computed over the full dataset rather than per fold: six scalars
are a token leak, and holding them identical across every arm matters more,
since per-fold constants would vary preprocessing between folds and confound the
comparison they exist to serve.
"""
import copy
import os

import numpy as np
import torch

CLASSES = ["building", "platform", "aguada"]

# measured over all 2094 tiles; see scripts/chactun_pixel_stats.py
PIXEL_MEAN = [216.527, 198.453, 228.612]
PIXEL_STD = [26.915, 16.698, 21.212]

BAND_NAMES = ["sky-view factor", "positive openness", "slope"]


def load_image(path):
    """(H, W, 3) uint8 in stored band order: SVF, positive openness, slope."""
    import rasterio

    with rasterio.open(path) as src:
        a = src.read()                      # (3, H, W)
    return np.ascontiguousarray(a.transpose(1, 2, 0))


def register_fold(root="data/chactun", coco_dir=None, fold=0, prefix="chactun",
                  noedge=True):
    """Register train/val for one cross-validation fold.

    Returns the registered names. The noedge val set shares the same images as
    the ordinary val set but drops annotations touching a tile edge, so the same
    predictions can be scored both ways without a second inference pass.
    """
    from detectron2.data.datasets import register_coco_instances

    coco_dir = coco_dir or os.path.join(root, "coco")
    meta = {"thing_classes": list(CLASSES)}
    registered = []

    variants = [("train", "fold%d_train.json" % fold),
                ("val", "fold%d_val.json" % fold)]
    if noedge:
        variants.append(("val_noedge", "fold%d_val_noedge.json" % fold))

    for split, fname in variants:
        json_path = os.path.join(coco_dir, fname)
        if not os.path.isfile(json_path):
            continue
        name = "%s_fold%d_%s" % (prefix, fold, split)
        register_coco_instances(name, dict(meta), json_path, root)
        registered.append(name)
    return registered


class ChactunMapper:
    """DatasetMapper equivalent that loads 3-band GeoTIFFs in stored band order.

    Written out rather than subclassing DatasetMapper for the same reason as
    SpaceNetMapper: the only thing that needs to change is the image read, and
    DatasetMapper performs that read inline in __call__ with no hook to override.
    """

    def __init__(self, cfg, is_train):
        from detectron2.data import detection_utils as utils
        from detectron2.data import transforms as T

        self._utils = utils
        self._T = T
        self.augmentations = T.AugmentationList(
            utils.build_augmentation(cfg, is_train))
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        utils, T = self._utils, self._T
        dataset_dict = copy.deepcopy(dataset_dict)

        image = load_image(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations", [])
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image.shape[:2], mask_format=self.mask_format)
        # Tiles with no structures are kept deliberately: they are the negative
        # examples, and dropping them would bias the model toward always firing.
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
