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
from fvcore.transforms.transform import NoOpTransform, Transform

from detectron2.data.transforms import Augmentation

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



# ---------------------------------------------------------------------------
# D4 augmentation: the eight symmetries of the square.
#
# WHY IT IS LEGITIMATE HERE, AND WHEN IT WOULD NOT BE
# ---------------------------------------------------
# Rotating a HILLSHADE is wrong: a fixed illumination azimuth is baked into the
# pixel values, so the rotated image depicts terrain lit from an angle it never
# was. Sky-view factor, positive openness and slope are all computed
# isotropically -- over the full hemisphere, or as a gradient magnitude -- which
# is exactly why archaeological prospection prefers them to hillshade, and which
# is what makes rotation label-preserving here.
#
# NOTE the dependency: the band descriptions are absent from the rasters
# themselves (all None, no tags), so that identification comes from the dataset
# publication rather than from the files. If one band were in fact directional,
# these transforms would be invalid and a null result from the arm would be
# ambiguous between a bad prior and an invalid transform.
#
# D4 rather than arbitrary rotation is deliberate. Maya architecture is
# frequently aligned to the cardinal directions, and flips plus 90-degree
# multiples map cardinal directions onto cardinal directions, preserving that
# real prior. Arbitrary-angle rotation would destroy it, and would resample.
#
# Tiles are square, so np.rot90 is EXACT. detectron2's RandomRotation goes
# through an affine warp with bilinear interpolation, which would blur 25 px
# buildings to no purpose.


class Rot90Transform(Transform):
    """Rotate by k * 90 degrees counter-clockwise, without resampling."""

    def __init__(self, k, h, w):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        return np.ascontiguousarray(np.rot90(img, self.k))

    def apply_coords(self, coords):
        # np.rot90 CCW once maps input (x, y) to (y, W - 1 - x). Applying that
        # map k times keeps image and annotations in agreement; getting this
        # wrong detaches every polygon from its object silently, which is why
        # scripts/verify_d4.py checks it against a rasterised mask.
        coords = np.asarray(coords, dtype=float).reshape(-1, 2)
        w = self.w
        for _ in range(self.k % 4):
            x, y = coords[:, 0].copy(), coords[:, 1].copy()
            coords[:, 0] = y
            coords[:, 1] = w - 1 - x
        return coords

    def apply_segmentation(self, segmentation):
        return self.apply_image(segmentation)

    def inverse(self):
        return Rot90Transform((4 - self.k) % 4, self.w, self.h)


class RandomRot90(Augmentation):
    """Uniformly pick one of the four 90-degree rotations."""

    def get_transform(self, image):
        k = int(np.random.randint(4))
        h, w = image.shape[:2]
        if k == 0:
            return NoOpTransform()
        return Rot90Transform(k, h, w)


def build_augmentations(cfg, is_train, d4=False):
    """detectron2's stock list, optionally extended to the full D4 group.

    Stock training augmentation for these configs is ResizeShortestEdge plus a
    horizontal flip. Adding a vertical flip and a uniform 90-degree rotation
    generates all eight symmetries of the square: the two flips at p=0.5 and
    four rotations give sixteen combinations covering each of the eight group
    elements exactly twice, so the result is uniform over D4.
    """
    from detectron2.data import detection_utils as utils
    from detectron2.data import transforms as T

    augs = utils.build_augmentation(cfg, is_train)
    if is_train and d4:
        augs.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
        augs.append(RandomRot90())
    return augs

class ChactunMapper:
    """DatasetMapper equivalent that loads 3-band GeoTIFFs in stored band order.

    Written out rather than subclassing DatasetMapper for the same reason as
    SpaceNetMapper: the only thing that needs to change is the image read, and
    DatasetMapper performs that read inline in __call__ with no hook to override.
    """

    def __init__(self, cfg, is_train, d4=False):
        from detectron2.data import detection_utils as utils
        from detectron2.data import transforms as T

        self._utils = utils
        self._T = T
        self.augmentations = T.AugmentationList(
            build_augmentations(cfg, is_train, d4))
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
