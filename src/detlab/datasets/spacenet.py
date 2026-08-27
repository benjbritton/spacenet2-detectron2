"""SpaceNet 2 dataset registration and the 16-bit -> 8-bit load path.

THE PROBLEM THIS SOLVES
-----------------------
SN2 PS-RGB tiles are 650x650 UInt16 GeoTIFFs. detectron2 and the COCO-pretrained
weights expect 8-bit RGB. Something must convert, and the choice is consequential
rather than cosmetic: measured over every tile, the real data occupies roughly
1-2% of the nominal 16-bit range (98th percentiles land near 400-1250 of 65535).
A naive divide-by-256 therefore maps every building into the bottom ~2% of the
8-bit range and produces a near-black image that still trains, still converges,
and quietly underperforms.

NO 8-BIT FILES ARE WRITTEN. The stretch happens here, at load time. The
georeferenced UInt16 GeoTIFFs stay the only copy on disk, so nothing can drift
out of sync with a derivative, the 26 GB is not duplicated, and the stretch
parameters live in config where they are versioned with everything else.

TWO MODES, AND WHY BOTH EXIST
-----------------------------
per_image   Percentile computed from the tile being loaded. This is what
            SpaceNet's own baseline write-ups describe, and it is the closest
            available reference: Solaris, the implementation the FA26 plan names,
            cannot be installed (tensorflow==1.13.1, pyyaml==5.2, and a git://
            dependency GitHub disabled in March 2022). Solaris' own imread
            rescales per-image too, though with min/max rather than percentiles;
            percentiles are the same idea made robust to a single hot pixel.

per_city    Constants precomputed per AOI by scripts/spacenet_stats.py. Two tiles
            from the same city then receive the same stretch, so absolute
            brightness keeps carrying information across the dataset. per_image
            maximizes contrast tile by tile and destroys exactly that.

Neither is obviously correct, which is why the mode is a config value and not a
decision baked into a file on disk. per_image is the baseline-comparable run;
per_city is the experiment.

EMPTY TILES
-----------
2069 of 10592 tiles contain no buildings and they are KEPT in the COCO files on
purpose -- they are legitimate negatives, and Paris is 45% of them. detectron2
drops such images by default, so any config using this dataset must set

    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

or 20% of the training set vanishes silently.
"""

import copy
import json
import os

import numpy as np
import torch

CITIES = ["AOI_2_Vegas", "AOI_3_Paris", "AOI_4_Shanghai", "AOI_5_Khartoum"]


# --------------------------------------------------------------------------
# stretch
# --------------------------------------------------------------------------

def city_of(path):
    """AOI name from a SpaceNet filename, e.g. ..._AOI_3_Paris_PS-RGB_img10.tif."""
    base = os.path.basename(path)
    for c in CITIES:
        if c in base:
            return c
    raise ValueError("no AOI in filename: %s" % base)


class Stretch:
    """UInt16 -> uint8, either per-image or from per-city constants."""

    def __init__(self, mode="per_image", low=2.0, high=98.0, constants=None):
        if mode not in ("per_image", "per_city"):
            raise ValueError("mode must be per_image or per_city, got %r" % mode)
        if mode == "per_city" and not constants:
            raise ValueError("per_city needs constants; run scripts/spacenet_stats.py")
        self.mode = mode
        self.low = low
        self.high = high
        self.constants = constants or {}

    @classmethod
    def from_json(cls, path, mode):
        with open(path) as f:
            d = json.load(f)
        lo = d["percentiles"]["low"]
        hi = d["percentiles"]["high"]
        lo_key, hi_key = "p%.1f" % lo, "p%.1f" % hi
        consts = {}
        for city, s in d["cities"].items():
            consts[city] = [(float(s[b][lo_key]), float(s[b][hi_key]))
                            for b in ("R", "G", "B")]
        return cls(mode=mode, low=lo, high=hi, constants=consts)

    def bounds(self, arr, path):
        """Per-channel (lo, hi) in raw UInt16 units."""
        if self.mode == "per_city":
            return self.constants[city_of(path)]
        out = []
        for c in range(arr.shape[0]):
            band = arr[c]
            # Nonzero only: tiles that overrun the imaged strip carry a zero
            # nodata border, and including it drags the low percentile to 0,
            # which silently disables the low end of the stretch.
            v = band[band > 0]
            if v.size == 0:
                out.append((0.0, 1.0))
            else:
                out.append((float(np.percentile(v, self.low)),
                            float(np.percentile(v, self.high))))
        return out

    def apply(self, arr, path):
        """(3, H, W) UInt16 -> (H, W, 3) uint8 RGB."""
        bounds = self.bounds(arr, path)
        out = np.empty(arr.shape, dtype=np.uint8)
        for c, (lo, hi) in enumerate(bounds):
            span = max(hi - lo, 1.0)
            scaled = (arr[c].astype(np.float32) - lo) * (255.0 / span)
            out[c] = np.clip(scaled, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(out.transpose(1, 2, 0))

    def load(self, path):
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read([1, 2, 3])
        return self.apply(arr, path)


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def register(root="data/spacenet2", coco_dir=None, prefix="spacenet2"):
    """Register one detectron2 dataset per AOI.

    Names are prefix_<city>, e.g. spacenet2_AOI_3_Paris. Splitting into train and
    val is deliberately NOT done here: the competition test labels were never
    released, so how the labelled data is divided is a methodological decision
    that belongs in the training config, not hidden in a registration helper.
    """
    from detectron2.data.datasets import register_coco_instances

    coco_dir = coco_dir or os.path.join(root, "coco")
    registered = []
    for city in CITIES:
        json_path = os.path.join(coco_dir, "%s_train.json" % city)
        img_dir = os.path.join(root, city, "PS-RGB")
        if not (os.path.isfile(json_path) and os.path.isdir(img_dir)):
            continue
        name = "%s_%s" % (prefix, city)
        register_coco_instances(name, {"thing_classes": ["building"]},
                                json_path, img_dir)
        registered.append(name)
    return registered


# --------------------------------------------------------------------------
# mapper
# --------------------------------------------------------------------------

class SpaceNetMapper:
    """DatasetMapper equivalent that loads UInt16 GeoTIFFs through Stretch.

    Written out rather than subclassing DatasetMapper because the only thing
    needing to change is the image read, and DatasetMapper does the read inline
    in __call__ with no hook to override. Subclassing would mean copying this
    body anyway, with the inheritance hiding that fact.

    detectron2's default read path would not merely be suboptimal here, it would
    be wrong: utils.read_image goes through PIL, which returns the raw UInt16
    values for these files, and the downstream code assumes 8-bit.
    """

    def __init__(self, cfg, is_train, stretch):
        from detectron2.data import detection_utils as utils
        from detectron2.data import transforms as T

        self._utils = utils
        self._T = T
        self.augmentations = T.AugmentationList(utils.build_augmentation(cfg, is_train))
        self.image_format = cfg.INPUT.FORMAT
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.is_train = is_train
        self.stretch = stretch

    def __call__(self, dataset_dict):
        utils, T = self._utils, self._T
        dataset_dict = copy.deepcopy(dataset_dict)

        image = self.stretch.load(dataset_dict["file_name"])   # H, W, 3 RGB uint8
        if self.image_format == "BGR":
            image = image[:, :, ::-1]
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
        # Per-INSTANCE filtering only. A tile with zero buildings still passes
        # through with an empty Instances, which is the point of keeping the
        # 2069 empty tiles.
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
