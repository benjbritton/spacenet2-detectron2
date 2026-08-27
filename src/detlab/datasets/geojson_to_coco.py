"""Convert SpaceNet-style GeoJSON building footprints into COCO instance JSON.

WHY THIS EXISTS
---------------
SpaceNet distributes labels as GeoJSON polygons in *geographic* coordinates
(usually EPSG:4326) while the imagery is a GeoTIFF whose CRS varies by product.
detectron2 wants polygons in *pixel* coordinates. Two transforms are therefore
required, in order:

    geographic CRS  --reproject-->  raster CRS  --affine-->  pixel

Skipping the reprojection is the classic silent failure: if the GeoJSON and the
raster happen to share a CRS the code appears to work, and the moment an AOI in
a different UTM zone appears the footprints land in the wrong place -- offset,
not obviously broken. So reprojection is unconditional here even when the CRSs
match, and every raster CRS encountered is recorded in the output for auditing.

COORDINATE CONVENTIONS THAT BITE
--------------------------------
- COCO bbox is [x, y, WIDTH, HEIGHT]. detectron2 BoxMode.XYXY_ABS is
  [x1, y1, x2, y2]. register_coco_instances expects the COCO form and converts
  internally. Emitting XYXY here produces boxes that look plausible on
  inspection and evaluate as garbage.
- Polygon holes: the COCO polygon format cannot express interior rings, so
  buildings with courtyards lose their holes. Encode masks as RLE instead if
  that matters for a given AOI.
- One source footprint yields ONE annotation, even when it ends up as several
  disjoint pieces. Clipping at the tile edge, and make_valid repairing a
  self-intersecting outline, both split a polygon. Emitting each piece as its own
  annotation would invent buildings that do not exist -- on AOI_2_Vegas that was
  863 phantom instances, 0.8% -- and would make instance counts incomparable with
  published baselines. COCO segmentation is a LIST of polygons precisely so one
  instance can be disjoint, and a mask head can predict a disjoint mask, so the
  pieces are grouped with a bbox spanning all of them and area summed.
- Empty tiles are KEPT. A tile with no buildings is a legitimate negative and
  discarding those biases the background distribution.

REQUIRES: rasterio, shapely, pyproj -- NOT yet in m2/detectron2:cu124-torch251.
Add them to docker/Dockerfile.detectron2 before first use.

MEASURED AGAINST THE REAL DATA (SN2 PS-RGB, downloaded 2026-08-26)
------------------------------------------------------------------
- The PS-RGB rasters are **EPSG:4326**, i.e. geographic, not the projected UTM
  the note above anticipated. The reprojection step is therefore a no-op on this
  product. It stays unconditional anyway: the same code must survive an AOI or a
  product that is projected, and a silently-skipped reprojection is the failure
  this module exists to prevent.
- Pixel size is ~2.7e-6 degrees, about 0.3 m. Tiles are 650x650, three bands,
  **UInt16**. Bit depth is deliberately NOT handled here -- this module touches
  only geometry and raster metadata, never pixels. The 16-bit to 8-bit decision
  belongs in the dataloader mapper, where it is a versioned config value rather
  than baked into a derivative file.
- AOI_2_Vegas ships 3851 label files against 3850 images: img1000 has a footprint
  file and no PS-RGB tile. Upstream artifact, present in the bucket, not a bad
  download. Pairing is on the intersection and BOTH orphan directions are now
  reported -- see pair_by_key.

USAGE
-----
    python -m detlab.datasets.geojson_to_coco \
        --images  data/spacenet/AOI_2_Vegas/RGB-PanSharpen \
        --labels  data/spacenet/AOI_2_Vegas/geojson/buildings \
        --out     data/spacenet/AOI_2_Vegas/coco_train.json

Then in detectron2:

    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("spacenet_vegas_train", {},
                            ".../coco_train.json", ".../RGB-PanSharpen")
"""

import argparse
import json
import os
import re
from datetime import datetime

CATEGORY_ID = 1
CATEGORY_NAME = "building"


def _lazy_imports():
    """Imported here so this module can be read without the geo stack installed."""
    import rasterio
    from rasterio.warp import transform_geom
    from shapely.geometry import Polygon, box, shape
    from shapely.validation import make_valid

    return rasterio, transform_geom, shape, box, make_valid, Polygon


def pair_by_key(image_dir, label_dir, image_ext=".tif", label_ext=".geojson",
                key_regex=r"(img\d+)"):
    """Match imagery to labels on a key shared by their filenames.

    SpaceNet names files with different prefixes but a common image key, e.g.
    RGB-PanSharpen_AOI_2_Vegas_img1.tif alongside
    buildings_AOI_2_Vegas_img1.geojson.
    """
    pattern = re.compile(key_regex)

    def index(directory, ext):
        out = {}
        for name in sorted(os.listdir(directory)):
            if not name.endswith(ext):
                continue
            m = pattern.search(name)
            if m:
                out[m.group(1)] = os.path.join(directory, name)
        return out

    images, labels = index(image_dir, image_ext), index(label_dir, label_ext)
    common = sorted(set(images) & set(labels),
                    key=lambda k: int(re.sub(r"\D", "", k) or 0))

    # BOTH directions matter and they fail differently.
    #
    # An image with no label would be silently treated as an empty tile -- a
    # negative example that is really an unlabelled positive, which actively
    # teaches the model to miss buildings.
    #
    # A label with no image is simply unusable, but it must still be reported:
    # AOI_2_Vegas genuinely ships one (img1000, 3851 labels against 3850
    # images). Dropping it without a word makes the final tile count
    # unexplainable later, which is how a real data loss gets mistaken for a
    # known quirk.
    unlabelled = sorted(set(images) - set(labels))
    unimaged = sorted(set(labels) - set(images))
    if unlabelled:
        print(f"WARNING: {len(unlabelled)} images have NO label file "
              f"(treated as absent, not as empty tiles): {unlabelled[:5]}")
    if unimaged:
        print(f"NOTE: {len(unimaged)} label files have NO image and are skipped: "
              f"{unimaged[:5]}")
    print(f"pairing: {len(images)} images, {len(labels)} labels, "
          f"{len(common)} paired")
    return [(k, images[k], labels[k]) for k in common], {
        "images_found": len(images),
        "labels_found": len(labels),
        "paired": len(common),
        "images_without_labels": unlabelled,
        "labels_without_images": unimaged,
    }


def geoms_to_pixel(label_path, src_raster, deps, stats=None):
    """Read GeoJSON, reproject into the raster CRS, convert to pixel coords.

    Every discard is counted into `stats`. Geometry conversion drops features for
    several legitimate reasons and one illegitimate one, and without a tally the
    two are indistinguishable: a converter that silently loses 30% of footprints
    produces a model that trains, converges, and underperforms for no visible
    reason.
    """
    rasterio, transform_geom, shape, box, make_valid, Polygon = deps
    if stats is None:
        stats = {}

    def bump(k):
        stats[k] = stats.get(k, 0) + 1

    with open(label_path) as f:
        gj = json.load(f)

    features = gj.get("features", [])
    stats["source_features"] = stats.get("source_features", 0) + len(features)
    if not features:
        return []

    # RFC 7946: GeoJSON is EPSG:4326 unless it explicitly declares otherwise.
    crs_field = gj.get("crs", {}).get("properties", {}).get("name")
    src_crs = crs_field or "EPSG:4326"

    inv = ~src_raster.transform          # world -> pixel
    frame = box(0, 0, src_raster.width, src_raster.height)

    instances = []
    for feat in features:
        # Pieces accumulate per source feature, not globally: one footprint is
        # one building however many fragments it survives as.
        feature_pieces = []
        geom = feat.get("geometry")
        if geom is None:
            bump("dropped_null_geometry")
            continue

        geom = transform_geom(src_crs, src_raster.crs.to_string(), geom)
        poly = shape(geom)
        if not poly.is_valid:
            bump("repaired_invalid_source")
            poly = make_valid(poly)

        parts = list(poly.geoms) if poly.geom_type.startswith("Multi") else [poly]
        for part in parts:
            if part.geom_type != "Polygon" or part.is_empty:
                bump("dropped_non_polygon_part")
                continue

            # SpaceNet footprints carry a Z ordinate, so coordinates arrive as
            # (x, y, z) and unpacking two names raises ValueError. Take the first
            # two explicitly rather than unpacking: this must not depend on whether
            # a given AOI happens to ship 2D or 3D geometry.
            pixel_poly = Polygon([inv * (c[0], c[1])
                                  for c in part.exterior.coords])
            if not pixel_poly.is_valid:
                bump("repaired_invalid_after_transform")
                pixel_poly = make_valid(pixel_poly)
                if pixel_poly.geom_type != "Polygon":
                    bump("dropped_unrepairable")
                    continue

            # Footprints can run past the tile edge; clip rather than discard.
            clipped = pixel_poly.intersection(frame)
            if clipped.is_empty:
                bump("dropped_outside_frame")
                continue
            pieces = clipped.geoms if clipped.geom_type.startswith("Multi") else [clipped]
            for piece in pieces:
                if piece.geom_type != "Polygon":
                    bump("dropped_non_polygon_after_clip")
                elif piece.area <= 1.0:
                    bump("dropped_sliver_under_1px")
                else:
                    feature_pieces.append(piece)

        if feature_pieces:
            if len(feature_pieces) > 1:
                bump("instances_kept_whole_despite_splitting")
            instances.append(feature_pieces)
        else:
            bump("dropped_feature_entirely")
    return instances


def build_coco(image_dir, label_dir, out_path, image_ext=".tif",
               label_ext=".geojson", key_regex=r"(img\d+)"):
    deps = _lazy_imports()
    rasterio = deps[0]

    pairs, pairing = pair_by_key(image_dir, label_dir, image_ext, label_ext,
                                 key_regex)

    images, annotations = [], []
    ann_id = 1
    empty_tiles = 0
    crs_seen = set()
    geom_stats = {}

    for image_id, (key, img_path, lbl_path) in enumerate(pairs, start=1):
        with rasterio.open(img_path) as src:
            instances = geoms_to_pixel(lbl_path, src, deps, geom_stats)
            crs_seen.add(src.crs.to_string())
            images.append({
                "id": image_id,
                "file_name": os.path.basename(img_path),
                "width": src.width,
                "height": src.height,
            })

        if not instances:
            empty_tiles += 1

        for pieces in instances:
            # bbox spans every piece; area sums them. Using only the largest
            # piece would under-report area, and area is what COCOeval buckets
            # small/medium/large on.
            x0 = min(p.bounds[0] for p in pieces)
            y0 = min(p.bounds[1] for p in pieces)
            x1 = max(p.bounds[2] for p in pieces)
            y1 = max(p.bounds[3] for p in pieces)
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": CATEGORY_ID,
                # COCO bbox is [x, y, w, h] -- NOT [x1, y1, x2, y2].
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": sum(p.area for p in pieces),
                "segmentation": [[c for xy in p.exterior.coords for c in xy]
                                 for p in pieces],
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {
            "description": "SpaceNet buildings converted from GeoJSON",
            "date_created": datetime.now().isoformat(timespec="seconds"),
            "source_images": image_dir,
            "source_labels": label_dir,
            "raster_crs_seen": sorted(crs_seen),
            # Recorded in the output so a tile or footprint count can always be
            # reconciled against the source directories without a rerun.
            "pairing": pairing,
            "geometry_stats": geom_stats,
            "empty_tiles_kept": empty_tiles,
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}],
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f)

    src_feats = geom_stats.get("source_features", 0)
    kept = len(annotations)
    print(f"images        : {len(images)}")
    print(f"annotations   : {kept}")
    print(f"source feats  : {src_feats}"
          + (f"  ({100.0 * kept / src_feats:.2f}% retained as instances)"
             if src_feats else ""))
    for k in sorted(geom_stats):
        if k != "source_features":
            print(f"  {k:34s}: {geom_stats[k]}")
    print(f"empty tiles   : {empty_tiles}  (kept -- negatives are training signal)")
    print(f"raster CRSs   : {sorted(crs_seen)}")
    print(f"wrote         : {out_path}")
    return coco


def main():
    p = argparse.ArgumentParser(
        description="Convert SpaceNet GeoJSON footprints to COCO instance JSON.")
    p.add_argument("--images", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--image-ext", default=".tif")
    p.add_argument("--label-ext", default=".geojson")
    p.add_argument("--key-regex", default=r"(img\d+)",
                   help="regex whose first group is the key shared by an image "
                        "filename and its label filename")
    args = p.parse_args()
    build_coco(args.images, args.labels, args.out,
               args.image_ext, args.label_ext, args.key_regex)


if __name__ == "__main__":
    main()
