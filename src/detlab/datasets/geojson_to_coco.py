"""Convert SpaceNet-style GeoJSON building footprints into COCO instance JSON.

WHY THIS EXISTS
---------------
SpaceNet distributes labels as GeoJSON polygons in *geographic* coordinates
(usually EPSG:4326) while the imagery is a GeoTIFF in a projected CRS (usually
UTM). detectron2 wants polygons in *pixel* coordinates. Two transforms are
therefore required, in order:

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
- Empty tiles are KEPT. A tile with no buildings is a legitimate negative and
  discarding those biases the background distribution.

REQUIRES: rasterio, shapely, pyproj -- NOT yet in m2/detectron2:cu124-torch251.
Add them to docker/Dockerfile.detectron2 before first use.

STATUS: untested against real SpaceNet data, which has not been downloaded yet.
Validate on a handful of tiles (overlay the pixel polygons on the imagery)
before trusting a full conversion.

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

    orphans = sorted(set(images) - set(labels))
    if orphans:
        print(f"WARNING: {len(orphans)} images have no matching label file "
              f"(first few: {orphans[:5]})")
    return [(k, images[k], labels[k]) for k in common]


def geoms_to_pixel(label_path, src_raster, deps):
    """Read GeoJSON, reproject into the raster CRS, convert to pixel coords."""
    rasterio, transform_geom, shape, box, make_valid, Polygon = deps

    with open(label_path) as f:
        gj = json.load(f)

    features = gj.get("features", [])
    if not features:
        return []

    # RFC 7946: GeoJSON is EPSG:4326 unless it explicitly declares otherwise.
    crs_field = gj.get("crs", {}).get("properties", {}).get("name")
    src_crs = crs_field or "EPSG:4326"

    inv = ~src_raster.transform          # world -> pixel
    frame = box(0, 0, src_raster.width, src_raster.height)

    polygons = []
    for feat in features:
        geom = feat.get("geometry")
        if geom is None:
            continue

        geom = transform_geom(src_crs, src_raster.crs.to_string(), geom)
        poly = shape(geom)
        if not poly.is_valid:
            poly = make_valid(poly)

        parts = list(poly.geoms) if poly.geom_type.startswith("Multi") else [poly]
        for part in parts:
            if part.geom_type != "Polygon" or part.is_empty:
                continue

            pixel_poly = Polygon([inv * (x, y) for x, y in part.exterior.coords])
            if not pixel_poly.is_valid:
                pixel_poly = make_valid(pixel_poly)
                if pixel_poly.geom_type != "Polygon":
                    continue

            # Footprints can run past the tile edge; clip rather than discard.
            clipped = pixel_poly.intersection(frame)
            if clipped.is_empty:
                continue
            pieces = clipped.geoms if clipped.geom_type.startswith("Multi") else [clipped]
            for piece in pieces:
                if piece.geom_type == "Polygon" and piece.area > 1.0:
                    polygons.append(piece)
    return polygons


def build_coco(image_dir, label_dir, out_path, image_ext=".tif",
               label_ext=".geojson", key_regex=r"(img\d+)"):
    deps = _lazy_imports()
    rasterio = deps[0]

    pairs = pair_by_key(image_dir, label_dir, image_ext, label_ext, key_regex)
    print(f"matched {len(pairs)} image/label pairs")

    images, annotations = [], []
    ann_id = 1
    empty_tiles = 0
    crs_seen = set()

    for image_id, (key, img_path, lbl_path) in enumerate(pairs, start=1):
        with rasterio.open(img_path) as src:
            polygons = geoms_to_pixel(lbl_path, src, deps)
            crs_seen.add(src.crs.to_string())
            images.append({
                "id": image_id,
                "file_name": os.path.basename(img_path),
                "width": src.width,
                "height": src.height,
            })

        if not polygons:
            empty_tiles += 1

        for poly in polygons:
            x0, y0, x1, y1 = poly.bounds
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": CATEGORY_ID,
                # COCO bbox is [x, y, w, h] -- NOT [x1, y1, x2, y2].
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": poly.area,
                "segmentation": [[c for xy in poly.exterior.coords for c in xy]],
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
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}],
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(coco, f)

    print(f"images      : {len(images)}")
    print(f"annotations : {len(annotations)}")
    print(f"empty tiles : {empty_tiles}  (kept -- negatives are training signal)")
    print(f"raster CRSs : {sorted(crs_seen)}")
    print(f"wrote       : {out_path}")
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
