#!/usr/bin/env python
"""Cross-model concordance on G-LiHT, where no ground truth exists.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The G-LiHT tile carries no annotations, so precision and recall cannot be
computed for any encoding. What DOES exist is a second, independently trained
detector's opinion of the same ground: the Q2000 catalogue, produced by a
separate model trained on separate hand annotations and run over this transect.

Agreement with it is NOT accuracy. Two models can be wrong together, and these
two are more likely to be wrong together than two arbitrary models, because
both read relief visualisations of the same kind. What agreement does provide
is a constraint that a raw count does not: if one encoding's extra detections
are systematically ignored by an independent detector, that is evidence they
are the encoding's artefacts rather than structures the encoding revealed.

Reported in both directions:
  concordance  the share of THIS arm's detections that an independent detector
               also called a structure
  recovery     the share of the independent detector's calls that THIS arm found

Q2000 is single-class ("Maya Structures"), so its calls are compared against
building and platform detections pooled. Aguada is excluded: the independent
detector has no such class and its silence there means nothing.
"""
import argparse
import json
import math
import os

import geopandas as gpd
from shapely.geometry import shape

Q2000_DEFAULT = os.path.join(
    "/mnt/c" if os.path.isdir("/mnt/c") else "C:/",
    "g1/Q2000_GliHT_Package/Q2000_master_060726.shp")
TRANSECT = "South_GLAS_l0s395"
RUNS = [("RVT, Table 3 stretch", "S395_spec_fixed.geojson"),
        ("G1 composite, as delivered", "S395_g1_fixed.geojson"),
        ("RVT, mean/sd matched", "S395_matched_fixed.geojson")]
STRUCTURE = {"building", "platform"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="outputs/gliht_spec")
    p.add_argument("--q2000", default=Q2000_DEFAULT)
    p.add_argument("--radius-m", type=float, default=20.0,
                   help="centroid separation counted as the same feature")
    p.add_argument("--out", default="outputs/gliht_spec/concordance.json")
    a = p.parse_args()

    q = gpd.read_file(a.q2000)
    q = q[q["transect"].astype(str) == TRANSECT]
    q = q.to_crs("EPSG:32615")
    qc = [(g.centroid.x, g.centroid.y) for g in q.geometry]
    print("independent detector (Q2000) on %s: %d calls" % (TRANSECT, len(qc)))
    print("radius counted as agreement: %.0f m" % a.radius_m)
    print()

    r2 = a.radius_m ** 2
    out = {"independent_calls": len(qc), "radius_m": a.radius_m, "arms": {}}
    hdr = "%-28s %6s %10s %10s" % ("encoding", "n", "concord", "recovery")
    print(hdr)
    print("-" * len(hdr))
    for label, gj in RUNS:
        feats = json.load(open(os.path.join(a.dir, gj)))["features"]
        cen = []
        for f in feats:
            if f["properties"].get("class") not in STRUCTURE:
                continue
            g = shape(f["geometry"]).centroid
            cen.append((g.x, g.y))

        matched = 0
        for x, y in cen:
            if any((x - qx) ** 2 + (y - qy) ** 2 <= r2 for qx, qy in qc):
                matched += 1
        recovered = 0
        for qx, qy in qc:
            if any((x - qx) ** 2 + (y - qy) ** 2 <= r2 for x, y in cen):
                recovered += 1

        conc = matched / len(cen) if cen else float("nan")
        rec = recovered / len(qc) if qc else float("nan")
        out["arms"][label] = {"n_structure": len(cen), "concordance": conc,
                              "recovery": rec, "matched": matched,
                              "recovered": recovered}
        print("%-28s %6d %9.1f%% %9.1f%%" % (label, len(cen), 100 * conc, 100 * rec))

    json.dump(out, open(a.out, "w"), indent=1)
    print()
    print("wrote %s" % a.out)
    print()
    print("Concordance is agreement between two models, not accuracy. Both read")
    print("relief visualisations of the same terrain and can err together.")


if __name__ == "__main__":
    main()
