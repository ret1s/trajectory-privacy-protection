"""Deterministic, tracked build of the Beijing road graph (verifier R2-005/R2-013).

Downloads the BBBike Beijing extract (if absent), decompresses it, builds a
drive/all graph, truncates to the GeoLife bbox with the CORRECT OSMnx-2.x order
(left,bottom,right,top), keeps the largest weakly-connected component, writes
data/raw/beijing_graph.pkl, and regenerates data/beijing_graph.manifest.json
(source hashes, node/edge/component counts, extent, dependency versions) from
the artifact itself — never hand-edited.

Run from repo root:  python -m data.build_beijing_graph
"""
import gzip
import hashlib
import json
import os
import platform
import shutil

import networkx as nx
import numpy as np
import osmnx as ox
import scipy

from data.geolife import BEIJING_BBOX

RAW = os.path.join(os.path.dirname(__file__), "raw")
GZ = os.path.join(RAW, "Beijing.osm.gz")
XML = os.path.join(RAW, "Beijing.osm")
PKL = os.path.join(RAW, "beijing_graph.pkl")
MANIFEST = os.path.join(os.path.dirname(__file__), "beijing_graph.manifest.json")
URL = "https://download.bbbike.org/osm/bbbike/Beijing/Beijing.osm.gz"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build():
    os.makedirs(RAW, exist_ok=True)
    if not os.path.exists(GZ):
        raise SystemExit(
            f"Missing {GZ}. Download first:\n  curl -L -o {GZ} '{URL}'"
        )
    if not os.path.exists(XML):
        with gzip.open(GZ, "rb") as fi, open(XML, "wb") as fo:
            shutil.copyfileobj(fi, fo)

    b = BEIJING_BBOX  # (min_lat, min_lon, max_lat, max_lon)
    G = ox.graph_from_xml(XML, simplify=True, retain_all=True)
    G = ox.truncate.truncate_graph_bbox(G, bbox=(b[1], b[0], b[3], b[2]))
    G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()

    import pickle
    with open(PKL, "wb") as f:
        pickle.dump(G, f)

    lats = [d["y"] for _, d in G.nodes(data=True)]
    lons = [d["x"] for _, d in G.nodes(data=True)]
    manifest = {
        "source_url": URL,
        "source_sha256": {
            "Beijing.osm.gz": _sha256(GZ),
            "Beijing.osm": _sha256(XML),
        },
        "build": "graph_from_xml(simplify=True, retain_all=True); "
                 "truncate_graph_bbox(left,bottom,right,top); largest weakly-connected component",
        "bbox_geolife_min_lat_min_lon_max_lat_max_lon": list(BEIJING_BBOX),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "weak_components": nx.number_weakly_connected_components(G),
        "extent": {"lat": [round(min(lats), 6), round(max(lats), 6)],
                   "lon": [round(min(lons), 6), round(max(lons), 6)]},
        "graph_sha256": _sha256(PKL),
        "versions": {"python": platform.python_version(), "osmnx": ox.__version__,
                     "networkx": nx.__version__, "numpy": np.__version__,
                     "scipy": scipy.__version__},
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"nodes={manifest['nodes']} edges={manifest['edges']} "
          f"wcc={manifest['weak_components']} graph_sha256={manifest['graph_sha256'][:16]}")
    print(f"wrote {PKL} and {MANIFEST}")


if __name__ == "__main__":
    build()
