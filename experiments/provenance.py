"""Experiment provenance + fail-closed graph check (verifier R3-003 / R3-008).

`msc-experiment-v1` schema records the source commit AND whether the working
tree was dirty before the run, so a reader can tell whether an output pins a
clean, committed source state. Also records the exact command, Python version,
graph hash, RNG schema, root seeds and the selected GeoLife record IDs so every
table cell is traceable.

`assert_graph_matches_manifest` fails CLOSED: an official experiment refuses a
graph whose bytes/node count do not match the committed manifest, so results
cannot silently come from a different graph.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

from experiments.rng_util import RNG_SCHEMA

MANIFEST = os.path.join("data", "beijing_graph.manifest.json")
GRAPH_PKL = os.path.join("data", "raw", "beijing_graph.pkl")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state():
    """Source commit + whether SOURCE (code/thesis/data) is dirty. The `outputs/`
    directory is excluded from the dirty check: regenerating results is expected
    and must not, by itself, mark the source state as dirty (so both experiments
    in a run can each report a clean source commit)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)outputs"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return commit, bool(porcelain)
    except Exception:
        return None, None


def load_manifest(path=MANIFEST):
    return json.load(open(path)) if os.path.exists(path) else {}


def assert_graph_matches_manifest(rn, graph_pkl=GRAPH_PKL, manifest_path=MANIFEST):
    """Refuse to run an official experiment on a graph that does not match the
    committed manifest (verifier R3-008). Returns the manifest on success."""
    man = load_manifest(manifest_path)
    if not man:
        raise SystemExit(f"missing manifest {manifest_path}; run data.build_beijing_graph")
    actual = sha256_file(graph_pkl)
    if man.get("graph_sha256") and actual != man["graph_sha256"]:
        raise SystemExit(
            f"GRAPH HASH MISMATCH — refusing to run.\n  file:     {actual}\n"
            f"  manifest: {man['graph_sha256']}\n"
            "Rebuild with `python -m data.build_beijing_graph` or restore the pinned graph.")
    if man.get("nodes") and len(rn) != man["nodes"]:
        raise SystemExit(
            f"GRAPH NODE COUNT MISMATCH — refusing to run: {len(rn)} != manifest {man['nodes']}")
    return man


def provenance(rn, epsilons, root_seeds, quick=False, extra=None):
    commit, dirty = _git_state()
    man = load_manifest()
    prov = {
        "schema": "msc-experiment-v1",
        "source_commit": commit,
        "source_dirty_before_run": dirty,
        "command": sys.argv,
        "mode": "quick" if quick else "full",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "graph_sha256": man.get("graph_sha256"),
        "graph_nodes": len(rn),
        "graph_source_sha256": man.get("source_sha256"),
        "graph_network_profile": man.get("network_profile", "all (unfiltered; see R3-013)"),
        "rng_schema": RNG_SCHEMA,
        "root_seeds": list(root_seeds),
        "epsilons": list(epsilons),
    }
    if extra:
        prov.update(extra)
    return prov
