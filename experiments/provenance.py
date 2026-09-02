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
from pathlib import Path

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


def _env_lock():
    """Resolved versions of the packages that shape the numbers (verifier
    R4-008), so a rerun can pin the same environment."""
    import importlib.metadata as im
    out = {}
    for pkg in (
        "numpy",
        "scipy",
        "networkx",
        "osmnx",
        "pyproj",
        "shapely",
        "haversine",
        "eclipse-sumo",
        "sumo-data",
    ):
        try:
            out[pkg] = im.version(pkg)
        except Exception:
            out[pkg] = None
    return out


def _replay_command():
    """Return a command that can be copied to rerun the current entry point.

    When Python executes ``-m experiments.some_runner``, ``sys.argv[0]`` is the
    resolved file path, not the original module name.  Re-prepending ``-m`` to
    that path produces an invalid command.  Project-local ``.py`` entry points
    are therefore converted back to dotted module names; external scripts fall
    back to direct file execution.
    """

    project_root = Path(__file__).resolve().parents[1]
    entry_point = Path(sys.argv[0]).resolve()
    try:
        relative = entry_point.relative_to(project_root)
    except ValueError:
        return [sys.executable, str(entry_point), *sys.argv[1:]]

    if relative.suffix == ".py" and all(
        part.isidentifier() for part in relative.with_suffix("").parts
    ):
        module = ".".join(relative.with_suffix("").parts)
        return [sys.executable, "-m", module, *sys.argv[1:]]
    return [sys.executable, str(entry_point), *sys.argv[1:]]


def begin_run():
    """Capture the immutable run context BEFORE the computation loop (verifier
    R4-008): source commit + dirty flag, start time, interpreter, cwd, platform,
    and resolved package versions. Pass the returned dict to `provenance(...)`."""
    commit, dirty = _git_state()
    return {
        "source_commit": commit,
        "source_dirty_before_run": dirty,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "interpreter": sys.executable,
        "cwd": os.getcwd(),
        "platform": platform.platform(),
        "command": _replay_command(),
        "package_versions": _env_lock(),
    }


def provenance(rn, epsilons, root_seeds, quick=False, extra=None, begin=None):
    ctx = begin or begin_run()
    man = load_manifest()
    prov = {
        "schema": "msc-experiment-v1",
        "source_commit": ctx.get("source_commit"),
        "source_dirty_before_run": ctx.get("source_dirty_before_run"),
        "command": ctx.get("command", sys.argv),
        "interpreter": ctx.get("interpreter"),
        "cwd": ctx.get("cwd"),
        "platform": ctx.get("platform"),
        "mode": "quick" if quick else "full",
        "started_at_utc": ctx.get("started_at_utc"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": ctx.get("python", platform.python_version()),
        "package_versions": ctx.get("package_versions"),
        "graph_sha256": man.get("graph_sha256"),
        "graph_nodes": len(rn),
        "graph_source_sha256": man.get("source_sha256"),
        "graph_network_profile": man.get("network_profile", "all (unfiltered; see R3-013)"),
        "rng_schema": RNG_SCHEMA,
        "root_seeds": list(root_seeds),
        "epsilons": list(epsilons),
        "dataset": {"name": "GeoLife v1.3", "raw_bytes_in_git": False,
                    "note": "raw GeoLife/graph are gitignored; selected record IDs "
                            "pinned below, per-file byte hashing is future work (R4-008)"},
    }
    if extra:
        prov.update(extra)
    return prov
