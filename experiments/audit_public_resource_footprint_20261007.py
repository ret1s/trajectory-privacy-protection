"""Read-only array/storage accounting for the retained public road context.

This does not generate GPS/Q, score utility, measure peak RSS or claim phone
performance. NPZ disk size and NPY payload sizes are distinct. Prefix/cache
rows are exact derived storage if materialized, not measured live allocations.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/benchmarks/public_resource_footprint_20261007_v1'
NPZ = 'artifacts/benchmarks/dynamic_provider_status_20261006_v1/public_reply60.npz'
CONFIG = 'artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1/recommended_configuration.json'
SOURCES = (NPZ, CONFIG, 'benchmark/public_poi_context.py', 'benchmark/query_purpose.py',
           'experiments/qplanner_study_20261006_v2.py',
           'experiments/audit_public_resource_footprint_20261007.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header(archive, name):
    with archive.open(name) as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0): reader = np.lib.format.read_array_header_1_0
        elif version == (2, 0): reader = np.lib.format.read_array_header_2_0
        else: raise ValueError('Unsupported retained NPY header version')
        shape, fortran, dtype = reader(stream)
    if dtype.hasobject or fortran: raise ValueError('Numeric C-order public array required')
    return {'shape': list(shape), 'dtype': str(dtype), 'itemsize': dtype.itemsize,
            'payload_bytes': math.prod(shape) * dtype.itemsize}


def build():
    path = ROOT / NPZ
    with zipfile.ZipFile(path) as archive:
        sig, access = [header(archive, name) for name in ('signatures.npy', 'access.npy')]
    n, categories, depth = sig['shape']
    assert (n, categories, depth) == (66189, 6, 60)
    assert sig['dtype'] == access['dtype'] == 'int32' and access['shape'] == [n]
    config = json.loads((ROOT / CONFIG).read_text())['configuration']
    assert config['Q_planner'] == 'legacy_l10' and config['planner_signature_L'] == 10
    tree = ast.parse((ROOT / 'benchmark/query_purpose.py').read_text())
    init = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MultiPurposeRoadRanking')
    init = next(node for node in init.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    default_limit = ast.literal_eval(init.args.defaults[-1])
    tree = ast.parse((ROOT / 'experiments/qplanner_study_20261006_v2.py').read_text())
    limits = [ast.literal_eval(k.value) for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Name) and node.func.id == 'MultiPurposeRoadRanking'
              for k in node.keywords if k.arg == 'cache_limit']
    assert default_limit == 256 and limits == [4096]
    return {'schema': 'public-array-footprint-accounting-v1', 'as_of': '2026-10-07',
            'source_sha256': {name: sha(ROOT / name) for name in SOURCES},
            'archive_bytes_on_disk': path.stat().st_size, 'arrays': {'signatures': sig, 'access': access},
            'signature_prefix_storage_if_materialized': {
                str(L): n * categories * L * sig['itemsize'] for L in (5, 10, 20, 30, 60)},
            'distance_cache_storage_if_full_float64': {
                'ranking_default_256_entries': default_limit * n * 8,
                'offline_evaluator_4096_entries': limits[0] * n * 8},
            'scope': 'Exact retained array/archive sizes and analytic prefix/full-cache payloads only. Excludes Python/NetworkX objects, graph matrices, temporary arrays, OS memory and peak process RSS. L60 is a provider/evaluator fixture, not required current client L10 context; local reply L30 is not a full-map client table. No CPU, latency, network or battery measurement.',
            'model_or_private_state_modified': False, 'private_keys_read': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); result = build(); target = OUT / 'readout.json'
    if args.check:
        assert json.loads(target.read_text()) == result
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        with target.open('x') as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print('PASS: public archive/array accounting; prefix/cache rows are analytic, not live memory benchmarks')


if __name__ == '__main__': main()
