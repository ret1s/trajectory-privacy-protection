"""Application gate on the frozen REM/Planar development pilot.

No protection generation, defense selection, timing simulation or historical
evidence mutation. Compare the same four local POI purposes using frozen
current/epoch response pools versus one full static catalogue per public epoch.
"""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose
from data.lane_states import build_lane_states
from evaluation.lane_travel import LanePoiService
from evaluation.static_catalogue_control import catalogue_payload, evaluate_local_purposes

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT/'artifacts/benchmarks/jisa_native_anchor_ablation_20261006_v1'
DATA = ROOT/'artifacts/datasets/future_controlled_20261005_v2/dataset.json.gz'
REPLY = ROOT/'artifacts/benchmarks/future_native_depth_20261005_v1/public_reply40.npz'
OUT = ROOT/'artifacts/benchmarks/jisa_static_catalogue_control_20261006_v1'
METHODS = ('raw', 'rem_epoch8', 'planar_epoch8')
ANSWERS = ('full_catalogue',) + tuple(m+'_'+p for m in METHODS for p in ('current_only', 'epoch_cache'))
INPUT_FILES = ('protocol.json', 'protocol.sha256', 'results.json', 'public_transcripts.json.gz', 'wire_rows.json.gz')
PUBLIC_EPOCH = {'epoch_id': 'jisa-native-matched-eight-trip-epoch', 'start_s': 0., 'end_s': 12000.}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_text())


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve evidence: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode()
    path.write_bytes(gzip.compress(content, mtime=0) if path.suffix == '.gz' else content)


def source_closure():
    pending = [str(Path(__file__).relative_to(ROOT)), 'evaluation/static_catalogue_control.py']
    found = set()
    while pending:
        name = pending.pop()
        if name in found: continue
        found.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                       [node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else [])
            for module in modules:
                for target in (module.replace('.', '/')+'.py', module.replace('.', '/')+'/__init__.py'):
                    if (ROOT/target).is_file() and target not in found: pending.append(target)
    return sorted(found)


def summarize(rows, answer):
    # Equal events/categories within session; sessions within family; families
    # within split. Empty reference queries are omitted, but counted explicitly.
    groups = {}
    for row in rows:
        groups.setdefault((row['family_id'], row['slot']), []).append(row['recall'][answer])
    session = {key: float(np.mean([v for v in values if v is not None]))
               if any(v is not None for v in values) else None for key, values in groups.items()}
    families = {family: float(np.mean([v for (f, _), v in session.items() if f == family and v is not None]))
                if any(f == family and v is not None for (f, _), v in session.items()) else None
                for family in sorted({r['family_id'] for r in rows})}
    defined = [v for v in families.values() if v is not None]
    usable = [v for v in session.values() if v is not None]
    return {'family_macro_recall5': float(np.mean(defined)) if defined else None,
        'family_values': families, 'represented_family_count': len(families),
        'defined_family_count': len(defined), 'represented_session_count': len(session),
        'undefined_session_count': sum(v is None for v in session.values()),
        'minimum_defined_session_recall5': min(usable) if usable else None,
        'reference_defined_queries': sum(r['reference'] != [] for r in rows),
        'empty_reference_queries': sum(r['reference'] == [] for r in rows), 'total_queries': len(rows)}


def declare(out, pilot):
    if out.exists(): raise FileExistsError('Use a fresh application-control directory')
    out.mkdir(parents=True)
    source_files = source_closure()
    protocol = {'schema': 'jisa-static-catalogue-application-control-v1', 'date': '2026-10-06',
        'status': 'DEVELOPMENT on all 24 previously inspected native pilot families; no confirmation or defense selection',
        'pilot': str(pilot.relative_to(ROOT)) if pilot.is_relative_to(ROOT) else str(pilot),
        'pilot_sha256': {name: sha(pilot/name) for name in INPUT_FILES},
        'source_sha256': {name: sha(ROOT/name) for name in source_files},
        'dataset_sha256': sha(DATA), 'reply40_file_sha256': sha(REPLY),
        'poi_source_sha256': sha(ROOT/'artifacts/benchmarks/research_loop/resources.json'),
        'public_epoch': PUBLIC_EPOCH, 'methods': METHODS, 'answers': ANSWERS,
        'session_rule': 'all eight frozen publicly scheduled sessions of all24 pilot families, no private draw regeneration',
        'purposes': [p.value for p in QueryPurpose], 'reference_k': 5, 'private_radius_m': 1000.,
        'private_destination': 'nearest road state of actual final session GPS; evaluator/local utility only, never Q/planner/request',
        'reference': 'full fixed static catalogue under exact existing directed-road QuerySpec definitions and lexical ties',
        'service_control': 'received topL20-by-distance category IDs from frozen pilot wire rows; current-only or union from same epoch',
        'full_catalogue_control': 'bulk fetch all id/category/lat/lon records at public epoch start0 before local input; no position/purpose/category/route in request; no later local-query requests',
        'cost': 'actual compact UTF8 JSON byte lengths; full catalogue once per independent family epoch vs frozen pilot all per-Q request/full record reply estimates; no HTTP/TLS/map downloads/CPU/GNSS/latency',
        'assumptions': ['provider permits bulk access to full418-POI catalogue', 'fixed public region and map known independently of private GPS',
                        'catalogue version remains unchanged during declared epoch', 'all POIs static, no availability/price/live travel-time data',
                        'same local exact GPS/query/destination oracle for all utility controls'],
        'weighting': 'conditional nonempty query recall: query average within session, equal defined sessions within family, equal defined families within split; counts retained',
        'failure_policy': 'retain all completed or failed evidence; no replacing families or outputs'}
    write(out/'protocol.json', protocol)
    (out/'protocol.sha256').write_text(sha(out/'protocol.json')+'\n')
    snapshots = out/'source_snapshot'
    for name in source_files:
        target = snapshots/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    return protocol


def run(out, pilot):
    protocol = declare(out, pilot)
    pilot_protocol, pilot_result = read(pilot/'protocol.json'), read(pilot/'results.json')
    assert sha(pilot/'protocol.json') == (pilot/'protocol.sha256').read_text().strip()
    for name, digest in pilot_result['file_sha256'].items(): assert sha(pilot/name) == digest, name
    data, public, wire = read(DATA), read(pilot/'public_transcripts.json.gz'), read(pilot/'wire_rows.json.gz')['rows']
    assert len(data['families']) == len(public['groups']) == 24
    rn = build_lane_states(ROOT/data['network']['compressed_path'], spacing_m=40.)
    pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')} for p in
            read(ROOT/'artifacts/benchmarks/research_loop/resources.json')['pois_used']]
    reply = PublicPoiContext(LanePoiService(rn, pois, k=40), REPLY)
    assert reply.sha256 == pilot_result['resources']['reply40_sha256'] and len(reply.pois) == 418
    local = MultiPurposeRoadRanking(reply, cache_limit=256)
    bulk = catalogue_payload(reply.pois, epoch_id=PUBLIC_EPOCH['epoch_id'], public_start_s=0.)
    write(out/'bulk_request.json', bulk['request']); write(out/'bulk_response.json', bulk['response'])
    full_ids = set(range(local.n)); by_wire = {(r['family_id'], r['slot'], r['method'], r['event_id']): r for r in wire}
    assert len(by_wire) == len(wire)
    rows, costs = [], []
    for group, family in zip(public['groups'], data['families']):
        cached = {m: set() for m in METHODS}
        assert group['public_context'] == family['public_context']
        family_wire = [r for r in wire if r['family_id'] == family['family_id']]
        costs.append({'family_id': family['family_id'], 'split': family['split'], 'method': 'full_catalogue',
                      'requests': 1, 'request_bytes': bulk['request_bytes'], 'reply_bytes': bulk['response_bytes']})
        for method in METHODS:
            selected = [r for r in family_wire if r['method'] == method]
            costs.append({'family_id': family['family_id'], 'split': family['split'], 'method': method,
                'requests': sum(r['requests'] for r in selected), 'request_bytes': sum(r['request_bytes'] for r in selected),
                'reply_bytes': sum(r['reply_bytes'] for r in selected)})
        for slot, spec in enumerate(family['evaluator_only']['sessions']):
            trace = {int(p['time_s']): p for p in data['traces'][spec['session_id']]}
            end = trace[600]; destination = rn.nearest(end['lat'], end['lon'])[0]
            clocks = [e['timestamp_s'] for e in group['streams']['raw'][slot]['events']]
            assert all([e['timestamp_s'] for e in group['streams'][m][slot]['events']] == clocks for m in METHODS)
            for event in group['streams']['raw'][slot]['events']:
                t = int(event['timestamp_s']); absolute = spec['depart_s']+t
                assert PUBLIC_EPOCH['start_s'] <= absolute < PUBLIC_EPOCH['end_s']
                pools = {'full_catalogue': full_ids}
                for method in METHODS:
                    record = by_wire[family['family_id'], slot, method, event['event_id']]
                    current = {i for ids in record['reply_poi_indices_by_Q'] for i in ids}
                    cached[method].update(current)
                    pools[method+'_current_only'], pools[method+'_epoch_cache'] = current, cached[method]
                gps = trace[t]; state = rn.nearest(gps['lat'], gps['lon'])[0]
                for answer in evaluate_local_purposes(local, state, pools, destination_state=destination):
                    assert all(v in (None, 1.) for key, v in answer['recall'].items() if key == 'full_catalogue')
                    rows.append(dict(answer, family_id=family['family_id'], split=family['split'], slot=slot,
                        session_id=spec['session_id'], event_id=event['event_id'], timestamp_s=t,
                        absolute_public_time_s=absolute, local_state_evaluator_only=int(state),
                        local_destination_evaluator_only=int(destination)))
        print('Static catalogue control family complete', family['family_id'], flush=True)
    write(out/'utility_rows.json.gz', {'rows': rows}); write(out/'cost_rows.json', costs)
    utility, cost = {}, {}
    for split in pilot_protocol['splits']:
        subset = [r for r in rows if r['split'] == split]
        utility[split] = {answer: {purpose.value: summarize([r for r in subset if r['purpose'] == purpose.value], answer)
                                for purpose in QueryPurpose} for answer in ANSWERS}
        cost[split] = {method: {'requests': sum(r['requests'] for r in costs if r['split'] == split and r['method'] == method),
            'request_bytes': sum(r['request_bytes'] for r in costs if r['split'] == split and r['method'] == method),
            'reply_bytes': sum(r['reply_bytes'] for r in costs if r['split'] == split and r['method'] == method)}
                       for method in ('full_catalogue', *METHODS)}
        for entry in cost[split].values(): entry['total_bytes'] = entry['request_bytes']+entry['reply_bytes']
    write(out/'results.json', {'schema': 'jisa-static-catalogue-control-readout-v1', 'status': protocol['status'],
        'protocol_sha256': sha(out/'protocol.json'), 'source_pilot_unchanged': all(sha(pilot/name) == digest for name, digest in protocol['pilot_sha256'].items()),
        'file_sha256': {name: sha(out/name) for name in ('utility_rows.json.gz', 'cost_rows.json', 'bulk_request.json', 'bulk_response.json')},
        'poi_count': local.n, 'catalogue_version': bulk['catalogue_version'], 'utility': utility, 'cost': cost,
        'family_count': len(data['families']), 'sessions': sum(len(f['evaluator_only']['sessions']) for f in data['families']),
        'local_query_count': len(rows), 'protected_draws_generated': 0, 'timing_measured': False,
        'claim': 'With fixed public region/version, bulk access and full static catalogue, local-only exactly recovers every nonempty four-purpose reference with one public transfer per epoch. This gate does not establish real-provider feasibility or dynamic freshness.'})
    write(out/'runtime.json', {'python': sys.version, 'numpy': np.__version__, 'timing_measured': False})
    print('Saved static application gate', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT); parser.add_argument('--pilot', type=Path, default=PILOT)
    args = parser.parse_args(); run(args.out, args.pilot)
