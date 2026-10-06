"""Independent native/source integrity check, without defense or test score.

Reparse the archived 1Hz SUMO FCD instead of trusting generated trajectory
dicts. Check group exclusions, split/public-speed balance and counterfactual
clock provenance. This does not evaluate or inspect protection performance.
"""
from pathlib import Path
import argparse
import collections
import gzip
import hashlib
import json
import tempfile
import xml.etree.ElementTree as ET

import sumolib

from data.scenario_suite.mobility import parse_fcd
from data.lane_states import build_lane_states
from evaluation.scenario_metrics import PoiService

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'artifacts/datasets/qplanner_fresh_native_20261006_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def public_overlap(a, b):
    a, b = set(a), set(b)
    return len(a & b)/len(a | b) if a or b else 1.


def archived_native(entries, temporary):
    restored = {}
    for name, meta in entries.items():
        source = ROOT/meta['compressed_path']
        assert sha(source) == meta['compressed_sha256'], ('archive_changed', source)
        content = gzip.decompress(source.read_bytes())
        assert hashlib.sha256(content).hexdigest() == meta['native_sha256'], ('native_source_changed', source)
        target = temporary/name
        target.write_bytes(content)
        restored[name] = target
    return restored


def verify(output=DEFAULT):
    output = Path(output).resolve()
    protocol = json.loads((output/'protocol.json').read_text())
    manifest = json.loads((output/'manifest.json').read_text())
    assert sha(output/'protocol.json') == (output/'protocol.sha256').read_text().strip()
    assert sha(output/'protocol.json') == manifest['protocol_sha256']
    assert sha(output/'dataset.json.gz') == manifest['dataset_sha256']
    for name, expected in protocol['source_sha256'].items():
        assert sha(ROOT/name) == expected, ('current_source_changed', name)
        assert sha(output/'source_snapshot'/name) == expected, ('snapshot_changed', name)
    old_path = ROOT/'artifacts/datasets/future_controlled_20261005_v2/dataset.json.gz'
    assert sha(old_path) == protocol['old_dataset_sha256']
    old = load(old_path)
    old_public_path = ROOT/old['public_calibration_source']
    assert sha(old_public_path) == protocol['old_public_calibration_sha256']
    old_public = load(old_public_path)
    poi_path = ROOT/'artifacts/benchmarks/research_loop/resources.json'
    assert sha(poi_path) == protocol['public_poi_source_sha256']
    source_pois = json.loads(poi_path.read_text())['pois_used']
    source_poi_count = len(source_pois)
    assert source_poi_count == 419
    data = load(output/'dataset.json.gz')
    assert data['protocol_sha256'] == manifest['protocol_sha256']
    assert data['source_sha256'] == protocol['source_sha256']
    assert data['choice_key_exported'] is False and manifest['private_choice_key_exported'] is False
    assert manifest['protection_or_attacker_scores_inspected'] is False
    assert data['families'] and len(data['families']) == data['family_count'] == manifest['family_count'] == 60
    assert data['session_count'] == manifest['session_count'] == 480
    assigned = {r['family_id']: r for r in protocol['assignments']}
    assert len(assigned) == len(protocol['assignments']) == 60
    assert {f['family_id'] for f in data['families']} == set(assigned)
    balances = collections.Counter((f['split'], f['public_speed_m_s']) for f in data['families'])
    assert balances == {('train', 6.): 12, ('train', 8.): 12,
        ('selection', 6.): 6, ('selection', 8.): 6, ('test', 6.): 12, ('test', 8.): 12}
    assert not any(k in data for k in ('master_hex', 'private_key', 'rng_state', 'seed_state'))
    eligible = json.loads((output/'public_eligibility.json').read_text())
    assert eligible['retained_family_count'] == 60
    assert eligible['screened_fork_count'] == 60+len(eligible['rejections']) <= 800
    assert data['public_eligibility_rejections'] == eligible['rejections']
    attempt_ids = [f['public_attempt'] for f in data['families']]+[r['attempt'] for r in eligible['rejections']]
    assert len(attempt_ids) == len(set(attempt_ids)) == eligible['screened_fork_count']
    assert set(attempt_ids) == set(range(eligible['screened_fork_count']))
    native_checks = {'families': 0, 'sessions': 0, 'native_1Hz_fixes': 0,
        'public_counterfactual_calibration_trips': 0, 'native_source_archives': 0}
    with tempfile.TemporaryDirectory(prefix='qplanner-native-integrity-') as td:
        temporary = Path(td)
        net_meta = data['network']
        compressed_net = ROOT/net_meta['compressed_path']
        assert sha(compressed_net) == net_meta['compressed_sha256']
        net = temporary/'native.net.xml'
        net.write_bytes(gzip.decompress(compressed_net.read_bytes()))
        assert sha(net) == protocol['network_sha256'] == net_meta['native_sha256'] == old['network']['native_sha256']
        network = sumolib.net.readNet(str(net), withInternal=True)
        rn = build_lane_states(net, spacing_m=40.)
        public_pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')}
            for p in source_pois]
        service = PoiService(rn, public_pois, k=5, max_access_m=250.)
        assert len(service.pois) == 418 and len(service.excluded) == 1
        excluded = {network.getEdge(f['fork_edge']).getToNode().getID() for f in old['families']}
        assert excluded == set(eligible['excluded_old_public_junctions'])
        groups = set()
        routes = [route for c in old_public['public_calibrations'] for route in c['proposal']['routes']]
        calibrations = {c['family_id']: c for c in data['public_calibrations']}
        assert set(calibrations) == set(assigned)
        for family in data['families']:
            fid = family['family_id']
            a = assigned[fid]
            assert {k: family[k] for k in a} == a
            fork = network.getEdge(family['fork_edge'])
            junction = fork.getToNode().getID()
            assert family['public_junction_group'] == junction
            assert junction not in excluded and junction not in groups
            groups.add(junction)
            public = calibrations[fid]
            proposal = public['proposal']
            assert proposal['fork_edge'] == fork.getID() and proposal['public_junction_group'] == junction
            assert proposal['public_choices'] == family['public_context']['choices']
            for route in proposal['routes']:
                assert all(public_overlap(route, prior) < .8 for prior in routes)
            routes.extend(proposal['routes'])
            public_dir = temporary/'calibration'
            public_dir.mkdir(exist_ok=True)
            archived = archived_native(public['native_source_archives'], public_dir)
            public_traces = parse_fcd(archived['fcd.xml'], network)
            assert set(public_traces) == {f'calibration_s{speed}_b{branch}' for speed in (6, 8) for branch in (0, 1)}
            native_checks['public_counterfactual_calibration_trips'] += 4
            native_checks['native_source_archives'] += 3
            for speed in (6, 8):
                branch_traces = []
                for branch in (0, 1):
                    original_trace = public_traces[f'calibration_s{speed}_b{branch}']
                    origin = original_trace[0]['time_s']
                    branch_traces.append([{**p, 'time_s': p['time_s']-origin} for p in original_trace])
                before = [{int(p['time_s']) for p in trace if p['edge_id'] == fork.getID()
                    and p['lane_pos_m'] <= fork.getLength()-80.} for trace in branch_traces]
                shared = max(before[0] & before[1])
                inside = [{int(p['time_s']) for p in trace
                    if p['lane_id'] in proposal['public_choices'][branch]['via_lane_ids']}
                    for branch, trace in enumerate(branch_traces)]
                overlap = sorted(inside[0] & inside[1])
                visible = overlap[len(overlap)//2]
                clocks = public['clocks_by_speed'][str(speed)]
                assert clocks == {'shared_fork_t': shared, 'turn_visible_t': visible,
                    'internal_overlap_s': len(overlap), 'raw_shared_prefix_identical': True}
                prefixes = [[(p['time_s'], p['lat'], p['lon']) for p in trace if p['time_s'] <= shared]
                    for trace in branch_traces]
                assert prefixes[0] == prefixes[1]
            assert family['clocks'] == public['clocks_by_speed'][str(int(family['public_speed_m_s']))]
            actual_dir = temporary/'actual'
            actual_dir.mkdir(exist_ok=True)
            archived = archived_native(family['evaluator_only']['native_source_archives'], actual_dir)
            parsed = parse_fcd(archived['fcd.xml'], network)
            specifications = family['evaluator_only']['sessions']
            assert len(specifications) == 8
            assert set(parsed) == {s['session_id'] for s in specifications}
            native_checks['native_source_archives'] += 3
            xml = ET.parse(archived['routes.rou.xml']).getroot()
            xml_vehicles = {v.attrib['id']: v for v in xml.findall('vehicle')}
            for slot, spec in enumerate(specifications):
                assert spec['day'] == slot+1 and spec['depart_s'] == slot*1500.
                assert spec['public_speed_m_s'] == family['public_speed_m_s']
                assert spec['route_edges'] == proposal['routes'][spec['choice_index']]
                vehicle = xml_vehicles[spec['session_id']]
                assert vehicle.attrib['type'] == f'controlled-{int(family["public_speed_m_s"])}'
                assert vehicle.find('route').attrib['edges'].split() == spec['route_edges']
                assert float(vehicle.find('stop').attrib['until']) == spec['depart_s']+650.
                trace = parsed[spec['session_id']]
                origin = trace[0]['time_s']
                normalized = [{**p, 'time_s': p['time_s']-origin} for p in trace]
                assert normalized == data['traces'][spec['session_id']], ('GPS_not_native_FCD', fid, slot)
                by_time = {p['time_s']: p for p in normalized}
                assert len(by_time) == len(normalized)
                assert set(range(601)).issubset(by_time) and by_time[600.]['speed_m_s'] == 0.
                assert all(0 <= p['speed_m_s'] <= family['public_speed_m_s']+.02 for p in normalized)
                visible = family['clocks']['turn_visible_t']
                assert by_time[visible]['lane_id'] in proposal['public_choices'][spec['choice_index']]['via_lane_ids']
                native_checks['sessions'] += 1
                native_checks['native_1Hz_fixes'] += len(normalized)
            query = [data['traces'][s['session_id']] for s in specifications[6:]]
            cut = family['clocks']['shared_fork_t']
            assert [[(p['time_s'], p['lat'], p['lon']) for p in trace if p['time_s'] <= cut] for trace in query][0] == [
                (p['time_s'], p['lat'], p['lon']) for p in query[1] if p['time_s'] <= cut]
            assert sorted(s['choice_index'] for s in specifications[6:]) == [0, 1]
            native_checks['families'] += 1
    assert native_checks['families'] == 60 and native_checks['sessions'] == 480
    assert native_checks['native_source_archives'] == 360
    assert set(data['traces']) == {s['session_id'] for f in data['families'] for s in f['evaluator_only']['sessions']}
    return {'schema': 'fresh-qplanner-native-integrity-v1', 'passed': True,
        'dataset_sha256': manifest['dataset_sha256'], 'protocol_sha256': manifest['protocol_sha256'],
        'verifier_sha256': sha(Path(__file__)), 'checks': native_checks,
        'public_poi_source_count': source_poi_count, 'native_service_poi_count': len(service.pois),
        'native_service_public_access_exclusions': len(service.excluded),
        'distinct_new_junction_groups': len(groups),
        'old_sources_preserved': True, 'test_performance_inspected': False,
        'source_group_disjoint': True, 'same_map_same_generator': True,
        'cross_city_or_real_GPS_confirmation': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--validation-output', type=Path)
    args = parser.parse_args()
    receipt = verify(args.output)
    destination = args.validation_output or args.output/'validation.json'
    if destination.exists():
        existing = json.loads(destination.read_text())
        assert existing['dataset_sha256'] == receipt['dataset_sha256']
        assert existing['protocol_sha256'] == receipt['protocol_sha256']
        print('Integrity recheck passed; existing validation retained')
    else:
        with destination.open('x') as stream:
            stream.write(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
