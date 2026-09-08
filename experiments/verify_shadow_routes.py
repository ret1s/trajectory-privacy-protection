"""Independent native SUMO, leakage and profile checks for auxiliary release."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

from data.scenario_store.store import validate_bundle, content_hash
from data.sumo_demo import _load_sumolib
from experiments.run_service_cover import read, write
from experiments.scenario_v2_checks import meters, outgoing, sha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/datasets/urban_shadow_v1/dataset.json'


def verify(path=DATA):
    path = Path(path)
    d = read(path)
    validate_bundle(d)
    assert d['purpose'] == 'auxiliary_shadow_training_and_holdout_only'
    assert len(d['families']) == 80 and len(d['records']) == 320 and len(d['traces']) == 160
    assert [f['seed'] for f in d['families']] == list(range(1001, 1081))
    assert {c['scenario'] for c in d['catalogue']} == {'AUX'}
    assert len({tuple(f['departure_cell']) for f in d['families']}) == 80
    for p, h in d['source_sha256'].items():
        assert sha(ROOT / p) == h, p
    assert sha(ROOT / d['network']['path']) == d['network']['sha256']
    old = read(ROOT / 'artifacts/datasets/urban_scenarios_v3/dataset.json')
    assert sha(ROOT / 'artifacts/datasets/urban_scenarios_v3/dataset.json') == d['base_dataset_sha256']
    network = _load_sumolib().net.readNet(str(ROOT / d['network']['path']), withInternal=True)
    geometry = _load_sumolib().geomhelper
    sessions = {s['session_id']: s for f in d['families'] for s in f['sessions']}
    planned_stops = [f['sessions'][1]['stops'][0] for f in d['families']]
    assert len({s['lane'] for s in planned_stops}) == 80
    stop_cells = {tuple(math.floor(v) for v in geometry.positionAtShapeOffset(
        network.getLane(s['lane']).getShape(), s['endPos'])) for s in planned_stops}
    assert len(stop_cells) == 80
    previous_sessions = {s['session_id']: s for f in old['families'] for s in f['sessions']}
    old_routes = {tuple(s['route_edges']) for s in previous_sessions.values()}
    new_routes = {tuple(s['route_edges']) for s in sessions.values()}
    assert not sessions.keys() & previous_sessions.keys()
    assert len(new_routes) == 160 and not new_routes & old_routes
    for key in ('person_id', 'device_id', 'physical_vehicle_id'):
        assert not {s[key] for s in sessions.values()} & {s[key] for s in previous_sessions.values()}
    old_edges = {e for route in old_routes for e in route}
    new_edges = {e for route in new_routes for e in route}
    checked = 0
    largest_jump = largest_lane_error = 0.
    for f in d['families']:
        expected_split = 'development_train' if f['seed'] <= 1064 else 'development_validation'
        assert f['split'] == expected_split
        assert f['role'] == ('auxiliary_training' if f['seed'] <= 1064 else 'auxiliary_holdout')
        assert 2500 <= f['base_route_length_m'] <= 4000
        first_edge = network.getEdge(f['sessions'][0]['route_edges'][0])
        lane = next(l for l in first_edge.getLanes() if l.allows('passenger'))
        xy = geometry.positionAtShapeOffset(lane.getShape(), geometry.polyLength(lane.getShape()) / 2)
        assert [math.floor(v / 120) for v in xy] == f['departure_cell']
        assert len(f['simulation_runs']) == 1
        for p, h in f['source_files'].items():
            assert sha(ROOT / p) == h
        files = {Path(p).name: ROOT / p for p in f['source_files']}
        assert all(s.attrib['parking'] == 'false' for s in ET.parse(files['routes.rou.xml']).getroot().iter('stop'))
        vehicles = {v.attrib['id']: v for v in ET.parse(files['vehicle_routes.xml']).getroot().findall('vehicle')}
        assert set(vehicles) == {s['session_id'] for s in f['sessions']}
        for sid, v in vehicles.items():
            assert dict(v.attrib) == f['actual_vehicles'][sid]
            route = v.find('route').attrib['edges'].split()
            assert route == sessions[sid]['route_edges']
            assert all(b in outgoing(network.getEdge(a)) for a, b in zip(route, route[1:]))
            assert float(v.attrib['arrival']) > d['traces'][sid][-1]['time_s']
            assert float(v.attrib['arrival']) - d['traces'][sid][-1]['time_s'] <= 1.01
        native_stops = ET.parse(files['stops.xml']).getroot().findall('stopinfo')
        returned = f['sessions'][1]
        stops = [s for s in native_stops if s.attrib['id'] == returned['session_id']]
        assert len(stops) == 3
        assert [float(s.attrib['ended']) - float(s.attrib['started']) for s in stops] == [180., 1., 180.]
        indices = Counter()
        for _, step in ET.iterparse(files['fcd.xml'], events=('end',)):
            if step.tag != 'timestep':
                continue
            t = float(step.attrib['time'])
            for v in step.findall('vehicle'):
                a = v.attrib
                sid = a['id']
                lane = network.getLane(a['lane'])
                actual = {'time_s': t, 'lat': float(a['y']), 'lon': float(a['x']),
                    'speed_m_s': float(a['speed']), 'lane_id': a['lane'],
                    'edge_id': lane.getEdge().getID(), 'lane_pos_m': float(a['pos']), 'angle_deg': float(a['angle'])}
                assert actual == d['traces'][sid][indices[sid]]
                assert lane.allows('passenger') and 0 <= actual['speed_m_s'] <= 8.01
                assert 0 <= actual['lane_pos_m'] <= lane.getLength() + .05
                px, py = network.convertLonLat2XY(actual['lon'], actual['lat'])
                ex, ey = geometry.positionAtShapeOffset(lane.getShape(), actual['lane_pos_m'])
                error = math.hypot(px - ex, py - ey)
                largest_lane_error = max(largest_lane_error, error)
                assert error <= 4.1, (sid, indices[sid], 'lane geometry', error)
                if indices[sid]:
                    prev = d['traces'][sid][indices[sid] - 1]
                    assert t - prev['time_s'] == 1
                    jump = meters(prev, actual)
                    largest_jump = max(largest_jump, jump)
                    assert jump <= 10.1, (sid, 'jump', jump)
                indices[sid] += 1
                checked += 1
            step.clear()
        assert dict(indices) == {sid: len(d['traces'][sid]) for sid in vehicles}
        print(f"Verified native family {f['seed']}", flush=True)
    assert checked == d['summary']['raw_fcd_samples']
    old_windows = set()
    for r in old['records']:
        for sid, indices in zip(r['session_ids'], r['observed_indices']):
            t = old['traces'][sid]
            old_windows.add(tuple((t[i]['lat'], t[i]['lon'], t[i]['time_s'] - t[indices[0]]['time_s']) for i in indices[:12]))
    train_xy, holdout_xy, seen = [], [], set()
    for r in d['records']:
        t = d['traces'][r['session_ids'][0]]
        indices = r['observed_indices'][0]
        profile = r['profile']
        window = tuple((t[i]['lat'], t[i]['lon'], t[i]['time_s'] - t[indices[0]]['time_s']) for i in indices)
        assert window not in old_windows and window not in seen
        seen.add(window)
        times = [p[2] for p in window]
        gaps = np.diff(times)
        if profile.startswith('cruise'):
            dt = int(profile[6:])
            assert np.all(gaps == dt) and 2 <= len(indices) <= 12
            expected_start = min(len(t) // 4, max(0, len(t) - 221)) if dt == 20 else 0
            assert indices == list(range(expected_start, min(len(t), expected_start + (221 if dt == 20 else 661)), dt))
        else:
            assert len(indices) == 12
            assert all(t[i]['speed_m_s'] <= .05 for i in indices)
            if profile == 'stop5':
                assert np.all(gaps == 5)
                assert max(meters(t[indices[0]], t[i]) for i in indices) <= 1
            else:
                assert profile == 'return5'
                assert np.all(np.r_[gaps[:5], gaps[6:]] == 5) and gaps[5] > 120
                assert any(p['speed_m_s'] > 1 for p in t[indices[5] + 1:indices[6]])
                assert max(meters(t[indices[0]], t[i]) for i in indices) <= 1
        target = train_xy if r['split'] == 'development_train' else holdout_xy
        target.extend(network.convertLonLat2XY(t[i]['lon'], t[i]['lat']) for i in indices)
    train_xy, holdout_xy = np.asarray(train_xy), np.asarray(holdout_xy)
    counts = {}
    for label, xy in (('training', train_xy), ('holdout', holdout_xy)):
        counts[label] = {'observations': len(xy), 'unique_xy': len(np.unique(xy, axis=0)),
            'occupied_120m_cells': len(np.unique(np.floor(xy / 120), axis=0))}
    distances = cKDTree(train_xy).query(holdout_xy)[0]
    result = {'verified': True, 'dataset_sha256': sha(path), 'dataset_content_sha256': content_hash(d),
        'source_sha256': sha(__file__), 'raw_points_compared': checked,
        'families': 80, 'training_families': 64, 'holdout_families': 16,
        'sessions': 160, 'profiles': 320, 'support': counts,
        'max_step_m': largest_jump, 'max_lane_center_error_m': largest_lane_error,
        'exact_old_route_overlap': 0, 'exact_old_window_overlap': 0,
        'unique_external_edges': len(new_edges), 'external_edges_shared_with_v3': len(new_edges & old_edges),
        'holdout_to_training_label_distance_m': {'median': float(np.median(distances)), 'p95': float(np.percentile(distances, 95))},
        'scope': 'auxiliary synthetic data, not calibrated population or fresh core-scenario confirmation'}
    write(path.parent / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA)
    verify(parser.parse_args().data)
