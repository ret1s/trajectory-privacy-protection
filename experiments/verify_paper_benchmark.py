"""Independent arithmetic, raw-source and selection audit of paper-v2."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'artifacts/benchmarks/paper_benchmark/results.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a, b):
    assert math.isclose(a, b, abs_tol=1e-6, rel_tol=1e-9), (a, b)


def verify(path=DEFAULT):
    path = Path(path)
    assert sha(path) == path.with_suffix('.sha256').read_text().strip()
    data = json.loads(path.read_text())
    assert data['protocol_version'] == 'paper-v2'
    for name, expected in data['source_sha256'].items():
        assert sha(ROOT / name) == expected, f'stale source: {name}'
    samples = {(s['seed'], s['record_id']): s for s in data['dataset_records']}
    manifests = {m['seed']: m for m in data['manifests']}
    for sample in samples.values():
        t, fcd = sample['times'], sample['fcd']
        assert len(t) == len(fcd) == len(sample['points']) <= 12
        assert all(b - a >= 20 for a, b in zip(t, t[1:]))
        assert sample['points'] == [[p['lat'], p['lon']] for p in fcd]
        if sample['scenario'] == 'S1':
            assert len(t) == 1
        elif sample['scenario'] == 'S2':
            assert t[-1] - t[0] >= 120 and all(p['speed_m_s'] <= .05 for p in fcd)
            assert sample['checks']['max_radius_from_first_m'] <= 5
        elif sample['scenario'] == 'S3':
            assert t[-1] - t[0] >= 60 and all(p['speed_m_s'] > .05 for p in fcd)
        else:
            assert sample['scenario'] in {'S9', 'S10'} and len(t) == 12
    for m in manifests.values():
        roles = list(m['split'].values())
        assert len(roles) == 5
        assert all(set(a).isdisjoint(b) for i, a in enumerate(roles) for b in roles[i+1:])
        for selection in m['attacker_selection']:
            for values in selection['cases'].values():
                mae, hit = values['validation_mae'], values['validation_hit100']
                if mae:
                    assert values['mae_attack'] == min(mae, key=lambda n: (mae[n], n))
                    assert values['hit_attack'] == min(hit, key=lambda n: (-hit[n], n))
        for selection in m['defender_selection']:
            for option in selection['candidates']:
                s = option['validation_summary']
                near(option['min_case_recall'], min(r['poi_delivered_recall_at_k'] for r in s))
                near(option['macro_hit100'], np.mean([r['location_hit_100m'] for r in s]))
                assert len(s) == 5 and all(r['completed'] > 0 and r['failures'] == 0 for r in s)
            feasible = [o for o in selection['candidates'] if o['min_case_recall'] >= .9]
            assert selection['utility_feasible'] == bool(feasible)
            winner = min(feasible, key=lambda o: (o['macro_hit100'], -o['min_case_recall'], o['spec']['id'])) if feasible else min(selection['candidates'], key=lambda o: (-o['min_case_recall'], o['macro_hit100'], o['spec']['id']))
            assert winner['spec'] == selection['selected']
    groups, keys = defaultdict(list), set()
    events, queries = 0, 0
    for row in data['rows']:
        key = row['seed'], row['record_id'], row['method'], row['k']
        assert key not in keys
        keys.add(key)
        groups[row['scenario'], row['method'], row['k']].append(row)
        sample = samples[row['seed'], row['record_id']]
        manifest = manifests[row['seed']]
        assert sample['vehicle_id'] in manifest['split']['test']
        if row['method'] == 'br_selected':
            assert row['selected_spec'] == next(s['selected'] for s in manifest['defender_selection'] if s['k'] == row['k'])
        if row['status'] != 'ok':
            assert row['status'] in {'failed', 'not_applicable'} and row['error']
            continue
        public, metrics = row['public'], row['metrics']
        def check_keys(value):
            if isinstance(value, dict):
                assert not {'truth', 'truth_input', 'hidden_target', 'seed', 'anchors', 'real_candidate_ids', 'vehicle_id'} & value.keys()
                for v in value.values():
                    check_keys(v)
            elif isinstance(value, list):
                for v in value:
                    check_keys(v)
        check_keys(public)
        assert row['truth'] == sample['points'] and row['truth_input'] == sample['points_input']
        assert [e['timestamp_s'] for e in public['events']] == sample['times']
        scale = math.pi * 6371000 / 180
        def xy(points):
            return np.array([[lon * scale * math.cos(math.radians(manifest['projection_lat0'])), lat * scale] for lat, lon in points])
        truth = xy(row['truth'])
        predictions, hits = np.array(row['attack_xy']), np.array(row['hit_attack_xy'])
        if row['scenario'] in {'S9', 'S10'}:
            idx = 0 if row['scenario'] == 'S9' else -1
            truth = xy([sample['hidden_target']])
            predictions, hits = predictions[[idx]], hits[[idx]]
        errors = np.linalg.norm(predictions - truth, axis=1)
        hit_errors = np.linalg.norm(hits - truth, axis=1)
        near(errors.mean(), metrics['location_mae_m'])
        near(np.median(errors), metrics['location_median_error_m'])
        near(np.quantile(errors, .9), metrics['location_p90_error_m'])
        assert np.allclose(errors, metrics['per_event_error_m'], atol=1e-6)
        assert np.allclose(hit_errors, metrics['hit_decoder_errors_m'], atol=1e-6)
        for radius in (50, 100, 200):
            near((hit_errors <= radius).mean(), metrics[f'location_hit_{radius}m'])
        selected = next(s for s in manifest['attacker_selection'] if s['spec'] == row['selected_spec'] and s['k'] == row['k'])['cases'][row['scenario']]
        assert row['attacker_selected'] == selected['mae_attack'] and row['hit_attacker_selected'] == selected['hit_attack']
        recalls, completed, extra = [], [], []
        for u in row['utility']['poi_rows']:
            assert len(u['returned']) == len(set(u['returned'])) <= 5
            if u['reference']:
                value = len(set(u['reference']) & set(u['returned'])) / len(u['reference'])
                near(value, u['recall']); recalls.append(value)
                complete = len(u['reference']) == len(u['returned'])
                assert u['complete'] == complete
                completed.append(complete)
                if complete:
                    assert u['extra_distance_m'] >= 0
                    extra.append(u['extra_distance_m'])
                else:
                    assert u['extra_distance_m'] is None
            else:
                assert u['recall'] is None
        for field, values in [('poi_recall_at_k', recalls), ('poi_complete_rate', completed), ('poi_extra_distance_m', extra)]:
            if values:
                near(np.mean(values), metrics[field])
            else:
                assert metrics[field] is None
        n = len(public['events'])
        near(metrics['coordinates_per_request'], np.mean([len(e['candidates']) for e in public['events']]))
        expected_bytes = len(json.dumps({'events': public['events'], 'query_categories': public['query_categories']}, separators=(',', ':')).encode())
        near(metrics['payload_json_bytes_per_request'], (expected_bytes + sum(u['response_json_bytes'] for u in row['utility']['poi_rows'])) / n)
        if row['method'].startswith('br_'):
            spent = min(n, 12) * .02 - (.01 if row['selected_spec'].get('anchor_mode') == 'private_reuse' else 0)
            near(metrics['privacy_budget_bound'], spent)
            assert metrics['privacy_budget_bound'] <= .24 + 1e-12
            if n > 1:
                near(metrics['directed_track_validity'], 1.)
        events += n; queries += len(row['utility']['poi_rows'])
    for summary in data['summary']:
        group = groups[summary['scenario'], summary['method'], summary['k']]
        good = [r for r in group if r['status'] == 'ok']
        assert summary['attempted'] == len(group) and summary['completed'] == len(good)
        assert summary['failures'] == sum(r['status'] == 'failed' for r in group)
        assert summary['not_applicable'] == sum(r['status'] == 'not_applicable' for r in group)
        for field in [f[:-2] for f in summary if f.endswith('_n')]:
            vals = [r['metrics'][field] for r in good if r['metrics'].get(field) is not None]
            assert summary[field + '_n'] == len(vals)
            if vals:
                near(np.mean(vals), summary[field])
        vals = [r['metrics']['poi_recall_at_k'] if r['status'] == 'ok' else 0. for r in group if r['status'] != 'not_applicable' and r['poi_reference_evaluable']]
        vals = [v for v in vals if v is not None]
        if vals:
            near(np.mean(vals), summary['poi_delivered_recall_at_k'])
    assert all(c['prefix_invariance'] for c in data['causal_checks'] if c['method'] != 'anotherme_adaptation')
    return {'artifact_sha256': sha(path), 'source_files': len(data['source_sha256']), 'records': len(samples),
            'rows': len(keys), 'events': events, 'category_queries': queries, 'summary_rows': len(data['summary']),
            'status': dict(Counter(r['status'] for r in data['rows'])),
            'prefix_total': len(data['causal_checks']), 'prefix_pass': sum(c['prefix_invariance'] for c in data['causal_checks'])}


def verify_raw(path):
    from data.sumo_demo import parse_fcd, load_sumo_road_network, _existing_default_osm
    from evaluation.scenario_metrics import read_osm_pois
    data = json.loads(Path(path).read_text())
    matched, endpoints, queries, transitions = 0, 0, 0, 0
    for m in data['manifests']:
        root = ROOT / 'cache/paper_benchmark' / f"seed_{m['seed']}"
        traces = parse_fcd(root / 'beijing_smoke.fcd.xml')
        completed = {v.attrib['id'] for v in ET.parse(root / 'beijing_smoke.vehroute.xml').getroot().findall('vehicle') if float(v.attrib.get('arrival', -1)) >= 0}
        for sample in data['dataset_records']:
            if sample['seed'] != m['seed']:
                continue
            trace = traces[sample['vehicle_id']]
            lookup = {p.timestamp_s: asdict(p) for p in trace}
            for fcd in sample['fcd']:
                assert lookup[fcd['timestamp_s']] == fcd
                matched += 1
            if sample['scenario'] in {'S9', 'S10'}:
                index = 0 if sample['scenario'] == 'S9' else -1
                assert sample['vehicle_id'] in completed
                assert asdict(trace[index]) == sample['hidden_target_fcd']
                assert abs(sample['times'][index] - trace[index].timestamp_s) >= 60
                endpoints += 1
        source = _existing_default_osm()
        assert sha(source) == m['provenance']['sha256']['osm']
        rn = load_sumo_road_network(root / 'beijing_smoke.net.xml')
        # Independently construct directed free-flow travel times, instead of
        # trusting the stored validity metric or importing its implementation.
        times_graph = nx.DiGraph()
        times_graph.add_nodes_from(rn.graph.nodes)
        for u, v, edge in rn.graph.edges(data=True):
            seconds = edge['length'] / min(25., edge['speed'])
            if not times_graph.has_edge(u, v) or seconds < times_graph[u][v]['weight']:
                times_graph.add_edge(u, v, weight=seconds)
        reached_cache = {}
        pois = []
        for p in read_osm_pois(source, m['config']['bbox']):
            vertex, offset = rn.nearest(p['lat'], p['lon'])
            if offset <= 250:
                pois.append({**p, 'node': rn.node_ids[vertex]})
        assert len(pois) == m['pois']['accepted']
        cache = {}
        def distances(point):
            node = rn.node_ids[rn.nearest(*point)[0]]
            if node not in cache:
                lengths = nx.single_source_dijkstra_path_length(rn.graph, node, weight='length')
                cache[node] = {p['id']: lengths[p['node']] for p in pois if p['node'] in lengths}
            return cache[node]
        def order(point, category):
            d = distances(point)
            return sorted([p['id'] for p in pois if p['category'] == category and p['id'] in d], key=lambda pid: (d[pid], pid))
        for row in data['rows']:
            if row['seed'] != m['seed'] or row['status'] != 'ok':
                continue
            if row['method'].startswith('br_'):
                previous = {}
                for event in row['public']['events']:
                    for c in event['candidates']:
                        node = rn.node_ids[rn.nearest(c['lat'], c['lon'])[0]]
                        if c['candidate_id'] in previous:
                            old, old_time = previous[c['candidate_id']]
                            key = old, event['timestamp_s'] - old_time
                            if key not in reached_cache:
                                reached_cache[key] = nx.single_source_dijkstra_path_length(times_graph, old, cutoff=key[1], weight='weight')
                            assert node in reached_cache[key]
                            transitions += 1
                        previous[c['candidate_id']] = node, event['timestamp_s']
            n = len(row['truth'])
            for j, u in enumerate(row['utility']['poi_rows']):
                truth = row['truth'][j % n]; event = row['public']['events'][j % n]
                category = u['category']; reference = order(truth, category)
                replies = [order((c['lat'], c['lon']), category)[:5] for c in event['candidates']]
                union = set().union(*map(set, replies))
                returned = [pid for pid in reference if pid in union][:5]
                assert u['reference'] == reference[:5] and u['returned'] == returned
                assert u['response_json_bytes'] == len(json.dumps(replies, separators=(',', ':')).encode())
                if u.get('complete'):
                    d = distances(truth)
                    near(u['extra_distance_m'], max(0., np.mean([d[p] for p in returned]) - np.mean([d[p] for p in reference[:5]])))
                queries += 1
    return {'raw_fcd_points': matched, 'hidden_endpoint_labels': endpoints, 'poi_queries_from_osm': queries,
            'br_directed_transitions_recomputed': transitions}


def deterministic_view(data):
    """Exclude measured clocks and machine/XML-header provenance, not outcomes."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if '_ms' not in k and k not in {'provenance', 'runtime', 'source_sha256'}}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    return strip(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=Path, default=DEFAULT)
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    result = verify(args.path)
    if args.raw:
        result.update(verify_raw(args.path))
    if args.compare:
        assert deterministic_view(json.loads(args.path.read_text())) == deterministic_view(json.loads(args.compare.read_text()))
        result['deterministic_replay'] = True
    print(json.dumps(result, indent=2))
