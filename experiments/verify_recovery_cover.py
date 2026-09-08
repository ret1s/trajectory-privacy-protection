"""Independent score, lineage, corridor and replay checks for the development cycle."""
import argparse
from collections import defaultdict
import json

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist

from benchmark.anchor_belief import AnchorBelief
from benchmark.engines.service_cover import ServiceCoverLaneDummy
from core.demo_protocol import TrajectoryPoint
from evaluation.lane_travel import matrix
from evaluation.research_protocol import utility_metrics
from experiments.run_contextual_lane import sha, estimators
from experiments.run_recovery_cover import prepare, generate, NEW, METHODS, PHASES, OUTPUT, OLD, ROOT, read
from experiments.verify_service_cover import shadow_features, near


def verify_diagnostic(rn, service, context, learned, flat, proto):
    p = read(OUTPUT/'diagnostic.json'); old = read(OLD/'confirmation.json')
    assert p['previous_confirmation_sha256'] == sha(OLD/'confirmation.json')
    for path, digest in p['source_sha256'].items(): assert sha(ROOT/path) == digest, path
    original = {r['record_id']: r for r in old['rows']
                if (r['method'], r['k'], r['replicate']) == ('service_cover', 5, 1)}
    records = {r['record_id']: r for r in old['records']}
    assert len(p['rows']) == sum(len(r['points']) for r in records.values()) == 227
    assert len({(r['record_id'], r['event']) for r in p['rows']}) == len(p['rows'])
    queries = 0
    for rid, record in records.items():
        rows = [r for r in p['rows'] if r['record_id'] == rid]
        assert [r['event'] for r in rows] == list(range(len(record['points'])))
        b, u = AnchorBelief(learned), AnchorBelief(flat)
        for row, x in zip(rows, record['points']):
            t = row['event']; o = original[rid]; anchor = o['evaluator_anchors'][t]
            w, v = b.update(anchor, x['timestamp_s']), u.update(anchor, x['timestamp_s'])
            truth_state, _ = rn.nearest(x['lat'], x['lon'])
            near(row['learned_mean_error_m'], np.linalg.norm(w @ learned.xy-rn.xy[truth_state]))
            near(row['uniform_mean_error_m'], np.linalg.norm(v @ flat.xy-rn.xy[truth_state]))
            near(row['anchor_error_m'], np.linalg.norm(rn.point_xy(*anchor)-rn.xy[truth_state]))
            weights = {'learned': w @ learned.poi_weights, 'uniform': v @ flat.poi_weights,
                       'anchor': context.reference_weights(anchor),
                       'truth_oracle': context.reference_weights((x['lat'], x['lon']))}
            assert row['outputs']['learned/reachable']['states'] == o['evaluator_states'][t]
            assert set(row['outputs']) == {f'{m}/{c}' for m in weights for c in ('reachable', 'free')}
            for name, out in row['outputs'].items():
                belief_kind, constraint = name.split('/')
                ids = out['states']; assert len(ids) == 5 and all(proto.viable[i] for i in ids)
                if t and constraint == 'reachable':
                    dt = x['timestamp_s']-record['points'][t-1]['timestamp_s']
                    assert all(v in proto.travel.reachable(a, dt) for a, v in zip(o['evaluator_states'][t-1], ids))
                covered = set(i for state in ids for category in context.query_indices(state) for i in category if i >= 0)
                near(out['objective'], np.asarray(weights[belief_kind])[list(covered)].sum())
                public = {'output_kind': 'dummy_only', 'events': [{'timestamp_s': x['timestamp_s'],
                    'candidates': [{'candidate_id': str(j), 'lat': rn.latlon(s)[0], 'lon': rn.latlon(s)[1]}
                                   for j, s in enumerate(ids)]}]}
                for category, value in zip(context.categories, out['category_recall']):
                    expected = service.evaluate(public, [(x['lat'], x['lon'])], category)['poi_rows'][0]['recall']
                    if expected is None: assert value is None
                    else: near(value, expected)
                    queries += 1
    for key, value in p['summary'].items():
        if key.endswith('error_m'):
            per_record = [np.mean([r[key] for r in p['rows'] if r['record_id'] == rid]) for rid in records]
        else:
            per_record = [np.mean([v for r in p['rows'] if r['record_id'] == rid
                                   for v in r['outputs'][key]['category_recall'] if v is not None]) for rid in records]
        near(value, np.mean(per_record))
    return {'events': len(p['rows']), 'category_queries': queries,
            'mean_error_reference': 'nearest_native_state_not_raw_GPS',
            'greedy_optimality_claim': False}


def verify(replay=False):
    phases = {p: read(OUTPUT/f'{p}.json') for p in PHASES}
    selection = read(OUTPUT/'selection.json')
    records, rn, service, context, prior, learned, flat, provenance, _ = prepare('development')
    assert selection['validation_sha256'] == sha(OUTPUT/'validation.json')
    assert selection['training_sha256'] == sha(OUTPUT/'training.json')
    for p in (*phases.values(), selection):
        for key in ('source_sha256', 'dataset_content_sha256', 'uniform_model_sha256', 'diagnostic_sha256'):
            assert p[key] == provenance[key], key
    assert np.allclose(flat.prior, 1 / len(flat.prior), rtol=1e-12, atol=1e-15)
    near(flat.log_normalizers, learned.log_normalizers)
    for dt in (5., 20., 60., 120.):
        near(np.asarray(flat.transition(dt).sum(axis=1)).ravel(), np.ones(len(flat.xy)))
    training = phases['training']; lookup = {r['record_id']: r for r in training['records']}
    assert len(training['shadow_models']) == 6
    for name in METHODS:
        m = training['shadow_models'][f'{name}/5']
        group = [r for r in training['rows'] if r['method'] == name]
        x = np.concatenate([shadow_features(r['public'], rn) for r in group])
        y = np.concatenate([np.array([rn.point_xy(p['lat'], p['lon']) for p in lookup[r['record_id']]['points']]) for r in group])
        near(x, m['x']); near(y, m['y']); near(x.mean(axis=0), m['mean'])
        scale = x.std(axis=0); scale[scale < 1e-12] = 1; near(scale, m['scale'])
        assert m['provenance'] == {'families': ['family-101', 'family-102'], 'split': 'development_train',
                                  'row_keys': [[r['record_id'], r['replicate']] for r in group]}
    counts = dict(rows=0, scored_rows=0, events=0, replay_rows=0, prefix_checks=0, strict_prefix_checks=0,
                  category_queries=0, reused_rows=0, motion_transitions=0, corridor_events=0)
    transitions = defaultdict(set)
    reverse = matrix(rn).transpose().tocsr()
    proto = ServiceCoverLaneDummy(rn, belief_model=learned, k=5)
    viable = proto.viable_ids
    for phase, p in phases.items():
        original = read(OLD/f'{PHASES[phase]}.json')
        assert p['records'] == original['records']
        assert p['original_phase_sha256'] == sha(OLD/f'{PHASES[phase]}.json')
        assert p['scope'] == 'development_ablation_no_fresh_confirmation'
        assert p['historical_split_not_new_confirmation'] == (phase == 'development')
        if phase != 'training': assert p['training_sha256'] == sha(OUTPUT/'training.json')
        if phase == 'development': assert p['selection_sha256'] == sha(OUTPUT/'selection.json')
        lookup = {r['record_id']: r for r in p['records']}
        baseline = {(r['record_id'], r['replicate'], r['method']): r for r in original['rows'] if r['k'] == 5}
        expected = {(rid, name, rep) for rid in lookup for name in METHODS for rep in (1, 2, 3)}
        keys = [(r['record_id'], r['method'], r['replicate']) for r in p['rows']]
        assert len(keys) == len(set(keys)) and set(keys) == expected
        for n, row in enumerate(p['rows']):
            name, rep = row['method'], row['replicate']; r = lookup[row['record_id']]
            control = baseline[r['record_id'], rep, 'baseline']
            public, states = row['public'], row['evaluator_states']
            assert set(public) == {'mechanism', 'output_kind', 'public_parameters', 'events'}
            assert public['output_kind'] == 'dummy_only'
            assert row['k'] == 5 and row['rng_seed'] == control['rng_seed']
            assert row['evaluator_anchors'] == control['evaluator_anchors']
            assert all(row[key] == r[key] for key in ('family_id', 'split', 'case_id'))
            assert len(states) == len(r['points']) == len(public['events'])
            assert len(row['step_ms']) == len(states) and min(row['step_ms']) >= 0
            near(row['spent_bound'], .01 + .02 * (len(states)-1))
            for t, (x, event, ids) in enumerate(zip(r['points'], public['events'], states)):
                assert set(event) == {'event_id', 'timestamp_s', 'candidates'}
                assert len(ids) == len(event['candidates']) == 5
                assert [c['candidate_id'] for c in event['candidates']] == [f'candidate_{j:04d}' for j in range(5)]
                assert all(0 <= i < len(rn) and proto.viable[i] for i in ids)
                assert event['timestamp_s'] == x['timestamp_s']
                for i, c in zip(ids, event['candidates']):
                    assert set(c) == {'candidate_id', 'lat', 'lon'}
                    assert rn.latlon(i) == (c['lat'], c['lon'])
                if t:
                    dt = x['timestamp_s'] - r['points'][t-1]['timestamp_s']
                    for u, v in zip(states[t-1], ids):
                        transitions[u, dt].add(v); counts['motion_transitions'] += 1
            if name not in NEW:
                assert row == baseline[r['record_id'], rep, name]
                counts['reused_rows'] += 1
            else:
                params = public['public_parameters']
                kind, slack = NEW[name]
                assert params['prior_kind'] == kind and params['corridor_slack_m'] == slack
                assert not set(params) & {'offset_m', 'temperature_m', 'route_weight', 'coverage_weight', 'center_mode'}
                model = flat if kind == 'uniform_public_cells' else learned
                assert params['belief_model_sha256'] == model.sha256
                b = AnchorBelief(model)
                for t, (x, ids, evidence) in enumerate(zip(r['points'], states, row['evaluator_objective'])):
                    weights = b.update(row['evaluator_anchors'][t], x['timestamp_s']) @ model.poi_weights
                    covered = set(i for state in ids for row_ids in context.query_indices(state) for i in row_ids if i >= 0)
                    near(sum(evidence['greedy_gains']), np.asarray(weights)[list(covered)].sum())
                    near(evidence['value'], sum(evidence['greedy_gains']))
                    assert len(evidence['greedy_gains']) == 5 and min(evidence['greedy_gains']) >= 0
                    if slack is not None:
                        anchor_xy = rn.point_xy(*row['evaluator_anchors'][t])
                        goal = viable[np.argmin(np.linalg.norm(rn.xy[viable]-anchor_xy, axis=1))]
                        potential = dijkstra(reverse, directed=True, indices=goal)
                        near(evidence['selected_goal_distance_m'], potential[ids])
                        if t:
                            dt = x['timestamp_s']-r['points'][t-1]['timestamp_s']
                            groups = [np.array(sorted(i for i in proto.travel.reachable(s, dt) if proto.viable[i])) for s in states[t-1]]
                        else: groups = [viable]*5
                        lower = [float(potential[g].min()) for g in groups]
                        near(evidence['minimum_goal_distance_m'], lower)
                        assert evidence['reachable_counts'] == list(map(len, groups))
                        assert evidence['corridor_counts'] == [int(np.sum(potential[g] <= m+slack)) for g, m in zip(groups, lower)]
                        assert np.all(potential[ids] <= np.asarray(lower)+slack)
                        counts['corridor_events'] += 1
                if replay and rep == 1:
                    points = tuple(TrajectoryPoint(**x) for x in r['points'])
                    again = generate(name, points, rn, learned, flat, row['rng_seed'])
                    for field in ('public', 'evaluator_states', 'evaluator_anchors', 'evaluator_objective', 'spent_bound'):
                        assert again[field] == row[field], (phase, n, field)
                    short = generate(name, points[:3], rn, learned, flat, row['rng_seed'])
                    assert short['public']['events'] == public['events'][:3]
                    counts['replay_rows'] += 1; counts['prefix_checks'] += 1
                    counts['strict_prefix_checks'] += int(len(points) > 3)
            if phase != 'training':
                truth = np.array([rn.point_xy(x['lat'], x['lon']) for x in r['points']])
                predictions = estimators(public, rn, prior, r['scenario'])
                m = training['shadow_models'][f'{name}/5']; mean, scale = np.array(m['mean']), np.array(m['scale'])
                distance = cdist((shadow_features(public, rn)-mean)/scale, (np.array(m['x'])-mean)/scale)
                order = np.argsort(distance, axis=1, kind='stable')
                for k in (1, 5, 15): predictions[f'shadow_knn_{k}'] = np.asarray(m['y'])[order[:, :k]].mean(axis=1)
                assert predictions.keys() == row['errors_by_attack'].keys()
                for a, xy in predictions.items(): near(row['errors_by_attack'][a], np.linalg.norm(xy-truth, axis=1))
                assert utility_metrics(service, public, [(x['lat'], x['lon']) for x in r['points']]) == row['utility']
                counts['scored_rows'] += 1; counts['category_queries'] += len(row['utility']['poi_rows'])
            counts['rows'] += 1; counts['events'] += len(states)
            if (n+1) % 108 == 0: print(f'{phase}: {n+1}/{len(p["rows"])} verified', flush=True)
        if phase == 'training': continue
        assert len(p['summaries']) == 54
        assert {(s['method'], s['case_id']) for s in p['summaries']} == {
            (name, f'S{s}.{c}') for name in METHODS for s in (1, 2, 3) for c in 'ABC'}
        for s in p['summaries']:
            g = [r for r in p['rows'] if (r['method'], r['case_id']) == (s['method'], s['case_id'])]
            assert len(g) == s['families']*3 and s['families'] == (4 if phase == 'development' else 2)
            near(s['poi_recall'], np.mean([r['utility']['poi_recall_at_5'] for r in g]))
            near(s['poi_complete'], np.mean([r['utility']['poi_complete_rate'] for r in g]))
            for a in s['mae_by_attack']:
                near(s['mae_by_attack'][a], np.mean([np.mean(r['errors_by_attack'][a]) for r in g]))
                near(s['hit_by_attack'][a], np.mean([np.mean(np.asarray(r['errors_by_attack'][a]) <= 100) for r in g]))
            near(s['audit_min_mae_m'], min(s['mae_by_attack'].values()))
            near(s['audit_max_hit100'], max(s['hit_by_attack'].values()))
            assert set(s['by_category']) == set(context.categories)
            for c, value in s['by_category'].items():
                qs = [[q for q in r['utility']['poi_rows'] if q['category'] == c] for r in g]
                rs = [[q['recall'] for q in row if q['recall'] is not None] for row in qs]
                near(value['recall'], np.mean([np.mean(v) for v in rs if v]))
                assert value['valid_query_n'] == sum(map(len, rs))
                assert value['empty_reference_n'] == sum(map(len, qs))-sum(map(len, rs))
            a = selection['attackers'][f'{s["method"]}/{s["case_id"]}']
            if phase == 'validation':
                assert a == {'mae': min(s['mae_by_attack'], key=lambda k: (s['mae_by_attack'][k], k)),
                             'hit': min(s['hit_by_attack'], key=lambda k: (-s['hit_by_attack'][k], k))}
            else:
                assert s['selected_mae_attack'] == a['mae'] and s['selected_hit_attack'] == a['hit']
                near(s['selected_mae_m'], s['mae_by_attack'][a['mae']]); near(s['selected_hit100'], s['hit_by_attack'][a['hit']])
    choices = []
    for name in METHODS:
        g = [s for s in phases['validation']['summaries'] if s['method'] == name]
        choices.append({'method': name, 'min_case_recall': min(s['poi_recall'] for s in g),
                        'macro_hit100': float(np.mean([s['audit_max_hit100'] for s in g]))})
    eligible = [c for c in choices if round(c['min_case_recall'], 12) >= .9]
    chosen = min(eligible, key=lambda c: (round(c['macro_hit100'], 12), c['method'])) if eligible else min(
        choices, key=lambda c: (-round(c['min_case_recall'], 12), round(c['macro_hit100'], 12), c['method']))
    assert selection['method_selection'] == {'chosen': chosen, 'utility_feasible': bool(eligible), 'candidates': choices}
    travel = matrix(rn, time=True)
    for (u, dt), targets in transitions.items():
        dist = dijkstra(travel, directed=True, indices=u, limit=dt+1e-8)
        assert all(dist[v] <= dt+1e-8 for v in targets)
    diagnostic = verify_diagnostic(rn, service, context, learned, flat, proto)
    return {'verified': True, **counts, 'shadow_models': 6, 'replay_new_methods_replicate_1': replay,
        'diagnostic': diagnostic,
        'source_sha256': sha(ROOT/'experiments/verify_recovery_cover.py'),
        'artifacts': {p: sha(OUTPUT/f'{p}.json') for p in (*PHASES, 'selection', 'diagnostic')},
        'scope': 'development_only; no fresh confirmation; no optimal-adversary or all-scenario claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--replay', action='store_true')
    result = verify(parser.parse_args().replay)
    (OUTPUT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
