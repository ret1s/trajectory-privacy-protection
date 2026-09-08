"""Independent numerics/lineage/selection/replay audit of prior-factor evidence."""
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
from experiments.run_prior_factors import OUTPUT, PREVIOUS, ROOT, METHODS, CONTROLS, NEW, prepare, generate, read, sha
from experiments.run_contextual_lane import estimators
from experiments.verify_service_cover import shadow_features, near


def loss_oracle(support):
    """Enumerate finite MAE actions and radius-100 disk support intersections."""
    mean = np.mean(support, axis=0)
    base = np.concatenate([support, mean[None]])
    d = np.sqrt(np.square(base[:, None]-support[None]).sum(axis=2))
    mae = base[np.argmin(d.mean(axis=1))]
    left, right = np.triu_indices(len(support), k=1)
    diff = support[right]-support[left]
    length2 = np.square(diff).sum(axis=1)
    ok = (length2 > 0) & (length2 <= 40000.)
    middle = .5*(support[left[ok]]+support[right[ok]])
    vec = diff[ok]; length2 = length2[ok]
    normal = np.column_stack([-vec[:, 1], vec[:, 0]])
    offset = normal*np.sqrt(np.maximum(0, (10000.-length2/4)/length2))[:, None]
    actions = np.concatenate([base, middle+offset, middle-offset])
    dist = np.sqrt(np.square(actions[:, None]-support[None]).sum(axis=2))
    mass = np.mean(dist <= 100.+1e-7, axis=1)
    return {'mean': mean, 'mae_action': mae, 'hit_action': actions[np.argmax(mass)],
            'mae_risk': float(d.mean(axis=1).min()), 'mean_risk': float(d[-1].mean()),
            'hit_mass': float(mass.max()), 'mean_hit_mass': float(mass[len(support)])}


def verify():
    phases = {p: read(OUTPUT/f'{p}.json') for p in ('training', 'validation', 'development')}
    selected = read(OUTPUT/'selection.json')
    records, rn, service, prior, models, provenance, _ = prepare('development')
    for p in (*phases.values(), selected):
        for key in ('source_sha256', 'dataset_content_sha256', 'factor_model_sha256'):
            assert p[key] == provenance[key], key
        for path, h in p['source_sha256'].items(): assert sha(ROOT/path) == h, path
    assert selected['training_sha256'] == sha(OUTPUT/'training.json')
    assert selected['validation_sha256'] == sha(OUTPUT/'validation.json')
    cell_model = models['balanced_cover'].initial
    _, inv = np.unique(np.floor(rn.xy/cell_model.spacing_m).astype(int), axis=0, return_inverse=True)
    density_means = np.array([prior[inv == i].mean() for i in range(inv.max()+1)])
    near(cell_model.prior, density_means/density_means.sum())
    uniform = models['uniform_initial_learned_motion'].initial
    near(uniform.prior, np.ones(len(uniform.prior))/len(uniform.prior))
    near(models['mixed_cover'].initial.prior, .5*(cell_model.prior+uniform.prior))
    for model in models.values():
        for dt in (5., 20., 60., 120.):
            near(np.asarray(model.transition(dt).sum(axis=1)).ravel(), np.ones(len(model.prior)))
        near(model.predict(model.prior, 10000.), model.motion.prior)
    train = phases['training']; tr = {r['record_id']: r for r in train['records']}
    assert set(train['shadow_models']) == {f'{n}/5' for n in METHODS}
    for name in METHODS:
        rows = [r for r in train['rows'] if r['method'] == name]
        m = train['shadow_models'][f'{name}/5']
        x = np.concatenate([shadow_features(r['public'], rn) for r in rows])
        y = np.concatenate([np.array([rn.point_xy(p['lat'], p['lon']) for p in tr[r['record_id']]['points']]) for r in rows])
        near(x, m['x']); near(y, m['y']); near(x.mean(axis=0), m['mean'])
        scale = x.std(axis=0); scale[scale < 1e-12] = 1
        near(scale, m['scale'])
        assert m['provenance'] == {'families': ['family-101', 'family-102'], 'split': 'development_train',
                                  'row_keys': [[r['record_id'], r['replicate']] for r in rows]}
    counts = defaultdict(int); transitions = defaultdict(set)
    proto = ServiceCoverLaneDummy(rn, belief_model=models['balanced_cover'], k=5)
    full_errors = {}
    for phase, data in phases.items():
        old = read(PREVIOUS/f'{phase}.json')
        assert data['records'] == old['records']
        assert data['previous_phase_sha256'] == sha(PREVIOUS/f'{phase}.json')
        assert data['scope'] == 'development_only_no_fresh_confirmation'
        assert data['role'] == {'training': 'training', 'validation': 'development_validation', 'development': 'reused_development'}[phase]
        if phase != 'training': assert data['training_sha256'] == sha(OUTPUT/'training.json')
        if phase == 'development': assert data['selection_sha256'] == sha(OUTPUT/'selection.json')
        lookup = {r['record_id']: r for r in data['records']}
        previous = {(r['record_id'], r['replicate'], r['method']): r for r in old['rows']}
        keys = [(r['record_id'], r['replicate'], r['method']) for r in data['rows']]
        assert len(keys) == len(set(keys))
        assert set(keys) == {(r, k, n) for r in lookup for k in (1, 2, 3) for n in METHODS}
        for row in data['rows']:
            rid, rep, name = row['record_id'], row['replicate'], row['method']
            rec = lookup[rid]; points = rec['points']; public = row['public']; states = row['evaluator_states']
            control = previous[rid, rep, 'baseline']
            assert row['rng_seed'] == control['rng_seed'] and row['k'] == 5
            assert row['evaluator_anchors'] == control['evaluator_anchors']
            assert all(row[k] == rec[k] for k in ('case_id', 'split', 'family_id'))
            assert len(states) == len(points) == len(public['events'])
            assert set(public) == {'mechanism', 'output_kind', 'public_parameters', 'events'}
            assert public['output_kind'] == 'dummy_only'
            near(row['spent_bound'], .01+.02*(len(points)-1))
            if name in NEW:
                m = models[name]; params = public['public_parameters']
                assert params['belief_model_sha256'] == m.sha256
                assert params['initial_prior_model_sha256'] == m.initial.sha256
                assert params['motion_prior_model_sha256'] == m.motion.sha256
                assert not set(params) & {'offset_m', 'temperature_m', 'route_weight', 'coverage_weight', 'center_mode'}
                matching_initial = {'learned_initial_uniform_motion': 'service_cover',
                                    'uniform_initial_learned_motion': 'uniform_cover'}.get(name)
                if matching_initial:
                    assert public['events'][0] == previous[rid, rep, matching_initial]['public']['events'][0]
                    counts['first_event_motion_invariance'] += 1
                belief = AnchorBelief(m)
                assert len(row['evaluator_objective']) == len(points)
                for t, (p, ss, evidence) in enumerate(zip(points, states, row['evaluator_objective'])):
                    weights = belief.update(row['evaluator_anchors'][t], p['timestamp_s']) @ m.poi_weights
                    covered = set(int(v) for s in ss for ids in m.context.query_indices(s) for v in ids if v >= 0)
                    near(evidence['value'], np.asarray(weights)[list(covered)].sum())
                    near(evidence['value'], sum(evidence['greedy_gains']))
                    assert len(evidence['greedy_gains']) == 5 and min(evidence['greedy_gains']) >= 0
                    counts['objective_events'] += 1
            if name in CONTROLS:
                for key, value in previous[rid, rep, name].items():
                    if key == 'errors_by_attack':
                        assert all(row[key][a] == e for a, e in value.items())
                    else: assert row[key] == value, (phase, name, key)
                counts['reused_rows'] += 1
            for i, (p, event, ss) in enumerate(zip(points, public['events'], states)):
                assert set(event) == {'event_id', 'timestamp_s', 'candidates'}
                assert event['timestamp_s'] == p['timestamp_s']
                assert len(ss) == len(event['candidates']) == 5
                for j, (s, c) in enumerate(zip(ss, event['candidates'])):
                    assert proto.viable[s] and set(c) == {'candidate_id', 'lat', 'lon'}
                    assert c['candidate_id'] == f'candidate_{j:04d}'
                    assert rn.latlon(s) == (c['lat'], c['lon'])
                if i:
                    for a, b in zip(states[i-1], ss):
                        transitions[a].add((b, p['timestamp_s']-points[i-1]['timestamp_s']))
                        counts['directed_transitions'] += 1
            if phase != 'training':
                truth = np.array([rn.point_xy(p['lat'], p['lon']) for p in points])
                expected = estimators(public, rn, prior, rec['scenario'])
                m = train['shadow_models'][f'{name}/5']
                x, y = np.array(m['x']), np.array(m['y'])
                q = shadow_features(public, rn)
                dist = cdist((q-np.array(m['mean']))/np.array(m['scale']),
                             (x-np.array(m['mean']))/np.array(m['scale']))
                order = np.argsort(dist, axis=1, kind='stable')
                for n in (1, 5, 15): expected[f'shadow_knn_{n}'] = y[order[:, :n]].mean(axis=1)
                for n in (15, 45):
                    ids = order[:, :min(n, len(y))]; audit = row['loss_audit'][str(n)]
                    assert ids.tolist() == audit['neighbors']
                    oracle = [loss_oracle(y[idx]) for idx in ids]
                    for key in ('mae_risk', 'mean_risk', 'hit_mass', 'mean_hit_mass'):
                        near(audit[key], [v[key] for v in oracle])
                    for key in ('mae_action', 'hit_action'):
                        attack = f'shadow_loss_{key}_{n}'
                        expected[attack] = np.array([v[key] for v in oracle])
                        near(row['loss_predictions'][attack], expected[attack])
                    if n == 45:
                        expected['shadow_mean_45'] = np.array([v['mean'] for v in oracle])
                        near(row['loss_predictions']['shadow_mean_45'], expected['shadow_mean_45'])
                    assert all(v['mae_risk'] <= v['mean_risk']+1e-9 and v['hit_mass'] >= v['mean_hit_mass'] for v in oracle)
                    counts['loss_decisions'] += len(ids)
                errors = {a: np.linalg.norm(p-truth, axis=1) for a, p in expected.items()}
                assert set(errors) == set(row['errors_by_attack'])
                for a, e in errors.items(): near(row['errors_by_attack'][a], e)
                full_errors[phase, rid, rep, name] = errors
                u = utility_metrics(service, public, [(p['lat'], p['lon']) for p in points])
                assert u == row['utility']
                counts['category_queries'] += 6*len(points)
                counts['scored_rows'] += 1
            if name in NEW and rep == 1:
                pp = tuple(TrajectoryPoint(**p) for p in points)
                replay = generate(name, pp, rn, models, row['rng_seed'])
                for key in ('public', 'evaluator_states', 'evaluator_anchors', 'evaluator_objective', 'spent_bound'):
                    assert replay[key] == row[key], (phase, rid, name, key)
                length = max(1, len(pp)//2)
                prefix = generate(name, pp[:length], rn, models, row['rng_seed'])
                assert prefix['public']['events'] == public['events'][:length]
                counts['replays'] += 1; counts['prefix_checks'] += 1
                counts['strict_prefix_checks'] += int(length < len(pp))
            counts['rows'] += 1; counts['events'] += len(points)
        if phase != 'training':
            assert len(data['summaries']) == 63
            assert {(s['method'], s['case_id']) for s in data['summaries']} == {
                (n, f'S{s}.{c}') for n in METHODS for s in (1, 2, 3) for c in 'ABC'}
            for s in data['summaries']:
                group = [r for r in data['rows'] if (r['method'], r['case_id']) == (s['method'], s['case_id'])]
                assert len(group) == (12 if phase == 'development' else 6)
                for attack in s['mae_by_attack']:
                    errors = [full_errors[phase, r['record_id'], r['replicate'], r['method']][attack] for r in group]
                    near(s['mae_by_attack'][attack], np.mean([e.mean() for e in errors]))
                    near(s['hit_by_attack'][attack], np.mean([(e <= 100).mean() for e in errors]))
                near(s['poi_recall'], np.mean([r['utility']['poi_recall_at_5'] for r in group]))
                near(s['poi_complete'], np.mean([r['utility']['poi_complete_rate'] for r in group]))
                near(s['audit_min_mae_m'], min(s['mae_by_attack'].values()))
                near(s['audit_max_hit100'], max(s['hit_by_attack'].values()))
                for c, v in s['by_category'].items():
                    lists = [[q['recall'] for q in r['utility']['poi_rows'] if q['category'] == c and q['recall'] is not None] for r in group]
                    near(v['recall'], np.mean([np.mean(vv) for vv in lists if vv]))
                    assert v['valid_query_n'] == sum(map(len, lists))
                    assert v['empty_reference_n'] == sum(q['category'] == c and q['recall'] is None for r in group for q in r['utility']['poi_rows'])
                a = selected['attackers'][f'{s["method"]}/{s["case_id"]}']
                if phase == 'validation':
                    assert a['mae'] == min(s['mae_by_attack'], key=lambda x: (s['mae_by_attack'][x], x))
                    assert a['hit'] == min(s['hit_by_attack'], key=lambda x: (-s['hit_by_attack'][x], x))
                else:
                    assert s['selected_mae_attack'] == a['mae'] and s['selected_hit_attack'] == a['hit']
                    near(s['selected_mae_m'], s['mae_by_attack'][a['mae']])
                    near(s['selected_hit100'], s['hit_by_attack'][a['hit']])
        print(f'Checked {phase}', flush=True)
    adj = matrix(rn, time=True)
    for start, pairs in transitions.items():
        dd = dijkstra(adj, indices=start, directed=True, limit=max(dt for _, dt in pairs)+1e-8)
        assert all(dd[end] <= dt+1e-8 for end, dt in pairs)
    candidates = []
    for n in METHODS:
        ss = [s for s in phases['validation']['summaries'] if s['method'] == n]
        candidates.append({'method': n, 'min_case_recall': min(s['poi_recall'] for s in ss),
                           'macro_hit100': float(np.mean([s['audit_max_hit100'] for s in ss]))})
    assert candidates == selected['method_selection']['candidates']
    feasible = [c for c in candidates if round(c['min_case_recall'], 12) >= .90]
    key = (lambda c: (round(c['macro_hit100'], 12), c['method'])) if feasible else (
           lambda c: (-round(c['min_case_recall'], 12), round(c['macro_hit100'], 12), c['method']))
    assert selected['method_selection']['chosen'] == min(feasible or candidates, key=key)
    assert selected['method_selection']['utility_feasible'] == bool(feasible)
    receipt = {'verified': True, **counts, 'shadow_models': len(METHODS),
               'source_sha256': sha(__file__), 'artifact_sha256': {p: sha(OUTPUT/f'{p}.json') for p in (*phases, 'selection')},
               'scope': 'development_only; empirical_shadow_not_exact_likelihood; no_new_privacy_theorem'}
    (OUTPUT/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__': verify()
