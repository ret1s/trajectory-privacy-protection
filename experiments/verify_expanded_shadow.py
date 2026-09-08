"""Independent auxiliary lineage, forest refit, inference, selection and replay QA."""
from collections import defaultdict
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesRegressor

from core.demo_protocol import TrajectoryPoint
from data.scenario_store import ScenarioStore
from evaluation.lane_travel import matrix
from experiments.run_expanded_shadow import ROOT, OUTPUT, OLD, DATA, DB, RELEASE, METHODS, setup, generate
from experiments.run_service_cover import read, write, sha
from experiments.run_contextual_lane import estimators
from experiments.verify_service_cover import shadow_features, near
from experiments.verify_prior_factors import loss_oracle
from experiments.rng_util import rng_from_key


def empirical(model, q, prefix):
    x, y = np.asarray(model['x']), np.asarray(model['y'])
    mean, scale = np.asarray(model['mean']), np.asarray(model['scale'])
    order = np.argsort(cdist((q - mean) / scale, (x - mean) / scale), axis=1, kind='stable')
    result = {prefix + f'knn_{k}': y[order[:, :k]].mean(axis=1) for k in (1, 5, 15)}
    for k in (15, 45):
        decisions = [loss_oracle(y[ids]) for ids in order[:, :k]]
        for kind in ('mae_action', 'hit_action'):
            result[prefix + f'shadow_loss_{kind}_{k}'] = np.asarray([v[kind] for v in decisions])
        if k == 45:
            result[prefix + 'shadow_mean_45'] = np.asarray([v['mean'] for v in decisions])
    if prefix == '':
        for k in (1, 5, 15):
            result[f'shadow_knn_{k}'] = result.pop(f'knn_{k}')
    return result


def independent_choices(ss):
    choices, global_choices = {}, {}
    for s in ss:
        choices[s['method'] + '/' + s['case_id']] = {
            'mae': sorted(s['mae_by_attack'], key=lambda a: (s['mae_by_attack'][a], a))[0],
            'hit': sorted(s['hit_by_attack'], key=lambda a: (-s['hit_by_attack'][a], a))[0]}
    for method in METHODS:
        group = [s for s in ss if s['method'] == method]
        common = set.intersection(*(set(s['mae_by_attack']) for s in group))
        global_choices[method] = {
            'mae': sorted(common, key=lambda a: (sum(s['mae_by_attack'][a] for s in group), a))[0],
            'hit': sorted(common, key=lambda a: (-sum(s['hit_by_attack'][a] for s in group), a))[0]}
    return choices, global_choices


def verify(output=OUTPUT, receipt_path=None):
    OUTPUT = Path(output)
    data = {p: read(OUTPUT / f'{p}.json') for p in ('training', 'validation', 'development', 'holdout')}
    training = data['training']
    selection = read(OUTPUT / 'selection.json')
    _, rn, prior, models, provenance = setup('training')
    aux, registry = read(DATA / 'dataset.json'), read(DATA / 'registry.json')
    native = read(DATA / 'verification.json')
    with ScenarioStore(DB) as store:
        assert store.verify()['status'] == 'passed'
        assert store.head() == RELEASE and len(store.releases()) == 4
        assert store.releases()[:3] == registry['old_release_logs_preserved']
        assert store.export_bundle(RELEASE) == aux
        for p in ('training', 'holdout'):
            for record in data[p]['records']:
                expected = [{'timestamp_s': q['time_s'], 'lat': q['lat'], 'lon': q['lon']}
                            for q in store.device_view(RELEASE, record['record_id'])]
                assert expected == record['points']
    for item in (*data.values(), selection):
        for key in ('source_sha256', 'auxiliary_content_sha256', 'parent_training_sha256', 'tree_params'):
            assert item[key] == provenance[key]
        for p, h in item['source_sha256'].items():
            assert sha(ROOT / p) == h, p
    assert selection['training_sha256'] == sha(OUTPUT / 'training.json')
    assert selection['validation_sha256'] == sha(OUTPUT / 'validation.json')
    assert selection['defender_selection'] == 'unchanged_from_prior_factors; not_reselected'
    families = {p: {r['family_id'] for r in d['records']} for p, d in data.items()}
    assert [len(families[p]) for p in data] == [64, 2, 4, 16]
    assert all(not families[a] & families[b] for a in families for b in families if a < b)
    parent_train = read(OLD / 'training.json')
    lookup = {r['record_id']: r for r in training['records']}
    refits = {}
    counts = defaultdict(int)
    for method in METHODS:
        old = parent_train['shadow_models'][method + '/5']
        rows = [r for r in training['rows'] if r['method'] == method]
        model = training['shadow_models'][method]
        x = np.concatenate([np.asarray(old['x'])] + [shadow_features(r['public'], rn) for r in rows])
        y = np.concatenate([np.asarray(old['y'])] + [np.asarray([rn.point_xy(p['lat'], p['lon']) for p in lookup[r['record_id']]['points']]) for r in rows])
        near(x, model['x']); near(y, model['y'])
        near(x.mean(axis=0), model['mean'])
        scale = x.std(axis=0); scale[scale < 1e-12] = 1
        near(scale, model['scale'])
        assert model['provenance'] == {'parent_rows': 360, 'auxiliary_families': sorted(families['training']),
            'auxiliary_records': [r['record_id'] for r in rows], 'holdout_used': False}
        assert len(x) == 360 + native['support']['training']['observations']
        params = training['tree_params']
        assert params == dict(n_estimators=128, min_samples_leaf=5, max_depth=18, max_features=1,
                             bootstrap=False, random_state=20260908, n_jobs=1)
        assert type(params['max_features']) is int
        # Refit sklearn independently, including residual target construction.
        center = x[:, :10].reshape(-1, 5, 2).mean(axis=1) * 1000
        z = (x - model['mean']) / model['scale']
        info = training['forests'][method]
        assert sha(OUTPUT / info['path']) == info['sha256']
        with np.load(OUTPUT / info['path'], allow_pickle=False) as arrays:
            assert len(arrays.files) == 2 * 128 * 5
            for mode, target in (('direct', y), ('residual', y - center)):
                forest = ExtraTreesRegressor(**params).fit(z, target)
                refits[method, mode] = forest
                for i, estimator in enumerate(forest.estimators_):
                    tree = estimator.tree_
                    for key, value in (('left', tree.children_left), ('right', tree.children_right),
                                       ('feature', tree.feature), ('threshold', tree.threshold), ('value', tree.value[:, :, 0])):
                        assert np.array_equal(value, arrays[f'{mode}/{i}/{key}'])
                        counts['forest_arrays_refit'] += 1
        print(f'Refit and checked {method}', flush=True)
    transitions = defaultdict(set)
    for phase, evidence in data.items():
        if phase != 'training':
            assert evidence['training_sha256'] == sha(OUTPUT / 'training.json')
        if phase in ('development', 'holdout'):
            assert evidence['selection_sha256'] == sha(OUTPUT / 'selection.json')
        records = {r['record_id']: r for r in evidence['records']}
        parent = read(OLD / f'{phase}.json') if phase in ('validation', 'development') else None
        if parent:
            assert evidence['records'] == parent['records']
            assert evidence['parent_phase_sha256'] == sha(OLD / f'{phase}.json')
        old_rows = {(r['record_id'], r['method'], r['replicate']): r for r in parent['rows']} if parent else {}
        row_keys = [(r['record_id'], r['method'], r['replicate']) for r in evidence['rows']]
        reps = (1, 2, 3) if parent else (1,)
        assert len(row_keys) == len(set(row_keys))
        assert set(row_keys) == {(rid, name, rep) for rid in records for name in METHODS for rep in reps}
        anchors = {}
        for ri, row in enumerate(evidence['rows']):
            record = records[row['record_id']]
            points, public, method = record['points'], row['public'], row['method']
            assert all(row[k] == record[k] for k in ('family_id', 'case_id', 'split'))
            assert set(public) == {'mechanism', 'output_kind', 'public_parameters', 'events'}
            assert public['output_kind'] == 'dummy_only' and row['k'] == 5
            assert len(points) == len(public['events']) == len(row['evaluator_states'])
            near(row['spent_bound'], .01 + .02 * (len(points) - 1))
            anchor_key = (row['record_id'], row['replicate'])
            assert anchors.setdefault(anchor_key, row['evaluator_anchors']) == row['evaluator_anchors']
            for t, (p, event, states) in enumerate(zip(points, public['events'], row['evaluator_states'])):
                assert set(event) == {'event_id', 'timestamp_s', 'candidates'}
                assert event['timestamp_s'] == p['timestamp_s'] and len(event['candidates']) == len(states) == 5
                for j, (state, candidate) in enumerate(zip(states, event['candidates'])):
                    assert candidate == {'candidate_id': f'candidate_{j:04d}', 'lat': rn.latlon(state)[0], 'lon': rn.latlon(state)[1]}
                if t:
                    for a, b in zip(row['evaluator_states'][t-1], states):
                        transitions[a].add((b, p['timestamp_s'] - points[t-1]['timestamp_s']))
                        counts['directed_transitions'] += 1
            q = shadow_features(public, rn)
            if parent:
                old = old_rows[row['record_id'], method, row['replicate']]
                for key, value in old.items():
                    if key == 'errors_by_attack':
                        assert all(row[key][a] == e for a, e in value.items())
                    else:
                        assert row[key] == value, (phase, key)
                assert len(row['errors_by_attack']) == len(old['errors_by_attack']) + 10
                counts['frozen_rows_unchanged'] += 1
            else:
                seed = int(rng_from_key(record['record_id'], schema='expanded-shadow-v1').integers(0, 2**31))
                assert row['rng_seed'] == seed
                # Predeclared deterministic replay subset: 8 train + 2 holdout families.
                if int(row['family_id'].split('-')[-1]) % 8 == 1:
                    pp = tuple(TrajectoryPoint(**p) for p in points)
                    replay = generate(method, pp, rn, models, seed)
                    for k in ('public', 'evaluator_states', 'evaluator_anchors', 'spent_bound'):
                        assert replay[k] == row[k]
                    if 'evaluator_objective' in row:
                        assert replay['evaluator_objective'] == row['evaluator_objective']
                    length = max(1, len(pp) // 2)
                    prefix = generate(method, pp[:length], rn, models, seed)
                    assert prefix['public']['events'] == public['events'][:length]
                    counts['replay_rows'] += 1; counts['strict_prefix_checks'] += int(length < len(pp))
            if phase != 'training':
                model = training['shadow_models'][method]
                expected = empirical(model, q, 'expanded_')
                z = (q - model['mean']) / model['scale']
                center = q[:, :10].reshape(-1, 5, 2).mean(axis=1) * 1000
                expected['expanded_tree_direct'] = refits[method, 'direct'].predict(z)
                expected['expanded_tree_residual'] = refits[method, 'residual'].predict(z) + center
                assert set(expected) == set(row['expanded_predictions']) and len(expected) == 10
                truth = np.asarray([rn.point_xy(p['lat'], p['lon']) for p in points])
                for a, pred in expected.items():
                    near(pred, row['expanded_predictions'][a])
                    near(np.linalg.norm(pred - truth, axis=1), row['errors_by_attack'][a])
                    counts['new_point_estimates'] += len(pred)
                # Check prefix feature extraction without future candidates.
                prefix_public = {**public, 'events': public['events'][:max(1, len(points)//2)]}
                near(shadow_features(prefix_public, rn), q[:len(prefix_public['events'])])
                if phase == 'holdout':
                    previous = estimators(public, rn, prior, 'S3')
                    previous.update(empirical(parent_train['shadow_models'][method + '/5'], q, ''))
                    assert set(previous) | set(expected) == set(row['errors_by_attack'])
                    for a, pred in previous.items():
                        near(np.linalg.norm(pred - truth, axis=1), row['errors_by_attack'][a])
                    counts['old_bank_holdout_rows'] += 1
                counts['scored_rows'] += 1
            counts['rows'] += 1; counts['events'] += len(points)
            if ri % 140 == 0:
                print(f'Checked {phase} {ri}/{len(evidence["rows"])}', flush=True)
        if phase != 'training':
            expected_groups = {(r['method'], r['case_id']) for r in evidence['rows']}
            assert len(evidence['summaries']) == len(expected_groups)
            assert {(s['method'], s['case_id']) for s in evidence['summaries']} == expected_groups
            for s in evidence['summaries']:
                rows = [r for r in evidence['rows'] if (r['method'], r['case_id']) == (s['method'], s['case_id'])]
                assert s['rows'] == len(rows) and s['families'] == len({r['family_id'] for r in rows})
                for a in rows[0]['errors_by_attack']:
                    errors = [np.asarray(r['errors_by_attack'][a]) for r in rows]
                    near(s['mae_by_attack'][a], np.mean([e.mean() for e in errors]))
                    near(s['hit_by_attack'][a], np.mean([(e <= 100).mean() for e in errors]))
                near(s['envelope_mae_m'], min(s['mae_by_attack'].values()))
                near(s['envelope_hit100'], max(s['hit_by_attack'].values()))
                old_attacks = [a for a in s['mae_by_attack'] if not a.startswith('expanded_')]
                assert s['envelope_mae_m'] <= min(s['mae_by_attack'][a] for a in old_attacks) + 1e-9
                assert s['envelope_hit100'] >= max(s['hit_by_attack'][a] for a in old_attacks) - 1e-12
                if phase != 'validation':
                    choice = selection['global_attackers'][s['method']] if phase == 'holdout' else selection['attackers'][s['method'] + '/' + s['case_id']]
                    assert s['selected_mae_attack'] == choice['mae'] and s['selected_hit_attack'] == choice['hit']
                    near(s['selected_mae_m'], s['mae_by_attack'][choice['mae']])
                    near(s['selected_hit100'], s['hit_by_attack'][choice['hit']])
            if phase == 'validation':
                cases, global_choices = independent_choices(evidence['summaries'])
                assert selection['attackers'] == cases and selection['global_attackers'] == global_choices
    adj = matrix(rn, time=True)
    for start, pairs in transitions.items():
        dd = dijkstra(adj, indices=start, directed=True, limit=max(dt for _, dt in pairs) + 1e-8)
        assert all(dd[end] <= dt + 1e-8 for end, dt in pairs)
    result = {'verified': True, **counts, 'forest_models': 14, 'training_families_total': 66,
        'auxiliary_holdout_families': 16, 'defender_reselected': False,
        'artifact_sha256': {p: sha(OUTPUT / f'{p}.json') for p in (*data, 'selection')},
        'auxiliary_dataset_sha256': sha(DATA / 'dataset.json'),
        'auxiliary_native_verification_sha256': sha(DATA / 'verification.json'),
        'source_sha256': sha(__file__),
        'limitations': 'empirical inference only; reused core development; auxiliary holdout differs from core cases'}
    write(Path(receipt_path) if receipt_path else OUTPUT / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT, help='Existing evidence directory to verify')
    parser.add_argument('--receipt', type=Path, help='Fresh receipt path; default is evidence/verification.json')
    args = parser.parse_args()
    verify(args.output, args.receipt)
