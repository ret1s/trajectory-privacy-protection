"""Locked defenders, expanded auxiliary training, nested inference-only audit."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import sklearn

from core.demo_protocol import TrajectoryPoint
from data.scenario_store import ScenarioStore
from evaluation.service_shadow import features, fit, predict as knn_predict
from evaluation.loss_aware_shadow import predict as loss_predict
from evaluation.expanded_shadow import fit_trees, predict as expanded_predict, TREE_PARAMS
from experiments.run_prior_factors import prepare, generate as factor_generate, NEW, METHODS, OUTPUT as OLD
from experiments.run_service_cover import generate as cover_generate, read, write, sha
from experiments.run_recovery_cover import generate as recovery_generate
from experiments.run_contextual_lane import estimators
from experiments.rng_util import rng_from_key

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'artifacts/benchmarks/expanded_shadow'
DATA = ROOT / 'artifacts/datasets/urban_shadow_v1'
DB = ROOT / 'artifacts/datasets/scenarios.sqlite3'
RELEASE = 'urban-shadow-v1'
SOURCE_PATHS = ('evaluation/expanded_shadow.py', 'experiments/run_expanded_shadow.py',
                'tests/test_expanded_shadow.py', 'requirements.txt',
                'thesis/notes/expanded_shadow_protocol.md', 'thesis/notes/shadow_route_quality_amendment.md')


def setup(phase):
    _, rn, service, prior, models, provenance, _ = prepare('training')
    registry = read(DATA / 'registry.json')
    dataset = read(DATA / 'dataset.json')
    verified = read(DATA / 'verification.json')
    assert verified['source_sha256'] == sha(ROOT / 'experiments/verify_shadow_routes.py')
    assert verified['dataset_sha256'] == sha(DATA / 'dataset.json')
    assert sklearn.__version__ == '1.7.1'
    for p, h in dataset['source_sha256'].items():
        assert sha(ROOT / p) == h
    records = []
    if phase in ('training', 'holdout'):
        split = 'development_train' if phase == 'training' else 'development_validation'
        with ScenarioStore(DB) as store:
            assert store._release(RELEASE)['content_sha256'] == registry['content_sha256']
            for r in dataset['records']:
                if r['split'] != split:
                    continue
                stream = store.device_view(RELEASE, r['record_id'])
                records.append({**{k: r[k] for k in ('record_id', 'family_id', 'case_id', 'profile', 'split')},
                    'points': [{'timestamp_s': x['time_s'], 'lat': x['lat'], 'lon': x['lon']} for x in stream]})
        assert len(records) == (256 if phase == 'training' else 64)
    pins = set(provenance['source_sha256']) | set(dataset['source_sha256']) | set(SOURCE_PATHS)
    provenance.update(source_sha256={p: sha(ROOT / p) for p in sorted(pins)},
        auxiliary_release=RELEASE, auxiliary_content_sha256=registry['content_sha256'],
        auxiliary_dataset_sha256=sha(DATA / 'dataset.json'), auxiliary_verification_sha256=sha(DATA / 'verification.json'),
        auxiliary_registry_sha256=sha(DATA / 'registry.json'),
        parent_training_sha256=sha(OLD / 'training.json'),
        parent_selection_sha256=sha(OLD / 'selection.json'),
        tree_params=TREE_PARAMS, sklearn_version=sklearn.__version__)
    return records, rn, prior, models, provenance


def generate(name, points, rn, models, seed):
    learned = models['learned_initial_uniform_motion'].initial
    uniform = models['uniform_initial_learned_motion'].initial
    if name in NEW:
        return factor_generate(name, points, rn, models, seed)
    if name == 'uniform_cover':
        return recovery_generate(name, points, rn, learned, uniform, seed)
    return cover_generate(name, points, rn, learned.context, learned, 5, seed)


def summaries(rows, selection=None, holdout=False):
    groups = defaultdict(list)
    for r in rows:
        groups[r['method'], r['case_id']].append(r)
    result = []
    for (method, case), group in sorted(groups.items()):
        attacks = sorted(group[0]['errors_by_attack'])
        assert all(set(r['errors_by_attack']) == set(attacks) for r in group)
        mae = {a: float(np.mean([np.mean(r['errors_by_attack'][a]) for r in group])) for a in attacks}
        hit = {a: float(np.mean([np.mean(np.asarray(r['errors_by_attack'][a]) <= 100) for r in group])) for a in attacks}
        s = {'method': method, 'case_id': case, 'rows': len(group),
             'families': len({r['family_id'] for r in group}), 'mae_by_attack': mae,
             'hit_by_attack': hit, 'envelope_mae_m': min(mae.values()), 'envelope_hit100': max(hit.values())}
        if selection:
            a = selection['global_attackers'][method] if holdout else selection['attackers'][method + '/' + case]
            s.update(selected_mae_attack=a['mae'], selected_hit_attack=a['hit'],
                     selected_mae_m=mae[a['mae']], selected_hit100=hit[a['hit']])
        result.append(s)
    return result


def choose(summaries):
    per_case = {s['method'] + '/' + s['case_id']: {
        'mae': min(s['mae_by_attack'], key=lambda a: (s['mae_by_attack'][a], a)),
        'hit': min(s['hit_by_attack'], key=lambda a: (-s['hit_by_attack'][a], a))} for s in summaries}
    global_choices = {}
    for method in METHODS:
        ss = [s for s in summaries if s['method'] == method]
        # Full-window-only attacks absent for S1 are not global candidates.
        attacks = set.intersection(*(set(s['mae_by_attack']) for s in ss))
        global_choices[method] = {
            'mae': min(attacks, key=lambda a: (np.mean([s['mae_by_attack'][a] for s in ss]), a)),
            'hit': min(attacks, key=lambda a: (-np.mean([s['hit_by_attack'][a] for s in ss]), a))}
    return {'attackers': per_case, 'global_attackers': global_choices}


def run(phase, output=OUTPUT):
    output = Path(output)
    if (output / f'{phase}.json').exists():
        raise FileExistsError('Use a fresh evidence destination')
    records, rn, prior, models, provenance = setup(phase)
    parent_train = read(OLD / 'training.json')
    training = read(output / 'training.json') if phase != 'training' else None
    selection = read(output / 'selection.json') if phase in ('development', 'holdout') else None
    for earlier in (training, selection):
        if earlier:
            for k in ('source_sha256', 'auxiliary_content_sha256', 'parent_training_sha256', 'tree_params'):
                assert earlier[k] == provenance[k], k
    if selection:
        assert selection['training_sha256'] == sha(output / 'training.json')
        assert selection['validation_sha256'] == sha(output / 'validation.json')
    forests = {}
    if training:
        for name, info in training['forests'].items():
            p = output / info['path']
            assert sha(p) == info['sha256']
            with np.load(p, allow_pickle=False) as arrays:
                forests[name] = {k: arrays[k] for k in arrays.files}
    rows = []
    if phase in ('training', 'holdout'):
        for record in records:
            points = tuple(TrajectoryPoint(**p) for p in record['points'])
            seed = int(rng_from_key(record['record_id'], schema='expanded-shadow-v1').integers(0, 2**31))
            anchors = None
            for name in METHODS:
                generated = generate(name, points, rn, models, seed)
                if anchors is not None:
                    assert generated['evaluator_anchors'] == anchors
                anchors = generated['evaluator_anchors']
                # Keep raw auxiliary truth in records, not inside public transcript.
                rows.append({**{k: record[k] for k in ('record_id', 'family_id', 'case_id', 'profile', 'split')},
                    'method': name, 'k': 5, 'replicate': 1, 'rng_seed': seed, **generated})
            print(f"Generated {record['record_id']}", flush=True)
    else:
        previous = read(OLD / f'{phase}.json')
        records = deepcopy(previous['records'])
        rows = deepcopy(previous['rows'])
        provenance['parent_phase_sha256'] = sha(OLD / f'{phase}.json')
    lookup = {r['record_id']: r for r in records}
    result = {'schema': 'expanded-shadow-v1', 'scope': 'inference_audit_frozen_defenders',
        'phase': phase, **provenance, 'records': records, 'rows': rows,
        'training_sha256': sha(output / 'training.json') if training else None,
        'selection_sha256': sha(output / 'selection.json') if selection else None}
    if phase == 'training':
        result['shadow_models'], result['forests'] = {}, {}
        output.mkdir(parents=True, exist_ok=True)
        for name in METHODS:
            old = parent_train['shadow_models'][name + '/5']
            group = [r for r in rows if r['method'] == name]
            x = np.concatenate([np.asarray(old['x'])] + [features(r['public'], rn) for r in group])
            y = np.concatenate([np.asarray(old['y'])] + [np.array([rn.point_xy(p['lat'], p['lon']) for p in lookup[r['record_id']]['points']]) for r in group])
            model = fit(x, y, {'parent_rows': len(old['x']), 'auxiliary_families': sorted({r['family_id'] for r in group}),
                'auxiliary_records': [r['record_id'] for r in group], 'holdout_used': False})
            arrays = fit_trees(model)
            p = output / f'{name}_forests.npz'
            with p.open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            result['shadow_models'][name] = model
            result['forests'][name] = {'path': p.name, 'sha256': sha(p)}
            print(f"Fit {name}: {len(x)} observations, {len(np.unique(y, axis=0))} unique XY", flush=True)
    else:
        for i, row in enumerate(rows):
            name = row['method']
            x = features(row['public'], rn)
            truth = np.array([rn.point_xy(p['lat'], p['lon']) for p in lookup[row['record_id']]['points']])
            if phase == 'holdout':
                predictions = estimators(row['public'], rn, prior, 'S3')
                old_model = parent_train['shadow_models'][name + '/5']
                predictions.update({f'shadow_knn_{k}': knn_predict(old_model, x, k) for k in (1, 5, 15)})
                old_loss, _ = loss_predict(old_model, x)
                predictions.update(old_loss)
                row['errors_by_attack'] = {a: np.linalg.norm(v - truth, axis=1).tolist() for a, v in predictions.items()}
            additional = expanded_predict(training['shadow_models'][name], forests[name], x)
            assert not additional.keys() & row['errors_by_attack'].keys()
            row['expanded_predictions'] = {a: v.tolist() for a, v in additional.items()}
            row['errors_by_attack'].update({a: np.linalg.norm(v - truth, axis=1).tolist() for a, v in additional.items()})
            if i % 49 == 0:
                print(f"Scored {phase}: {i}/{len(rows)}", flush=True)
        result['summaries'] = summaries(rows, selection, phase == 'holdout')
    write(output / f'{phase}.json', result)
    if phase == 'validation':
        write(output / 'selection.json', {**provenance, **choose(result['summaries']),
            'training_sha256': sha(output / 'training.json'), 'validation_sha256': sha(output / 'validation.json'),
            'defender_selection': 'unchanged_from_prior_factors; not_reselected',
            'role': 'validation_only_attacker_selection'})
    print(f'Saved {phase}: {len(rows)} rows', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('training', 'validation', 'development', 'holdout'), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    a = parser.parse_args()
    run(a.phase, a.output)
