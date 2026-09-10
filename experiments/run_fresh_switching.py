"""Frozen training -> fresh validation -> sealed selection -> confirmation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from benchmark.engines.switching_cover import SwitchingCoverLaneDummy
from core.demo_protocol import TrajectoryPoint
from data.scenario_store import ScenarioStore
from evaluation.expanded_shadow import fit_trees, predict as expanded_predict
from evaluation.loss_aware_shadow import predict as loss_predict
from evaluation.service_shadow import features, fit, predict as knn_predict
from evaluation.retrieval_frontier import evaluate_retrieval
from experiments.run_contextual_lane import estimators
from experiments.run_coverage_frontier import prepare, make_model as old_model, generate as pooled_generate, compact_utility, summarize, recover_unsealed
from experiments.run_service_cover import ROOT, read, write, sha, CASES
from experiments.publish_fresh_scenarios import DB, RELEASE, FOLDER
from experiments.rng_util import rng_from_key

OUTPUT = ROOT/'artifacts/benchmarks/fresh_switching'
PARENT = ROOT/'artifacts/benchmarks/coverage_frontier/b0.24'
METHODS = ('geometric', 'mean_greedy', 'mean_exchange', 'switching_exchange')
PATHS = ('benchmark/switching_belief.py', 'benchmark/engines/switching_cover.py',
    'tests/test_switching_cover.py', 'experiments/run_fresh_switching.py',
    'thesis/notes/fresh_switching_protocol.md', 'thesis/notes/fresh_simulation_amendment.md')


def make_model(method, rn, belief, seed):
    if method != 'switching_exchange': return old_model(method, rn, belief, .24, seed)
    return SwitchingCoverLaneDummy(rn, belief_model=belief, budget=.24, horizon=12,
        k=5, theta_m=200, max_exchanges=3, category_cap=None,
        rng=rng_from_key(seed, schema='lane-comparison-v1'))


def generate(method, points, rn, belief, seed, pool=None):
    if pool is None: pool = {method: make_model(method, rn, belief, seed)}
    return pooled_generate(method, points, rn, belief, .24, seed, pool=pool)


def choose(summaries):
    assert {(s['method'], s['case_id']) for s in summaries} == {(m, c) for m in METHODS for c in CASES}
    attackers = {s['method']+'/'+s['case_id']: {
        'mae': min(s['mae_by_attack'], key=lambda a: (s['mae_by_attack'][a], a)),
        'hit': min(s['hit_by_attack'], key=lambda a: (-s['hit_by_attack'][a], a))} for s in summaries}
    choices = {}
    for depth in ('5', '10'):
        candidates = []
        for method in METHODS:
            group = [s for s in summaries if s['method'] == method]
            candidates.append({'method': method,
                'min_case_recall': min(s['utility'][depth]['poi_recall_at_5'] for s in group),
                'macro_hit100': float(np.mean([s['envelope_hit100'] for s in group]))})
        eligible = [c for c in candidates if round(c['min_case_recall'], 12) >= .90]
        choices[depth] = {'candidates': candidates, 'utility_feasible': bool(eligible),
            'chosen': min(eligible, key=lambda c: (round(c['macro_hit100'], 12), c['method'])) if eligible else None}
    return {'attackers': attackers, 'method_selection_by_depth': choices}


def fresh_records(phase):
    receipt = read(FOLDER/'registry.json')
    split = 'development_validation' if phase == 'validation' else 'confirmation'
    with ScenarioStore(DB) as store:
        assert store._release(RELEASE)['content_sha256'] == receipt['content_sha256']
        assert store.verify()['status'] == 'passed'
        records = []
        for row in store.connection.execute('SELECT record_id,case_id,scenario,family_id,split FROM records '
            'WHERE release_id=? AND split=? ORDER BY ordinal', (RELEASE, split)):
            if row['case_id'] not in CASES: continue
            points = store.device_view(RELEASE, row['record_id'])
            records.append({**dict(row), 'available_events': len(points),
                'points': [{'timestamp_s': p['time_s'], 'lat': p['lat'], 'lon': p['lon']} for p in points[:12]]})
    assert len({r['family_id'] for r in records}) == 6
    assert {r['case_id'] for r in records} == set(CASES)
    return records, receipt


def run(phase, output=OUTPUT):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/f'{phase}.json').exists(): raise FileExistsError('Completed phase is immutable')
    _, rn, prior, belief, service, deeper, provenance = prepare(.24)
    provenance.update(source_sha256={p: sha(ROOT/p) for p in sorted(set(provenance['source_sha256']) | set(PATHS))},
        methods=METHODS, parent_training_sha256=sha(PARENT/'training.json'),
        scope='same_city_new_family_confirmation; S1-S3_only; internal_ablations_not_SOTA')
    parent = read(PARENT/'training.json')
    trained = read(output/'training.json') if phase != 'training' else None
    selection = read(output/'selection.json') if phase == 'confirmation' else None
    if trained:
        for k in ('source_sha256', 'parent_training_sha256'): assert trained[k] == provenance[k]
        records, receipt = fresh_records(phase)
        provenance.update(fresh_dataset_content_sha256=receipt['content_sha256'],
            fresh_registry_sha256=sha(FOLDER/'registry.json'), fresh_verification_sha256=sha(FOLDER/'verification.json'))
    else: records = parent['records']
    if selection:
        assert selection['training_sha256'] == sha(output/'training.json')
        assert selection['validation_sha256'] == sha(output/'validation.json')
        for k in ('source_sha256', 'fresh_dataset_content_sha256', 'fresh_registry_sha256'):
            assert selection[k] == provenance[k]
    shadow_models = trained['shadow_models'] if trained else {}
    forests = {}
    if trained:
        for m, info in trained['forests'].items():
            path = ROOT/info['path']; assert sha(path) == info['sha256']
            with np.load(path, allow_pickle=False) as arrays: forests[m] = {k: arrays[k] for k in arrays.files}
    pool = {m: make_model(m, rn, belief, 0) for m in METHODS}
    fingerprint = hashlib.sha256(json.dumps({k: v for k, v in provenance.items() if not k.endswith('_ms')}, sort_keys=True).encode()).hexdigest()
    checkpoints = ROOT/'cache/fresh_switching_checkpoints'/fingerprint
    checkpoints.mkdir(parents=True, exist_ok=True)
    old_rows = {(r['record_id'], r['replicate'], r['method']): r for r in parent['rows']}
    rows = []
    for index, record in enumerate(records):
        checkpoint = checkpoints/f'{phase}_{index:04d}.json'
        recover_unsealed(checkpoint)
        if checkpoint.exists():
            saved = read(checkpoint); assert saved['record'] == record and saved['fingerprint'] == fingerprint
            rows.extend(saved['rows']); continue
        start = len(rows)
        points = tuple(TrajectoryPoint(**p) for p in record['points'])
        truth = np.array([rn.point_xy(p.lat, p.lon) for p in points])
        aux = record['case_id'].startswith('AUX')
        for rep in ((1,) if aux else (1, 2, 3)):
            seed = int((rng_from_key(record['record_id'], schema='expanded-shadow-v1') if aux else
                rng_from_key(record['record_id'], 5, rep, schema='service-cover-row-v1')).integers(0, 2**31))
            anchors = None
            for method in METHODS:
                if phase == 'training' and method != 'switching_exchange':
                    row = old_rows[record['record_id'], rep, method]
                    assert row['rng_seed'] == seed
                else:
                    row = {**{k: record[k] for k in ('record_id', 'family_id', 'case_id', 'split')},
                        'method': method, 'k': 5, 'replicate': rep, 'rng_seed': seed,
                        **generate(method, points, rn, belief, seed, pool)}
                assert anchors is None or row['evaluator_anchors'] == anchors
                anchors = row['evaluator_anchors']
                if trained:
                    x = features(row['public'], rn)
                    predictions = estimators(row['public'], rn, prior, record['scenario'])
                    core = shadow_models[method]['core']
                    predictions.update({f'shadow_knn_{k}': knn_predict(core, x, k) for k in (1, 5, 15)})
                    predictions.update(loss_predict(core, x)[0])
                    predictions.update(expanded_predict(shadow_models[method]['expanded'], forests[method], x))
                    row['predictions'] = {a: v.tolist() for a, v in predictions.items()}
                    row['errors_by_attack'] = {a: np.linalg.norm(v-truth, axis=1).tolist() for a, v in predictions.items()}
                    row['utility'] = compact_utility(evaluate_retrieval(service, deeper, row['public'],
                        [(p.lat, p.lon) for p in points]))
                rows.append(row)
        write(checkpoint, {'fingerprint': fingerprint, 'record': record, 'rows': rows[start:]})
        if index % 10 == 0 or index == len(records)-1: print(f'{phase}: {index+1}/{len(records)} records', flush=True)
    payload = {'schema': 'fresh-switching-v1', 'phase': phase, **provenance,
        'records': records, 'rows': rows,
        'training_sha256': sha(output/'training.json') if trained else None,
        'selection_sha256': sha(output/'selection.json') if selection else None}
    if phase == 'training':
        forest_info = {}
        for m in METHODS[:-1]:
            shadow_models[m] = parent['shadow_models'][m]
            info = parent['forests'][m]
            forest_info[m] = {'path': str((PARENT/info['path']).relative_to(ROOT)), 'sha256': info['sha256']}
        m = 'switching_exchange'; shadow_models[m] = {}; lookup = {r['record_id']: r for r in records}
        for kind in ('core', 'expanded'):
            group = [r for r in rows if r['method'] == m and (kind == 'expanded' or not r['case_id'].startswith('AUX'))]
            x = np.concatenate([features(r['public'], rn) for r in group])
            y = np.concatenate([np.array([rn.point_xy(p['lat'], p['lon']) for p in lookup[r['record_id']]['points']]) for r in group])
            shadow_models[m][kind] = fit(x, y, {'families': sorted({r['family_id'] for r in group}),
                'row_keys': [[r['record_id'], r['replicate']] for r in group], 'holdout_used': False})
            assert len(x) == (360 if kind == 'core' else 3233)
        arrays = fit_trees(shadow_models[m]['expanded']); path = output/'switching_forests.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                assert set(saved.files) == set(arrays) and all(np.array_equal(saved[k], v) for k, v in arrays.items())
        else:
            with path.open('xb') as stream: np.savez_compressed(stream, **arrays)
        forest_info[m] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
        payload.update(shadow_models=shadow_models, forests=forest_info)
    else: payload['summaries'] = summarize(rows, selection)
    write(output/f'{phase}.json', payload)
    if phase == 'validation':
        result = {**provenance, **choose(payload['summaries']),
            'training_sha256': sha(output/'training.json'), 'validation_sha256': sha(output/'validation.json')}
        write(output/'selection.json', result)
        print(json.dumps(result['method_selection_by_depth'], indent=2), flush=True)
    print(f'Sealed {phase}: {len(rows)} runs', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('training', 'validation', 'confirmation'), required=True)
    p.add_argument('--output', type=Path, default=OUTPUT)
    a = p.parse_args(); run(a.phase, a.output)
