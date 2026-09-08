"""Declared prior-factor development, with a nested loss-aware attack bank."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.prior_factors import FactorizedAnchorModel, PriorFactorCover, cell_balanced_prior
from core.demo_protocol import TrajectoryPoint
from evaluation.loss_aware_shadow import predict as loss_predict
from evaluation.research_protocol import utility_metrics
from evaluation.service_shadow import features, fit, predict
from experiments.run_contextual_lane import sha, choose_configuration, estimators
from experiments.run_recovery_cover import prepare as previous_prepare, OUTPUT as PREVIOUS
from experiments.run_recovery_cover import PHASES, ROOT, read
from experiments.run_service_cover import write
from experiments.rng_util import rng_from_key

OUTPUT = ROOT/'artifacts/benchmarks/prior_factors'
CONTROLS = ('baseline', 'service_cover', 'uniform_cover')
NEW = {'learned_initial_uniform_motion': ('L', 'U'),
       'uniform_initial_learned_motion': ('U', 'L'),
       'balanced_cover': ('C', 'C'), 'mixed_cover': ('M', 'M')}
METHODS = CONTROLS + tuple(NEW)
SOURCE_PATHS = ('benchmark/prior_factors.py', 'evaluation/loss_aware_shadow.py',
                'experiments/run_prior_factors.py', 'tests/test_prior_factors.py',
                'tests/test_loss_aware_shadow.py', 'thesis/notes/prior_factor_protocol.md')


def prepare(phase):
    records, rn, service, context, prior, learned, uniform, provenance, _ = previous_prepare(phase)
    balanced = cell_balanced_prior(rn, prior, learned.spacing_m)
    from experiments.diagnose_service_cover import uniform_cell_prior
    flat = uniform_cell_prior(rn, learned.spacing_m)
    c = PublicAnchorModel(rn, context, balanced, cache_path=ROOT/'cache/prior_factors/balanced.npz')
    m = PublicAnchorModel(rn, context, .5*balanced+.5*flat, cache_path=ROOT/'cache/prior_factors/mixed.npz')
    base = {'L': learned, 'U': uniform, 'C': c, 'M': m}
    models = {name: FactorizedAnchorModel(base[a], base[b]) for name, (a, b) in NEW.items()}
    previous = read(PREVIOUS/f'{phase}.json')
    assert previous['records'] == records
    paths = set(provenance['source_sha256']) | set(SOURCE_PATHS)
    provenance.update(source_sha256={p: sha(ROOT/p) for p in sorted(paths)},
                      factor_model_sha256={n: m.sha256 for n, m in models.items()},
                      previous_phase_sha256=sha(PREVIOUS/f'{phase}.json'))
    return records, rn, service, prior, models, provenance, previous


def generate(name, points, rn, models, seed):
    started = time.perf_counter()
    m = PriorFactorCover(rn, belief_model=models[name], budget=.24, horizon=12, k=5,
                         theta_m=200, rng=rng_from_key(seed, schema='lane-comparison-v1'))
    init_ms = (time.perf_counter()-started)*1000
    started = time.perf_counter(); run = m.protect_run(points)
    return {'public': run.to_attacker_dict(), 'evaluator_states': m.evaluator_states,
            'evaluator_anchors': m.evaluator_anchors, 'evaluator_objective': m.evaluator_objective,
            'spent_bound': m.spent_bound, 'step_ms': m.step_ms, 'init_ms': init_ms,
            'generation_ms': (time.perf_counter()-started)*1000}


def summarize(rows):
    from experiments.run_recovery_cover import summarize as original
    return original(rows)


def run(phase, output=OUTPUT):
    output = Path(output); target = output/f'{phase}.json'
    if target.exists(): raise FileExistsError('Preserve evidence; choose a fresh output directory')
    records, rn, service, prior, models, provenance, previous = prepare(phase)
    training = read(output/'training.json') if phase != 'training' else None
    selection = read(output/'selection.json') if phase == 'development' else None
    for earlier in (training, selection):
        if earlier:
            for key in ('source_sha256', 'dataset_content_sha256', 'factor_model_sha256'):
                assert earlier[key] == provenance[key], key
    if selection:
        assert selection['validation_sha256'] == sha(output/'validation.json')
        assert selection['training_sha256'] == sha(output/'training.json')
    old = {(r['record_id'], r['replicate'], r['method']): r for r in previous['rows']}
    rows = []
    for r in records:
        points = tuple(TrajectoryPoint(**p) for p in r['points'])
        truth = np.array([rn.point_xy(p.lat, p.lon) for p in points])
        for rep in (1, 2, 3):
            control = old[r['record_id'], rep, 'baseline']; seed = control['rng_seed']
            for name in METHODS:
                if name in CONTROLS:
                    row = copy.deepcopy(old[r['record_id'], rep, name])
                else:
                    row = {**{k: r[k] for k in ('record_id', 'case_id', 'split', 'family_id')},
                           'method': name, 'k': 5, 'replicate': rep, 'rng_seed': seed,
                           **generate(name, points, rn, models, seed)}
                    if training:
                        predictions = estimators(row['public'], rn, prior, r['scenario'])
                        for n in (1, 5, 15):
                            predictions[f'shadow_knn_{n}'] = predict(training['shadow_models'][f'{name}/5'], features(row['public'], rn), n)
                        row['errors_by_attack'] = {a: np.linalg.norm(p-truth, axis=1).tolist() for a, p in predictions.items()}
                        row['utility'] = utility_metrics(service, row['public'], [(p.lat, p.lon) for p in points])
                assert row['evaluator_anchors'] == control['evaluator_anchors']
                if training:
                    additional, audit = loss_predict(training['shadow_models'][f'{name}/5'], features(row['public'], rn))
                    row['loss_predictions'] = {a: p.tolist() for a, p in additional.items()}
                    row['loss_audit'] = audit
                    row['errors_by_attack'].update({a: np.linalg.norm(p-truth, axis=1).tolist() for a, p in additional.items()})
                rows.append(row)
        print(f'{phase}: {r["record_id"]}', flush=True)
    result = {'schema': 'prior-factor-v1', 'scope': 'development_only_no_fresh_confirmation',
              'phase': phase, **provenance, 'records': records, 'rows': rows, 'methods': METHODS,
              'training_sha256': sha(output/'training.json') if training else None,
              'selection_sha256': sha(output/'selection.json') if selection else None}
    if phase == 'training':
        models = {k: v for k, v in previous['shadow_models'].items() if k in {f'{n}/5' for n in CONTROLS}}
        lookup = {r['record_id']: r for r in records}
        for name in NEW:
            group = [r for r in rows if r['method'] == name]
            x = np.concatenate([features(r['public'], rn) for r in group])
            y = np.concatenate([np.array([rn.point_xy(p['lat'], p['lon']) for p in lookup[r['record_id']]['points']]) for r in group])
            models[f'{name}/5'] = fit(x, y, {'families': ['family-101', 'family-102'], 'split': 'development_train',
                                           'row_keys': [[r['record_id'], r['replicate']] for r in group]})
        result['shadow_models'] = models
    else:
        result['summaries'] = summarize(rows)
        if selection:
            for s in result['summaries']:
                a = selection['attackers'][f'{s["method"]}/{s["case_id"]}']
                s.update(selected_mae_attack=a['mae'], selected_hit_attack=a['hit'],
                         selected_mae_m=s['mae_by_attack'][a['mae']], selected_hit100=s['hit_by_attack'][a['hit']])
    write(target, result)
    if phase == 'validation':
        candidates = [{'method': name, 'min_case_recall': min(s['poi_recall'] for s in result['summaries'] if s['method'] == name),
                       'macro_hit100': float(np.mean([s['audit_max_hit100'] for s in result['summaries'] if s['method'] == name]))} for name in METHODS]
        attackers = {f'{s["method"]}/{s["case_id"]}': {
            'mae': min(s['mae_by_attack'], key=lambda a: (s['mae_by_attack'][a], a)),
            'hit': min(s['hit_by_attack'], key=lambda a: (-s['hit_by_attack'][a], a))} for s in result['summaries']}
        chosen = choose_configuration(candidates)
        write(output/'selection.json', {**provenance, 'schema': 'prior-factor-selection-v1',
              'training_sha256': sha(output/'training.json'), 'validation_sha256': sha(target),
              'attackers': attackers, 'method_selection': chosen})
        print(json.dumps(chosen, indent=2), flush=True)
    print(f'Saved {len(rows)} rows to {target}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=tuple(PHASES), required=True)
    p.add_argument('--output', type=Path, default=OUTPUT)
    a = p.parse_args(); run(a.phase, a.output)
