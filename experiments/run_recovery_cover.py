"""Small, explicitly development-only prior/corridor ablation on pinned SQL v3."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.recovery_cover import RecoveryCoverLaneDummy
from core.demo_protocol import TrajectoryPoint
from evaluation.research_protocol import utility_metrics
from evaluation.service_shadow import features, fit, predict
from experiments.diagnose_service_cover import uniform_cell_prior, OUTPUT
from experiments.run_contextual_lane import sha, choose_configuration, estimators
from experiments.run_service_cover import prepare as original_prepare, read, write, ROOT, OUTPUT as OLD
from experiments.rng_util import rng_from_key

METHODS = ('baseline', 'belief24', 'service_cover', 'uniform_cover', 'learned_corridor', 'uniform_corridor')
NEW = {'uniform_cover': ('uniform_public_cells', None),
       'learned_corridor': ('training_occupancy', 200.),
       'uniform_corridor': ('uniform_public_cells', 200.)}
PHASES = {'training': 'training', 'validation': 'validation', 'development': 'confirmation'}


def prepare(phase):
    records, rn, service, context, prior, learned, provenance = original_prepare(PHASES[phase])
    flat = PublicAnchorModel(rn, context, uniform_cell_prior(rn, learned.spacing_m),
                             cache_path=ROOT/'cache/service_recovery/uniform.npz')
    paths = set(provenance['source_sha256']) | {
        'benchmark/engines/recovery_cover.py', 'experiments/diagnose_service_cover.py',
        'experiments/run_recovery_cover.py', 'tests/test_recovery_cover.py',
        'thesis/notes/service_recovery_protocol.md', 'thesis/notes/service_recovery_candidate_protocol.md'}
    provenance.update(source_sha256={p: sha(ROOT/p) for p in sorted(paths)},
        uniform_model_sha256=flat.sha256, diagnostic_sha256=sha(OUTPUT/'diagnostic.json'),
        role='training' if phase == 'training' else 'development_validation' if phase == 'validation' else 'reused_development',
        historical_split_not_new_confirmation=phase == 'development')
    original = read(OLD/f'{PHASES[phase]}.json')
    assert original['records'] == records
    provenance['original_phase_sha256'] = sha(OLD/f'{PHASES[phase]}.json')
    return records, rn, service, context, prior, learned, flat, provenance, original


def generate(name, points, rn, learned, flat, seed):
    prior_kind, slack = NEW[name]
    started = time.perf_counter()
    model = RecoveryCoverLaneDummy(rn, belief_model=flat if prior_kind == 'uniform_public_cells' else learned,
        prior_kind=prior_kind, corridor_m=slack, budget=.24, horizon=12, k=5, theta_m=200,
        rng=rng_from_key(seed, schema='lane-comparison-v1'))
    init_ms = (time.perf_counter()-started)*1000
    started = time.perf_counter(); run = model.protect_run(points)
    return {'public': run.to_attacker_dict(), 'evaluator_states': model.evaluator_states,
        'evaluator_anchors': model.evaluator_anchors, 'evaluator_objective': model.evaluator_objective,
        'spent_bound': model.spent_bound, 'step_ms': model.step_ms, 'init_ms': init_ms,
        'generation_ms': (time.perf_counter()-started)*1000}


def summarize(rows):
    groups = defaultdict(list)
    for r in rows: groups[r['method'], r['case_id']].append(r)
    result = []
    for (method, case), group in sorted(groups.items()):
        mae = {a: float(np.mean([np.mean(r['errors_by_attack'][a]) for r in group])) for a in group[0]['errors_by_attack']}
        hit = {a: float(np.mean([np.mean(np.asarray(r['errors_by_attack'][a]) <= 100) for r in group])) for a in mae}
        categories = {}
        for c in sorted({q['category'] for r in group for q in r['utility']['poi_rows']}):
            values = [[q['recall'] for q in r['utility']['poi_rows'] if q['category'] == c and q['recall'] is not None] for r in group]
            categories[c] = {'recall': float(np.mean([np.mean(v) for v in values if v])) if any(values) else None,
                'valid_query_n': sum(map(len, values)),
                'empty_reference_n': sum(sum(q['category'] == c and q['recall'] is None for q in r['utility']['poi_rows']) for r in group)}
        result.append({'method': method, 'case_id': case, 'k': 5,
            'families': len({r['family_id'] for r in group}), 'replicates': 3,
            'mae_by_attack': mae, 'hit_by_attack': hit, 'audit_min_mae_m': min(mae.values()),
            'audit_max_hit100': max(hit.values()),
            'poi_recall': float(np.mean([r['utility']['poi_recall_at_5'] for r in group])),
            'poi_complete': float(np.mean([r['utility']['poi_complete_rate'] for r in group])),
            'by_category': categories})
    return result


def run(phase, output=OUTPUT):
    output = Path(output); target = output/f'{phase}.json'
    if target.exists(): raise FileExistsError(target)
    records, rn, service, context, prior, learned, flat, provenance, original = prepare(phase)
    training = read(output/'training.json') if phase != 'training' else None
    selection = read(output/'selection.json') if phase == 'development' else None
    for earlier in (training, selection):
        if earlier:
            for key in ('source_sha256', 'dataset_content_sha256', 'uniform_model_sha256', 'diagnostic_sha256'):
                assert earlier[key] == provenance[key], key
    if selection:
        assert selection['training_sha256'] == sha(output/'training.json')
        assert selection['validation_sha256'] == sha(output/'validation.json')
    existing = {(r['record_id'], r['replicate'], r['method']): r for r in original['rows'] if r['k'] == 5}
    rows = []
    for r in records:
        points = tuple(TrajectoryPoint(**x) for x in r['points'])
        truth = np.array([rn.point_xy(p.lat, p.lon) for p in points])
        for rep in (1, 2, 3):
            seed = int(rng_from_key(r['record_id'], 5, rep, schema='service-cover-row-v1').integers(0, 2**31))
            paired = None
            for name in METHODS:
                if name not in NEW:
                    row = existing[r['record_id'], rep, name]
                    assert row['rng_seed'] == seed
                else:
                    generated = generate(name, points, rn, learned, flat, seed)
                    row = {**{key: r[key] for key in ('record_id', 'case_id', 'split', 'family_id')},
                        'method': name, 'k': 5, 'replicate': rep, 'rng_seed': seed, **generated}
                    if training:
                        predictions = estimators(generated['public'], rn, prior, r['scenario'])
                        for count in (1, 5, 15):
                            predictions[f'shadow_knn_{count}'] = predict(training['shadow_models'][f'{name}/5'], features(generated['public'], rn), count)
                        row['errors_by_attack'] = {a: np.linalg.norm(p-truth, axis=1).tolist() for a, p in predictions.items()}
                        row['utility'] = utility_metrics(service, generated['public'], [(p.lat, p.lon) for p in points])
                if paired is not None: assert row['evaluator_anchors'] == paired
                paired = row['evaluator_anchors']; rows.append(row)
        print(f'{phase}: {r["record_id"]}', flush=True)
    result = {'schema': 'service-recovery-v1', 'scope': 'development_ablation_no_fresh_confirmation',
        'phase': phase, **provenance, 'records': records, 'rows': rows, 'methods': METHODS,
        'training_sha256': sha(output/'training.json') if training else None,
        'selection_sha256': sha(output/'selection.json') if selection else None}
    if phase == 'training':
        models = {key: value for key, value in original['shadow_models'].items() if key in {f'{m}/5' for m in METHODS}}
        lookup = {r['record_id']: r for r in records}
        for name in NEW:
            group = [r for r in rows if r['method'] == name]
            x = np.concatenate([features(r['public'], rn) for r in group])
            y = np.concatenate([np.array([rn.point_xy(p['lat'], p['lon']) for p in lookup[r['record_id']]['points']]) for r in group])
            models[f'{name}/5'] = fit(x, y, {'families': ['family-101', 'family-102'],
                'split': 'development_train', 'row_keys': [[r['record_id'], r['replicate']] for r in group]})
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
        write(output/'selection.json', {**provenance, 'schema': 'service-recovery-selection-v1',
            'training_sha256': sha(output/'training.json'), 'validation_sha256': sha(target),
            'attackers': attackers, 'method_selection': chosen})
        print(json.dumps(chosen, indent=2), flush=True)
    print(f'Saved {len(rows)} rows to {target}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=tuple(PHASES), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(); run(args.phase, args.output)
