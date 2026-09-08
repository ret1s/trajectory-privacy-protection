"""DB-pinned, three-phase service-cover experiment. Never overwrite evidence."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.service_cover import ServiceCoverLaneDummy
from benchmark.public_poi_context import PublicPoiContext
from core.demo_protocol import TrajectoryPoint
from data.lane_states import build_lane_states
from data.scenario_store import ScenarioStore
from evaluation.lane_travel import LanePoiService
from evaluation.research_protocol import utility_metrics
from evaluation.scenario_metrics import read_osm_pois
from evaluation.service_shadow import features, fit, predict
from experiments.run_belief_suite import generate as old_generate
from experiments.run_contextual_lane import sha, choose_configuration, estimators
from experiments.run_lane_comparison import ROOT, spatial_prior
from experiments.rng_util import rng_from_key

DB = ROOT / 'artifacts/datasets/scenarios.sqlite3'
RELEASE = 'urban-scenarios-v3'
CONTENT = '213886fc2722bbecf5e55f61e8978d2c2842bb6018b7f65f5021a261bd79358d'
OUTPUT = ROOT / 'artifacts/benchmarks/service_cover'
METHODS = ('baseline', 'belief24', 'service_cover', 'prior_cover')
CASES = tuple(f'S{s}.{c}' for s in (1, 2, 3) for c in 'ABC')
SPLITS = {'training': 'development_train', 'validation': 'development_validation', 'confirmation': 'confirmation'}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    with path.with_suffix('.sha256').open('x') as stream:
        stream.write(sha(path) + '\n')


def read(path):
    assert sha(path) == path.with_suffix('.sha256').read_text().strip()
    return json.loads(path.read_text())


def prepare(phase):
    previous = read(ROOT / 'artifacts/benchmarks/belief_suite/confirmation.json')
    with ScenarioStore(DB) as store:
        audit = store.verify()
        release = store._release(RELEASE)
        assert release['content_sha256'] == CONTENT
        metadata = json.loads(release['metadata_json'])
        records = []
        for row in store.connection.execute('SELECT record_id,case_id,scenario,family_id,split FROM records '
                'WHERE release_id=? AND split=? ORDER BY ordinal', (RELEASE, SPLITS[phase])):
            if row['case_id'] not in CASES: continue
            stream = store.device_view(RELEASE, row['record_id'])
            records.append({**dict(row), 'available_events': len(stream),
                'points': [{'timestamp_s': x['time_s'], 'lat': x['lat'], 'lon': x['lon']} for x in stream[:12]]})
        train = {}
        for row in store.connection.execute('''SELECT p.session_id,p.lat,p.lon FROM points p
            JOIN sessions s USING(release_id,session_id)
            JOIN families f ON f.release_id=s.release_id AND f.family_id=s.family_id
            WHERE p.release_id=? AND f.split='development_train' ORDER BY p.session_id,p.point_index''', (RELEASE,)):
            train.setdefault(row['session_id'], []).append({'lat': row['lat'], 'lon': row['lon']})
        sources = dict(store.connection.execute('SELECT path,sha256 FROM source_hashes WHERE release_id=?', (RELEASE,)))
    for path, digest in {**previous['source_sha256'], **sources}.items():
        assert sha(ROOT / path) == digest, path
    assert sha(ROOT / previous['network_path']) == previous['network_sha256']
    assert sha(ROOT / previous['service']['osm_path']) == previous['service']['osm_sha256']
    verified = json.loads((ROOT / 'artifacts/datasets/urban_scenarios_v3/verification.json').read_text())
    assert verified['dataset_sha256'] == release['source_sha256']
    assert verified['raw_fcd_points_compared'] == metadata['summary']['raw_fcd_samples']
    n = 4 if phase == 'confirmation' else 2
    assert len(records) == 9 * n and all(sum(r['case_id'] == c for r in records) == n for c in CASES)
    started = time.perf_counter()
    rn = build_lane_states(ROOT / previous['network_path'])
    service = LanePoiService(rn, read_osm_pois(ROOT / previous['service']['osm_path'], tuple(metadata['network']['bbox_lon_lat'])))
    context = PublicPoiContext(service, ROOT / 'cache/contextual_lane_v1/public_poi.npz')
    prior = spatial_prior(rn, [train[s] for s in sorted(train)])
    belief = PublicAnchorModel(rn, context, prior, cache_path=ROOT / 'cache/belief_suite_v1/model.npz')
    assert belief.sha256 == previous['belief_model_sha256']  # reused development data must agree
    paths = set(previous['source_sha256']) | set(sources) | {
        'benchmark/engines/service_cover.py', 'evaluation/service_shadow.py',
        'experiments/run_service_cover.py', 'data/scenario_store/store.py',
        'data/scenario_store/schema.sql', 'tests/test_service_cover.py'}
    provenance = {'dataset_release': RELEASE, 'dataset_content_sha256': CONTENT,
        'dataset_source_sha256': release['source_sha256'], 'network_path': previous['network_path'],
        'network_sha256': previous['network_sha256'], 'service': previous['service'],
        'belief_model_sha256': belief.sha256, 'context_sha256': context.sha256,
        'source_sha256': {p: sha(ROOT / p) for p in sorted(paths)},
        'training_sessions': sorted(train), 'registry_releases': audit['releases'],
        'prepare_ms': (time.perf_counter() - started) * 1000}
    return records, rn, service, context, prior, belief, provenance


def generate(name, points, rn, context, belief, k, seed):
    if name in ('baseline', 'belief24'):
        return old_generate(name, points, rn, context, belief, k, seed)
    started = time.perf_counter()
    model = ServiceCoverLaneDummy(rn, belief_model=belief, prior_only=name == 'prior_cover',
        budget=.24, horizon=12, k=k, theta_m=200, rng=rng_from_key(seed, schema='lane-comparison-v1'))
    init_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter(); run = model.protect_run(points)
    return {'public': run.to_attacker_dict(), 'evaluator_states': model.evaluator_states,
        'evaluator_anchors': model.evaluator_anchors, 'spent_bound': model.spent_bound,
        'evaluator_objective': model.evaluator_objective, 'init_ms': init_ms,
        'generation_ms': (time.perf_counter() - started) * 1000, 'step_ms': model.step_ms}


def summarize(rows):
    result = []
    for name in METHODS:
        for k in (3, 5):
            for case in CASES:
                group = [r for r in rows if (r['method'], r['k'], r['case_id']) == (name, k, case)]
                assert len(group) in (6, 12)
                errors = {a: float(np.mean([np.mean(r['errors_by_attack'][a]) for r in group])) for a in group[0]['errors_by_attack']}
                hits = {a: float(np.mean([np.mean(np.asarray(r['errors_by_attack'][a]) <= 100) for r in group])) for a in errors}
                categories = {}
                for category in sorted({p['category'] for r in group for p in r['utility']['poi_rows']}):
                    values = [[p['recall'] for p in r['utility']['poi_rows'] if p['category'] == category and p['recall'] is not None] for r in group]
                    categories[category] = {'recall': float(np.mean([np.mean(v) for v in values if v])) if any(values) else None,
                        'valid_query_n': sum(map(len, values)),
                        'empty_reference_n': sum(sum(p['category'] == category and p['recall'] is None for p in r['utility']['poi_rows']) for r in group)}
                result.append({'method': name, 'k': k, 'case_id': case,
                    'families': len({r['family_id'] for r in group}), 'replicates': 3,
                    'mae_by_attack': errors, 'hit_by_attack': hits,
                    'audit_min_mae_m': min(errors.values()), 'audit_max_hit100': max(hits.values()),
                    'poi_recall': float(np.mean([r['utility']['poi_recall_at_5'] for r in group])),
                    'poi_complete': float(np.mean([r['utility']['poi_complete_rate'] for r in group])),
                    'by_category': categories,
                    'step_mean_ms': float(np.mean([np.mean(r['step_ms']) for r in group]))})
    return result


def select(summaries):
    choices = {}
    for k in (3, 5):
        candidates = []
        for name in METHODS[:-1]:  # prior-only is a negative control, not a candidate
            group = [s for s in summaries if (s['method'], s['k']) == (name, k)]
            candidates.append({'method': name, 'min_case_recall': min(s['poi_recall'] for s in group),
                'macro_hit100': float(np.mean([s['audit_max_hit100'] for s in group]))})
        choices[str(k)] = choose_configuration(candidates)
    return choices


def run(phase, output=OUTPUT):
    output = Path(output); target = output / f'{phase}.json'
    if target.exists(): raise FileExistsError('Use a fresh output directory; evidence is immutable')
    records, rn, service, context, prior, belief, provenance = prepare(phase)
    training = read(output / 'training.json') if phase != 'training' else None
    selection = read(output / 'selection.json') if phase == 'confirmation' else None
    for previous in (training, selection):
        if previous:
            assert previous['source_sha256'] == provenance['source_sha256']
            assert previous['dataset_content_sha256'] == CONTENT
    if selection:
        assert selection['training_sha256'] == sha(output / 'training.json')
        assert selection['validation_sha256'] == sha(output / 'validation.json')
    rows = []
    for record in records:
        points = tuple(TrajectoryPoint(**p) for p in record['points'])
        truth = np.array([rn.point_xy(p.lat, p.lon) for p in points])
        for k in (3, 5):
            for replicate in (1, 2, 3):
                seed = int(rng_from_key(record['record_id'], k, replicate, schema='service-cover-row-v1').integers(0, 2**31))
                anchors = []
                for name in METHODS:
                    generated = generate(name, points, rn, context, belief, k, seed)
                    row = {**{key: record[key] for key in ('record_id', 'case_id', 'split', 'family_id')},
                        'method': name, 'k': k, 'replicate': replicate, 'rng_seed': seed, **generated}
                    if training:
                        predictions = estimators(generated['public'], rn, prior, record['scenario'])
                        for count in (1, 5, 15):
                            predictions[f'shadow_knn_{count}'] = predict(training['shadow_models'][f'{name}/{k}'], features(generated['public'], rn), count)
                        row['errors_by_attack'] = {a: np.linalg.norm(v - truth, axis=1).tolist() for a, v in predictions.items()}
                        row['utility'] = utility_metrics(service, generated['public'], [(p.lat, p.lon) for p in points])
                    rows.append(row); anchors.append(generated['evaluator_anchors'])
                assert all(a == anchors[0] for a in anchors)
        print(f'{phase}: {record["record_id"]} {record["case_id"]}', flush=True)
    payload = {'schema': 'service-cover-v1', 'phase': phase, **provenance, 'records': records, 'rows': rows,
               'method_grid': METHODS, 'training_sha256': sha(output / 'training.json') if training else None,
               'selection_sha256': sha(output / 'selection.json') if selection else None}
    if phase == 'training':
        models = {}
        for name in METHODS:
            for k in (3, 5):
                group = [r for r in rows if (r['method'], r['k']) == (name, k)]
                x = np.concatenate([features(r['public'], rn) for r in group])
                y = np.concatenate([np.array([rn.point_xy(p['lat'], p['lon']) for p in next(s for s in records if s['record_id'] == r['record_id'])['points']]) for r in group])
                models[f'{name}/{k}'] = fit(x, y, {'families': sorted({r['family_id'] for r in group}),
                    'split': SPLITS[phase], 'row_keys': [[r['record_id'], r['replicate']] for r in group]})
        payload['shadow_models'] = models
    else:
        payload['summaries'] = summarize(rows)
        if selection:
            for s in payload['summaries']:
                a = selection['attackers'][f'{s["method"]}/{s["k"]}/{s["case_id"]}']
                s.update(selected_mae_attack=a['mae'], selected_hit_attack=a['hit'],
                    selected_mae_m=s['mae_by_attack'][a['mae']], selected_hit100=s['hit_by_attack'][a['hit']])
    write(target, payload)
    if phase == 'validation':
        attackers = {f'{s["method"]}/{s["k"]}/{s["case_id"]}': {
            'mae': min(s['mae_by_attack'], key=lambda a: (s['mae_by_attack'][a], a)),
            'hit': min(s['hit_by_attack'], key=lambda a: (-s['hit_by_attack'][a], a))} for s in payload['summaries']}
        write(output / 'selection.json', {**provenance, 'schema': 'service-cover-selection-v1',
            'training_sha256': sha(output / 'training.json'), 'validation_sha256': sha(target),
            'attackers': attackers, 'method_selection': select(payload['summaries'])})
    print(f'{len(rows)} rows saved to {target}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=tuple(SPLITS), required=True)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args(); run(args.phase, args.output)
