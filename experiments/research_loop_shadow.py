"""Mechanism-aware first-query inversion for full-session S9 development."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist
from benchmark.engines.quotient_cover import QuotientCoverLaneDummy
from evaluation.service_shadow import fit, features
from evaluation.expanded_shadow import fit_trees, predict, TREE_PARAMS
from evaluation.loss_aware_shadow import decision
from experiments.research_loop_resources import ROOT, load, sha

SOURCE = ROOT/'artifacts/benchmarks/research_loop/iteration09_switching.json'
SHADOW = ROOT/'artifacts/benchmarks/research_loop/iteration11_shadow_training.json'
OUT = ROOT/'artifacts/benchmarks/research_loop/iteration11_shadow_attack.json'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, service, context, belief, metadata = load()
    if not SHADOW.exists():
        rng = np.random.default_rng(24092611)
        _, inv, counts = np.unique(np.floor(rn.xy/120.).astype(np.int64), axis=0,
                                    return_inverse=True, return_counts=True)
        prior = 1./counts[inv]; prior /= prior.sum()
        ids = rng.choice(len(rn), 2000, p=prior)
        model = QuotientCoverLaneDummy(rn, belief_model=belief, budget=.24, horizon=12,
                                      k=5, rng=np.random.default_rng(24092612))
        x, y, public_rows = [], [], []
        for j, state in enumerate(ids):
            model.reset()
            lat, lon = rn.latlon(int(state))
            coordinates = model.protect_step(lat, lon, 0.)
            public = {'events': [{'timestamp_s': 0., 'candidates': [
                {'candidate_id': f'candidate_{i:04d}', 'lat': a, 'lon': b}
                for i, (a, b) in enumerate(coordinates)]}]}
            x.append(features(public, rn)[0].tolist()); y.append(rn.xy[state].tolist())
            public_rows.append(public)
            if (j+1) % 100 == 0:
                print('synthetic public-prior shadow', j+1, flush=True)
        training = {'schema': 'first-query-shadow-public-simulation-v1',
                    'resources': metadata, 'seed_locations': 24092611, 'seed_mechanism': 24092612,
                    'prior': 'uniform_120m_cells_then_uniform_road_states_within_cell',
                    'source': 'public road simulation, no development endpoint labels',
                    'code_sha256': sha(Path(__file__)), 'state_ids': ids.tolist(),
                    'x': x, 'y': y, 'public': public_rows}
        SHADOW.write_text(json.dumps(training, indent=2)+'\n')
    training = json.loads(SHADOW.read_text())
    assert training['resources']['resource_sha256'] == metadata['resource_sha256']
    model = fit(training['x'], training['y'], {'training_sha256': sha(SHADOW)})
    arrays = fit_trees(model)
    source = json.loads(SOURCE.read_text())
    data = json.loads((ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json').read_text())
    records = [r for r in source['rows'] if r['method'] != 'raw']
    x = np.array([features({'events': r['public']['events'][:1]}, rn)[0] for r in records])
    preds = predict(model, arrays, x)
    dist = cdist((x-model['mean'])/model['scale'],
                 (np.array(model['x'])-model['mean'])/model['scale'])
    ordering = np.argsort(dist, axis=1, kind='stable')
    for n in (15, 45):
        supports = np.array(model['y'])[ordering[:, :n]]
        for radius in (50, 200, 500):
            preds[f'shadow_hit{radius}_{n}'] = np.array([decision(p, radius)['hit_action'] for p in supports])
    rows = []
    for i, r in enumerate(records):
        p = data['traces'][r['session_id']][0]
        truth = np.array(rn.point_xy(p['lat'], p['lon']))
        errors = dict(r['errors']['S9'])
        errors.update({a: float(np.linalg.norm(v[i]-truth)) for a, v in preds.items()})
        rows.append({'family_id': r['family_id'], 'split': r['split'], 'session_id': r['session_id'],
                     'method': r['method'], 'errors': errors,
                     'shadow_predictions': {a: v[i].tolist() for a, v in preds.items()}})
    summaries = []
    for method in sorted({r['method'] for r in rows}):
        train = [r for r in rows if r['method'] == method and r['split'] == 'development_train']
        val = [r for r in rows if r['method'] == method and r['split'] == 'development_validation']
        names = sorted(train[0]['errors'])
        for loss in ('mae', 50, 100, 200, 500):
            def score(name):
                errors = [r['errors'][name] for r in train]
                return ((0. if loss == 'mae' else -np.mean(np.array(errors) <= loss)), np.mean(errors), name)
            best = min(names, key=score)
            summaries.append({'method': method, 'loss': loss, 'selected_attack': best,
                              'validation_families': 2, 'validation_sessions': len(val),
                              'mae_m': float(np.mean([r['errors'][best] for r in val])),
                              'hit': None if loss == 'mae' else float(np.mean([r['errors'][best] <= loss for r in val]))})
    result = {'schema': 'first-query-shadow-S9-development-v1', 'source_sha256': sha(SOURCE),
              'training_sha256': sha(SHADOW), 'code_sha256': sha(Path(__file__)),
              'scope': 'S9 full-session first-query inverse; synthetic public-road shadow prior; exposed development only',
              'source_code_sha256': {p: sha(ROOT/p) for p in (
                  'evaluation/service_shadow.py', 'evaluation/expanded_shadow.py', 'evaluation/loss_aware_shadow.py',
                  'benchmark/engines/quotient_cover.py')},
              'tree_params': TREE_PARAMS, 'rows': rows, 'summaries': summaries}
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
