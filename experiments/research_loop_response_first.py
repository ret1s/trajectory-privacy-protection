"""Retain the first-query inversion challenge after changing the selector."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from benchmark.engines.matched_filter import MatchedFilteredProgressCoverLaneDummy
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from benchmark.public_poi_context import PublicPoiContext
from evaluation.lane_travel import LanePoiService
from evaluation.service_shadow import fit, features
from evaluation.expanded_shadow import fit_trees, predict, TREE_PARAMS
from evaluation.loss_aware_shadow import decision
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_sequence_attack import select

BASE = ROOT/'artifacts/benchmarks/research_loop'
TRAINING = BASE/'iteration16_first_query_training.json'
OUT = BASE/'iteration16_first_query_attack.json'
CORE = BASE/'iteration15_response_cases.json'


def prediction_bank(model, trees, x):
    result = predict(model, trees, x)
    dist = cdist((x-model['mean'])/model['scale'], (np.array(model['x'])-model['mean'])/model['scale'])
    order = np.argsort(dist, axis=1, kind='stable')
    for n in (15, 45):
        support = np.array(model['y'])[order[:, :n]]
        for radius in (50, 200, 500):
            result[f'shadow_hit{radius}_{n}'] = np.array([decision(p, radius)['hit_action'] for p in support])
    return result


def main(generate_only=False):
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, _, context, base, metadata = load()
    if not TRAINING.exists():
        original = json.loads((BASE/'iteration11_shadow_training.json').read_text())
        ids = original['state_ids']
        reply = PublicPoiContext(LanePoiService(rn, list(context.pois), k=10), CACHE/'poi10.npz')
        belief = ResponseAwareAnchorModel(base, reply)
        engine = MatchedFilteredProgressCoverLaneDummy(rn, belief_model=belief, budget=.24, horizon=12,
                                                       k=5, rng=np.random.default_rng(24092612))
        x, y, public_rows = [], [], []
        for ordinal, state in enumerate(ids):
            engine.reset(); lat, lon = rn.latlon(int(state))
            coords = engine.protect_step(lat, lon, 0.)
            public = {'events': [{'timestamp_s': 0., 'candidates': [{'lat': a, 'lon': b} for a, b in coords]}]}
            x.append(features(public, rn)[0].tolist()); y.append(rn.xy[state].tolist()); public_rows.append(public)
            if (ordinal+1) % 100 == 0:
                print('Public-road first-query training', ordinal+1, flush=True)
        TRAINING.write_text(json.dumps({'scope': 'public-road prior simulation; no core endpoint labels',
            'locations_source_sha256': sha(BASE/'iteration11_shadow_training.json'), 'state_ids': ids,
            'resources_sha256': metadata['resource_sha256'], 'code_sha256': sha(Path(__file__)),
            'response_model_sha256': belief.sha256, 'seed_mechanism': 24092612,
            'x': x, 'y': y, 'public': public_rows}, indent=2, allow_nan=False)+'\n')
    if generate_only:
        return
    training = json.loads(TRAINING.read_text())
    assert training['resources_sha256'] == metadata['resource_sha256']
    model = fit(training['x'], training['y'], {'training_sha256': sha(TRAINING)})
    trees = fit_trees(model)
    manifest_path = BASE/'response_shadows/manifest.json'; manifest = json.loads(manifest_path.read_text())
    holdout_sources = []
    for item in manifest['shards']:
        if item['split'] != 'auxiliary_selection':
            continue
        path = manifest_path.parent/item['file']; assert sha(path) == item['sha256']
        shard = json.loads(gzip.decompress(path.read_bytes()))
        holdout_sources.extend({**r, 'family_id': shard['family_id']} for r in shard['runs'] if r['method'] == 'response_progress')
    x = np.array([features({'events': r['public']['events'][:1]}, rn)[0] for r in holdout_sources])
    pred = prediction_bank(model, trees, x)
    holdout = [{'family_id': r['family_id'], 'errors': {a: [float(np.linalg.norm(p[i]-r['evaluation_truth']['origin_xy']))]
                 for a, p in pred.items()}} for i, r in enumerate(holdout_sources)]
    chosen = select(holdout)
    source = json.loads(CORE.read_text()); data = json.loads((ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json').read_text())
    executions = [r for r in source['executions'] if r['method'] in ('response_progress', 'response_paced')]
    first = {}
    for ex in executions:
        event = next(iter(ex['events'].values())); key = ex['session_id'], ex['rep']
        assert key not in first or first[key] == event
        first[key] = event
    x = np.array([features({'events': [next(iter(r['events'].values()))]}, rn)[0] for r in executions])
    pred = prediction_bank(model, trees, x); rows = []
    for i, r in enumerate(executions):
        p = data['traces'][r['session_id']][0]; truth = rn.point_xy(p['lat'], p['lon'])
        rows.append({'method': r['method'], 'family_id': r['family_id'], 'split': r['split'],
            'session_id_evaluator_only': r['session_id'], 'rep': r['rep'], 'selected_attack': chosen,
            'errors': {a: [float(np.linalg.norm(p[i]-truth))] for a, p in pred.items()}})
    summaries = []
    for method in ('response_progress', 'response_paced'):
        val = [r for r in rows if r['method'] == method and r['split'] == 'development_validation']
        for name in sorted(val[0]['errors']):
            e = np.array([r['errors'][name][0] for r in val])
            summaries.append({'method': method, 'attack': name, 'mae_m': float(e.mean()),
                'hits': {str(rad): float(np.mean(e <= rad)) for rad in (50, 100, 200, 500)},
                'source_sessions': len({r['session_id_evaluator_only'] for r in val}),
                'family_count': len({r['family_id'] for r in val}), 'RNG_repetitions': 2})
    OUT.write_text(json.dumps({'scope': 'common full-session first-query S9 challenge; chosen on 16 auxiliary families; exposed core development',
        'training_sha256': sha(TRAINING), 'core_source_sha256': sha(CORE), 'shadow_manifest_sha256': sha(manifest_path),
        'code_sha256': sha(Path(__file__)), 'tree_params': TREE_PARAMS, 'source_sha256': {p: sha(ROOT/p) for p in (
            'evaluation/service_shadow.py', 'evaluation/expanded_shadow.py', 'evaluation/loss_aware_shadow.py')},
        'selection': chosen, 'auxiliary_selection_rows': holdout, 'core_rows': rows, 'all_attack_summaries': summaries,
        'first_query_identical_across_both_candidates': True}, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'selection': chosen, 'all_attack_summaries': summaries}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--generate-only', action='store_true')
    main(parser.parse_args().generate_only)
