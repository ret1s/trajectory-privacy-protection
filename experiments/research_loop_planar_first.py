"""Mechanism-matched first-query inversion; not masked S9 case evaluation."""
from pathlib import Path
import json
import numpy as np
from benchmark.planar_anchor import PlanarAnchorModel
from benchmark.engines.planar_paced import PlanarPacedLaneDummy
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from benchmark.public_poi_context import PublicPoiContext
from evaluation.lane_travel import LanePoiService
from evaluation.service_shadow import fit, features
from evaluation.expanded_shadow import fit_trees, TREE_PARAMS
from experiments.research_loop_response_first import prediction_bank
from experiments.research_loop_sequence_attack import select
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.rng_util import rng_from_key

BASE = ROOT/'artifacts/benchmarks/research_loop'
TRAINING = BASE/'iteration26_first_query_training.json'
TREES = BASE/'iteration26_first_query_trees.npz'
OUT = BASE/'iteration26_first_query_attack.json'
CORE = BASE/'iteration26_planar_anchor_cases.json'
AUX = ROOT/'artifacts/datasets/research_loop_shadow_v1/dataset.json'
DATA = ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json'


def main():
    if any(p.exists() for p in (TRAINING, TREES, OUT)):
        raise FileExistsError('Preserve evidence; do not silently regenerate a partial run')
    rn, _, context, base, metadata = load()
    reply = PublicPoiContext(LanePoiService(rn, list(context.pois), k=10), CACHE/'poi10.npz')
    belief = PlanarAnchorModel(ResponseAwareAnchorModel(base, reply))
    engine = PlanarPacedLaneDummy(rn, belief_model=belief, budget=.24, horizon=12,
                                  k=5, utility_slack=0., rng=np.random.default_rng(24092612))
    def first_query(lat, lon):
        engine.reset(); coords = engine.protect_step(lat, lon, 0.)
        return {'events': [{'timestamp_s': 0., 'candidates': [
            {'candidate_id': f'candidate_{j:04d}', 'lat': a, 'lon': b} for j, (a, b) in enumerate(coords)]}]}
    locations = BASE/'iteration11_shadow_training.json'
    ids = json.loads(locations.read_text())['state_ids']
    x, y, public = [], [], []
    for ordinal, state in enumerate(ids):
        view = first_query(*rn.latlon(int(state)))
        x.append(features(view, rn)[0].tolist()); y.append(rn.xy[state].tolist()); public.append(view)
        if (ordinal+1) % 200 == 0:
            print('Planar first-query public training', ordinal+1, flush=True)
    sources = (Path(__file__), locations, AUX, DATA, CORE,
        ROOT/'benchmark/planar_anchor.py', ROOT/'benchmark/engines/planar_paced.py',
        ROOT/'experiments/research_loop_response_first.py', ROOT/'evaluation/service_shadow.py',
        ROOT/'evaluation/expanded_shadow.py', ROOT/'evaluation/loss_aware_shadow.py')
    provenance = {'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sources},
                  'resources': metadata, 'belief_model_sha256': belief.sha256}
    training = {'scope': 'fixed 2000 public-road locations, planar-generated views; no core labels used',
                'provenance': provenance, 'state_ids': ids, 'seed_mechanism': 24092612, 'x': x, 'y': y, 'public': public}
    TRAINING.write_text(json.dumps(training, indent=2, allow_nan=False)+'\n')
    model = fit(x, y, {'training_sha256': sha(TRAINING)}); trees = fit_trees(model)
    np.savez_compressed(TREES, **trees)
    aux = json.loads(AUX.read_text()); selection = []
    for family in aux['families']:
        if family['split'] != 'auxiliary_selection':
            continue
        for session in family['sessions']:
            sid = session['session_id']; p = aux['traces'][sid][0]
            seeds = rng_from_key(sid, schema='research-loop-sequential-shadow-v1').integers(0, 2**63, 2, dtype=np.int64)
            engine.anchor_rng, engine.dummy_rng = (np.random.default_rng(int(s)) for s in seeds)
            view = first_query(p['lat'], p['lon'])
            selection.append({'family_id': family['family_id'], 'session_id_evaluator_only': sid,
                              'public': view, 'truth_xy': list(rn.point_xy(p['lat'], p['lon']))})
    predictions = prediction_bank(model, trees, np.array([features(r['public'], rn)[0] for r in selection]))
    for i, r in enumerate(selection):
        r['errors'] = {a: [float(np.linalg.norm(p[i]-r['truth_xy']))] for a, p in predictions.items()}
    chosen = select(selection)
    source = json.loads(CORE.read_text()); data = json.loads(DATA.read_text())
    executions = [e for e in source['executions'] if e['method'] == 'planar_paced']
    slack = {(e['session_id'], e['rep']): e for e in source['executions'] if e['method'] == 'planar_paced_slack03'}
    views = []
    for ex in executions:
        first = next(iter(ex['events'].values()))
        assert first == next(iter(slack[ex['session_id'], ex['rep']]['events'].values()))
        views.append({'events': [first]})
    predictions = prediction_bank(model, trees, np.array([features(v, rn)[0] for v in views]))
    rows = []
    for i, ex in enumerate(executions):
        p = data['traces'][ex['session_id']][0]; target = rn.point_xy(p['lat'], p['lon'])
        rows.append({'family_id': ex['family_id'], 'split': ex['split'], 'session_id_evaluator_only': ex['session_id'],
            'rep': ex['rep'], 'public': views[i], 'errors': {a: [float(np.linalg.norm(v[i]-target))] for a, v in predictions.items()},
            'raw_first_query_error_m': 0.})
    val = [r for r in rows if r['split'] == 'development_validation']
    summary = {'selection': chosen,
        'selected_mae_m': float(np.mean([r['errors'][chosen['mae']][0] for r in val])),
        'selected_hits': {str(rad): float(np.mean([r['errors'][chosen[f'hit{rad}']][0] <= rad for r in val])) for rad in (50,100,200,500)},
        'validation_families': len({r['family_id'] for r in val}),
        'validation_source_sessions': len({r['session_id_evaluator_only'] for r in val}),
        'validation_record_RNG_pairs': len(val)}
    OUT.write_text(json.dumps({'scope': 'full-session FIRST query S9 challenge, not S9.A/B/C masked views; auxiliary-selected attack; exposed core development; both planar slack variants have identical first query',
        'provenance': provenance, 'training_sha256': sha(TRAINING), 'trees_sha256': sha(TREES), 'tree_params': TREE_PARAMS,
        'new_first_query_only_executions': len(ids)+len(selection), 'auxiliary_selection_rows': selection,
        'core_rows': rows, 'summary': summary}, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
