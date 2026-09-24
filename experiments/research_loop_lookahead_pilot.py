"""Fixed two-slice planner pilot on all core rare-POI source sessions."""
from functools import partial
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from benchmark.engines.lookahead_cover import LookaheadCoverLaneDummy
from benchmark.empirical_mobility import EmpiricalMobilityModel
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from evaluation.lane_travel import LanePoiService
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_cases import recall_pair, mean_optional
from experiments.rng_util import rng_from_key

BASE = ROOT/'artifacts/benchmarks/research_loop'
SOURCE = BASE/'iteration20_mobility_cases.json'
FIT = BASE/'iteration20_mobility_fit.json'
DATA = ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json'
OUT = BASE/'iteration21_lookahead_pilot.json'
CONTROLS = ('response_empirical_paced', 'response_empirical_paced_slack03')


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, _, reference, base, metadata = load()
    reply = PublicPoiContext(LanePoiService(rn, list(reference.pois), k=10), CACHE/'poi10.npz')
    fit = json.loads(FIT.read_text()); chosen = fit['selected_generator']
    path = BASE/chosen['file']; assert sha(path) == chosen['sha256']
    belief = EmpiricalMobilityModel(ResponseAwareAnchorModel(base, reply), load_npz(path),
        {'fit_evidence_sha256': sha(FIT), 'training_sha256': fit['training_sha256'],
         'pseudo_exposure_s': chosen['pseudo_exposure_s'], 'auxiliary_fit_families': 64,
         'auxiliary_selection_families': 16, 'target_case_labels_used': False})
    factories = {'lookahead120_floor0': partial(LookaheadCoverLaneDummy, current_objective_slack=0.),
        'lookahead120_floor03': partial(LookaheadCoverLaneDummy, current_objective_slack=.03),
        'lookahead120_static_floor03': partial(LookaheadCoverLaneDummy, current_objective_slack=.03, forecast_belief=False)}
    models = {m: cls(rn, belief_model=belief, k=5, budget=.24, horizon=12, rng=np.random.default_rng(0))
              for m, cls in factories.items()}
    source = json.loads(SOURCE.read_text()); data = json.loads(DATA.read_text())
    assert source['mobility_model_sha256'] == belief.sha256
    records = [r for r in data['records'] if r['case_id'] == 'S1.C']; assert len(records) == 4
    lookup = {(ex['session_id'], ex['rep'], ex['method']): ex for ex in source['executions']}
    executions, rows = [], []
    for record in records:
        sid = record['session_ids'][0]; trace = data['traces'][sid]; target = record['observed_indices'][0][0]
        for rep in range(2):
            local = [lookup[sid, rep, m] for m in CONTROLS]
            clock = local[0]['clock_indices']
            seeds = rng_from_key(sid+f'/{rep}', schema='persistent-exact-case-clock-v1').integers(0, 2**63, 2, dtype=np.int64)
            for method, model in models.items():
                model.anchor_rng, model.dummy_rng = (np.random.default_rng(int(s)) for s in seeds)
                model.reset(); events, utility, diagnostics = {}, {}, {}
                for i in clock:
                    p = trace[i]; t = p['time_s']-trace[0]['time_s']
                    coords = model.protect_step(p['lat'], p['lon'], t)
                    events[str(i)] = {'timestamp_s': t, 'candidates': [
                        {'candidate_id': f'candidate_{j:04d}', 'lat': lat, 'lon': lon}
                        for j, (lat, lon) in enumerate(coords)]}
                    states = [rn.nearest(lat, lon)[0] for lat, lon in coords]
                    utility[str(i)] = recall_pair(reference, reply, states, rn.nearest(p['lat'], p['lon'])[0])
                    diagnostics[str(i)] = {k: v for k, v in model.evaluator_objective[-1].items()
                        if k.startswith(('lookahead_', 'current_objective_', 'future_objective_', 'two_slice_'))}
                assert model.evaluator_ledger == local[0]['evaluator_ledger']
                digest = hashlib.sha256(json.dumps(model.evaluator_anchors).encode()).hexdigest()
                assert digest == local[0]['evaluator_anchor_sha256']
                ex = {'session_id': sid, 'family_id': record['family_id'], 'split': record['split'],
                    'rep': rep, 'method': method, 'clock_indices': clock, 'events': events,
                    'utility_by_index': utility, 'budget_bound': model.spent_bound,
                    'evaluator_ledger': model.evaluator_ledger, 'evaluator_anchor_sha256': digest,
                    'step_ms': list(model.step_ms), 'lookahead_diagnostics': diagnostics}
                local.append(ex)
                print('Ran', sid, rep, method, 'S1.C', round(utility[str(target)]['10'], 4), flush=True)
            for ex in local:
                rows.append({'record_id': record['record_id'], 'case_id': 'S1.C', 'session_id': sid,
                    'family_id': record['family_id'], 'split': record['split'], 'rep': rep, 'method': ex['method'],
                    'recall_L10': ex['utility_by_index'][str(target)]['10'],
                    'whole_session_recall_L10': mean_optional([v['10'] for v in ex['utility_by_index'].values()])})
            executions.extend(local)
    summaries = []
    for method in (*CONTROLS, *factories):
        for split in ('development_train', 'development_validation'):
            selected = [r for r in rows if r['method'] == method and r['split'] == split]
            summaries.append({'method': method, 'split': split, 'families': 2, 'RNG_repetitions': 2,
                'recall_L10': float(np.mean([r['recall_L10'] for r in selected])),
                'whole_session_recall_L10': float(np.mean([r['whole_session_recall_L10'] for r in selected]))})
    OUT.write_text(json.dumps({'scope': 'all four core S1.C sources, development pilot only; not all-case or learned privacy evidence',
        'new_executions': 24, 'replayed_controls': 16, 'source_sha256': sha(SOURCE), 'dataset_sha256': sha(DATA),
        'mobility_fit_sha256': sha(FIT), 'mobility_model_sha256': belief.sha256,
        'code_sha256': sha(Path(__file__)), 'implementation_sha256': {p: sha(ROOT/p) for p in (
            'benchmark/engines/lookahead_cover.py', 'benchmark/empirical_mobility.py',
            'benchmark/engines/empirical_paced.py', 'benchmark/engines/paced_guard.py',
            'benchmark/engines/progress_cover.py', 'benchmark/engines/quotient_cover.py',
            'benchmark/engines/fair_cover.py', 'benchmark/engines/service_cover.py')},
        'executions': executions, 'rows': rows, 'summaries': summaries}, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
