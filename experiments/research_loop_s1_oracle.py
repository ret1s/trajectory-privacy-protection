"""Posthoc S1.C failure decomposition; oracle labels stay in the evaluator."""
from pathlib import Path
import json
import numpy as np
from benchmark.engines.paced_guard import PacedProgressLaneDummy
from benchmark.engines.matched_filter import MatchedFilteredProgressCoverLaneDummy
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from benchmark.engines.fair_cover import CoverageObjective
from evaluation.coverage_oracle import solve
from evaluation.lane_travel import LanePoiService
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.rng_util import rng_from_key

SOURCE = ROOT/'artifacts/benchmarks/research_loop/iteration15_response_cases.json'
DATA = ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json'
OUT = ROOT/'artifacts/benchmarks/research_loop/iteration16_s1_oracle_diagnosis.json'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, service, reference, base, metadata = load()
    reply = PublicPoiContext(LanePoiService(rn, list(reference.pois), k=10), CACHE/'poi10.npz')
    response = ResponseAwareAnchorModel(base, reply)
    factories = {'filter_paced': (PacedProgressLaneDummy, base),
                 'response_progress': (MatchedFilteredProgressCoverLaneDummy, response),
                 'response_paced': (PacedProgressLaneDummy, response)}
    models = {m: cls(rn, belief_model=b, k=5, horizon=12, budget=.24, rng=np.random.default_rng(0))
              for m, (cls, b) in factories.items()}
    data, source = json.loads(DATA.read_text()), json.loads(SOURCE.read_text())
    lookup = {(r['session_id'], r['rep'], r['method']): r for r in source['executions']}
    rows = []
    for record in data['records']:
        if record['case_id'] != 'S1.C':
            continue
        sid = record['session_ids'][0]; target = record['labels']['target_index']; trace = data['traces'][sid]
        for rep in range(2):
            seeds = rng_from_key(sid+f'/{rep}', schema='persistent-exact-case-clock-v1').integers(0, 2**63, 2, dtype=np.int64)
            for method, model in models.items():
                ex = lookup[sid, rep, method]
                model.anchor_rng, model.dummy_rng = (np.random.default_rng(int(s)) for s in seeds)
                model.reset()
                for i in ex['clock_indices']:
                    p = trace[i]; t = p['time_s']-trace[0]['time_s']
                    previous, last_t = model.previous, model.last_t
                    coordinates = model.protect_step(p['lat'], p['lon'], t)
                    expected = [(c['lat'], c['lon']) for c in ex['events'][str(i)]['candidates']]
                    assert list(coordinates) == expected
                    if i == target:
                        break
                assert i == target and previous is not None
                # Everything below is evaluator-only; no feedback into model.
                truth, _ = rn.nearest(p['lat'], p['lon'])
                weights = reference.reference_weights((p['lat'], p['lon']))
                objective = CoverageObjective(reply.signatures, reply.access, weights)
                selected = model.previous
                actual = objective.value(selected)
                assert abs(actual-ex['utility_by_index'][str(target)]['10']) < 1e-12
                groups = [np.array(sorted(s for s in model.travel.reachable(state, t-last_t) if model.viable[s]), dtype=int)
                          for state in previous]
                oracle = solve(groups, reply.signatures, reply.access, weights, time_limit_s=10.)
                goals = model.evaluator_objective[-1]['progress_goals']
                protected_weights = np.asarray(model.belief.weights @ model.belief_model.poi_weights).ravel()
                protected_objective = CoverageObjective(reply.signatures, reply.access, protected_weights)
                near = np.linalg.norm(model.belief_model.xy-rn.xy[truth], axis=1)
                def categories(states):
                    answer = {}
                    for ci, name in enumerate(reference.categories):
                        refs = set(reference.signatures[truth, ci])-{-1}
                        got = set(reply.signatures[reply.access[states], ci].ravel())-{-1}
                        answer[name] = len(got & refs)/len(refs) if refs else None
                    return answer
                rows.append({'method': method, 'family_id': record['family_id'], 'split': record['split'],
                    'session_id_evaluator_only': sid, 'rep': rep, 'record_id': record['record_id'],
                    'target_time_s': t, 'last_query_gap_s': t-last_t, 'actual_recall_L10': actual,
                    'actual_category_recall': categories(selected), 'reachable_true_reference_oracle': oracle,
                    'oracle_category_recall': categories(oracle['selected_states']),
                    'global_protected_plan_true_recall': objective.value(goals),
                    'predicted_recall_selected': protected_objective.value(selected),
                    'predicted_recall_global_plan': protected_objective.value(goals),
                    'belief_mass_within_200m': float(model.belief.weights[near <= 200].sum()),
                    'belief_mass_within_500m': float(model.belief.weights[near <= 500].sum()),
                    'anchor_error_m': float(np.linalg.norm(rn.point_xy(*model.last_anchor)-rn.xy[truth])),
                    'belief_mean_error_m': float(np.linalg.norm(model.belief.weights @ model.belief_model.xy-rn.xy[truth])),
                    'budget_spent': model.spent_bound, 'privacy_reads': sum(r['private_read'] for r in model.evaluator_ledger),
                    'private_ledger_at_target': model.evaluator_ledger[-1], 'prefix_matches_saved_outputs': True})
                print(sid, rep, method, 'actual/oracle/global', round(actual, 3), round(oracle['upper_bound'], 3),
                      round(objective.value(goals), 3), 'optimal', oracle['optimal'], flush=True)
    OUT.write_text(json.dumps({'scope': 'posthoc failure diagnosis, true-reference oracle not a private method; exposed development',
        'source_sha256': sha(SOURCE), 'dataset_sha256': sha(DATA), 'code_sha256': sha(Path(__file__)),
        'oracle_source_sha256': sha(ROOT/'evaluation/coverage_oracle.py'), 'rows': rows}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
