"""Development ablation: auxiliary-fitted mobility with response-aware paced planning."""
from collections import defaultdict
from functools import partial
from pathlib import Path
import json
import hashlib
from scipy.sparse import load_npz
import numpy as np
from benchmark.anchor_belief import PublicAnchorModel
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from benchmark.engines.matched_filter import MatchedFilteredProgressCoverLaneDummy
from benchmark.engines.paced_guard import PacedProgressLaneDummy, PacedOriginGuardLaneDummy
from benchmark.engines.empirical_paced import EmpiricalPacedProgressLaneDummy, EmpiricalPacedSlackProgressLaneDummy
from benchmark.empirical_mobility import EmpiricalMobilityModel
from evaluation.lane_travel import LanePoiService
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.rng_util import rng_from_key
from experiments.research_loop_cases import (public_view, geometric_predictions, target_xy,
    mean_optional, recall_pair, passes_recall_gate)

SOURCE = ROOT/'artifacts/benchmarks/research_loop/iteration17_paced_slack_cases.json'
FIT = ROOT/'artifacts/benchmarks/research_loop/iteration20_mobility_fit.json'
DATA = ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json'
OUT = ROOT/'artifacts/benchmarks/research_loop/iteration20_mobility_cases.json'
CONTROLS = ('raw', 'response_paced', 'response_paced_slack03')


def summaries(rows):
    result = []
    for method in sorted({r['method'] for r in rows}):
        for case in sorted({r['case_id'] for r in rows}):
            train = [r for r in rows if r['method'] == method and r['case_id'] == case and r['split'] == 'development_train']
            val = [r for r in rows if r['method'] == method and r['case_id'] == case and r['split'] == 'development_validation']
            names = sorted(train[0]['errors'])
            mae = lambda a: float(np.mean([np.mean(r['errors'][a]) for r in train]))
            selected = {'mae': min(names, key=lambda a: (mae(a), a))}
            hits = {}
            for radius in (50, 100, 200, 500):
                name = min(names, key=lambda a: (-np.mean([np.mean(np.array(r['errors'][a]) <= radius) for r in train]), mae(a), a))
                selected[f'hit{radius}'] = name
                hits[str(radius)] = float(np.mean([np.mean(np.array(r['errors'][name]) <= radius) for r in val]))
            recall = {L: mean_optional([r['recall'][L] for r in val]) for L in ('5', '10')}
            result.append({'method': method, 'case_id': case,
                'validation_families': len({r['family_id'] for r in val}), 'selected_attack': selected,
                'mae_m': float(np.mean([np.mean(r['errors'][selected['mae']]) for r in val])),
                'hits': hits, 'recall': recall, 'pass_90pct_case_recall_L10': passes_recall_gate(recall['10']),
                'privacy_scope': 'finite geometric bank only; mechanism-matched learned attack not yet run'})
    return result


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, service, context, base, metadata = load()
    deeper = PublicPoiContext(LanePoiService(rn, list(context.pois), k=10), CACHE/'poi10.npz')
    fit = json.loads(FIT.read_text()); chosen = fit['selected_generator']
    assert fit['improves_auxiliary_NLL'] and fit['resource_sha256'] == metadata['resource_sha256']
    path = FIT.parent/chosen['file']; assert sha(path) == chosen['sha256']
    belief = EmpiricalMobilityModel(ResponseAwareAnchorModel(base, deeper), load_npz(path),
        {'fit_evidence_sha256': sha(FIT), 'training_sha256': fit['training_sha256'],
         'pseudo_exposure_s': chosen['pseudo_exposure_s'], 'auxiliary_fit_families': 64,
         'auxiliary_selection_families': 16, 'target_case_labels_used': False})
    factories = {'response_empirical_paced': EmpiricalPacedProgressLaneDummy,
                 'response_empirical_paced_slack03': partial(EmpiricalPacedSlackProgressLaneDummy, utility_slack=.03)}
    models = {name: cls(rn, belief_model=belief, k=5, budget=.24, horizon=12, rng=np.random.default_rng(0))
              for name, cls in factories.items()}
    source = json.loads(SOURCE.read_text()); data = json.loads(DATA.read_text())
    assert source['dataset_sha256'] == sha(DATA) and source['resources'] == metadata
    executions = [r for r in source['executions'] if r['method'] in CONTROLS]
    clocks = [r for r in source['executions'] if r['method'] == 'raw']
    for ordinal, control in enumerate(clocks):
        sid, rep = control['session_id'], control['rep']; trace = data['traces'][sid]
        seeds = rng_from_key(sid+f'/{rep}', schema='persistent-exact-case-clock-v1').integers(0, 2**63, 2, dtype=np.int64)
        coupled_anchor_digest = None
        baseline_ledger = next(r['evaluator_ledger'] for r in executions if r['method'] == 'response_paced' and r['session_id'] == sid and r['rep'] == rep)
        for method, model in models.items():
            model.anchor_rng, model.dummy_rng = (np.random.default_rng(int(s)) for s in seeds)
            model.reset(); events, utilities, item_counts = {}, {}, {}
            for i in control['clock_indices']:
                p = trace[i]; t = p['time_s']-trace[0]['time_s']
                coordinates = model.protect_step(p['lat'], p['lon'], t)
                events[i] = {'timestamp_s': t, 'candidates': [
                    {'candidate_id': f'candidate_{j:04d}', 'lat': lat, 'lon': lon}
                    for j, (lat, lon) in enumerate(coordinates)]}
                states = [rn.nearest(lat, lon)[0] for lat, lon in coordinates]
                truth, _ = rn.nearest(p['lat'], p['lon'])
                utilities[i] = recall_pair(context, deeper, states, truth)
                item_counts[i] = {str(L): int(np.sum(deeper.signatures[states, :, :L] >= 0)) for L in (5, 10)}
            assert model.spent_bound <= .23+1e-12
            assert model.evaluator_ledger == baseline_ledger
            digest = hashlib.sha256(json.dumps(model.evaluator_anchors).encode()).hexdigest()
            assert coupled_anchor_digest is None or digest == coupled_anchor_digest
            coupled_anchor_digest = digest
            executions.append({'session_id': sid, 'family_id': control['family_id'], 'split': control['split'],
                'rep': rep, 'method': method, 'clock_indices': control['clock_indices'], 'events': events,
                'utility_by_index': utilities, 'budget_bound': model.spent_bound,
                'whole_session_recall': {L: mean_optional([u[L] for u in utilities.values()]) for L in ('5', '10')},
                'reply_items_by_index': item_counts, 'step_ms': list(model.step_ms),
                'quarter_units': False, 'evaluator_ledger': model.evaluator_ledger,
                'evaluator_anchor_sha256': digest})
        print('Completed source/repetition', ordinal+1, '/', len(clocks), sid, rep, flush=True)
    lookup = {(r['session_id'], r['rep'], r['method']): r for r in executions}
    rows = [r for r in source['rows'] if r['method'] in CONTROLS]
    for record in data['records']:
        if record['scenario'] not in {'S1', 'S2', 'S3', 'S9', 'S10'}:
            continue
        for rep in range(2):
            for method in factories:
                predictions, targets, utility, public = [], [], [], []
                for slot, (sid, indices) in enumerate(zip(record['session_ids'], record['observed_indices'])):
                    ex = lookup[sid, rep, method]
                    p = public_view(ex['events'], indices); public.append(p)
                    predictions.append(geometric_predictions(p, record['scenario'], rn))
                    targets.append(target_xy(record, slot, data['traces'][sid], rn))
                    utility.extend(ex['utility_by_index'][i] for i in indices)
                values = {a: np.concatenate([p[a] for p in predictions]) for a in predictions[0]}
                truth = np.concatenate(targets)
                if record['case_id'] in {'S9.C', 'S10.C'}:
                    values.update({a+'_joint_mean': np.repeat(p.mean(axis=0, keepdims=True), 2, axis=0) for a, p in list(values.items())})
                rows.append({'record_id': record['record_id'], 'case_id': record['case_id'], 'scenario': record['scenario'],
                    'family_id': record['family_id'], 'split': record['split'], 'rep': rep, 'method': method,
                    'public_views': public, 'composition_bound': .23*len(set(record['session_ids'])),
                    'errors': {a: np.linalg.norm(p-truth, axis=1).tolist() for a, p in values.items()},
                    'eligible_events': sum(u['5'] is not None for u in utility),
                    'empty_reference_events': sum(u['5'] is None for u in utility),
                    'recall': {L: mean_optional([u[L] for u in utility]) for L in ('5', '10')}})
    result = {'schema': 'empirical-mobility-case-development-v1',
        'scope': 'auxiliary-fitted mobility, same L10 service and top-five target; matched ledger; development, new matched attacks required',
        'control_source_sha256': sha(SOURCE), 'mobility_fit_sha256': sha(FIT), 'mobility_model_sha256': belief.sha256, 'dataset_sha256': sha(DATA), 'resources': metadata,
        'reference_depth': 5, 'optimized_reply_depth': 10, 'K': 5, 'tight_session_cap': .23,
        'new_executions': len(clocks)*len(factories), 'replayed_control_executions': len(clocks)*len(CONTROLS),
        'code_sha256': sha(Path(__file__)), 'source_sha256': {p: sha(ROOT/p) for p in (
            'benchmark/response_aware_belief.py', 'benchmark/engines/paced_guard.py', 'benchmark/engines/paced_slack.py',
            'benchmark/engines/slack_progress.py', 'benchmark/empirical_mobility.py', 'benchmark/engines/empirical_paced.py',
            'benchmark/engines/quotient_cover.py', 'benchmark/engines/progress_cover.py',
            'benchmark/engines/matched_filter.py', 'experiments/research_loop_cases.py')},
        'executions': executions, 'rows': rows, 'summaries': summaries(rows)}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for row in result['summaries']:
        print(row['method'], row['case_id'], row['recall'], flush=True)


if __name__ == '__main__':
    main()
