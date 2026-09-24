"""Exact case views of persistent sessions; conditional fixed request clock.

See iteration 10 in docs/research/algorithm_improvement_loop.md. Two development
families per split and finite attacks cannot establish thesis readiness.
"""
from collections import defaultdict
from functools import partial
from benchmark.engines.paced_guard import PacedProgressLaneDummy,PacedOriginGuardLaneDummy
from benchmark.anchor_belief import PublicAnchorModel
from pathlib import Path
import json
import numpy as np
from benchmark.engines.matched_filter import MatchedFilteredProgressCoverLaneDummy
from benchmark.engines.switching_progress import (
    SwitchingQuotientCoverLaneDummy, MatchedSwitchingProgressCoverLaneDummy)
from benchmark.public_poi_context import PublicPoiContext
from evaluation.lane_travel import LanePoiService
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.rng_util import rng_from_key
from experiments.contribution_stress import endpoint_predictions

DATA = ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json'
OUT = ROOT/'artifacts/benchmarks/research_loop/iteration13_paced_cases.json'
METHODS = {'raw': None, 'switching_core': SwitchingQuotientCoverLaneDummy,
           'filter_progress': MatchedFilteredProgressCoverLaneDummy,
           'filter_paced': PacedProgressLaneDummy, 'filter_paced_origin': PacedOriginGuardLaneDummy}
SCENARIOS = {'S1', 'S2', 'S3', 'S9', 'S10'}


def public_view(events, indices):
    """Whitelist coordinates/time only; remove original position in session."""
    chosen = [events[i] for i in indices]
    epoch = chosen[0]['timestamp_s']
    return {'events': [{'event_id': f'e{j:04d}',
                        'timestamp_s': e['timestamp_s']-epoch,
                        'candidates': e['candidates']}
                       for j, e in enumerate(chosen)]}


def geometric_predictions(public, scenario, rn):
    if scenario in {'S9', 'S10'}:
        result = {k: v[None, :] for k, v in
                  endpoint_predictions(public, scenario, rn.proj.lat0).items()}
    else:
        tracks = np.array([[rn.point_xy(c['lat'], c['lon']) for c in e['candidates']]
                           for e in public['events']])
        result = {'mean': tracks.mean(axis=1), 'median': np.median(tracks, axis=1)}
        result.update({f'track_{i}': tracks[:, i] for i in range(tracks.shape[1])})
        if scenario == 'S2':
            result = {key: value.mean(axis=0, keepdims=True) for key, value in result.items()}
    result.update({key+'_road': rn.xy[rn.tree.query(value)[1]]
                   for key, value in list(result.items())})
    return result


def target_xy(record, slot, trace, rn):
    label = record['labels']
    indices = (record['observed_indices'][slot] if record['scenario'] == 'S3' else
               [label.get('target_index', label.get('target_indices', [None]*len(record['session_ids']))[slot])])
    return np.array([rn.point_xy(trace[i]['lat'], trace[i]['lon']) for i in indices])


def mean_optional(values):
    eligible = [v for v in values if v is not None]
    return float(np.mean(eligible)) if eligible else None


def passes_recall_gate(value):
    # A mathematical 0.9 may round to 0.8999999999999999 under nested means.
    # Absolute numerical tolerance only; no service-threshold relaxation.
    return value is not None and (value >= .9 or abs(value-.9) <= 1e-12)


def recall_pair(context, deeper, states, truth):
    scores = {5: [], 10: []}
    for ci in range(len(context.categories)):
        ref = set(context.signatures[truth, ci]); ref.discard(-1)
        if not ref:
            continue
        for L in scores:
            got = set(deeper.signatures[states, ci, :L].ravel()); got.discard(-1)
            scores[L].append(len(ref & got)/len(ref))
    return {str(L): mean_optional(v) for L, v in scores.items()}


def select_and_summarize(rows):
    summaries = []
    for method in METHODS:
        for case in sorted({r['case_id'] for r in rows}):
            train = [r for r in rows if r['method'] == method and r['case_id'] == case
                     and r['split'] == 'development_train']
            val = [r for r in rows if r['method'] == method and r['case_id'] == case
                   and r['split'] == 'development_validation']
            if not train or not val:
                summaries.append({'method': method, 'case_id': case, 'status': 'missing_split'})
                continue
            names = sorted(train[0]['errors'])
            # Equal weight per case record/repetition, not per observed point.
            mae = min(names, key=lambda a: (np.mean([np.mean(r['errors'][a]) for r in train]), a))
            selected = {'mae': mae}
            hits = {}
            for radius in (50, 100, 200, 500):
                name = min(names, key=lambda a: (-np.mean([np.mean(np.array(r['errors'][a]) <= radius) for r in train]),
                                                np.mean([np.mean(r['errors'][a]) for r in train]), a))
                selected[f'hit{radius}'] = name
                hits[str(radius)] = float(np.mean([np.mean(np.array(r['errors'][name]) <= radius) for r in val]))
            summaries.append({'method': method, 'case_id': case, 'status': 'evaluated_finite_attack_bank',
                'validation_records': len({r['record_id'] for r in val}),
                'validation_families': len({r['family_id'] for r in val}),
                'rng_repetitions': 2, 'selected_attack': selected,
                'reference_eligible_events_per_repetition': sum(r.get('eligible_events', 0) for r in val)/2,
                'reference_empty_events_per_repetition': sum(r.get('empty_reference_events', 0) for r in val)/2,
                'mae_m': float(np.mean([np.mean(r['errors'][mae]) for r in val])),
                'hits': hits,
                'recall': {L: mean_optional([r['recall'][L] for r in val]) for L in ('5', '10')},
                'pass_90pct_case_recall_L10': bool(passes_recall_gate(mean_optional([r['recall']['10'] for r in val])))})
    return summaries


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, service, context, belief, metadata = load()
    deeper = PublicPoiContext(LanePoiService(rn, list(context.pois), k=10), CACHE/'poi10.npz')
    assert np.array_equal(deeper.signatures[:, :, :5], context.signatures)
    data = json.loads(DATA.read_text())
    records = [r for r in data['records'] if r['scenario'] in SCENARIOS]
    clocks = defaultdict(set)
    session_info = {s['session_id']: (f['family_id'], f['split'])
                    for f in data['families'] for s in f['sessions']}
    for r in records:
        for sid, ids in zip(r['session_ids'], r['observed_indices']):
            clocks[sid].update(ids)
    for sid in clocks:
        clocks[sid].update(range(0, len(data['traces'][sid]), 20))
    _, inv, counts = np.unique(np.floor(rn.xy/120.).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    prior = 1./counts[inv]; prior /= prior.sum()
    early = PublicAnchorModel(rn, context, prior, epsilon_release=.0025, epsilon_test=.0025, cache_path=CACHE/'belief_origin_quarter.npz')
    factories = {**METHODS, 'filter_paced_origin': partial(PacedOriginGuardLaneDummy, early_belief_model=early, guard_seconds=0.)}
    models = {name: cls(rn, belief_model=belief, k=5, budget=.24, horizon=12,
                        rng=np.random.default_rng(0)) for name, cls in factories.items() if cls}
    executions = []
    lookup = {}
    for sid, clock in sorted(clocks.items()):
        trace = data['traces'][sid]
        for rep in range(2):
            seeds = rng_from_key(sid+f'/{rep}', schema='persistent-exact-case-clock-v1').integers(0, 2**63, 2, dtype=np.int64)
            for method, cls in METHODS.items():
                model = models.get(method)
                if model:
                    model.anchor_rng, model.dummy_rng = (np.random.default_rng(int(s)) for s in seeds)
                    model.reset()
                events, utilities = {}, {}
                for i in sorted(clock):
                    p = trace[i]; t = p['time_s']-trace[0]['time_s']
                    coordinates = model.protect_step(p['lat'], p['lon'], t) if model else [(p['lat'], p['lon'])]
                    events[i] = {'timestamp_s': t, 'candidates': [
                        {'candidate_id': f'candidate_{j:04d}', 'lat': lat, 'lon': lon}
                        for j, (lat, lon) in enumerate(coordinates)]}
                    states = [rn.nearest(lat, lon)[0] for lat, lon in coordinates]
                    truth, _ = rn.nearest(p['lat'], p['lon'])
                    utilities[i] = recall_pair(context, deeper, states, truth)
                bound = model.spent_bound if model else None
                assert bound is None or bound <= .23+1e-12
                execution = {'session_id': sid, 'family_id': session_info[sid][0], 'split': session_info[sid][1],
                             'rep': rep, 'method': method, 'clock_indices': sorted(clock),
                             'events': events, 'utility_by_index': utilities, 'budget_bound': bound,
                             'whole_session_recall': {L: mean_optional([u[L] for u in utilities.values()]) for L in ('5', '10')},
                             'step_ms': list(model.step_ms) if model else [],
                             'quarter_units': method=='filter_paced_origin', 'evaluator_ledger': model.evaluator_ledger if model and method.startswith('filter') else None}
                executions.append(execution)
                lookup[sid, rep, method] = execution
        print(sid, len(clock), 'queries; completed 10 method/repetition runs', flush=True)
    rows = []
    for r in records:
        for rep in range(2):
            for method in METHODS:
                predictions, targets, utility, public = [], [], [], []
                for slot, (sid, ids) in enumerate(zip(r['session_ids'], r['observed_indices'])):
                    ex = lookup[sid, rep, method]
                    view = public_view(ex['events'], ids)
                    public.append(view)
                    predictions.append(geometric_predictions(view, r['scenario'], rn))
                    targets.append(target_xy(r, slot, data['traces'][sid], rn))
                    utility.extend(ex['utility_by_index'][i] for i in ids)
                names = sorted(predictions[0])
                values = {a: np.concatenate([p[a] for p in predictions]) for a in names}
                truth = np.concatenate(targets)
                if r['case_id'] in {'S9.C', 'S10.C'}:
                    assert len(predictions) == 2 and len(truth) == 2
                    values.update({a+'_joint_mean': np.repeat(v.mean(axis=0, keepdims=True), 2, axis=0)
                                   for a, v in list(values.items())})
                rows.append({'record_id': r['record_id'], 'case_id': r['case_id'], 'scenario': r['scenario'],
                             'family_id': r['family_id'], 'split': r['split'], 'rep': rep, 'method': method,
                             'session_ids_evaluator_only': r['session_ids'], 'public_views': public,
                             'composition_bound': .23*len(set(r['session_ids'])) if method != 'raw' else None,
                             'errors': {a: np.linalg.norm(v-truth, axis=1).tolist() for a, v in values.items()},
                             'eligible_events': sum(u['5'] is not None for u in utility),
                             'empty_reference_events': sum(u['5'] is None for u in utility),
                             'recall': {L: mean_optional([u[L] for u in utility]) for L in ('5', '10')}})
    result = {'schema': 'persistent-paced-exact-case-development-v1',
              'scope': 'fixed benchmark query clocks; no per-window reset; finite attacks; development only; static service issue unresolved',
              'resources': metadata, 'dataset_sha256': sha(DATA), 'code_sha256': sha(Path(__file__)),
              'source_sha256': {p: sha(ROOT/p) for p in (
                  'benchmark/engines/quotient_cover.py', 'benchmark/engines/filtered_cover.py',
                  'benchmark/engines/matched_filter.py', 'benchmark/engines/progress_cover.py',
                  'benchmark/engines/switching_progress.py', 'benchmark/engines/paced_guard.py', 'benchmark/engines/origin_guard.py', 'benchmark/switching_belief.py',
                  'experiments/contribution_stress.py')},
              'executions': executions, 'rows': rows, 'summaries': select_and_summarize(rows)}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for s in result['summaries']:
        print(s['method'], s['case_id'], s['status'], s.get('recall'), s.get('mae_m'), s.get('hits'), flush=True)


if __name__ == '__main__':
    main()
