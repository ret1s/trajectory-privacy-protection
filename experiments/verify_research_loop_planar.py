"""Verify planar primitive replay, fixed service contract and exact case views."""
import json
import hashlib
import numpy as np
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from evaluation.lane_travel import LanePoiService, SparseTravel
from benchmark.planar_anchor import PlanarAnchorModel, PredictivePlanarLaplace
from experiments.rng_util import rng_from_key
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_planar_cases import summaries, CONTROLS
from experiments.research_loop_cases import public_view, geometric_predictions, target_xy, recall_pair, mean_optional
from experiments.verify_research_loop_extended import ledger_check

BASE = ROOT/'artifacts/benchmarks/research_loop'


def replay_primitive(rn, ex, traces):
    seeds = rng_from_key(ex['session_id']+f"/{ex['rep']}", schema='persistent-exact-case-clock-v1').integers(0, 2**63, 2, dtype=np.int64)
    primitive = PredictivePlanarLaplace(rn, epsilon_release=.01, epsilon_test=.01,
                                       theta=200., rng=np.random.default_rng(int(seeds[0])))
    trace = traces[ex['session_id']]
    spent, last_read, anchor = 0, None, None
    anchors, ledger = [], []
    for index in ex['clock_indices']:
        p = trace[index]; t = p['time_s']-trace[0]['time_s']
        cost = 0; read = False
        if last_read is not None and t-last_read < 60.:
            branch = 'public_clock_skip'
        elif spent+(1 if anchor is None else 2) > 23:
            branch = 'postprocess'
        else:
            read = True; first = anchor is None; before = primitive.n_resample
            anchor = primitive.perturb(p['lat'], p['lon'], t)
            fresh = primitive.n_resample > before
            cost = (0 if first else 1)+int(fresh)
            branch = 'fresh' if fresh else 'reuse'
            last_read = t
        spent += cost
        anchors.append(list(anchor))
        ledger.append({'cost_units': cost, 'spent_units': spent, 'branch': branch, 'private_read': read})
    assert ex['evaluator_anchors'] == anchors
    assert ex['evaluator_ledger'] == ledger
    assert ex['evaluator_anchor_sha256'] == hashlib.sha256(json.dumps(anchors).encode()).hexdigest()


def check():
    path = BASE/'iteration26_planar_anchor_cases.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text()); json.dumps(result, allow_nan=False)
    assert result['code_sha256'] == sha(ROOT/'experiments/research_loop_planar_cases.py')
    for name, digest in result['source_sha256'].items():
        assert sha(ROOT/name) == digest
    source_path = BASE/'iteration17_paced_slack_cases.json'
    data_path = ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json'
    assert result['control_source_sha256'] == sha(source_path)
    assert result['dataset_sha256'] == sha(data_path)
    source = json.loads(source_path.read_text()); data = json.loads(data_path.read_text())
    controls = {(r['session_id'], r['rep'], r['method']): r for r in source['executions'] if r['method'] in CONTROLS}
    rn, _, context, belief, _ = load(); travel = SparseTravel(rn)
    deeper = PublicPoiContext(LanePoiService(rn, list(context.pois), k=10), CACHE/'poi10.npz')
    model = PlanarAnchorModel(ResponseAwareAnchorModel(belief, deeper))
    assert result['belief_model_sha256'] == model.sha256
    lookup = {}; new = replayed = 0
    for ex in result['executions']:
        key = ex['session_id'], ex['rep'], ex['method']; assert key not in lookup
        lookup[key] = ex; ledger_check(ex, False)
        if ex['method'] in CONTROLS:
            assert ex == controls[key]; replayed += 1; continue
        new += 1; sid, rep, method = key
        replay_primitive(rn, ex, data['traces'])
        assert ex['clock_indices'] == controls[sid, rep, 'raw']['clock_indices']
        assert len(ex['evaluator_states']) == len(ex['events']) == len(ex['planner_diagnostics'])
        previous, previous_t = None, None
        for index, states, diagnostic in zip(ex['clock_indices'], ex['evaluator_states'], ex['planner_diagnostics']):
            event = ex['events'][str(index)]; t = event['timestamp_s']; p = data['traces'][sid][index]
            coords = [[c['lat'], c['lon']] for c in event['candidates']]
            assert len(coords) == 5
            assert coords == [list(rn.latlon(s)) for s in states]
            if previous is not None:
                assert all(b in travel.reachable(a, t-previous_t) for a, b in zip(previous, states))
            previous, previous_t = states, t
            query_states = [rn.nearest(*c)[0] for c in coords]
            truth = rn.nearest(p['lat'], p['lon'])[0]
            assert ex['utility_by_index'][str(index)] == recall_pair(context, deeper, query_states, truth)
            assert ex['reply_items_by_index'][str(index)] == {str(L): int(np.sum(deeper.signatures[query_states, :, :L] >= 0)) for L in (5, 10)}
            if 'pre_progress_states' in diagnostic:
                assert np.array_equal(deeper.signatures[deeper.access[diagnostic['pre_progress_states']]], deeper.signatures[deeper.access[diagnostic['progress_states']]])
            slack = .03 if method.endswith('slack03') else 0.
            if 'objective_after_slack' in diagnostic:
                assert diagnostic['slack_states_after'] == states
                assert diagnostic['objective_after_slack'] >= diagnostic['objective_before_slack']-slack-1e-12
    assert new == result['new_executions'] == 132 and replayed == result['replayed_control_executions'] == 198
    records = {r['record_id']: r for r in data['records']}
    source_rows = {(r['record_id'], r['rep'], r['method']): r for r in source['rows'] if r['method'] in CONTROLS}
    for row in result['rows']:
        if row['method'] in CONTROLS:
            assert row == source_rows[row['record_id'], row['rep'], row['method']]; continue
        record = records[row['record_id']]; predictions, truth, utilities = [], [], []
        for slot, (sid, indices) in enumerate(zip(record['session_ids'], record['observed_indices'])):
            ex = lookup[sid, row['rep'], row['method']]
            view = public_view({int(i): e for i, e in ex['events'].items()}, indices)
            assert view == row['public_views'][slot]
            full = geometric_predictions(view, record['scenario'], rn)
            predictions.append(full); truth.append(target_xy(record, slot, data['traces'][sid], rn))
            utilities.extend(ex['utility_by_index'][str(i)] for i in indices)
        values = {a: np.concatenate([p[a] for p in predictions]) for a in predictions[0]}
        if record['case_id'] in {'S9.C', 'S10.C'}:
            values.update({a+'_joint_mean': np.repeat(p.mean(axis=0, keepdims=True), 2, axis=0) for a, p in list(values.items())})
        target = np.concatenate(truth)
        assert row['errors'] == {a: np.linalg.norm(p-target, axis=1).tolist() for a, p in values.items()}
        assert row['recall'] == {L: mean_optional([u[L] for u in utilities]) for L in ('5', '10')}
        assert row['composition_bound'] == .23*len(set(record['session_ids']))
    assert result['summaries'] == summaries(result['rows'])
    return {'file': path.name, 'sha256': sha(path), 'new_executions': new, 'replayed_controls': replayed,
            'case_rows': len(result['rows']), 'primitive_and_ledger_replayed': True, 'server_coordinate_service_and_road_feasibility_checked': True,
            'scope': 'exposed core development; finite geometric bank only; not privacy confirmation'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
