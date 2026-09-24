"""Verify capped-service objectives, immutable controls and exact case views."""
import json
import numpy as np
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from evaluation.lane_travel import LanePoiService, SparseTravel
from benchmark.grouped_capped_service import GroupedCappedServiceIndex
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_capped_cases import summaries, CONTROLS
from experiments.research_loop_cases import public_view, geometric_predictions, target_xy, recall_pair, mean_optional
from experiments.verify_research_loop_extended import ledger_check

BASE = ROOT/'artifacts/benchmarks/research_loop'


def check_profiles():
    results = []
    for filename in ('iteration25_capped_profile.json', 'iteration25_capped_grouped_profile.json'):
        path = BASE/filename; data = json.loads(path.read_text())
        for name, digest in data['source_sha256'].items():
            source = ROOT/name
            if filename == 'iteration25_capped_profile.json' and name == 'benchmark/engines/capped_service.py':
                source = BASE/'sources/iteration25_capped_service_v1.py'
            assert sha(source) == digest, (filename, name)
        for row in data['rows']:
            assert len(row['step_ms']) == row['steps'] == 12
            assert row['mean_ms'] == float(np.mean(row['step_ms']))
            assert row['p95_ms'] == float(np.percentile(row['step_ms'], 95))
        if 'grouped' in filename:
            rows = {r['method']: r for r in data['rows']}
            assert rows['capped90_sparse']['states'] == rows['capped90_grouped']['states']
            assert len({r['anchors_sha256'] for r in rows.values()}) == 1
            assert [rows[m]['reference_rows'] for m in ('capped90_sparse', 'capped90_grouped')] == [2073, 458]
        results.append({'file': filename, 'sha256': sha(path), 'scope': data['scope']})
    return results


def check():
    path = BASE/'iteration25_capped_service_cases.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text()); json.dumps(result, allow_nan=False)
    assert result['code_sha256'] == sha(ROOT/'experiments/research_loop_capped_cases.py')
    for name, digest in result['source_sha256'].items():
        assert sha(ROOT/name) == digest
    source_path = BASE/'iteration17_paced_slack_cases.json'
    anchor_path = BASE/'iteration20_mobility_cases.json'
    data_path = ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json'
    assert result['control_source_sha256'] == sha(source_path)
    assert result['paired_anchor_source_sha256'] == sha(anchor_path)
    assert result['dataset_sha256'] == sha(data_path)
    source = json.loads(source_path.read_text()); data = json.loads(data_path.read_text())
    controls = {(r['session_id'], r['rep'], r['method']): r for r in source['executions'] if r['method'] in CONTROLS}
    anchor_source = json.loads(anchor_path.read_text())
    anchors = {(r['session_id'], r['rep']): r['evaluator_anchor_sha256'] for r in anchor_source['executions'] if r['method'] == 'response_empirical_paced'}
    rn, _, context, belief, _ = load(); travel = SparseTravel(rn)
    deeper = PublicPoiContext(LanePoiService(rn, list(context.pois), k=10), CACHE/'poi10.npz')
    model = ResponseAwareAnchorModel(belief, deeper)
    assert result['belief_model_sha256'] == model.sha256
    index = GroupedCappedServiceIndex(deeper.signatures, deeper.access, model.poi_weights)
    assert index.original_latent_count == 2073 and index.reference.shape[0] == 458
    assert np.array_equal(index.reference[index.latent_groups].toarray(), model.poi_weights.toarray())
    for setup in result['setup_costs'].values():
        assert setup['reference_utility_groups'] == 458 and setup['original_latent_states'] == 2073
    lookup = {}; new = replayed = 0
    for ex in result['executions']:
        key = ex['session_id'], ex['rep'], ex['method']; assert key not in lookup
        lookup[key] = ex; ledger_check(ex, False)
        if ex['method'] in CONTROLS:
            assert ex == controls[key]; replayed += 1; continue
        new += 1; sid, rep, method = key
        assert ex['evaluator_ledger'] == controls[sid, rep, 'response_paced']['evaluator_ledger']
        assert ex['evaluator_anchor_sha256'] == anchors[sid, rep]
        assert ex['clock_indices'] == controls[sid, rep, 'raw']['clock_indices']
        assert len(ex['evaluator_states']) == len(ex['events']) == len(ex['planner_diagnostics'])
        previous, previous_t = None, None
        for index, states, diagnostic in zip(ex['clock_indices'], ex['evaluator_states'], ex['planner_diagnostics']):
            event = ex['events'][str(index)]; t = event['timestamp_s']; p = data['traces'][sid][index]
            coords = [[c['lat'], c['lon']] for c in event['candidates']]
            assert len(coords) == 5
            assert diagnostic['location_service_cap'] == .9
            assert -.000000000001 <= diagnostic['value'] <= .9+1e-12
            assert diagnostic['states_after_slack'] == states
            assert coords == [list(rn.latlon(s)) for s in states]
            if previous is not None:
                assert all(b in travel.reachable(a, t-previous_t) for a, b in zip(previous, states))
            previous, previous_t = states, t
            query_states = [rn.nearest(*c)[0] for c in coords]
            truth = rn.nearest(p['lat'], p['lon'])[0]
            assert ex['utility_by_index'][str(index)] == recall_pair(context, deeper, query_states, truth)
            assert ex['reply_items_by_index'][str(index)] == {str(L): int(np.sum(deeper.signatures[query_states, :, :L] >= 0)) for L in (5, 10)}
            assert np.array_equal(deeper.signatures[deeper.access[diagnostic['states_before_progress']]], deeper.signatures[deeper.access[diagnostic['states_before_slack']]])
            slack = .03 if method.endswith('slack03') else 0.
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
            'case_rows': len(result['rows']), 'paired_anchors_and_ledgers': True,
            'public_reference_grouping_exact': True, 'server_coordinate_service_and_road_feasibility_checked': True,
            'engineering_profiles': check_profiles(),
            'scope': 'exposed core development; finite geometric bank only; not privacy confirmation'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
