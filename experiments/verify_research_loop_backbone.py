"""Verify fixed-backbone ablation, exact controls and mechanism-aware case views."""
import json
import numpy as np
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from evaluation.lane_travel import LanePoiService, SparseTravel
from evaluation.public_cover import fit_public_cover
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_public_backbone import summaries, CONTROLS
from experiments.research_loop_cases import public_view, geometric_predictions, target_xy, recall_pair, mean_optional
from experiments.verify_research_loop_extended import ledger_check

BASE = ROOT/'artifacts/benchmarks/research_loop'


def check():
    path = BASE/'iteration23_public_backbone_cases.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text()); json.dumps(result, allow_nan=False)
    assert result['code_sha256'] == sha(ROOT/'experiments/research_loop_public_backbone.py')
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
    fixed_plan = json.loads(json.dumps(fit_public_cover(rn, model, deeper, 2)))
    fixed_coords = fixed_plan['coordinates']; lookup = {}; new = replayed = 0
    for ex in result['executions']:
        key = ex['session_id'], ex['rep'], ex['method']; assert key not in lookup
        lookup[key] = ex; ledger_check(ex, False)
        if ex['method'] in CONTROLS:
            assert ex == controls[key]; replayed += 1; continue
        new += 1; sid, rep, method = key
        assert ex['evaluator_ledger'] == controls[sid, rep, 'response_paced']['evaluator_ledger']
        assert ex['evaluator_anchor_sha256'] == anchors[sid, rep]
        assert ex['public_backbone_plan'] == fixed_plan
        assert ex['clock_indices'] == controls[sid, rep, 'raw']['clock_indices']
        assert len(ex['evaluator_states']) == len(ex['events']) == len(ex['planner_diagnostics'])
        previous, previous_t = None, None
        for index, states, diagnostic in zip(ex['clock_indices'], ex['evaluator_states'], ex['planner_diagnostics']):
            event = ex['events'][str(index)]; t = event['timestamp_s']; p = data['traces'][sid][index]
            coords = [[c['lat'], c['lon']] for c in event['candidates']]
            assert len(coords) == 5 and coords[:2] == fixed_coords
            assert states[:2] == fixed_plan['states']
            assert coords == [list(rn.latlon(s)) for s in states]
            if previous is not None:
                assert all(b in travel.reachable(a, t-previous_t) for a, b in zip(previous, states))
            previous, previous_t = states, t
            query_states = [rn.nearest(*c)[0] for c in coords]
            truth = rn.nearest(p['lat'], p['lon'])[0]
            assert ex['utility_by_index'][str(index)] == recall_pair(context, deeper, query_states, truth)
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
            moving_view = {'events': [{**e, 'candidates': e['candidates'][2:]} for e in view['events']]}
            full.update({'adaptive_only_'+k: v for k, v in geometric_predictions(moving_view, record['scenario'], rn).items()})
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
            'fixed_tracks_known_to_geometric_attacker': True, 'server_coordinate_service_and_road_feasibility_checked': True,
            'scope': 'exposed core development; finite geometric bank only; not privacy confirmation'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
