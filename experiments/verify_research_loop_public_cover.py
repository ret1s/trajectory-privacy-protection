"""Rebuild public plans and service scores with the actual coordinate interface."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from evaluation.public_cover import fit_public_cover
from benchmark.public_poi_context import PublicPoiContext
from evaluation.lane_travel import LanePoiService
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_cases import recall_pair, mean_optional, passes_recall_gate, SCENARIOS


def check():
    base = ROOT/'artifacts/benchmarks/research_loop'
    path = base/'iteration22_public_cover.json'
    if not path.exists():
        return None
    x = json.loads(path.read_text()); json.dumps(x, allow_nan=False)
    for name, digest in x['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    correction = x['service_access_correction']
    assert sha(base/correction['original_artifact']) == correction['original_sha256']
    old = json.loads((base/correction['original_artifact']).read_text())
    assert old['source_sha256']['experiments/research_loop_public_cover.py'] == sha(base/'sources/iteration22_public_cover_v1.py')
    data = json.loads((ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json').read_text())
    records = [r for r in data['records'] if r['scenario'] in SCENARIOS]
    clocks = defaultdict(set)
    for r in records:
        for sid, indices in zip(r['session_ids'], r['observed_indices']):
            clocks[sid].update(indices)
    for sid in clocks:
        clocks[sid].update(range(0, len(data['traces'][sid]), 20))
    assert len(records) == 173 and len(clocks) == x['physical_source_sessions'] == 102
    assert len(x['plans']) == 5 and x['deterministic_protocol_session_evaluations'] == 510
    rn, _, reference, belief, _ = load()
    service = LanePoiService(rn, list(reference.pois), k=10)
    reply = PublicPoiContext(service, CACHE/'poi10.npz')
    for ex in x['sources']:
        sid = ex['session_id']; indices = sorted(clocks[sid]); trace = data['traces'][sid]
        assert ex['clock_indices'] == indices
        assert ex['timestamps_s'] == [trace[i]['time_s']-trace[0]['time_s'] for i in indices]
        assert ex['reference_states'] == [int(rn.nearest(trace[i]['lat'], trace[i]['lon'])[0]) for i in indices]
    for plan, prior_plan in zip(x['plans'], old['plans']):
        k = plan['k']; rebuilt = fit_public_cover(rn, belief, reply, k)
        for key, value in rebuilt.items():
            assert json.loads(json.dumps(value)) == plan[key], (k, key)
        assert plan['states'] == prior_plan['states']
        states = [int(rn.nearest(*coord)[0]) for coord in plan['coordinates']]
        assert states == plan['server_access_states']
        for coord, state in zip(plan['coordinates'], states):
            for ci, category in enumerate(reference.categories):
                expected = [reply.pois[int(i)]['id'] for i in reply.signatures[state, ci] if i >= 0]
                assert service.query(tuple(coord), category) == expected
        table = plan['utility_by_reference_state']
        for state, utility in table.items():
            assert utility == recall_pair(reference, reply, states, int(state))
        lookup = {ex['session_id']: dict(zip(ex['clock_indices'],
                  [table[str(s)] for s in ex['reference_states']])) for ex in x['sources']}
        for r in records:
            utilities = [lookup[sid][i] for sid, ids in zip(r['session_ids'], r['observed_indices']) for i in ids]
            row = next(v for v in x['case_rows'] if v['k'] == k and v['record_id'] == r['record_id'])
            assert row['events'] == len(utilities)
            assert row['eligible_events'] == sum(u['10'] is not None for u in utilities)
            assert row['recall'] == {L: mean_optional([u[L] for u in utilities]) for L in ('5', '10')}
        for s in [s for s in x['summaries'] if s['k'] == k]:
            rows = [r for r in x['case_rows'] if r['k'] == k and r['case_id'] == s['case_id']]
            value = mean_optional([r['recall']['10'] for r in rows])
            assert value == s['recall_L10'] and s['pass_90pct'] == passes_recall_gate(value)
        for s in [s for s in x['sessions'] if s['k'] == k]:
            assert s['recall_L10'] == mean_optional([u['10'] for u in lookup[s['session_id']].values()])
    source = json.loads((base/'iteration18_expanded_screening.json').read_text())
    for comparison in x['matched_K5_comparisons']:
        control = next(s for s in source['summaries'] if s['method'] == comparison['method'] and s['case_id'] == comparison['case_id'])
        fixed = next(s for s in x['summaries'] if s['k'] == 5 and s['case_id'] == comparison['case_id'])
        assert control['recall']['10'] == comparison['adaptive_recall_L10']
        assert fixed['recall_L10'] == comparison['fixed_public_K5_recall_L10']
        assert comparison['adaptive_minus_fixed'] == control['recall']['10']-fixed['recall_L10']
    paired_path = base/'iteration22_public_cover_comparisons.json'
    if paired_path.exists():
        paired = json.loads(paired_path.read_text())
        for name, digest in paired['source_sha256'].items():
            assert sha(ROOT/name) == digest
        assert len(paired['rows']) == 30
        for row in paired['rows']:
            control = next(s for s in source['summaries'] if s['method'] == row['method'] and s['case_id'] == row['case_id'])
            fixed_rows = [r for r in x['case_rows'] if r['k'] == 5 and r['case_id'] == row['case_id']]
            families = sorted(control['family_L10'])
            delta = np.array([control['family_L10'][f]-np.mean([r['recall']['10'] for r in fixed_rows if r['family_id'] == f]) for f in families])
            draws = np.random.default_rng(paired['seed']).integers(0, len(delta), (paired['draws'], len(delta)))
            assert row['family_deltas'] == dict(zip(families, delta.tolist()))
            assert row['adaptive_minus_public_K5_recall'] == float(delta.mean())
            assert row['family_bootstrap_percentile_95'] == np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()
    return {'file': path.name, 'sha256': sha(path), 'public_plans_rebuilt': 5,
            'server_coordinate_interface_checked': True, 'source_sessions': 102,
            'case_rows': len(x['case_rows']), 'paired_comparisons': 30 if paired_path.exists() else None,
            'scope': 'public-only conditional fixed-clock service control; not confirmation'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
