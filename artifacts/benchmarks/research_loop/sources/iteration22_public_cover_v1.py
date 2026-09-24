"""Fixed public-query bandwidth frontier on every expanded target case.

No fitting on user traces. Expanded scores are exposed development, not a new
confirmation set. Stationary coordinate streams are stored losslessly as one
public query set plus each session's fixed request clock.
"""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from evaluation.public_cover import fit_public_cover
from benchmark.public_poi_context import PublicPoiContext
from evaluation.lane_travel import LanePoiService
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_cases import recall_pair, mean_optional, passes_recall_gate, SCENARIOS

BASE = ROOT/'artifacts/benchmarks/research_loop'
DATA = ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'
SOURCE = BASE/'iteration18_expanded_screening.json'
OUT = BASE/'iteration22_public_cover.json'
KS = (1, 3, 5, 8, 12)


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, _, reference, belief, meta = load()
    reply = PublicPoiContext(LanePoiService(rn, list(reference.pois), k=10), CACHE/'poi10.npz')
    # Freeze all public plans before opening private target traces.
    plans = [fit_public_cover(rn, belief, reply, k) for k in KS]
    data = json.loads(DATA.read_text())
    records = [r for r in data['records'] if r['scenario'] in SCENARIOS]
    clocks = defaultdict(set)
    info = {s['session_id']: f['family_id'] for f in data['families'] for s in f['sessions']}
    for r in records:
        for sid, indices in zip(r['session_ids'], r['observed_indices']):
            clocks[sid].update(indices)
    for sid in clocks:
        clocks[sid].update(range(0, len(data['traces'][sid]), 20))
    sources = []
    for sid, clock in sorted(clocks.items()):
        trace = data['traces'][sid]; indices = sorted(clock)
        sources.append({'session_id': sid, 'family_id': info[sid], 'clock_indices': indices,
            'timestamps_s': [trace[i]['time_s']-trace[0]['time_s'] for i in indices],
            'reference_states': [int(rn.nearest(trace[i]['lat'], trace[i]['lon'])[0]) for i in indices]})
    unique = sorted({s for ex in sources for s in ex['reference_states']})
    rows, sessions, summaries = [], [], []
    sources_by_id = {ex['session_id']: ex for ex in sources}
    for plan in plans:
        k, states = plan['k'], plan['states']
        table = {str(s): recall_pair(reference, reply, states, s) for s in unique}
        plan['utility_by_reference_state'] = table
        plan['coordinate_requests_per_event'] = k
        plan['category_queries_per_event'] = k*len(reference.categories)
        plan['reply_POI_slots_per_event'] = int(np.sum(reply.signatures[states] >= 0))
        plan['reply_POI_slot_upper_bound_per_event'] = k*10*len(reference.categories)
        lookup = {}
        for ex in sources:
            utilities = [table[str(s)] for s in ex['reference_states']]
            lookup[ex['session_id']] = dict(zip(ex['clock_indices'], utilities))
            sessions.append({'k': k, 'session_id': ex['session_id'], 'family_id': ex['family_id'],
                'events': len(utilities), 'eligible_events': sum(u['10'] is not None for u in utilities),
                'recall_L10': mean_optional([u['10'] for u in utilities])})
        for r in records:
            utilities = [lookup[sid][i] for sid, ids in zip(r['session_ids'], r['observed_indices']) for i in ids]
            rows.append({'k': k, 'record_id': r['record_id'], 'case_id': r['case_id'], 'family_id': r['family_id'],
                'events': len(utilities), 'eligible_events': sum(u['10'] is not None for u in utilities),
                'recall': {L: mean_optional([u[L] for u in utilities]) for L in ('5', '10')}})
        for case in sorted({r['case_id'] for r in records}):
            subset = [r for r in rows if r['k'] == k and r['case_id'] == case]
            recall = mean_optional([r['recall']['10'] for r in subset])
            summaries.append({'k': k, 'case_id': case, 'families': len({r['family_id'] for r in subset}),
                'records': len(subset), 'recall_L10': recall, 'pass_90pct': passes_recall_gate(recall)})
        print('K', k, 'gates', sum(s['pass_90pct'] for s in summaries if s['k'] == k),
              'whole-session Recall', mean_optional([s['recall_L10'] for s in sessions if s['k'] == k]), flush=True)
    comparisons = []
    for s in json.loads(SOURCE.read_text())['summaries']:
        if s['method'] not in ('response_paced', 'response_paced_slack03'):
            continue
        fixed = next(r for r in summaries if r['k'] == 5 and r['case_id'] == s['case_id'])
        comparisons.append({'method': s['method'], 'case_id': s['case_id'],
            'adaptive_recall_L10': s['recall']['10'], 'fixed_public_K5_recall_L10': fixed['recall_L10'],
            'adaptive_minus_fixed': s['recall']['10']-fixed['recall_L10']})
    result = {'schema': 'public-fixed-cover-service-audit-v1', 'resources': meta,
        'scope': 'all expanded development cases; deterministic conditional fixed-clock GPS-independent control',
        'coordinate_privacy_statement': 'Same fixed coordinates for every GPS trajectory conditional on public region/catalogue/request clock; no claim about timing, identity or session boundaries.',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), DATA, SOURCE,
            ROOT/'evaluation/public_cover.py', ROOT/'benchmark/engines/fair_cover.py',
            ROOT/'benchmark/engines/quotient_cover.py', ROOT/'benchmark/engines/service_cover.py')},
        'physical_source_sessions': len(sources), 'deterministic_protocol_session_evaluations': len(sources)*len(KS),
        'independent_confirmation': False, 'plans': plans, 'sources': sources, 'sessions': sessions,
        'case_rows': rows, 'summaries': summaries, 'matched_K5_comparisons': comparisons}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
