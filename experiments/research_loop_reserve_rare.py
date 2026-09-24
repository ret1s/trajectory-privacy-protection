"""Unchanged reserve candidates on every expanded S1.C source session."""
from functools import partial
import gzip
import json
from pathlib import Path
import numpy as np
from benchmark.engines.reserve_paced import ReservePacedProgressLaneDummy, ReservePacedSlackProgressLaneDummy
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from evaluation.lane_travel import LanePoiService
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_cases import recall_pair, mean_optional
from experiments.rng_util import rng_from_key

BASE = ROOT/'artifacts/benchmarks/research_loop'
DATA = ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'
SOURCE = BASE/'iteration18_expanded_screening.json'
OUT = BASE/'iteration19_expanded_rare_diagnostic.json'
CONTROLS = ('response_paced', 'response_paced_slack03')


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, _, reference, base, metadata = load()
    reply = PublicPoiContext(LanePoiService(rn, list(reference.pois), k=10), CACHE/'poi10.npz')
    belief = ResponseAwareAnchorModel(base, reply)
    factories = {'response_reserve': ReservePacedProgressLaneDummy,
                 'response_reserve_slack03': partial(ReservePacedSlackProgressLaneDummy, utility_slack=.03)}
    models = {m: cls(rn, belief_model=belief, k=5, budget=.24, horizon=12, rng=np.random.default_rng(0))
              for m, cls in factories.items()}
    data = json.loads(DATA.read_text()); source = json.loads(SOURCE.read_text())
    records = {r['family_id']: r for r in data['records'] if r['case_id'] == 'S1.C'}
    assert len(records) == 12
    executions, rows = [], []
    for item in source['shards']:
        path = BASE/'expanded_screening'/item['file']; assert sha(path) == item['sha256']
        shard = json.loads(gzip.decompress(path.read_bytes()))
        record = records[item['family_id']]
        sid = record['session_ids'][0]; indices = record['observed_indices'][0]; trace = data['traces'][sid]
        local = [r for r in shard['executions'] if r['session_id'] == sid and r['method'] in CONTROLS]
        for rep in range(2):
            clock = next(r['clock_indices'] for r in local if r['rep'] == rep)
            seeds = rng_from_key(sid+f'/{rep}', schema='persistent-exact-case-clock-v1').integers(0, 2**63, 2, dtype=np.int64)
            for method, model in models.items():
                model.anchor_rng, model.dummy_rng = (np.random.default_rng(int(s)) for s in seeds)
                model.reset(); events, utility = {}, {}
                for i in clock:
                    p = trace[i]; t = p['time_s']-trace[0]['time_s']
                    coordinates = model.protect_step(p['lat'], p['lon'], t)
                    events[str(i)] = {'timestamp_s': t, 'candidates': [
                        {'candidate_id': f'candidate_{j:04d}', 'lat': lat, 'lon': lon}
                        for j, (lat, lon) in enumerate(coordinates)]}
                    states = [rn.nearest(lat, lon)[0] for lat, lon in coordinates]
                    truth = rn.nearest(p['lat'], p['lon'])[0]
                    utility[str(i)] = recall_pair(reference, reply, states, truth)
                assert model.spent_bound <= .23+1e-12
                local.append({'session_id': sid, 'family_id': item['family_id'], 'split': 'expanded_development',
                    'rep': rep, 'method': method, 'clock_indices': clock, 'events': events,
                    'utility_by_index': utility, 'budget_bound': model.spent_bound,
                    'evaluator_ledger': model.evaluator_ledger, 'step_ms': list(model.step_ms)})
        for ex in local:
            target_time = ex['events'][str(indices[0])]['timestamp_s']
            reads = [e['timestamp_s'] for e, l in zip(ex['events'].values(), ex['evaluator_ledger'])
                     if l['private_read'] and e['timestamp_s'] <= target_time]
            rows.append({'family_id': item['family_id'], 'session_id': sid, 'record_id': record['record_id'],
                'rep': ex['rep'], 'method': ex['method'], 'case_id': 'S1.C',
                'recall_L10': mean_optional([ex['utility_by_index'][str(i)]['10'] for i in indices]),
                'whole_session_recall_L10': mean_optional([v['10'] for v in ex['utility_by_index'].values()]),
                'target_time_s': target_time, 'last_private_read_before_target_s': max(reads),
                'read_age_at_target_s': target_time-max(reads), 'budget_bound': ex['budget_bound']})
        executions.extend(local)
        print('Completed', item['family_id'], len(executions), 'executions including replayed controls', flush=True)
    summaries = []
    for method in (*CONTROLS, *factories):
        subset = [r for r in rows if r['method'] == method]
        summaries.append({'method': method, 'families': 12, 'RNG_repetitions': 2,
            'recall_L10': float(np.mean([r['recall_L10'] for r in subset])),
            'whole_session_recall_L10': float(np.mean([r['whole_session_recall_L10'] for r in subset])),
            'family_recall_L10': {f: float(np.mean([r['recall_L10'] for r in subset if r['family_id'] == f])) for f in sorted(records)},
            'mean_read_age_at_target_s': float(np.mean([r['read_age_at_target_s'] for r in subset]))})
    OUT.write_text(json.dumps({'scope': 'all expanded S1.C source sessions; utility diagnostic, NOT all-case or learned privacy confirmation',
        'new_executions': 48, 'replayed_control_executions': 48, 'source_sha256': sha(SOURCE),
        'dataset_sha256': sha(DATA), 'resources': metadata, 'code_sha256': sha(Path(__file__)),
        'implementation_sha256': {p: sha(ROOT/p) for p in ('benchmark/engines/reserve_paced.py',
            'benchmark/engines/paced_guard.py', 'benchmark/engines/paced_slack.py',
            'benchmark/engines/slack_progress.py', 'benchmark/engines/matched_filter.py',
            'benchmark/engines/progress_cover.py', 'benchmark/response_aware_belief.py')},
        'executions': executions, 'rows': rows, 'summaries': summaries}, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
