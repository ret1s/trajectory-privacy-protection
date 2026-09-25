"""New-family check plus public-epoch refresh cost for every fixed-plan control."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from evaluation.category_cover import reply_mask
from evaluation.live_poi import RankedRoadPois, AvailabilityWorld, LivePointService, score_returned
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_cases import SCENARIOS, mean_optional
from experiments.research_loop_live_service import encoded_size
from experiments.research_loop_category_cover import summarize

BASE = ROOT/'artifacts/benchmarks/research_loop'
PROTOCOL = BASE/'iteration30_protocol.json'
DATA = ROOT/'artifacts/datasets/research_loop_confirmation_v1/dataset.json'
OUT = BASE/'iteration30_category_confirmation.json'


def public_plans(ranking):
    plans = json.loads((BASE/'iteration29_plans.json').read_text())['plans']
    existing = json.loads((BASE/'iteration22_public_cover.json').read_text())['plans']
    for k in (5, 12):
        p = next(p for p in existing if p['k'] == k)
        plans[f'fixed_K{k}'] = {'queries': [
            {'category_index': c, 'category': category, 'coordinate': xy, 'state': state}
            for xy, state in zip(p['coordinates'], p['server_access_states'])
            for c, category in enumerate(ranking.categories)],
            'shared_coordinates': p['coordinates'], 'distinct_coordinates': k,
            'category_queries': k*len(ranking.categories)}
    plans['bulk_current'] = None
    return plans


def sources(data, rn):
    records = [r for r in data['records'] if r['scenario'] in SCENARIOS]
    clocks = defaultdict(set)
    families = {s['session_id']: f['family_id'] for f in data['families'] for s in f['sessions']}
    for r in records:
        for sid, ids in zip(r['session_ids'], r['observed_indices']):
            clocks[sid].update(ids)
    result = []
    for sid, indices in sorted(clocks.items()):
        trace = data['traces'][sid]
        indices.update(range(0, len(trace), 20))
        indices = sorted(indices)
        result.append({'session_id': sid, 'family_id': families[sid], 'clock_indices': indices,
                       'timestamps_s': [trace[i]['time_s']-trace[0]['time_s'] for i in indices],
                       'reference_states': [int(rn.nearest(trace[i]['lat'], trace[i]['lon'])[0]) for i in indices]})
    return result, records


def response(plan, server, epoch, ranking):
    if plan is None:
        return server.world.at_epoch(epoch), {'category_queries': 0, 'distinct_coordinates': 0,
            'response_poi_slots': 0, 'native_request_bytes': encoded_size({'epoch': epoch, 'all_status': True}),
            'native_response_bytes': 8+int(np.ceil(ranking.n/8)),
            'canonical_request_bytes': encoded_size({'epoch': epoch, 'all_status': True}),
            'canonical_response_bytes': 8+int(np.ceil(ranking.n/8))}
    replies = [server.query(q['state'], epoch)[q['category_index']] for q in plan['queries']]
    canonical_request = {'epoch': epoch, 'queries': [
        {'category': q['category'], 'point': q['coordinate']} for q in plan['queries']]}
    canonical_response = {'epoch': epoch, 'results': replies}
    if 'shared_coordinates' in plan:
        native_request = {'epoch': epoch, 'points': plan['shared_coordinates']}
        c = len(ranking.categories)
        native_response = {'epoch': epoch, 'topL': [replies[i:i+c] for i in range(0, len(replies), c)]}
    else:
        native_request, native_response = canonical_request, canonical_response
    return reply_mask(replies, ranking.n), {
        'category_queries': len(replies), 'distinct_coordinates': plan['distinct_coordinates'],
        'response_poi_slots': sum(len(x) for x in replies),
        'native_request_bytes': encoded_size(native_request), 'native_response_bytes': encoded_size(native_response),
        'canonical_request_bytes': encoded_size(canonical_request), 'canonical_response_bytes': encoded_size(canonical_response)}


def evaluate(data, sources_, records, plans, ranking, protocol):
    rows, sessions = [], []
    for probability in protocol['probabilities']:
        for seed in protocol['world_seeds']:
            server = LivePointService(ranking, AvailabilityWorld(ranking.n, seed, probability, 60), 10)
            for method, plan in plans.items():
                cache, lookup = {}, {}
                for source in sources_:
                    previous_epoch = None
                    costs = {m: defaultdict(int) for m in ('every_event', 'epoch_refresh')}
                    scores = {}
                    for i, timestamp, state in zip(source['clock_indices'], source['timestamps_s'], source['reference_states']):
                        epoch = server.world.epoch(timestamp)
                        if epoch not in cache:
                            cache[epoch] = response(plan, server, epoch, ranking)
                        known, cost = cache[epoch]
                        available = server.world.at_epoch(epoch)
                        score = score_returned(ranking.top(state, available, 5), ranking.top(state, known, 5), available)
                        assert score['unavailable_returned_items'] == 0
                        scores[i] = score
                        for mode in costs:
                            costs[mode]['events'] += 1
                            if mode == 'every_event' or epoch != previous_epoch:
                                costs[mode]['refreshes'] += 1
                                for k, v in cost.items(): costs[mode][k] += v
                        previous_epoch = epoch
                    lookup[source['session_id']] = scores
                    sessions.append({'method': method, 'probability': probability, 'world_seed': seed,
                                     'session_id': source['session_id'], 'family_id': source['family_id'],
                                     'costs': {m: dict(v) for m, v in costs.items()}})
                for r in records:
                    values = [lookup[sid][i] for sid, ids in zip(r['session_ids'], r['observed_indices']) for i in ids]
                    rows.append({'method': method, 'probability': probability, 'world_seed': seed,
                                 'case_id': r['case_id'], 'record_id': r['record_id'], 'family_id': r['family_id'],
                                 'recall': mean_optional([v['recall'] for v in values]),
                                 'empty_reference_categories': sum(v['empty_reference_categories'] for v in values)})
            print('Checked frozen plans on world', probability, seed, flush=True)
    return {'case_rows': rows, 'sessions': sessions, 'summaries': summarize(rows)}


def main():
    if OUT.exists(): raise FileExistsError('Preserve completed evidence')
    protocol = json.loads(PROTOCOL.read_text())
    rn, service, _, _, _ = load()
    ranking = RankedRoadPois(service, CACHE/'live_poi_full_rank_v1.npy')
    plans = public_plans(ranking)
    data = json.loads(DATA.read_text())
    assert data['source_sha256'][str((BASE/'iteration29_plans.json').relative_to(ROOT))] == sha(BASE/'iteration29_plans.json')
    assert [f['seed'] for f in data['families']] == protocol['family_seeds']
    sources_, records = sources(data, rn)
    result = evaluate(data, sources_, records, plans, ranking, protocol)
    result.update(schema='frozen-category-new-family-check-v1', protocol=protocol,
                  source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [
                      PROTOCOL, DATA, BASE/'iteration29_plans.json', BASE/'iteration22_public_cover.json',
                      Path(__file__), ROOT/'evaluation/category_cover.py', ROOT/'evaluation/live_poi.py']},
                  family_count=len(data['families']), source_sessions=len(sources_), target_records=len(records),
                  new_SUMO_completed_sessions=len(data['traces']),
                  case_counts={c: sum(r['case_id'] == c for r in records) for c in sorted({r['case_id'] for r in records})},
                  scope=protocol['confirmation_scope'])
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for method in plans:
        selected = [s for s in result['summaries'] if s['probability'] == .8 and s['method'] == method]
        print(method, 'macro', np.mean([s['recall'] for s in selected]), 'gates', sum(s['pass_90pct'] for s in selected), flush=True)


if __name__ == '__main__': main()
