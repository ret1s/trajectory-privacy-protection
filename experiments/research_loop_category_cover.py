"""Freeze public plans, then evaluate all expanded cases on nine live worlds."""
import argparse
from collections import defaultdict
import json
import time

import networkx as nx
import numpy as np

from evaluation.category_cover import (public_category_profiles, greedy_category_cover,
                                       attach_coordinates, query_category_plan, reply_mask)
from evaluation.live_poi import RankedRoadPois, AvailabilityWorld, LivePointService, score_returned
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_cases import SCENARIOS, mean_optional, passes_recall_gate
from experiments.research_loop_live_service import encoded_size

BASE = ROOT/'artifacts/benchmarks/research_loop'
PROTOCOL = BASE/'iteration29_protocol.json'
PLANS = BASE/'iteration29_plans.json'
OUT = BASE/'iteration29_category_cover.json'
DATA = ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'


def prepare():
    if PLANS.exists():
        raise FileExistsError('Preserve frozen public plans')
    protocol = json.loads(PROTOCOL.read_text())
    started = time.perf_counter()
    rn, service, _, _, meta = load()
    ranking = RankedRoadPois(service, CACHE/'live_poi_full_rank_v1.npy')
    nodes = max(nx.strongly_connected_components(rn.graph), key=len)
    viable = [i for i, node in enumerate(rn.node_ids) if node in nodes]
    profiles, targets = public_category_profiles(rn, ranking, viable, protocol['response_L'])
    plans = {}
    for name, budget in [('public_category_budget30', protocol['category_query_budget']),
                         ('public_category_full_cover', None)]:
        plan = attach_coordinates(greedy_category_cover(profiles, targets, budget), rn, ranking,
                                  protocol['response_L'])
        assert all(int(rn.nearest(*q['coordinate'])[0]) == q['state'] for q in plan['queries'])
        plans[name] = plan
    result = {'protocol_sha256': sha(PROTOCOL), 'resources_sha256': meta['resource_sha256'],
              'ranking_sha256': sha(CACHE/'live_poi_full_rank_v1.npy'),
              'planner_sha256': sha(ROOT/'evaluation/category_cover.py'),
              'candidate_states': len(viable), 'categories': list(ranking.categories),
              'profiles_per_category': [len(p) for p in profiles],
              'fit_seconds': time.perf_counter()-started, 'plans': plans,
              'private_trajectories_opened_during_fit': False}
    PLANS.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name, plan in plans.items():
        print(name, 'queries', plan['category_queries'], 'distinct coordinates', plan['distinct_coordinates'],
              'uncovered', [len(v) for v in plan['uncovered_ids']], flush=True)


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r['method'], r['probability'], r['case_id']].append(r)
    summaries = []
    for (method, probability, case), group in sorted(groups.items()):
        families = sorted({r['family_id'] for r in group})
        values = {f: mean_optional([r['recall'] for r in group if r['family_id'] == f]) for f in families}
        recall = mean_optional(list(values.values()))
        summaries.append({'method': method, 'probability': probability, 'case_id': case,
                          'recall': recall, 'pass_90pct': passes_recall_gate(recall),
                          'family_count': len(families), 'record_count': len({r['record_id'] for r in group}),
                          'family_recall': values})
    return summaries


def evaluate():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    protocol, plans = json.loads(PROTOCOL.read_text()), json.loads(PLANS.read_text())
    assert plans['protocol_sha256'] == sha(PROTOCOL)
    assert plans['planner_sha256'] == sha(ROOT/'evaluation/category_cover.py')
    rn, service, _, _, meta = load()
    assert plans['resources_sha256'] == meta['resource_sha256']
    ranking = RankedRoadPois(service, CACHE/'live_poi_full_rank_v1.npy')
    assert plans['ranking_sha256'] == sha(CACHE/'live_poi_full_rank_v1.npy')
    data = json.loads(DATA.read_text())
    cover = json.loads((BASE/'iteration22_public_cover.json').read_text())
    assert cover['source_sha256'][str(DATA.relative_to(ROOT))] == sha(DATA)
    records = [r for r in data['records'] if r['scenario'] in SCENARIOS]
    rows, session_costs, certificates = [], [], []
    for probability in protocol['probabilities']:
        for seed in protocol['world_seeds']:
            world = AvailabilityWorld(ranking.n, seed, probability, 60)
            server = LivePointService(ranking, world, protocol['response_L'])
            for method, plan in plans['plans'].items():
                epoch_cache, lookup = {}, {}
                query_schema = [{'category': q['category'], 'point': q['coordinate']} for q in plan['queries']]
                for source in cover['sources']:
                    score_by_index = {}
                    cost = {'events': 0, 'category_queries': 0, 'coordinate_occurrences': 0,
                            'response_poi_slots': 0, 'request_json_bytes': 0, 'response_json_bytes': 0}
                    for i, timestamp, truth in zip(source['clock_indices'], source['timestamps_s'], source['reference_states']):
                        epoch = world.epoch(timestamp)
                        if epoch not in epoch_cache:
                            replies = query_category_plan(plan, server, epoch)
                            known = reply_mask(replies, ranking.n)
                            available = world.at_epoch(epoch)
                            assert not np.any(known & ~available)
                            reachable = np.zeros(ranking.n, dtype=bool)
                            for q in plan['queries']:
                                reachable[q['static_ids']] = True
                            assert np.all(known[reachable & available])
                            certificates.append({'method': method, 'probability': probability, 'world_seed': seed,
                                                 'epoch': epoch, 'available_pois': int(available.sum()),
                                                 'returned_unique_pois': int(known.sum()),
                                                 'all_available_catalogue_retrieved': bool(np.all(known[available]))})
                            epoch_cache[epoch] = (known, encoded_size({'epoch': epoch, 'queries': query_schema}),
                                                 encoded_size({'epoch': epoch, 'results': replies}),
                                                 sum(len(v) for v in replies))
                        known, request_bytes, response_bytes, slots = epoch_cache[epoch]
                        available = world.at_epoch(epoch)
                        target = ranking.top(truth, available, 5)
                        returned = ranking.top(truth, known, 5)
                        score = score_returned(target, returned, available)
                        assert score['unavailable_returned_items'] == 0
                        if plan['full_catalogue_cover']:
                            assert score['recall'] in (None, 1.)
                        score_by_index[i] = score
                        cost['events'] += 1
                        cost['category_queries'] += plan['category_queries']
                        cost['coordinate_occurrences'] += plan['category_queries']
                        cost['response_poi_slots'] += slots
                        cost['request_json_bytes'] += request_bytes
                        cost['response_json_bytes'] += response_bytes
                    lookup[source['session_id']] = score_by_index
                    session_costs.append({'method': method, 'probability': probability, 'world_seed': seed,
                                          'session_id': source['session_id'], 'family_id': source['family_id'],
                                          'communication': cost})
                for record in records:
                    scores = [lookup[sid][i] for sid, indices in zip(record['session_ids'], record['observed_indices']) for i in indices]
                    rows.append({'method': method, 'probability': probability, 'world_seed': seed,
                                 'record_id': record['record_id'], 'case_id': record['case_id'],
                                 'family_id': record['family_id'], 'events': len(scores),
                                 'recall': mean_optional([s['recall'] for s in scores]),
                                 'empty_reference_categories': sum(s['empty_reference_categories'] for s in scores)})
            print('Evaluated world', probability, seed, flush=True)
    summaries = summarize(rows)
    result = {'schema': 'category-public-cover-live-results-v1', 'protocol_sha256': sha(PROTOCOL),
              'plans_sha256': sha(PLANS), 'dataset_sha256': sha(DATA),
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                               [ROOT/'evaluation/category_cover.py', ROOT/'evaluation/live_poi.py',
                                BASE/'iteration22_public_cover.json', __import__('pathlib').Path(__file__)]},
              'physical_source_sessions': len(cover['sources']), 'target_records': len(records),
              'independent_confirmation': False, 'summaries': summaries, 'case_rows': rows,
              'sessions': session_costs, 'certificate_checks': certificates}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for method in plans['plans']:
        selected = [s for s in summaries if s['method'] == method and s['probability'] == .8]
        print(method, 'macro', np.mean([s['recall'] for s in selected]), 'gates', sum(s['pass_90pct'] for s in selected), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'evaluate'])
    args = parser.parse_args()
    (prepare if args.stage == 'prepare' else evaluate)()
