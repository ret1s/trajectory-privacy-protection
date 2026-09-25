"""Independent source, direct-distance, client-wire and aggregate checks."""
from collections import defaultdict
import json
from pathlib import Path
import networkx as nx
import numpy as np

from benchmark.category_client import PublicCategoryClient
from evaluation.category_cover import public_category_profiles, greedy_category_cover, attach_coordinates
from evaluation.live_poi import RankedRoadPois, AvailabilityWorld, LivePointService
from experiments.research_loop_resources import ROOT, CACHE, load, sha

BASE = ROOT/'artifacts/benchmarks/research_loop'
OUT = BASE/'iteration30_verification.json'


def main():
    plans = json.loads((BASE/'iteration29_plans.json').read_text())
    dev = json.loads((BASE/'iteration29_category_cover.json').read_text())
    confirm = json.loads((BASE/'iteration30_category_confirmation.json').read_text())
    readout = json.loads((BASE/'iteration30_readout.json').read_text())
    for doc in (dev, confirm, readout):
        for path, expected in doc['source_sha256'].items():
            assert sha(ROOT/path) == expected, path
    assert dev['plans_sha256'] == sha(BASE/'iteration29_plans.json')
    assert dev['protocol_sha256'] == plans['protocol_sha256'] == sha(BASE/'iteration29_protocol.json')
    rn, service, _, _, meta = load()
    assert plans['resources_sha256'] == meta['resource_sha256']
    ranking = RankedRoadPois(service, CACHE/'live_poi_full_rank_v1.npy')
    assert plans['ranking_sha256'] == sha(CACHE/'live_poi_full_rank_v1.npy')
    nodes = max(nx.strongly_connected_components(rn.graph), key=len)
    candidates = [i for i, node in enumerate(rn.node_ids) if node in nodes]
    profiles, targets = public_category_profiles(rn, ranking, candidates, 10)
    for method, budget in [('public_category_budget30', 30), ('public_category_full_cover', None)]:
        recomputed = attach_coordinates(greedy_category_cover(profiles, targets, budget), rn, ranking)
        assert recomputed == plans['plans'][method]
        assert not recomputed['full_catalogue_cover']  # Do not confuse observed 100% with a global certificate.
    checks = 0
    # Forward shortest paths, with independent category sorting/filtering; do
    # not just compare two calls to the reverse-rank evaluator.
    points = [q['coordinate'] for q in plans['plans']['public_category_full_cover']['queries']]
    points += [list(rn.latlon(i)) for i in np.linspace(0, len(rn)-1, 12, dtype=int)]
    for probability in (.5, .8, .95):
        world = AvailabilityWorld(ranking.n, 24093001, probability)
        for epoch in (0, 1, 5):
            mask = world.at_epoch(epoch)
            for point in points:
                state = int(rn.nearest(*point)[0])
                distances = service.distances(point)
                for k in (5, 10):
                    got = ranking.top(state, mask, k)
                    for c, category in enumerate(ranking.categories):
                        expected = sorted((i for i, p in enumerate(ranking.pois)
                                           if mask[i] and p['category'] == category and p['id'] in distances),
                                          key=lambda i: (distances[ranking.pois[i]['id']], ranking.pois[i]['id']))[:k]
                        assert got[c] == expected
                        checks += 1
    aggregation_checks = 0
    for doc in (dev, confirm):
        grouped = defaultdict(list)
        for row in doc['case_rows']:
            grouped[row['method'], row['probability'], row['case_id'], row['family_id']].append(row['recall'])
        for s in doc['summaries']:
            by_family = {}
            for key, values in grouped.items():
                if key[:3] != (s['method'], s['probability'], s['case_id']):
                    continue
                eligible = [v for v in values if v is not None]
                by_family[key[-1]] = float(np.mean(eligible)) if eligible else None
            assert by_family.keys() == s['family_recall'].keys()
            assert all((by_family[f] is None and s['family_recall'][f] is None) or
                       (by_family[f] is not None and abs(by_family[f]-s['family_recall'][f]) < 1e-12)
                       for f in by_family)
            eligible = [v for v in by_family.values() if v is not None]
            assert abs(float(np.mean(eligible))-s['recall']) < 1e-12
            assert s['pass_90pct'] == (s['recall'] >= .9-1e-12)
            aggregation_checks += 1
    newdata = json.loads((ROOT/'artifacts/datasets/research_loop_confirmation_v1/dataset.json').read_text())
    olddata = json.loads((ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json').read_text())
    assert not {f['family_id'] for f in newdata['families']} & {f['family_id'] for f in olddata['families']}
    assert len(newdata['families']) == 4 and len(newdata['traces']) == 88
    for f in newdata['families']:
        assert len(f['actual_vehicles']) == 22
        assert all(float(v['arrival']) >= 0 for v in f['actual_vehicles'].values())
    # Two clients may rank the same response differently using private GPS.
    # Their server wire remains identical on an identical public clock.
    plan = plans['plans']['public_category_budget30']
    server = LivePointService(ranking, AvailabilityWorld(ranking.n, 24093002, .8))
    wire_checks = 0
    for refresh in (False, True):
        clients = [PublicCategoryClient(plan, ranking.n, refresh_once_per_epoch=refresh) for _ in range(2)]
        previous = None
        for timestamp in (0, 5, 20, 59, 60, 80, 120):
            def query(request):
                state = int(rn.nearest(*request['coordinate'])[0])
                return server.query(state, request['epoch'])[request['category_index']]
            outputs = [c.step(timestamp, query) for c in clients]
            assert outputs[0]['requests'] == outputs[1]['requests']
            assert outputs[0]['replies'] == outputs[1]['replies']
            assert np.array_equal(outputs[0]['known'], outputs[1]['known'])
            epoch = int(timestamp//60)
            assert len(outputs[0]['requests']) == (0 if refresh and epoch == previous else 30)
            assert not np.any(outputs[0]['known'] & ~server.world.at_epoch(epoch))
            for index, output in enumerate(outputs):
                ranking.top(index*(len(rn)-1), output['known'], 5)
            previous = epoch
            wire_checks += 1
    report = {'status': 'passed', 'public_plans_rebuilt': True,
              'direct_forward_distance_category_checks': checks, 'summary_rows_recomputed': aggregation_checks,
              'equal_clock_wire_noninterference_checks': wire_checks,
              'new_families_disjoint': True, 'new_SUMO_arrivals': '88/88',
              'no_global_full_catalogue_certificate_claimed': True,
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
                  BASE/'iteration29_plans.json', BASE/'iteration29_category_cover.json',
                  BASE/'iteration30_category_confirmation.json', BASE/'iteration30_readout.json',
                  ROOT/'benchmark/category_client.py', Path(__file__)]},
              'scope': 'Implementation/evaluation verification and a fixed-clock information-flow check; not protection of metadata or superiority to six papers.'}
    OUT.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
