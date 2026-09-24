"""Audit provenance, all aggregates, matched cache contrasts and direct service."""
import json
from pathlib import Path

import numpy as np

from evaluation.live_poi import AvailabilityWorld, RankedRoadPois
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_public_supplement import digest
from experiments.research_loop_live_service import BASE, OUT, PROTOCOL, summarize


def check():
    if not OUT.exists():
        return None
    result = json.loads(OUT.read_text()); protocol = json.loads(PROTOCOL.read_text())
    assert result['protocol'] == protocol
    assert result['provenance']['protocol_sha256'] == sha(PROTOCOL)
    for file, expected in result['provenance']['source_sha256'].items():
        assert sha(ROOT/file) == expected
    for item in result['provenance']['parent_shards']:
        assert sha(ROOT/item['file']) == item['sha256']
    assert result['provenance']['dataset_sha256'] == sha(ROOT/protocol['dataset'])
    assert result['provenance']['screen_sha256'] == sha(BASE/protocol['source_transcripts'])
    assert result['provenance']['cover_sha256'] == sha(BASE/protocol['source_plans'])
    rows, sessions = [], []
    for manifest in result['shards']:
        path = ROOT/manifest['file']; assert sha(path) == manifest['sha256']
        shard = json.loads(path.read_text())
        assert shard['provenance'] == result['provenance']
        assert shard['summaries'] == summarize(shard['case_rows'])
        assert len(shard['case_rows']) == manifest['case_rows']
        assert len(shard['sessions']) == manifest['sessions']
        world = AvailabilityWorld(result['poi_count'], shard['world_seed'], shard['probability'], protocol['epoch_seconds'])
        for epoch, expected in shard['availability_sha256_by_epoch'].items():
            assert digest(world.at_epoch(int(epoch)).astype(int).tolist()) == expected
        mapping = {(r['method'], r['mode'], r['record_id'], r['rep']): r for r in shard['case_rows']}
        for row in shard['case_rows']:
            if row['mode'] == 'epoch_cache':
                fresh = mapping[row['method'], 'fresh', row['record_id'], row['rep']]
                assert fresh['recall'] is None or row['recall'] >= fresh['recall']-1e-12
                assert row['empty_reference_categories'] == fresh['empty_reference_categories']
                if row['method'].startswith('fixed_'):
                    assert fresh['recall'] == row['recall']
            if row['method'] in ('raw_current', 'bulk_current'):
                assert row['recall'] in (None, 1.)
        session_map = {(s['method'], s['mode'], s['session_id'], s['rep']): s for s in shard['sessions']}
        for s in shard['sessions']:
            if s['mode'] == 'epoch_cache':
                fresh = session_map[s['method'], 'fresh', s['session_id'], s['rep']]
                assert s['communication'] == fresh['communication']
                assert s['original_public_sha256'] == fresh['original_public_sha256']
                assert s['budget_bound'] == fresh['budget_bound']
        rows.extend(shard['case_rows']); sessions.extend(shard['sessions'])
    assert len(rows) == result['case_rows']
    assert len(sessions) == result['deterministic_service_session_evaluations']
    summaries = []
    for probability in protocol['sensitivity_probabilities']:
        summaries.extend({'probability': probability, **s} for s in summarize([r for r in rows if r['probability'] == probability]))
    assert summaries == result['summaries']
    rn, service, _, _, metadata = load()
    assert metadata['resource_sha256'] == result['provenance']['resources_sha256']
    ranking = RankedRoadPois(service, CACHE/'live_poi_full_rank_v1.npy')
    assert sha(CACHE/'live_poi_full_rank_v1.npy') == result['provenance']['ranking_sha256']
    # Independent forward Dijkstra and ID-based filtering; do not merely compare
    # the evaluator to a second invocation of the same ranking builder.
    points = [rn.latlon(i) for i in np.linspace(0, len(rn)-1, 12, dtype=int)]
    checks = 0
    for probability in protocol['sensitivity_probabilities']:
        world = AvailabilityWorld(ranking.n, protocol['world_seeds'][0], probability)
        for epoch in (0, 1, 5):
            mask = world.at_epoch(epoch)
            for point in points:
                state = rn.nearest(*point)[0]; distances = service.distances(point)
                got = ranking.top(state, mask, protocol['response_L'])
                for category, ids in zip(ranking.categories, got):
                    expected = sorted((i for i, p in enumerate(ranking.pois)
                        if mask[i] and p['category'] == category and p['id'] in distances),
                        key=lambda i: (distances[ranking.pois[i]['id']], ranking.pois[i]['id']))[:protocol['response_L']]
                    assert ids == expected
                    checks += 1
    report = {'status': 'passed', 'file': str(OUT.relative_to(ROOT)), 'sha256': sha(OUT),
              'case_rows_verified': len(rows), 'session_rows_verified': len(sessions),
              'direct_forward_shortest_path_category_checks': checks,
              'all_record_and_session_cache_monotonicity_and_cost_equality': True,
              'all_summaries_recomputed': True,
              'scope': 'evaluation correctness and provenance, not six-paper superiority or privacy confirmation',
              'verifier_sha256': sha(Path(__file__))}
    return report


if __name__ == '__main__':
    report = check()
    (BASE/'iteration28_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
