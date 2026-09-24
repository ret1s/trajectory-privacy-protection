"""Paired synthetic service evaluation of unchanged coordinate transcripts.

Nine frozen availability settings, causal local caching, point and bulk controls.
This changes the SERVICE workload, not the privacy mechanism or test population.
"""
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from evaluation.live_poi import AvailabilityWorld, RankedRoadPois, LivePointService, EpochResponseCache, score_returned
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_cases import mean_optional, passes_recall_gate, SCENARIOS
from experiments.research_loop_public_supplement import digest

BASE = ROOT/'artifacts/benchmarks/research_loop'
PROTOCOL = BASE/'iteration28_protocol.json'
OUT = BASE/'iteration28_live_service.json'
SHARDS = BASE/'live_service'


def encoded_size(obj):
    return len(json.dumps(obj, separators=(',', ':'), allow_nan=False).encode())


def sources(rn, protocol):
    data_path = ROOT/protocol['dataset']
    data = json.loads(data_path.read_text())
    screen_path, cover_path = BASE/protocol['source_transcripts'], BASE/protocol['source_plans']
    screen, cover = json.loads(screen_path.read_text()), json.loads(cover_path.read_text())
    assert screen['provenance']['dataset_sha256'] == sha(data_path)
    truth = {s['session_id']: s for s in cover['sources']}
    for sid, source in truth.items():
        for i, state, timestamp in zip(source['clock_indices'], source['reference_states'], source['timestamps_s']):
            point = data['traces'][sid][i]
            assert state == int(rn.nearest(point['lat'], point['lon'])[0])
            assert timestamp == point['time_s'] - data['traces'][sid][0]['time_s']
    executions, manifests = [], []
    for item in screen['shards']:
        path = BASE/'expanded_screening'/item['file']
        assert sha(path) == item['sha256']
        manifests.append({'file': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        shard = json.loads(gzip.decompress(path.read_bytes()))
        for ex in shard['executions']:
            if ex['method'] not in protocol['parents']:
                continue
            sid = ex['session_id']; source = truth[sid]
            assert ex['clock_indices'] == source['clock_indices']
            events = [ex['events'][str(i)] for i in ex['clock_indices']]
            assert [e['timestamp_s'] for e in events] == source['timestamps_s']
            coordinates = [[(c['lat'], c['lon']) for c in e['candidates']] for e in events]
            executions.append({'method': ex['method'], 'rep': ex['rep'], 'session_id': sid,
                'states': [[int(rn.nearest(*c)[0]) for c in cs] for cs in coordinates],
                'coordinates': coordinates, 'original_public_sha256': digest(events),
                'budget_bound': ex['budget_bound']})
    for sid, source in truth.items():
        for k in protocol['fixed_query_counts']:
            plan = next(p for p in cover['plans'] if p['k'] == k)
            states = [int(rn.nearest(*c)[0]) for c in plan['coordinates']]
            assert states == plan['server_access_states']
            executions.append({'method': f'fixed_K{k}', 'rep': 0, 'session_id': sid,
                'states': [states]*len(source['clock_indices']),
                'coordinates': [plan['coordinates']]*len(source['clock_indices']), 'budget_bound': 0.})
        executions.append({'method': 'raw_current', 'rep': 0, 'session_id': sid,
            'states': [[s] for s in source['reference_states']],
            'coordinates': [[(data['traces'][sid][i]['lat'], data['traces'][sid][i]['lon'])]
                            for i in source['clock_indices']], 'budget_bound': None})
        for name in ('static_local', 'stale_epoch0', 'bulk_current'):
            executions.append({'method': name, 'rep': 0, 'session_id': sid,
                'states': None, 'coordinates': None, 'budget_bound': 0.})
    provenance = {'dataset_sha256': sha(data_path), 'screen_sha256': sha(screen_path),
                  'cover_sha256': sha(cover_path), 'parent_shards': manifests}
    return truth, executions, [r for r in data['records'] if r['scenario'] in SCENARIOS], provenance


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['method'], row['mode'], row['case_id']].append(row)
    result = []
    for (method, mode, case), subset in sorted(groups.items()):
        families = sorted({r['family_id'] for r in subset})
        # RNG and service worlds are repeated observations, not new families.
        values = {f: mean_optional([r['recall'] for r in subset if r['family_id'] == f]) for f in families}
        recall = mean_optional(list(values.values()))
        result.append({'method': method, 'mode': mode, 'case_id': case,
            'recall': recall, 'pass_90pct': passes_recall_gate(recall),
            'family_count': len(families), 'record_count': len({r['record_id'] for r in subset}),
            'replicate_rows': len(subset), 'family_recall': values,
            'empty_reference_category_observations': sum(r['empty_reference_categories'] for r in subset),
            'category_recall': [mean_optional([r['category_recall'][c] for r in subset])
                                for c in range(len(subset[0]['category_recall']))]})
    return result


def score_world(ranking, truth, executions, records, protocol, probability, seed, verbose=False):
    world = AvailabilityWorld(ranking.n, seed, probability, protocol['epoch_seconds'])
    server = LivePointService(ranking, world, protocol['response_L'])
    lookup, sessions = {}, []
    all_available = np.ones(ranking.n, dtype=bool)
    for number, ex in enumerate(executions):
        sid, method, rep = ex['session_id'], ex['method'], ex['rep']
        source = truth[sid]; cache = EpochResponseCache(ranking.n)
        point_method = ex['states'] is not None
        modes = ('fresh', 'epoch_cache') if point_method and method != 'raw_current' else ('fresh',)
        scores = {m: {} for m in modes}
        cost = {'coordinate_queries': 0, 'response_poi_slots': 0, 'request_json_bytes': 0,
                'response_body_bytes': 0, 'status_bitmap_refreshes': 0}
        previous_epoch = None
        for j, (i, timestamp, actual_state) in enumerate(zip(source['clock_indices'], source['timestamps_s'], source['reference_states'])):
            epoch = world.epoch(timestamp); available = world.at_epoch(epoch)
            target = ranking.top(actual_state, available, protocol['reference_k'])
            if point_method:
                replies = [server.query(state, epoch) for state in ex['states'][j]]
                current, history = cache.receive(epoch, replies)
                masks = {'fresh': current, 'epoch_cache': history}
                cost['coordinate_queries'] += len(replies)
                cost['response_poi_slots'] += sum(len(ids) for reply in replies for ids in reply)
                cost['request_json_bytes'] += encoded_size({'epoch': epoch, 'points': ex['coordinates'][j]})
                cost['response_body_bytes'] += encoded_size({'epoch': epoch, 'topL': replies})
            else:
                masks = {'fresh': {'static_local': all_available,
                    'stale_epoch0': world.at_epoch(0), 'bulk_current': available}[method]}
                refresh = (method == 'bulk_current' and epoch != previous_epoch) or (method == 'stale_epoch0' and previous_epoch is None)
                if refresh:
                    cost['status_bitmap_refreshes'] += 1
                    cost['request_json_bytes'] += encoded_size({'epoch': epoch if method == 'bulk_current' else 0, 'all_status': True})
                    cost['response_body_bytes'] += 8 + int(np.ceil(ranking.n/8))
            for mode in modes:
                returned = ranking.top(actual_state, masks[mode], protocol['reference_k'])
                score = score_returned(target, returned, available)
                scores[mode][i] = score
                if method in ('raw_current', 'bulk_current'):
                    assert score['recall'] in (None, 1.)
                if point_method:
                    assert score['unavailable_returned_items'] == 0
            if len(modes) == 2:
                a, b = scores['fresh'][i]['recall'], scores['epoch_cache'][i]['recall']
                assert a is None or b >= a-1e-12
                if method.startswith('fixed_'):
                    assert scores['fresh'][i] == scores['epoch_cache'][i]
            previous_epoch = epoch
        for mode in modes:
            lookup[method, mode, sid, rep] = scores[mode]
            sessions.append({'method': method, 'mode': mode, 'session_id': sid,
                'family_id': source['family_id'], 'rep': rep, 'probability': probability, 'world_seed': seed,
                'events': len(scores[mode]), 'recall': mean_optional([r['recall'] for r in scores[mode].values()]),
                'returned_items': sum(r['returned_items'] for r in scores[mode].values()),
                'unavailable_returned_items': sum(r['unavailable_returned_items'] for r in scores[mode].values()),
                'event_scores_sha256': digest(scores[mode]), 'communication': cost,
                'budget_bound': ex['budget_bound'], 'original_public_sha256': ex.get('original_public_sha256')})
        if verbose and (number+1) % 200 == 0:
            print('service', probability, seed, number+1, '/', len(executions), flush=True)
    variants = sorted({(m, mode, rep) for m, mode, sid, rep in lookup})
    rows = []
    for record in records:
        for method, mode, rep in variants:
            selected = [lookup[method, mode, sid, rep][i]
                        for sid, indices in zip(record['session_ids'], record['observed_indices']) for i in indices]
            rows.append({'method': method, 'mode': mode, 'rep': rep, 'probability': probability,
                'world_seed': seed, 'record_id': record['record_id'], 'case_id': record['case_id'],
                'family_id': record['family_id'], 'events': len(selected),
                'eligible_events': sum(s['recall'] is not None for s in selected),
                'recall': mean_optional([s['recall'] for s in selected]),
                'category_recall': [mean_optional([s['category_recall'][c] for s in selected]) for c in range(len(ranking.categories))],
                'empty_reference_categories': sum(s['empty_reference_categories'] for s in selected)})
    return {'probability': probability, 'world_seed': seed, 'sessions': sessions,
            'case_rows': rows, 'summaries': summarize(rows),
            'availability_sha256_by_epoch': {str(e): digest(m.astype(int).tolist()) for e, m in sorted(world.snapshots.items())}}


def main():
    protocol = json.loads(PROTOCOL.read_text())
    rn, service, _, _, metadata = load()
    ranking = RankedRoadPois(service, CACHE/'live_poi_full_rank_v1.npy')
    truth, executions, records, provenance = sources(rn, protocol)
    provenance.update(protocol_sha256=sha(PROTOCOL), resources_sha256=metadata['resource_sha256'],
        ranking_sha256=sha(CACHE/'live_poi_full_rank_v1.npy'),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), ROOT/'evaluation/live_poi.py')})
    SHARDS.mkdir(exist_ok=True)
    manifest, combined = [], []
    for probability in protocol['sensitivity_probabilities']:
        for seed in protocol['world_seeds']:
            path = SHARDS/f'p{int(probability*100)}-seed{seed}.json'
            if path.exists():
                result = json.loads(path.read_text())
                assert result['provenance'] == provenance
            else:
                result = score_world(ranking, truth, executions, records, protocol, probability, seed, True)
                result['provenance'] = provenance
                path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
            combined.extend(result['case_rows'])
            manifest.append({'file': str(path.relative_to(ROOT)), 'sha256': sha(path),
                             'sessions': len(result['sessions']), 'case_rows': len(result['case_rows'])})
    summaries = []
    for probability in protocol['sensitivity_probabilities']:
        summaries.extend({'probability': probability, **s} for s in summarize([r for r in combined if r['probability'] == probability]))
    result = {'schema': 'live-service-development-v1', 'provenance': provenance, 'protocol': protocol,
        'scope': 'new synthetic service on unchanged development coordinates; no new privacy efficacy or independent confirmation claim',
        'categories': ranking.categories, 'poi_count': ranking.n, 'physical_source_sessions': len(truth),
        'target_records': len(records), 'new_stochastic_full_session_executions': 0,
        'deterministic_service_session_evaluations': sum(m['sessions'] for m in manifest),
        'case_rows': len(combined), 'shards': manifest, 'summaries': summaries}
    if OUT.exists():
        assert json.loads(OUT.read_text()) == json.loads(json.dumps(result))
    else:
        OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for probability in protocol['sensitivity_probabilities']:
        for method, mode in sorted({(r['method'], r['mode']) for r in summaries}):
            subset = [r for r in summaries if r['probability'] == probability and r['method'] == method and r['mode'] == mode]
            rare = next(r for r in subset if r['case_id'] == 'S1.C')
            print(probability, method, mode, 'gates', sum(r['pass_90pct'] for r in subset), 'S1.C', rare['recall'], flush=True)


if __name__ == '__main__':
    main()
