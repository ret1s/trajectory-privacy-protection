"""Frozen native Q utility: current-only, public epoch60 and static rolling TTL.

The protocol is declared before new cache scores. Only six selection families
select a rolling TTL. Existing test was inspected on previous days, so these
results are development evidence. No emission, sampler, planner or POI changes.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmark.static_poi_cache import StaticPoiReplyCache
from benchmark.public_poi_context import PublicPoiContext
from evaluation.live_poi import EpochResponseCache
from data.lane_states import build_lane_states
from evaluation.lane_travel import LanePoiService
from experiments.build_future_sumo_cohort import ROOT, sha, save
from experiments.future_sumo_eval import DATA, OUT as PRIMARY, compressed_save
from experiments.native_future_retrieval_depth import OUT as DEPTH, PHASES, load

OUT = ROOT/'artifacts/benchmarks/native_static_cache_20261006_v1'
METHOD_DEPTH = {'geoi_session_reset': 10, 'geoi_epoch8': 20}
POLICIES = ('current_only', 'public_epoch60', 'rolling60', 'rolling120', 'rolling180')


def declaration():
    paths = [DATA, PRIMARY/'public_transcripts.json.gz', PRIMARY/'private_accounting.json.gz', PRIMARY/'results.json',
             DEPTH/'results.json', DEPTH/'public_reply40.npz', DEPTH/'public_resource_archive.json',
             ROOT/'benchmark/static_poi_cache.py', ROOT/'evaluation/live_poi.py',
             ROOT/'benchmark/query_purpose.py', Path(__file__)]
    return {'schema': 'native-static-cache-protocol-v1',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths},
        'method_depth': METHOD_DEPTH, 'policies': POLICIES, 'rolling_TTL_s': [60, 120, 180],
        'public_epoch60': 'existing EpochResponseCache; retain replies within floor(relative public time/60); reset at boundary',
        'rolling': 'retain latest static POI records if 0<=public age<TTL; generally rolling60 includes three20s events; extra predeclared clocks also count',
        'preselection': 'minimum rolling TTL with selection family-macro Recall>=.90 and minimum family tail400..600 Recall>=.90 and exactly unchanged requests/reply bytes',
        'failure': 'no selected TTL if none passes; preserve all failed policies',
        'splits': {'selection': [f'native-{i:02d}' for i in range(13, 19)], 'test': [f'native-{i:02d}' for i in range(19, 25)]},
        'phases_s': PHASES, 'ranking': 'same reference5 category distance/tie ordering; cached IDs are ranked locally using current GPS',
        'public_requests': 'frozen Q and timestamps; all categories, previously selected L; every cache policy has identical service requests/replies',
        'static_scope': 'cache id/category/lat/lon only; native all-POIs static; no liveavailability or dynamictraffic claim',
        'dynamic_guard': 'live answer requires current fixed-public-epoch known/available IDs from public replies; missing/stale =>unknown; absent from cover=>unknown, not unavailable',
        'utility': 'conditional on nonempty reachable reference; report coverage and unavailable reference windows without imputation',
        'cost': 'same per-Q compactJSON full id/category/lat/lon reply-only serialization estimate; no new requests and no request suppression',
        'privacy': 'local postprocessing only; unchanged Q/order/epsilon/budget/attacker; no new privacy score improvement asserted',
        'status': 'development on already-inspected native test; new cache scores sealed before selection/test readout'}


def aggregate(rows):
    families = sorted({r['family_id'] for r in rows})
    means = {}
    session_values = []
    for family in families:
        values = [r['recall5'] for r in rows if r['family_id'] == family and r['recall5'] is not None]
        means[family] = float(np.mean(values)) if values else None
        for slot in range(8):
            subset = [r['recall5'] for r in rows if r['family_id'] == family and r['slot'] == slot and r['recall5'] is not None]
            if subset:
                session_values.append({'family_id': family, 'slot': slot, 'recall5': float(np.mean(subset))})
    values = [v for v in means.values() if v is not None]
    defined = sum(r['recall5'] is not None for r in rows)
    return {'family_macro_recall5': float(np.mean(values)), 'family_values': means,
        'minimum_family_recall5': min(values), 'maximum_family_recall5': max(values),
        'median_session_recall5': float(np.median([s['recall5'] for s in session_values])),
        'minimum_session': min(session_values, key=lambda s: s['recall5']),
        'reference_defined_windows': defined, 'total_windows': len(rows), 'reference_coverage': defined/len(rows),
        'undefined_session_count': len(families)*8-len(session_values),
        'mean_cached_poi_count': float(np.mean([r['cached_poi_count'] for r in rows])),
        'maximum_cached_poi_count': max(r['cached_poi_count'] for r in rows),
        'reply_bytes_total': sum(r['reply_bytes'] for r in rows), 'request_count': sum(r['requests'] for r in rows),
        'session_values': session_values}


def score():
    if (OUT/'results.json').exists() or (OUT/'utility_rows.json.gz').exists():
        raise FileExistsError('Preserve completed static-cache evidence')
    data, public, private = load(DATA), load(PRIMARY/'public_transcripts.json.gz'), load(PRIMARY/'private_accounting.json.gz')
    rn = build_lane_states(ROOT/data['network']['compressed_path'], spacing_m=40.)
    pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')} for p in
        json.loads((ROOT/'artifacts/benchmarks/research_loop/resources.json').read_text())['pois_used']]
    reply = PublicPoiContext(LanePoiService(rn, pois, k=40), DEPTH/'public_reply40.npz')
    resources = json.loads((DEPTH/'results.json').read_text())['resources']
    assert reply.sha256 == resources['reply40_sha256'] and rn.catalogue_sha256 == resources['catalogue_sha256']
    poi_ids = [p['id'] for p in reply.pois]
    index = {name: i for i, name in enumerate(poi_ids)}
    response_cache = {}
    def response(state, depth):
        key = int(state), depth
        if key not in response_cache:
            categories = reply.signatures[reply.access[int(state)], :, :depth]
            arrays = [v[v >= 0] for v in categories]
            ids = np.concatenate(arrays)
            records = [{k: reply.pois[int(i)][k] for k in ('id', 'category', 'lat', 'lon')} for i in ids]
            size = len(json.dumps({'results': records}, separators=(',', ':'), ensure_ascii=False).encode())
            response_cache[key] = arrays, records, size
        return response_cache[key]
    by_family = {f['family_id']: f for f in data['families']}
    rows = []
    for group, truth in zip(public['groups'], private['rows']):
        if truth['split'] not in ('selection', 'test'):
            continue
        family = by_family[truth['family_id']]
        for slot, spec in enumerate(family['evaluator_only']['sessions']):
            trace = {int(p['time_s']): p for p in data['traces'][spec['session_id']]}
            for method, depth in METHOD_DEPTH.items():
                epoch_cache = EpochResponseCache(len(poi_ids))
                rolling = {ttl: StaticPoiReplyCache(poi_ids, ttl_s=ttl) for ttl in (60, 120, 180)}
                for event in group['streams'][method][slot]['events']:
                    t = int(event['timestamp_s'])
                    # Only the service response lookup consumes public Q.
                    responses = [response(rn.nearest(q['lat'], q['lon'])[0], depth) for q in event['candidates']]
                    current = set(int(i) for categories, _, _ in responses for ids in categories for i in ids)
                    _, known = epoch_cache.receive(int(t//60), [r[0] for r in responses])
                    static_records = [p for _, records, _ in responses for p in records]
                    masks = {'current_only': current, 'public_epoch60': set(map(int, np.flatnonzero(known)))}
                    for ttl, cache in rolling.items():
                        masks[f'rolling{ttl}'] = {index[i] for i in cache.receive(t, static_records)}
                    # Evaluator truth only defines recall, never cache membership.
                    gps = trace[t];state, _ = rn.nearest(gps['lat'], gps['lon'])
                    refs = [set(map(int, ids[ids >= 0])) for ids in reply.signatures[state, :, :5] if np.any(ids >= 0)]
                    for policy, ids in masks.items():
                        recall = float(np.mean([len(ref & ids)/len(ref) for ref in refs])) if refs else None
                        rows.append({'family_id': truth['family_id'], 'split': truth['split'], 'slot': slot,
                            'method': method, 'L': depth, 'policy': policy, 't': t, 'recall5': recall,
                            'cached_poi_count': len(ids), 'nonempty_reference_categories': len(refs),
                            'reply_bytes': sum(r[2] for r in responses), 'requests': len(responses),
                            'cached_poi_ids': sorted(ids)})
        print('Static local cache evaluated', truth['family_id'], flush=True)
    results = {}
    for method in METHOD_DEPTH:
        results[method] = {}
        for split in ('selection', 'test'):
            results[method][split] = {policy: {phase: aggregate([r for r in rows if r['method'] == method and
                r['split'] == split and r['policy'] == policy and start <= r['t'] <= end])
                for phase, (start, end) in PHASES.items()} for policy in POLICIES}
    selected = {}
    for method in METHOD_DEPTH:
        metrics = results[method]['selection']
        baseline = metrics['current_only']['all_0_600']
        gates = {}
        for ttl in (60, 120, 180):
            policy = f'rolling{ttl}'
            overall, tail = metrics[policy]['all_0_600'], metrics[policy]['tail_400_600']
            same_cost = baseline['reply_bytes_total'] == overall['reply_bytes_total'] and baseline['request_count'] == overall['request_count']
            passed = overall['family_macro_recall5'] >= .9 and tail['minimum_family_recall5'] >= .9 and same_cost
            gates[str(ttl)] = {'passes': passed, 'family_macro_recall5': overall['family_macro_recall5'],
                'minimum_tail_family_recall5': tail['minimum_family_recall5'], 'unchanged_requests_and_reply_bytes': same_cost}
        eligible = [ttl for ttl in (60, 120, 180) if gates[str(ttl)]['passes']]
        ttl = min(eligible) if eligible else None
        selected[method] = {'selected_TTL_s': ttl, 'selection_gates': gates,
            'test_at_selected_TTL': results[method]['test'][f'rolling{ttl}'] if ttl is not None else None}
    compressed_save(OUT/'utility_rows.json.gz', {'schema': 'native-static-cache-rows-v1', 'rows': rows})
    save(OUT/'results.json', {'schema': 'native-static-cache-readout-v1',
        'protocol_sha256': sha(OUT/'protocol.json'), 'source_sha256': declaration()['source_sha256'],
        'utility_rows_sha256': sha(OUT/'utility_rows.json.gz'), 'results': results, 'selection': selected,
        'resources': resources, 'Q_and_primary_attacker_unchanged': True,
        'liveavailability_evaluated': False, 'scope': declaration()['status']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('declare', 'score', 'all'), default='all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = declaration()
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == json.loads(json.dumps(protocol))
    else:
        save(OUT/'protocol.json', protocol)
    if args.stage in ('score', 'all'):
        score()


if __name__ == '__main__':
    main()
