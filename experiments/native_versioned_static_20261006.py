"""Public-version static POI metadata within the existing eight-session epoch.

New development candidate after inspected TTL diagnosis. All Q, GPS, privacy
accounting, catalogue records, selected reply depths and attacker scores stay
unchanged. Only local use of causally received STATIC records is extended.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmark.versioned_static_poi_cache import VersionedStaticPoiCache
from benchmark.public_poi_context import PublicPoiContext
from data.lane_states import build_lane_states
from evaluation.lane_travel import LanePoiService
from experiments.build_future_sumo_cohort import ROOT, sha, save
from experiments.future_sumo_eval import DATA, OUT as PRIMARY, compressed_save
from experiments.native_future_retrieval_depth import OUT as DEPTH, load
from experiments.native_static_cache_20261006 import OUT as TTL, METHOD_DEPTH, PHASES, aggregate

OUT = ROOT/'artifacts/benchmarks/native_versioned_static_20261006_v1'
EPOCH_ID = 'native-eight-trip-public-epoch'


def declaration():
    paths = [DATA, PRIMARY/'public_transcripts.json.gz', PRIMARY/'private_accounting.json.gz', PRIMARY/'results.json',
        DEPTH/'results.json', DEPTH/'public_reply40.npz', TTL/'protocol.json', TTL/'results.json',
        TTL/'weakest_tail_diagnostic.json', ROOT/'benchmark/static_poi_cache.py',
        ROOT/'benchmark/versioned_static_poi_cache.py', Path(__file__)]
    return {'schema': 'native-versioned-static-protocol-v1',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in paths}, 'method_depth': METHOD_DEPTH,
        'public_epoch': {'id': EPOCH_ID, 'start_s': 0., 'end_s': 12000., 'session_starts_s': [1500*i for i in range(8)]},
        'cache': 'one public catalogue version per declared linked-subject eight-session epoch; causal latest static records persist across sessions; invalidate on version/epoch mismatch or expiry',
        'storage_limit': 'exact current public POI catalogue size418; unknown IDs rejected',
        'preacceptance': 'selection family-macro Recall>=.90, minimum family tail400..600 Recall>=.90, cachedIDs<=418 and exactly unchanged requests/reply bytes',
        'failure': 'retain failed candidate and do not recommend it if selection gates fail',
        'splits': {'selection': [f'native-{i:02d}' for i in range(13, 19)], 'test': [f'native-{i:02d}' for i in range(19, 25)]},
        'phases_s': PHASES, 'causality': 'public absolute time=1500*slot+relativetime; only replies already received at that time; no metadata from future sessions',
        'static_records': ['id', 'category', 'lat', 'lon'],
        'availability': 'not evaluated; current fixed60s public-status known/available mask required or unknown/error; static metadata never implies available',
        'wire': 'unchanged Q/order/timestamps/K/selectedL/allcategory schema/current replies; no extra requests, GPS reads or reply bytes; no request suppression',
        'cost': 'same compactJSON full POI metadata reply-only serialization estimate as prior depth run; not HTTP/request/latency measurements',
        'privacy': 'local metadata postprocessing; no sampler, belief, anchor, Q or primary attacker change; no privacy score gain asserted',
        'status': 'posthoc development after weak-test inspection; not fresh confirmation; prior TTL run immutable'}


def score():
    if (OUT/'results.json').exists() or (OUT/'utility_rows.json.gz').exists():
        raise FileExistsError('Preserve completed versioned-static evidence')
    data, public, private = load(DATA), load(PRIMARY/'public_transcripts.json.gz'), load(PRIMARY/'private_accounting.json.gz')
    rn = build_lane_states(ROOT/data['network']['compressed_path'], spacing_m=40.)
    pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')} for p in
        json.loads((ROOT/'artifacts/benchmarks/research_loop/resources.json').read_text())['pois_used']]
    reply = PublicPoiContext(LanePoiService(rn, pois, k=40), DEPTH/'public_reply40.npz')
    resources = json.loads((DEPTH/'results.json').read_text())['resources']
    assert reply.sha256 == resources['reply40_sha256']
    poi_ids = [p['id'] for p in reply.pois];assert len(poi_ids) == 418
    indices = {name: i for i, name in enumerate(poi_ids)}
    version = reply.sha256
    response_cache = {}
    def response(state, depth):
        key = int(state), depth
        if key not in response_cache:
            ids = reply.signatures[reply.access[int(state)], :, :depth].ravel();ids = ids[ids >= 0]
            records = [{k: reply.pois[int(i)][k] for k in ('id', 'category', 'lat', 'lon')} for i in ids]
            size = len(json.dumps({'results': records}, separators=(',', ':'), ensure_ascii=False).encode())
            response_cache[key] = records, size
        return response_cache[key]
    family_by_id = {f['family_id']: f for f in data['families']}
    rows = []
    for group, truth in zip(public['groups'], private['rows']):
        if truth['split'] not in ('selection', 'test'):
            continue
        family = family_by_id[truth['family_id']]
        caches = {m: VersionedStaticPoiCache(poi_ids, catalogue_version=version, epoch_id=EPOCH_ID,
            start_s=0., end_s=12000.) for m in METHOD_DEPTH}
        for slot, spec in enumerate(family['evaluator_only']['sessions']):
            assert spec['depart_s'] == slot*1500.
            trace = {int(p['time_s']): p for p in data['traces'][spec['session_id']]}
            for method, depth in METHOD_DEPTH.items():
                for event in group['streams'][method][slot]['events']:
                    t = int(event['timestamp_s']);absolute = slot*1500.+t
                    responses = [response(rn.nearest(q['lat'], q['lon'])[0], depth) for q in event['candidates']]
                    records = [p for response_records, _ in responses for p in response_records]
                    ids = {indices[name] for name in caches[method].receive(absolute, records, catalogue_version=version, epoch_id=EPOCH_ID)}
                    gps = trace[t];state, _ = rn.nearest(gps['lat'], gps['lon'])
                    refs = [set(map(int, v[v >= 0])) for v in reply.signatures[state, :, :5] if np.any(v >= 0)]
                    recall = float(np.mean([len(v & ids)/len(v) for v in refs])) if refs else None
                    rows.append({'family_id': truth['family_id'], 'split': truth['split'], 'slot': slot,
                        'method': method, 'L': depth, 'policy': 'versioned_epoch8', 't': t,
                        'absolute_public_time_s': absolute, 'recall5': recall, 'cached_poi_count': len(ids),
                        'nonempty_reference_categories': len(refs), 'reply_bytes': sum(r[1] for r in responses),
                        'requests': len(responses), 'cached_poi_ids': sorted(ids)})
        print('Versioned static epoch evaluated', truth['family_id'], flush=True)
    results = {m: {s: {phase: aggregate([r for r in rows if r['method'] == m and r['split'] == s and start <= r['t'] <= end])
        for phase, (start, end) in PHASES.items()} for s in ('selection', 'test')} for m in METHOD_DEPTH}
    ttl = json.loads((TTL/'results.json').read_text())
    acceptance = {}
    for method in METHOD_DEPTH:
        overall = results[method]['selection']['all_0_600'];tail = results[method]['selection']['tail_400_600']
        old = ttl['results'][method]['selection']['current_only']['all_0_600']
        unchanged = overall['request_count'] == old['request_count'] and overall['reply_bytes_total'] == old['reply_bytes_total']
        passed = overall['family_macro_recall5'] >= .9 and tail['minimum_family_recall5'] >= .9 and overall['maximum_cached_poi_count'] <= 418 and unchanged
        acceptance[method] = {'accepted_for_static_selection_scope': passed, 'selection_family_macro_recall5': overall['family_macro_recall5'],
            'selection_minimum_tail_family_recall5': tail['minimum_family_recall5'], 'maximum_cacheIDs': overall['maximum_cached_poi_count'],
            'unchanged_requests_and_reply_bytes': unchanged}
    compressed_save(OUT/'utility_rows.json.gz', {'schema': 'native-versioned-static-rows-v1', 'rows': rows})
    save(OUT/'results.json', {'schema': 'native-versioned-static-readout-v1', 'protocol_sha256': sha(OUT/'protocol.json'),
        'source_sha256': declaration()['source_sha256'], 'utility_rows_sha256': sha(OUT/'utility_rows.json.gz'),
        'resources': resources, 'results': results, 'selection_acceptance': acceptance,
        'static_catalogue_version': version, 'catalogue_size': 418, 'Q_and_primary_attacker_unchanged': True,
        'liveavailability_evaluated': False, 'scope': declaration()['status']})


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage', choices=('declare', 'score', 'all'), default='all')
    args = parser.parse_args();OUT.mkdir(parents=True, exist_ok=True);protocol = declaration()
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == json.loads(json.dumps(protocol))
    else:
        save(OUT/'protocol.json', protocol)
    if args.stage in ('score', 'all'):
        score()


if __name__ == '__main__':
    main()
