"""Independent causal native epoch-cache replay, utility and wire checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmark.public_poi_context import PublicPoiContext
from data.lane_states import build_lane_states
from evaluation.lane_travel import LanePoiService
from experiments.build_future_sumo_cohort import ROOT, sha, save
from experiments.future_sumo_eval import DATA, OUT as PRIMARY
from experiments.native_future_retrieval_depth import OUT as DEPTH, load
from experiments.native_static_cache_20261006 import OUT as TTL, METHOD_DEPTH, PHASES, aggregate
from experiments.native_versioned_static_20261006 import OUT, EPOCH_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-output', type=Path, help='New output path; never replace evidence')
    args = parser.parse_args();output = args.validation_output or OUT/'validation.json'
    if args.validation_output is not None and output.exists():
        raise FileExistsError(f'Preserve completed evidence: {output}')
    protocol = json.loads((OUT/'protocol.json').read_text())
    result = json.loads((OUT/'results.json').read_text())
    assert result['source_sha256'] == protocol['source_sha256']
    for path, expected in protocol['source_sha256'].items():
        assert sha(ROOT/path) == expected, path
    assert result['protocol_sha256'] == sha(OUT/'protocol.json')
    assert result['utility_rows_sha256'] == sha(OUT/'utility_rows.json.gz')
    assert result['Q_and_primary_attacker_unchanged'] and not result['liveavailability_evaluated']
    assert protocol['public_epoch']['id'] == EPOCH_ID
    assert protocol['public_epoch']['session_starts_s'] == [1500*i for i in range(8)]
    data, public, private = load(DATA), load(PRIMARY/'public_transcripts.json.gz'), load(PRIMARY/'private_accounting.json.gz')
    rows = load(OUT/'utility_rows.json.gz')['rows']
    lookup = {(r['family_id'], r['slot'], r['method'], r['t']): r for r in rows}
    assert len(lookup) == len(rows)
    old_rows = load(TTL/'utility_rows.json.gz')['rows']
    old_lookup = {(r['family_id'], r['slot'], r['method'], r['t']): r for r in old_rows if r['policy'] == 'rolling60'}
    rn = build_lane_states(ROOT/data['network']['compressed_path'], spacing_m=40.)
    pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')} for p in
        json.loads((ROOT/'artifacts/benchmarks/research_loop/resources.json').read_text())['pois_used']]
    reply = PublicPoiContext(LanePoiService(rn, pois, k=40), DEPTH/'public_reply40.npz')
    assert reply.sha256 == result['static_catalogue_version'] == result['resources']['reply40_sha256']
    assert rn.catalogue_sha256 == result['resources']['catalogue_sha256']
    assert len(reply.pois) == result['catalogue_size'] == 418
    family_by_id = {f['family_id']: f for f in data['families']}
    reply_cache = {}
    def response(state, depth):
        key = int(state), depth
        if key not in reply_cache:
            ids = reply.signatures[reply.access[int(state)], :, :depth].ravel();ids = ids[ids >= 0]
            records = [{k: reply.pois[int(i)][k] for k in ('id', 'category', 'lat', 'lon')} for i in ids]
            size = len(json.dumps({'results': records}, separators=(',', ':'), ensure_ascii=False).encode())
            reply_cache[key] = set(map(int, ids)), size
        return reply_cache[key]
    checked = 0
    for group, truth in zip(public['groups'], private['rows']):
        if truth['split'] not in ('selection', 'test'):
            continue
        assert truth['family_id'] in protocol['splits'][truth['split']]
        seen = {m: set() for m in METHOD_DEPTH};last = {m: -1. for m in METHOD_DEPTH}
        family = family_by_id[truth['family_id']]
        for slot, spec in enumerate(family['evaluator_only']['sessions']):
            assert spec['depart_s'] == protocol['public_epoch']['session_starts_s'][slot]
            trace = {int(p['time_s']): p for p in data['traces'][spec['session_id']]}
            for method, depth in METHOD_DEPTH.items():
                for event in group['streams'][method][slot]['events']:
                    t = int(event['timestamp_s']);absolute = slot*1500.+t
                    assert protocol['public_epoch']['start_s'] <= absolute < protocol['public_epoch']['end_s']
                    assert absolute > last[method]
                    last[method] = absolute
                    responses = [response(rn.nearest(q['lat'], q['lon'])[0], depth) for q in event['candidates']]
                    current = set().union(*(r[0] for r in responses));seen[method].update(current)
                    row = lookup[truth['family_id'], slot, method, t]
                    assert row['absolute_public_time_s'] == absolute and row['L'] == depth
                    assert row['policy'] == 'versioned_epoch8' and row['split'] == truth['split']
                    assert row['cached_poi_ids'] == sorted(seen[method])
                    assert row['cached_poi_count'] == len(seen[method]) <= 418
                    # Independent chronological union: future sessions are never
                    # visited until the earlier rows have already been checked.
                    old = old_lookup[truth['family_id'], slot, method, t]
                    assert set(old['cached_poi_ids']) <= seen[method]
                    gps = trace[t];state, _ = rn.nearest(gps['lat'], gps['lon'])
                    refs = [set(map(int, v[v >= 0])) for v in reply.signatures[state, :, :5] if np.any(v >= 0)]
                    expected = float(np.mean([len(v & seen[method])/len(v) for v in refs])) if refs else None
                    assert row['recall5'] is None if expected is None else np.isclose(row['recall5'], expected)
                    assert old['recall5'] is None if expected is None else expected+1e-12 >= old['recall5']
                    assert row['nonempty_reference_categories'] == len(refs)
                    assert row['reply_bytes'] == sum(r[1] for r in responses) == old['reply_bytes']
                    assert row['requests'] == len(responses) == old['requests'] == 5
                    checked += 1
    assert checked == len(rows) == 6044
    for method in METHOD_DEPTH:
        for split in ('selection', 'test'):
            for phase, (start, end) in PHASES.items():
                subset = [r for r in rows if r['method'] == method and r['split'] == split and start <= r['t'] <= end]
                assert aggregate(subset) == result['results'][method][split][phase]
        s = result['results'][method]['selection'];o, tail = s['all_0_600'], s['tail_400_600']
        old = json.loads((TTL/'results.json').read_text())['results'][method]['selection']['current_only']['all_0_600']
        same = o['request_count'] == old['request_count'] and o['reply_bytes_total'] == old['reply_bytes_total']
        passed = o['family_macro_recall5'] >= .9 and tail['minimum_family_recall5'] >= .9 and o['maximum_cached_poi_count'] <= 418 and same
        assert passed == result['selection_acceptance'][method]['accepted_for_static_selection_scope']
    memory = {}
    byte_cache = {}
    for method in METHOD_DEPTH:
        memory[method] = {}
        for split in ('selection', 'test'):
            subset = [r for r in rows if r['method'] == method and r['split'] == split]
            sizes = []
            for row in subset:
                key = tuple(row['cached_poi_ids'])
                if key not in byte_cache:
                    records = [{k: reply.pois[i][k] for k in ('id', 'category', 'lat', 'lon')} for i in key]
                    byte_cache[key] = len(json.dumps({'records': records}, separators=(',', ':'), ensure_ascii=False).encode())
                sizes.append(byte_cache[key])
            counts = [r['cached_poi_count'] for r in subset]
            memory[method][split] = {'minimum_cached_IDs': min(counts), 'maximum_cached_IDs': max(counts),
                'mean_cached_IDs': float(np.mean(counts)), 'maximum_static_JSON_payload_bytes': max(sizes),
                'scope': 'unique static records serialized locally; excludes Python container overhead; no new network bytes'}
    record = {'schema': 'native-versioned-static-validation-v1', 'status': 'passed', 'validator_sha256': sha(Path(__file__)),
        'checked_artifacts': {name: sha(OUT/name) for name in ('protocol.json', 'results.json', 'utility_rows.json.gz')},
        'chronological_event_method_rows': checked, 'all_eight_public_session_starts_checked': True,
        'causal_cross_session_static_membership_independently_recomputed': True,
        'no_future_session_metadata': True, 'no_extra_GPS_requests_or_reply_bytes': True,
        'source_GPS_Q_attacker_accounting_unchanged': True, 'reference_coverage_not_imputed': True,
        'acceptance_uses_six_selection_families_only': True, 'public_cache_limit_IDs': 418, 'memory': memory,
        'static_only_current_status_guard_unit_tested_not_live_evaluated': True,
        'limits': ['posthoc development after weak-test diagnosis', 'metadata crosses linked sessions in same public catalogue/epoch',
                   'static POI records only', 'minimum tail session can remain below90%', 'POI-reference coverage remains conditional']}
    if output.exists():
        print(f'Existing validation record retained unchanged: {output}', flush=True)
    else:
        save(output, record)
        print(f'Fresh validation record written: {output}', flush=True)
    print(f'Native versioned-static verification passed: {checked} chronological event/method rows across all8 public sessions', flush=True)


if __name__ == '__main__':
    main()
