"""Recompute frozen-Q utility/cost from the public native L40 archive."""
import argparse
import json
from pathlib import Path
import numpy as np
from benchmark.public_poi_context import PublicPoiContext
from data.lane_states import build_lane_states
from evaluation.lane_travel import LanePoiService
from experiments.build_future_sumo_cohort import ROOT, sha, save
from experiments.future_sumo_eval import DATA, OUT as PRIMARY, METHODS
from experiments.native_future_retrieval_depth import OUT, DEPTHS, PHASES, load, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-output', type=Path, help='Write to a NEW path, preserve old records')
    args = parser.parse_args()
    output = args.validation_output or OUT/'validation.json'
    if args.validation_output is not None and output.exists():
        raise FileExistsError(f'Preserve completed evidence: {output}')
    protocol = json.loads((OUT/'protocol.json').read_text())
    result = json.loads((OUT/'results.json').read_text())
    archive = json.loads((OUT/'public_resource_archive.json').read_text())
    for pins in (result['source_sha256'], result['diagnostic_source_sha256']):
        for path, expected in pins.items():
            assert sha(ROOT/path) == expected, path
    assert result['protocol_sha256'] == sha(OUT/'protocol.json')
    assert result['row_artifact_sha256'] == sha(OUT/'utility_rows.json.gz')
    assert archive['public_map_only'] and archive['no_GPS_or_protected_queries']
    assert sha(ROOT/archive['path']) == archive['sha256']
    assert archive['primary_result_sha256'] == sha(PRIMARY/'results.json')
    assert result['no_new_emissions'] and result['primary_attacker_unchanged']
    assert result['resources']['original_L10_prefix_parity'] and archive['reference5_ordered_prefix_parity']
    data, public, private = load(DATA), load(PRIMARY/'public_transcripts.json.gz'), load(PRIMARY/'private_accounting.json.gz')
    compressed = ROOT/data['network']['compressed_path']
    rn = build_lane_states(compressed, spacing_m=40.)
    assert rn.catalogue_sha256 == result['resources']['catalogue_sha256']
    pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')} for p in
        json.loads((ROOT/'artifacts/benchmarks/research_loop/resources.json').read_text())['pois_used']]
    reply = PublicPoiContext(LanePoiService(rn, pois, k=40), ROOT/archive['path'])
    assert reply.sha256 == result['resources']['reply40_sha256']
    rows = load(OUT/'utility_rows.json.gz')['rows']
    lookup = {(r['family_id'], r['slot'], r['method'], r['t'], r['L']): r for r in rows}
    assert len(lookup) == len(rows)
    assert {r['split'] for r in rows} == {'selection', 'test'}
    family_by_id = {f['family_id']: f for f in data['families']}
    response_cache = {}
    def response(state, depth):
        key = int(state), depth
        if key not in response_cache:
            ids = reply.signatures[reply.access[int(state)], :, :depth].ravel()
            ids = ids[ids >= 0]
            records = [{k: reply.pois[int(i)][k] for k in ('id', 'category', 'lat', 'lon')} for i in ids]
            size = len(json.dumps({'results': records}, separators=(',', ':'), ensure_ascii=False).encode())
            response_cache[key] = set(map(int, ids)), size, len(ids)
        return response_cache[key]
    checked_events = 0
    for group, truth in zip(public['groups'], private['rows']):
        if truth['split'] not in ('selection', 'test'):
            continue
        assert truth['family_id'] in protocol['splits'][truth['split']]
        family = family_by_id[truth['family_id']]
        for slot, spec in enumerate(family['evaluator_only']['sessions']):
            trace = {int(p['time_s']): p for p in data['traces'][spec['session_id']]}
            for method in METHODS:
                for event, primary in zip(group['streams'][method][slot]['events'], truth['evaluator_sessions'][slot]['utility'][method]):
                    time = int(event['timestamp_s'])
                    gps = trace[time]
                    state, _ = rn.nearest(gps['lat'], gps['lon'])
                    refs = [set(map(int, v[v >= 0])) for v in reply.signatures[state, :, :5] if np.any(v >= 0)]
                    qstates = [rn.nearest(q['lat'], q['lon'])[0] for q in event['candidates']]
                    previous_recall = None
                    previous_cost = 0
                    for depth in DEPTHS:
                        row = lookup[truth['family_id'], slot, method, time, depth]
                        assert set(row) == {'family_id', 'split', 'slot', 'method', 'L', 't', 'recall5', 'nonempty_categories', 'reply_bytes', 'returned_records'}
                        replies = [response(s, depth) for s in qstates]
                        union = set().union(*(r[0] for r in replies))
                        recall = float(np.mean([len(v & union)/len(v) for v in refs])) if refs else None
                        assert row['nonempty_categories'] == len(refs)
                        assert row['recall5'] is None if recall is None else np.isclose(row['recall5'], recall)
                        assert row['reply_bytes'] == sum(r[1] for r in replies) >= previous_cost
                        assert row['returned_records'] == sum(r[2] for r in replies)
                        if depth == 10:
                            assert primary['recall5'] is None if recall is None else np.isclose(primary['recall5'], recall)
                        if previous_recall is not None:
                            assert recall+1e-12 >= previous_recall
                        previous_recall, previous_cost = recall, row['reply_bytes']
                    checked_events += 1
    assert len(rows) == checked_events*3
    for method in METHODS:
        for split in ('selection', 'test'):
            for depth in DEPTHS:
                for phase, (start, end) in PHASES.items():
                    subset = [r for r in rows if r['method'] == method and r['split'] == split and r['L'] == depth and start <= r['t'] <= end]
                    assert aggregate(subset) == result['results'][method][split][str(depth)][phase]
    for method in METHODS[1:]:
        metrics = result['results'][method]['selection']
        baseline = metrics['10']['all_0_600']['mean_reply_bytes_per_event']
        eligible = []
        for depth in DEPTHS:
            m = metrics[str(depth)]['all_0_600']
            ratio = m['mean_reply_bytes_per_event']/baseline
            passed = m['family_macro_recall5'] >= .9 and m['median_session_recall5'] >= .9 and ratio <= 2.
            assert passed == result['selection'][method]['selection_gates'][str(depth)]['passes']
            if passed:
                eligible.append(depth)
        chosen = min(eligible) if eligible else None
        assert chosen == result['selection'][method]['chosen_L']
        assert result['selection'][method]['test_at_chosen_L'] == (
            result['results'][method]['test'][str(chosen)] if chosen is not None else None)
    record = {'schema': 'native-frozen-Q-depth-validation-v1', 'status': 'passed',
        'validator_sha256': sha(Path(__file__)),
        'checked_artifacts': {name: sha(OUT/name) for name in ('protocol.json', 'results.json', 'utility_rows.json.gz', 'public_resource_archive.json', 'public_reply40.npz')},
        'checked_event_method_rows': checked_events, 'checked_depth_rows': len(rows),
        'Q_and_primary_attacker_unchanged': True, 'L10_and_reference5_prefix_parity': True,
        'all_utility_and_reply_byte_rows_recomputed': True, 'monotone_reply_depth_utility': True,
        'selected_on_six_selection_families_only': True, 'unchanged_primary_test': True,
        'scope': 'posthoc development; reply-only JSON estimate; conditional nonempty POI reference; fixed L10 planner'}
    if output.exists():
        print(f'Existing validation record retained unchanged: {output}', flush=True)
    else:
        save(output, record)
        print(f'Fresh validation record written: {output}', flush=True)
    print(f'Native frozen-Q depth verification passed:{checked_events}event/method rows, {len(rows)}depth rows', flush=True)


if __name__ == '__main__':
    main()
