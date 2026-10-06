"""Recompute native static cache membership, utility, cost and selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from pyproj import Transformer
from benchmark.public_poi_context import PublicPoiContext
from data.lane_states import build_lane_states
from evaluation.lane_travel import LanePoiService
from experiments.build_future_sumo_cohort import ROOT, sha, save
from experiments.future_sumo_eval import DATA, OUT as PRIMARY
from experiments.native_future_retrieval_depth import OUT as DEPTH, load
from experiments.native_static_cache_20261006 import OUT, METHOD_DEPTH, POLICIES, PHASES, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-output', type=Path, help='New path only; preserve completed records')
    args = parser.parse_args()
    output = args.validation_output or OUT/'validation.json'
    if args.validation_output is not None and output.exists():
        raise FileExistsError(f'Preserve completed evidence: {output}')
    protocol = json.loads((OUT/'protocol.json').read_text())
    result = json.loads((OUT/'results.json').read_text())
    diagnosis = json.loads((OUT/'weakest_tail_diagnostic.json').read_text())
    assert result['source_sha256'] == protocol['source_sha256']
    for path, expected in protocol['source_sha256'].items():
        assert sha(ROOT/path) == expected, path
    for path, expected in diagnosis['source_sha256'].items():
        assert sha(ROOT/path) == expected, path
    assert result['protocol_sha256'] == sha(OUT/'protocol.json')
    assert result['utility_rows_sha256'] == sha(OUT/'utility_rows.json.gz')
    assert result['Q_and_primary_attacker_unchanged'] and not result['liveavailability_evaluated']
    data, public = load(DATA), load(PRIMARY/'public_transcripts.json.gz')
    private = load(PRIMARY/'private_accounting.json.gz')
    depth_rows = load(DEPTH/'utility_rows.json.gz')['rows']
    depth_index = {(r['family_id'], r['slot'], r['method'], r['t'], r['L']): r for r in depth_rows}
    rows = load(OUT/'utility_rows.json.gz')['rows']
    index = {(r['family_id'], r['slot'], r['method'], r['t'], r['policy']): r for r in rows}
    assert len(index) == len(rows)
    rn = build_lane_states(ROOT/data['network']['compressed_path'], spacing_m=40.)
    pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')} for p in
        json.loads((ROOT/'artifacts/benchmarks/research_loop/resources.json').read_text())['pois_used']]
    reply = PublicPoiContext(LanePoiService(rn, pois, k=40), DEPTH/'public_reply40.npz')
    assert reply.sha256 == result['resources']['reply40_sha256']
    assert rn.catalogue_sha256 == result['resources']['catalogue_sha256']
    assert result['resources'] == json.loads((DEPTH/'results.json').read_text())['resources']
    family_by_id = {f['family_id']: f for f in data['families']}
    response_cache = {}
    def response(state, depth):
        key = int(state), depth
        if key not in response_cache:
            ids = reply.signatures[reply.access[int(state)], :, :depth].ravel()
            ids = ids[ids >= 0]
            records = [{k: reply.pois[int(i)][k] for k in ('id', 'category', 'lat', 'lon')} for i in ids]
            size = len(json.dumps({'results': records}, separators=(',', ':'), ensure_ascii=False).encode())
            response_cache[key] = set(map(int, ids)), size
        return response_cache[key]
    event_methods = 0
    diagnostic_rows = []
    project = Transformer.from_crs(4326, 32650, always_xy=True)
    for group, truth in zip(public['groups'], private['rows']):
        if truth['split'] not in ('selection', 'test'):
            continue
        assert truth['family_id'] in protocol['splits'][truth['split']]
        family = family_by_id[truth['family_id']]
        for slot, spec in enumerate(family['evaluator_only']['sessions']):
            trace = {int(p['time_s']): p for p in data['traces'][spec['session_id']]}
            assert spec['depart_s'] % 60 == 0, 'Relative and absolute public epoch boundaries must match'
            for method, depth in METHOD_DEPTH.items():
                history = []
                for event in group['streams'][method][slot]['events']:
                    t = int(event['timestamp_s'])
                    replies = [response(rn.nearest(q['lat'], q['lon'])[0], depth) for q in event['candidates']]
                    current = set().union(*(r[0] for r in replies))
                    history.append((t, current))
                    masks = {'current_only': current, 'public_epoch60': set().union(*(ids for time, ids in history if time//60 == t//60))}
                    masks.update({f'rolling{ttl}': set().union(*(ids for time, ids in history if 0 <= t-time < ttl)) for ttl in (60, 120, 180)})
                    assert current <= masks['public_epoch60'] <= masks['rolling60'] <= masks['rolling120'] <= masks['rolling180']
                    gps = trace[t]
                    state, _ = rn.nearest(gps['lat'], gps['lon'])
                    refs = [set(map(int, v[v >= 0])) for v in reply.signatures[state, :, :5] if np.any(v >= 0)]
                    costs = sum(r[1] for r in replies)
                    for policy, ids in masks.items():
                        row = index[truth['family_id'], slot, method, t, policy]
                        assert set(row) == {'family_id', 'split', 'slot', 'method', 'L', 'policy', 't', 'recall5',
                                           'cached_poi_count', 'nonempty_reference_categories', 'reply_bytes', 'requests', 'cached_poi_ids'}
                        assert row['split'] == truth['split'] and row['L'] == depth
                        assert row['cached_poi_ids'] == sorted(ids) and row['cached_poi_count'] == len(ids)
                        assert row['nonempty_reference_categories'] == len(refs)
                        expected = float(np.mean([len(r & ids)/len(r) for r in refs])) if refs else None
                        assert row['recall5'] is None if expected is None else np.isclose(row['recall5'], expected)
                        assert row['reply_bytes'] == costs and row['requests'] == len(event['candidates']) == 5
                        if policy == 'current_only':
                            prior = depth_index[truth['family_id'], slot, method, t, depth]
                            assert prior['recall5'] is None if expected is None else np.isclose(prior['recall5'], expected)
                            assert prior['reply_bytes'] == costs
                    case = diagnosis['case']
                    if (truth['family_id'], slot, method) == (case['family_id'], case['slot'], case['method']) and 400 <= t <= 600:
                        ever = set().union(*(ids for time, ids in history))
                        source_row = next(r for r in diagnosis['rows'] if r['t'] == t)
                        account = truth['evaluator_sessions'][slot]['ledger'][method]
                        account_event = next(v for e, v in zip(group['streams'][method][slot]['events'], account['ledger']) if e['timestamp_s'] == t)
                        references = set().union(*refs)
                        expected_upper = float(np.mean([len(v & ever)/len(v) for v in refs]))
                        assert np.isclose(source_row['all_causal_previous_replies_recall5'], expected_upper)
                        assert source_row['reference_ids'] == sorted(references)
                        assert source_row['reference_never_received_so_far_ids'] == sorted(references-ever)
                        assert source_row['private_read'] == account_event['private_read']
                        assert source_row['spent_units'] == account_event['spent_units']
                        assert source_row['remaining_units'] == account['allocation']['max_units']-account_event['spent_units']
                        assert source_row['GPS_speed_m_s'] == gps['speed_m_s'] == 0.
                        gps_xy = project.transform(gps['lon'], gps['lat'])
                        q_xy = [project.transform(q['lon'], q['lat']) for q in event['candidates']]
                        distance = min(np.linalg.norm(np.asarray(q)-gps_xy) for q in q_xy)
                        assert np.isclose(source_row['nearest_Q_to_GPS_m'], distance)
                        chosen = index[truth['family_id'], slot, method, t, f'rolling{case["selection_fixed_TTL_s"]}']
                        assert np.isclose(source_row['selected_rolling_recall5'], chosen['recall5'])
                        assert np.isclose(source_row['current_recall5'], index[truth['family_id'], slot, method, t, 'current_only']['recall5'])
                        diagnostic_rows.append(source_row)
                    event_methods += 1
    assert len(rows) == event_methods*len(POLICIES)
    for method in METHOD_DEPTH:
        for split in ('selection', 'test'):
            for policy in POLICIES:
                for phase, (start, end) in PHASES.items():
                    subset = [r for r in rows if r['method'] == method and r['split'] == split and r['policy'] == policy and start <= r['t'] <= end]
                    assert aggregate(subset) == result['results'][method][split][policy][phase]
    for method in METHOD_DEPTH:
        scores = result['results'][method]['selection']
        baseline = scores['current_only']['all_0_600']
        eligible = []
        for ttl in (60, 120, 180):
            m = scores[f'rolling{ttl}']
            overall, tail = m['all_0_600'], m['tail_400_600']
            unchanged = baseline['reply_bytes_total'] == overall['reply_bytes_total'] and baseline['request_count'] == overall['request_count']
            passed = overall['family_macro_recall5'] >= .9 and tail['minimum_family_recall5'] >= .9 and unchanged
            assert passed == result['selection'][method]['selection_gates'][str(ttl)]['passes']
            if passed:
                eligible.append(ttl)
        selected = min(eligible) if eligible else None
        assert selected == result['selection'][method]['selected_TTL_s']
        assert result['selection'][method]['test_at_selected_TTL'] == (
            result['results'][method]['test'][f'rolling{selected}'] if selected is not None else None)
    assert len(diagnostic_rows) == diagnosis['tail_events'] == 11
    worst = result['results']['geoi_epoch8']['test']['current_only']['tail_400_600']['minimum_session']
    assert (worst['family_id'], worst['slot']) == (diagnosis['case']['family_id'], diagnosis['case']['slot'])
    for report, row_key in [('current_tail_recall5', 'current_recall5'),
                            ('selected_rolling_tail_recall5', 'selected_rolling_recall5'),
                            ('all_causal_previous_replies_tail_recall5', 'all_causal_previous_replies_recall5'),
                            ('nearest_Q_to_GPS_tail_mean_m', 'nearest_Q_to_GPS_m')]:
        assert np.isclose(diagnosis[report], np.mean([r[row_key] for r in diagnostic_rows]))
    assert diagnosis['spent_units_at_end'] == diagnostic_rows[-1]['spent_units'] == 21
    assert diagnosis['session_max_units'] == 23 and diagnosis['private_GPS_read_at600s']
    assert diagnosis['reference_POIs_never_received_by600s'] == diagnostic_rows[-1]['reference_never_received_so_far_ids']
    record = {'schema': 'native-static-cache-validation-v1', 'status': 'passed',
        'validator_sha256': sha(Path(__file__)),
        'checked_artifacts': {name: sha(OUT/name) for name in ('protocol.json', 'results.json', 'utility_rows.json.gz', 'weakest_tail_diagnostic.json')},
        'event_method_rows': event_methods, 'policy_rows': len(rows),
        'cache_membership_independently_recomputed_from_public_history_only': True,
        'current_only_unchanged_from_previous_depth_run': True, 'epoch60_matches_absolute_native_clock': True,
        'utility_bytes_and_coverage_recomputed': True, 'selection_uses_six_selection_families_only': True,
        'no_extra_requests_or_reply_bytes': True, 'source_GPS_Q_and_attacker_hashes_unchanged': True,
        'static_only_liveavailability_not_evaluated': True,
        'weakest_tail_stationary_GPS_with21_of23_units_spent': True,
        'cumulative_causal_reply_upper_bound_recomputed': True,
        'limits': ['already-inspected test; development evidence', 'conditional nonempty reference utility',
                   'minimum-family-tail>=90% selection gate does not guarantee test success',
                   'dynamic status must be current public epoch; missing/stale is unknown']}
    if output.exists():
        print(f'Existing validation record retained unchanged: {output}', flush=True)
    else:
        save(output, record)
        print(f'Fresh validation record written: {output}', flush=True)
    print(f'Native static-cache verification passed: {event_methods} event/method rows, {len(rows)} policy rows', flush=True)


if __name__ == '__main__':
    main()
