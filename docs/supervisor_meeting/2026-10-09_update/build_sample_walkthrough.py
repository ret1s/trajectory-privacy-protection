"""Extract illustrative events from frozen Epoch8/REM Q tapes, without sampling.

The first lexicographic family/draw and its first session are fixed before
looking at utility. The first actual noisy-test reuse in that same family is a
separate-session inset. Belief is reconstructed only from saved protected
anchors/private-read flags; GPS is evaluator illustration, never belief input.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MEETING = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark.anchor_belief import AnchorBelief
from benchmark.engines.fair_cover import CoverageObjective
from evaluation.lane_travel import SparseTravel
from experiments.future_sumo_eval import native_resources

BASE = ROOT / 'artifacts/benchmarks/qplanner_depth_base_q_generalization_20261006_v1'
DEPTH = ROOT / 'artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1'
METHOD = 'legacy_l10'
PUBLIC_WORK = Path('/private/tmp/qplanner-response-depth-generalization-20261006-v1')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    content = Path(path).read_bytes()
    return json.loads(gzip.decompress(content) if Path(path).suffix == '.gz' else content)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def pinned_protocol(directory):
    path = directory / 'protocol.json'
    assert sha(path) == (directory / 'protocol.sha256').read_text().strip()
    p = read(path)
    for name, digest in p['source_sha256'].items():
        assert sha(ROOT / name) == digest, 'Changed source: ' + name
    for name, digest in p['public_inputs_sha256'].items():
        assert sha(ROOT / name) == digest, 'Changed public input: ' + name
    receipt = read(directory / 'validation.json')
    assert receipt['status'] == 'pass' and receipt['protocol_sha256'] == sha(path)
    return p


def point(rn, latlon):
    latlon = list(map(float, latlon))
    return {'latlon': latlon, 'xy_m': list(map(float, rn.point_xy(*latlon)))}


def first_reuse(bundle):
    for session in bundle['evaluator_only']['sessions']:
        events = bundle['public']['streams'][METHOD][session['slot']]['events']
        for index, row in enumerate(session['ledger'][METHOD]['ledger']):
            if row['private_read'] and row['branch'] == 'reuse':
                return session['slot'], events[index]['timestamp_s']
    return None


def session_rows(bundle, slot, requested_times, rn, belief_model, travel):
    saved = bundle['evaluator_only']['sessions'][slot]['ledger'][METHOD]
    events = bundle['public']['streams'][METHOD][slot]['events']
    raw_events = bundle['public']['streams']['raw'][slot]['events']
    assert len(events) == len(raw_events) == len(saved['anchors']) == len(saved['ledger']) == len(saved['states'])
    allocation = saved['allocation']
    epsilon = allocation['unit_epsilon_per_m']
    assert epsilon == .00125 and allocation['max_units'] == 23
    belief = AnchorBelief(belief_model)
    results = []
    prior_spent = 0
    last_read = None
    previous_Z = None
    previous_states = None
    previous_t = None
    read_table = []
    for index, (event, raw, ledger, anchor, states) in enumerate(zip(events, raw_events, saved['ledger'], saved['anchors'], saved['states'])):
        t = event['timestamp_s']
        assert raw['timestamp_s'] == t and len(raw['candidates']) == 1
        gps = [raw['candidates'][0]['lat'], raw['candidates'][0]['lon']]
        read_flag = ledger['private_read']
        reserve = 1 if previous_Z is None else 2
        elapsed = None if last_read is None else t - last_read
        expected_read = (last_read is None or elapsed >= 60) and prior_spent + reserve <= 23
        assert read_flag == expected_read
        expected_cost = (1 if previous_Z is None else 2) if ledger['branch'] == 'fresh' else (1 if ledger['branch'] == 'reuse' else 0)
        assert ledger['cost_units'] == expected_cost
        assert ledger['spent_units'] == prior_spent + expected_cost
        assert read_flag == (expected_cost > 0)
        if not read_flag:
            assert anchor == previous_Z
        if ledger['branch'] == 'reuse':
            assert anchor == previous_Z and read_flag
        Z_before = None if previous_Z is None else point(rn, previous_Z)
        distance = None if not read_flag or previous_Z is None else float(np.linalg.norm(np.array(rn.point_xy(*gps)) - np.array(rn.point_xy(*previous_Z))))
        if read_flag:
            read_table.append({'t_s': t, 'branch': ledger['branch'], 'cost_units': expected_cost,
                               'spent_units': ledger['spent_units'], 'spent_per_m': ledger['spent_units'] * epsilon,
                               'remaining_session_cap_per_m': (23 - ledger['spent_units']) * epsilon})
        if t <= max(requested_times):
            weights = belief.update(anchor, t, observed=read_flag)
            assert abs(float(weights.sum()) - 1) < 1e-12
            objective = CoverageObjective(belief_model.context.signatures, belief_model.context.access,
                                          np.asarray(weights @ belief_model.poi_weights).ravel())
            saved_objective = saved['planner_objectives'][index]
            expected_value = saved_objective.get('objective_after_slack', saved_objective['value'])
            reconstructed_value = objective.value(states)
            assert math.isclose(reconstructed_value, expected_value, abs_tol=2e-10, rel_tol=2e-10), (slot, t, reconstructed_value, expected_value)
            if t in requested_times:
                qs = []
                for j, (state, candidate) in enumerate(zip(states, event['candidates'])):
                    coordinate = [candidate['lat'], candidate['lon']]
                    assert np.allclose(rn.latlon(state), coordinate, rtol=0, atol=1e-12)
                    reachable_seconds = None
                    if previous_states is not None:
                        reached = travel.reachable(previous_states[j], t - previous_t)
                        assert state in reached
                        reachable_seconds = reached[state]
                    qs.append(dict(track_index=j + 1, state_id=state, **point(rn, coordinate),
                                   previous_state_id=None if previous_states is None else previous_states[j],
                                   minimum_public_travel_seconds=reachable_seconds))
                top = sorted(range(len(weights)), key=lambda i: (-float(weights[i]), int(belief_model.state_ids[i])))[:12]
                positive = weights[weights > 0]
                belief_points = [dict(latent_index=i, road_state_id=int(belief_model.state_ids[i]),
                                     weight=float(weights[i]), **point(rn, rn.latlon(int(belief_model.state_ids[i])))) for i in top]
                results.append({
                    'slot': slot, 'event_id': event['event_id'], 't_s': t,
                    'sample_id': f"{bundle['evaluator_only']['family_id']}--draw{bundle['evaluator_only']['draw']}--slot{slot}--t{t:g}",
                    'gps_xy': list(map(float, rn.point_xy(*gps))),
                    'Z_xy': list(map(float, rn.point_xy(*anchor))),
                    'Q_xy': [q['xy_m'] for q in qs],
                    'belief_top_xy': [p['xy_m'] for p in belief_points],
                    'evaluator_GPS': point(rn, gps),
                    'local_ranking_position': dict(point(rn, gps), assumption='Exact local GPS at this event, as in the frozen static-service utility evaluator'),
                    'protection': {
                        'GPS_read': read_flag, 'GPS_observation': point(rn, gps) if read_flag else None,
                        'branch': ledger['branch'], 'reserve_before_GPS_units': reserve,
                        'elapsed_since_previous_read_s': elapsed,
                        'cost_units': expected_cost, 'cost_per_m': expected_cost * epsilon,
                        'spent_before_units': prior_spent, 'spent_after_units': ledger['spent_units'],
                        'spent_after_per_m': ledger['spent_units'] * epsilon,
                        'remaining_session_cap_per_m': (23 - ledger['spent_units']) * epsilon,
                        'Z_before': Z_before, 'Z_after': point(rn, anchor),
                        'distance_GPS_to_previous_Z_m': distance,
                        'Laplace_noise_m': None, 'Laplace_noise_recorded': False,
                        'noise_scale_m': 800., 'threshold_m': 200.,
                        'inferred_noise_condition': None if distance is None else {
                            'operator': '<=' if ledger['branch'] == 'reuse' else '>',
                            'bound_m': 200. - distance,
                            'meaning': 'Only an inequality inferred from the saved branch; the sampled noise value was not retained'},
                        'explanation_vi': ('Đọc lần đầu: REM tạo Z, chi 1 đơn vị.' if previous_Z is None else
                                           'Không đọc GPS bảo vệ: giữ Z, chi 0; vị trí local là luồng riêng.' if not read_flag else
                                           'Đã đọc GPS và thử có nhiễu: giữ Z cũ, chi 1.' if ledger['branch'] == 'reuse' else
                                           'Đã đọc GPS: phép thử không đạt, REM tạo Z mới; chi 2.')
                    },
                    'belief': {
                        'observed_emission_this_event': read_flag,
                        'update_vi': 'Dự đoán rồi cập nhật từ quan sát đã bảo vệ' if read_flag and previous_t is not None else 'Cập nhật từ Z đầu tiên' if read_flag else 'Chỉ dự đoán chuyển động; Z lặp không phải phép đo mới',
                        'mean_xy_m': list(map(float, belief.mean_xy())),
                        'entropy_nats': float(-np.sum(positive * np.log(positive))),
                        'effective_states': float(1 / np.sum(weights ** 2)),
                        'top_weights': belief_points, 'top_weights_mass': float(sum(weights[i] for i in top)),
                        'scope': 'Approximate public/protected-history POI-priority belief; not calibrated true-user or attacker posterior'
                    },
                    'Q_sent': qs, 'Q_are_not_independent_REM_samples': True,
                    'planner': {'signature_L': 10, 'surrogate_value_reconstructed': reconstructed_value,
                                'surrogate_value_saved': expected_value,
                                'reachable_counts_saved': saved_objective['reachable_counts'],
                                'slack_loss_saved': saved_objective.get('objective_loss', 0.), 'slack': .03},
                    'request_schema': {'K': 5, 'categories': list(belief_model.context.categories), 'L_per_Q_per_category': 30,
                                       'private_purpose_radius_destination_sent': False}
                })
        prior_spent = ledger['spent_units']
        previous_Z = anchor
        if read_flag:
            last_read = t
        previous_states = states
        previous_t = t
    assert [r['t_s'] for r in read_table] == saved['supplier_times_s']
    assert math.isclose(prior_spent * epsilon, saved['spent_per_m'], abs_tol=1e-12)
    assert {r['t_s'] for r in results} == set(requested_times)
    return {'slot': slot, 'events': results, 'all_protection_reads_budget_table': read_table,
            'session_actual_spend_per_m': saved['spent_per_m'], 'allocation': allocation}


def geometry(rn, rows, *, include_segments=True):
    xy = []
    for row in rows:
        xy.append(row['evaluator_GPS']['xy_m'])
        xy.append(row['protection']['Z_after']['xy_m'])
        xy.extend(q['xy_m'] for q in row['Q_sent'])
        xy.extend(p['xy_m'] for p in row['belief']['top_weights'])
    xy = np.asarray(xy)
    lo, hi = xy.min(axis=0) - 200., xy.max(axis=0) + 200.
    used = set()
    display_segments = {}
    crop_arcs = 0
    zero_arcs = 0
    for a, b in rn.graph.edges:
        p, q = rn.xy[a], rn.xy[b]
        if np.all(np.maximum(p, q) >= lo) and np.all(np.minimum(p, q) <= hi):
            crop_arcs += 1
            used.update((int(a), int(b)))
            first, second = tuple(map(float, p)), tuple(map(float, q))
            if first == second:
                zero_arcs += 1
                continue
            key = tuple(sorted((first, second)))
            display_segments.setdefault(key, [list(first), list(second)])
    result = {'viewport_xy_m': [*map(float, lo), *map(float, hi)], 'padding_m': 200,
              'xy_origin': [0., 0.], 'render_origin_xy_m': list(map(float, lo)),
              'public_vertices_in_crop_count': len(used), 'public_directed_arcs_in_crop_count': crop_arcs,
              'zero_length_display_arcs_omitted': zero_arcs,
              'nonzero_duplicate_display_arcs_omitted': crop_arcs - zero_arcs - len(display_segments),
              'exact_undirected_display_segment_count': len(display_segments),
              'catalogue_sha256': rn.catalogue_sha256,
              'scope': 'Display-only: segment bounding-box crop, zero-length lines omitted, exact undirected coordinate segments deduplicated without rounding. The unchanged model retains the full directed graph, coincident states and zero-length connections; these lines are not recovered physical dummy trajectories.'}
    if include_segments:
        result['road_segments_xy'] = list(display_segments.values())
    else:
        result['road_geometry_omitted_for_inset'] = True
    return result


def build(output=MEETING / 'sample_walkthrough.json', public_work=PUBLIC_WORK):
    output = Path(output)
    output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
    if not output.is_relative_to(MEETING.resolve()):
        raise ValueError('Output must stay inside this new meeting directory')
    if output.exists():
        raise FileExistsError('Write-once sample; use a new output filename for a recheck')
    public_work = Path(public_work).resolve()
    if public_work.is_relative_to(ROOT):
        raise ValueError('Public reconstruction caches must stay outside the repository')
    base_protocol = pinned_protocol(BASE)
    depth_protocol = pinned_protocol(DEPTH)
    assert depth_protocol['selected_depth'] == 30
    assert depth_protocol['base_q_protocol_sha256'] == sha(BASE / 'protocol.json')
    first = sorted((BASE / 'families').glob('*.json.gz'))[0]
    bundle = read(first)
    derived_path = DEPTH / 'families' / first.name
    derived = read(derived_path)
    assert derived['source_bundle_sha256'] == sha(first)
    assert derived['frozen_controls']['Q_not_regenerated'] and derived['frozen_controls']['private_reads_not_performed']
    assert bundle['evaluator_only']['draw'] == 1
    dataset_path = ROOT / base_protocol['dataset_path']
    assert sha(dataset_path) == base_protocol['dataset_sha256']
    data = read(dataset_path)
    print('Reconstructing public native map and protected belief; no sampler/key access', flush=True)
    rn, reference, legacy, beliefs, metadata = native_resources(data, public_work)
    saved_resources = read(BASE / 'resources.json')
    assert rn.catalogue_sha256 == saved_resources['catalogue']['sha256']
    assert reference.sha256 == saved_resources['reference_sha256']
    assert legacy.sha256 == saved_resources['reply_sha256']
    assert beliefs[.00125].base.sha256 == saved_resources['belief_sha256']['0.00125']
    model = beliefs[.00125]
    travel = SparseTravel(rn)
    main = session_rows(bundle, 0, [0., 20., 60.], rn, model, travel)
    reuse = first_reuse(bundle)
    inset = None if reuse is None else session_rows(bundle, reuse[0], [0., reuse[1]], rn, model, travel)
    family_id = bundle['evaluator_only']['family_id']
    sample_rows = main['events'] + ([] if inset is None else inset['events'])
    names = {
        Path(__file__), BASE / 'protocol.json', BASE / 'validation.json', BASE / 'resources.json', first,
        DEPTH / 'protocol.json', DEPTH / 'depth_freeze.json', DEPTH / 'validation.json', derived_path,
        dataset_path, ROOT / data['network']['compressed_path'],
        ROOT / 'docs/supervisor_meeting/2026-09-26_brief/walkthrough/walkthrough.json'
    }
    names.update(ROOT / name for name in base_protocol['source_sha256'])
    value = {
        'schema': 'current-frozen-GeoI-REM-sample-walkthrough-v1', 'prepared_date': '2026-10-09',
        'selection': {'family_id': family_id, 'split': bundle['evaluator_only']['split'], 'draw': 1,
                      'main_slot': 0, 'main_times_s': [0, 20, 60],
                      'rule': 'First lexicographic frozen family/draw, first session; first event, first no-read event, first later read. Inset: first actual noisy-test reuse by slot then event in this same family. No Recall or attack score is used.',
                      'first_noisy_reuse': None if reuse is None else {'slot': reuse[0], 't_s': reuse[1]}},
        'configuration': {'display_name': 'Geo-I / REM, L=30', 'Q_planner': METHOD, 'K': 5,
                          'planner_signature_L': 10, 'server_L_per_Q_per_category': 30, 'local_top_k': 5,
                          'epoch_budget': base_protocol['configuration']['budget'], 'theta_m': 200., 'slack': .03,
                          'endpoint_policy': 'No warmup, no delay, no holdback. Legacy Endpoint20 is a separate sample/configuration.'},
        'scope': {
            'synthetic_only': True, 'not_new_benchmark': True,
            'source_GPS': 'Frozen native SUMO trace; evaluator illustration, not real-person GPS.',
            'protection_GPS': 'Only the supplier-permitted observation is passed to REM/noisy reuse.',
            'local_GPS': 'Exact evaluator GPS at each shown event is the local-ranking assumption; independent of the protection supplier gate.',
            'belief': 'Reconstructed from saved Z and private_read flags plus pinned public resources; no raw GPS input, future route or utility score.',
            'Laplace_noise': 'Original numerical test draws were not saved. Only branch and its implied inequality are shown; no noise value is invented.',
            'privacy': 'Ideal-kernel coordinate guarantee scope unchanged; no finite-float/PRNG or S1–S10 certificate.',
            'legacy_walkthrough': 'Sep26 used per-session B=.24/m, u=.01/m, L10 and separate boundary examples; its traces/numbers are not substituted for this Epoch8/L30 sample.',
            'visual_crop': 'Post-extraction display-only crop; does not change full REM support, graph or protected outputs.'},
        'main_sample': main, 'test_reuse_inset': inset,
        'metadata': {'xy_origin': [0., 0.], 'xy_units': 'metres',
                     'projection': {'type': 'repository LocalProjection, equirectangular',
                                    'reference_latitude_deg': rn.proj.lat0,
                                    'm_per_deg_lat': rn.proj.m_per_deg_lat,
                                    'm_per_deg_lon': rn.proj.m_per_deg_lon,
                                    'formula': 'x=longitude*m_per_deg_lon; y=latitude*m_per_deg_lat'},
                     'display_rule': 'Use one scale for x and y; translate by the selected viewport origin, never independently stretch the two axes.'},
        'map_main': geometry(rn, main['events']),
        'map_reuse_inset': None if inset is None else geometry(rn, inset['events'], include_segments=False),
        'source_pins_sha256': {relative(path): sha(path) for path in sorted(names)},
        'validation': {'source_protocol_and_receipt_hashes': 'pass', 'saved_Q_coordinates_match_road_states': 'pass',
                       'prospective_supplier_gate_and_per_read_budget': 'pass',
                       'belief_reconstructed_POI_objective_matches_saved_values': 'pass',
                       'directed_Q_reachability_at_shown_events': 'pass', 'selected_event_count': len(sample_rows),
                       'sampler_executed': False, 'private_key_or_seed_read': False, 'old_sources_or_scores_changed': False}
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        # Compact machine-readable geometry avoids a large redundant demo file.
        json.dump(value, handle, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        handle.write('\n')
    print('Sample written:', relative(output), 'SHA256', sha(output), flush=True)
    print('Main:', family_id, 'slot0', [(r['t_s'], r['protection']['branch'], r['protection']['spent_after_units']) for r in main['events']], flush=True)
    print('First actual noisy reuse:', reuse, flush=True)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=MEETING / 'sample_walkthrough.json')
    parser.add_argument('--public-work', type=Path, default=PUBLIC_WORK)
    args = parser.parse_args()
    build(args.output, args.public_work)
