"""Posthoc weakest-tail diagnosis; no cache/model choice is changed."""
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
from experiments.native_static_cache_20261006 import OUT


def main():
    result = json.loads((OUT/'results.json').read_text())
    method = 'geoi_epoch8'
    worst = result['results'][method]['test']['current_only']['tail_400_600']['minimum_session']
    family_id, slot = worst['family_id'], worst['slot']
    selected = result['selection'][method]['selected_TTL_s']
    assert selected is not None
    cache_rows = load(OUT/'utility_rows.json.gz')['rows']
    by_key = {(r['policy'], r['t']): r for r in cache_rows if r['family_id'] == family_id and r['slot'] == slot and r['method'] == method}
    data, public, private = load(DATA), load(PRIMARY/'public_transcripts.json.gz'), load(PRIMARY/'private_accounting.json.gz')
    family = next(f for f in data['families'] if f['family_id'] == family_id)
    truth = next(f for f in private['rows'] if f['family_id'] == family_id)
    group = next(g for g in public['groups'] if g['public_scope'] == truth['public_scope'])
    spec = family['evaluator_only']['sessions'][slot]
    trace = {int(p['time_s']): p for p in data['traces'][spec['session_id']]}
    ledger = truth['evaluator_sessions'][slot]['ledger'][method]
    rn = build_lane_states(ROOT/data['network']['compressed_path'], spacing_m=40.)
    pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')} for p in
        json.loads((ROOT/'artifacts/benchmarks/research_loop/resources.json').read_text())['pois_used']]
    reply = PublicPoiContext(LanePoiService(rn, pois, k=40), DEPTH/'public_reply40.npz')
    project = Transformer.from_crs(4326, 32650, always_xy=True)
    ever = set();rows = []
    for event, accounting in zip(group['streams'][method][slot]['events'], ledger['ledger']):
        t = int(event['timestamp_s'])
        current = set(by_key['current_only', t]['cached_poi_ids'])
        ever.update(current)
        if not 400 <= t <= 600:
            continue
        p = trace[t];state, _ = rn.nearest(p['lat'], p['lon'])
        refs = [set(map(int, v[v >= 0])) for v in reply.signatures[state, :, :5] if np.any(v >= 0)]
        reference_ids = set().union(*refs)
        gps_xy = project.transform(p['lon'], p['lat'])
        q_xy = [project.transform(q['lon'], q['lat']) for q in event['candidates']]
        union_recall = float(np.mean([len(v & ever)/len(v) for v in refs])) if refs else None
        selected_row = by_key[f'rolling{selected}', t]
        rows.append({'t': t, 'GPS_speed_m_s': p['speed_m_s'], 'reference_ids': sorted(reference_ids),
            'current_recall5': by_key['current_only', t]['recall5'], 'selected_rolling_recall5': selected_row['recall5'],
            'all_causal_previous_replies_recall5': union_recall,
            'reference_never_received_so_far_ids': sorted(reference_ids-ever),
            'private_read': accounting['private_read'], 'spent_units': accounting['spent_units'],
            'remaining_units': ledger['allocation']['max_units']-accounting['spent_units'],
            'nearest_Q_to_GPS_m': float(min(np.linalg.norm(np.asarray(q)-gps_xy) for q in q_xy))})
    tail = [trace[r['t']] for r in rows]
    report = {'schema': 'native-weakest-static-cache-tail-diagnosis-v1',
        'source_sha256': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
            str((OUT/'results.json').relative_to(ROOT)): sha(OUT/'results.json'),
            str((OUT/'utility_rows.json.gz').relative_to(ROOT)): sha(OUT/'utility_rows.json.gz'),
            str(DATA.relative_to(ROOT)): sha(DATA),
            str((PRIMARY/'private_accounting.json.gz').relative_to(ROOT)): sha(PRIMARY/'private_accounting.json.gz')},
        'case': {'family_id': family_id, 'slot': slot, 'method': method, 'L': 20,
            'selection_fixed_TTL_s': selected, 'destination_role': spec['destination_role']},
        'selection': 'weakest original current-only test tail session, descriptive after test; not used to tune TTL/model',
        'tail_events': len(rows), 'tail_GPS_is_native_stationary': all(p['speed_m_s'] == 0. for p in tail),
        'tail_GPS_positions_identical': len({(p['lat'], p['lon']) for p in tail}) == 1,
        'spent_units_at_end': ledger['ledger'][-1]['spent_units'], 'session_max_units': ledger['allocation']['max_units'],
        'private_GPS_read_at600s': ledger['ledger'][-1]['private_read'],
        'current_tail_recall5': float(np.mean([r['current_recall5'] for r in rows])),
        'selected_rolling_tail_recall5': float(np.mean([r['selected_rolling_recall5'] for r in rows])),
        'all_causal_previous_replies_tail_recall5': float(np.mean([r['all_causal_previous_replies_recall5'] for r in rows])),
        'reference_POIs_never_received_by600s': rows[-1]['reference_never_received_so_far_ids'],
        'nearest_Q_to_GPS_tail_mean_m': float(np.mean([r['nearest_Q_to_GPS_m'] for r in rows])),
        'rows': rows,
        'interpretation': 'static cache cannot manufacture POIs never returned by any causal public Q; '
            'cumulative-unbounded cache is a diagnostic upper bound, not a selected policy or liveavailability result'}
    save(OUT/'weakest_tail_diagnostic.json', report)
    print('Weakest static native tail diagnosis', family_id, 'slot', slot,
          'current/selected/cumulative', report['current_tail_recall5'], report['selected_rolling_tail_recall5'],
          report['all_causal_previous_replies_tail_recall5'], 'units', report['spent_units_at_end'], '/', report['session_max_units'], flush=True)


if __name__ == '__main__':
    main()
