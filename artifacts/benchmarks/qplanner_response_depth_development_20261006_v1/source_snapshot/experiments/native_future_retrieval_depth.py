"""Posthoc fixed-Q L10/20/40 diagnostic; selection-only depth choice.

The native Geo-I mechanism, L10 belief/planner and every Q remain unchanged.
This measures larger static replies, never regenerates protected coordinates.
Tests were already inspected in the primary run: this is development evidence.
"""
import gzip
import json
from pathlib import Path
import numpy as np
from benchmark.public_poi_context import PublicPoiContext
from data.lane_states import build_lane_states
from evaluation.lane_travel import LanePoiService
from experiments.build_future_sumo_cohort import ROOT, sha, save
from experiments.future_sumo_eval import DATA, OUT as PRIMARY, WORK, METHODS, compressed_save

OUT = ROOT/'artifacts/benchmarks/future_native_depth_20261005_v1'
DEPTHS = (10, 20, 40)
PHASES = {'all_0_600': (0, 600), 'early_0_180': (0, 180),
          'middle_200_380': (200, 380), 'tail_400_600': (400, 600)}


def load(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def aggregate(rows):
    families = sorted({r['family_id'] for r in rows})
    family_values = {f: [r['recall5'] for r in rows if r['family_id'] == f and r['recall5'] is not None] for f in families}
    means = {f: float(np.mean(v)) if v else None for f, v in family_values.items()}
    sessions = sorted({(r['family_id'], r['slot']) for r in rows})
    session_means = []
    for f, slot in sessions:
        values = [r['recall5'] for r in rows if r['family_id'] == f and r['slot'] == slot and r['recall5'] is not None]
        if values:
            session_means.append(float(np.mean(values)))
    valid_means = [v for v in means.values() if v is not None]
    return {'family_macro_recall5': float(np.mean(valid_means)), 'family_values': means,
        'family_min_recall5': min(valid_means), 'family_max_recall5': max(valid_means),
        'median_session_recall5': float(np.median(session_means)),
        'min_session_recall5': min(session_means),
        'reference_defined_windows': sum(r['recall5'] is not None for r in rows),
        'total_windows': len(rows), 'undefined_session_count': len(sessions)-len(session_means),
        'mean_reply_bytes_per_event': float(np.mean([r['reply_bytes'] for r in rows])),
        'mean_returned_records_per_event': float(np.mean([r['returned_records'] for r in rows])),
        'reply_bytes_total': sum(r['reply_bytes'] for r in rows)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    declaration = {'schema': 'fixed-Q-native-retrieval-depth-protocol-v1', 'depths': DEPTHS,
        'source_sha256': {str(DATA.relative_to(ROOT)): sha(DATA),
            str((PRIMARY/'public_transcripts.json.gz').relative_to(ROOT)): sha(PRIMARY/'public_transcripts.json.gz'),
            str((PRIMARY/'private_accounting.json.gz').relative_to(ROOT)): sha(PRIMARY/'private_accounting.json.gz'),
            str((PRIMARY/'results.json').relative_to(ROOT)): sha(PRIMARY/'results.json')},
        'preselection': 'minimum L with selection family-macro Recall>=.90, median session Recall>=.90 and mean reply bytes<=2x L10, separately per protected method',
        'failure': 'no selected depth if no depth passes all three gates',
        'splits': {'selection': [f'native-{i:02d}' for i in range(13, 19)], 'test': [f'native-{i:02d}' for i in range(19, 25)]},
        'phases_s': PHASES, 'mechanism': 'frozen public Q, unchanged L10-calibrated belief/planner, reference5,K5',
        'public_catalogue': 'L40 public reverse shortest paths on exact native network; L20 and L10 are exact ordered prefixes; assert originalL10 parity',
        'reply_cost': 'UTF8 compact JSON per Q: results records(id,category,lat,lon); sum all Q replies including duplicate POIs; reply-only estimate, not measured HTTP or request bytes',
        'ranking': 'same category travel-distance order and lexical POI tie rule; all POIs available',
        'utility': 'conditional on nonempty reachable reference; count undefined windows, no zero imputation',
        'status': 'posthoc development extension; primary test already inspected, no fresh confirmation claim',
        'attacker': 'primary attacks/selection/predictions frozen, no new attacker fitting or emissions'}
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == json.loads(json.dumps(declaration))
    else:
        save(OUT/'protocol.json', declaration)
    if (OUT/'results.json').exists():
        raise FileExistsError('Preserve completed depth diagnostic')
    data, public, private = load(DATA), load(PRIMARY/'public_transcripts.json.gz'), load(PRIMARY/'private_accounting.json.gz')
    cache = WORK/'public_resources'
    net_path = cache/'native.net.xml'
    assert sha(net_path) == data['network']['native_sha256']
    rn = build_lane_states(net_path, spacing_m=40.)
    pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')} for p in
        json.loads((ROOT/'artifacts/benchmarks/research_loop/resources.json').read_text())['pois_used']]
    reference = PublicPoiContext(LanePoiService(rn, pois, k=5), cache/'reference5.npz')
    reply10 = PublicPoiContext(LanePoiService(rn, pois, k=10), cache/'reply10.npz')
    reply40 = PublicPoiContext(LanePoiService(rn, pois, k=40), cache/'reply40-depth-diagnostic.npz')
    assert reply10.pois == reply40.pois and reply10.categories == reply40.categories
    assert np.array_equal(reply10.access, reply40.access)
    assert np.array_equal(reply10.signatures, reply40.signatures[:, :, :10]), 'L10 ordered prefix parity failed; primary remains untouched'
    response_cache = {}
    def response(state, depth):
        key = int(state), depth
        if key not in response_cache:
            ids = reply40.signatures[reply40.access[int(state)], :, :depth].ravel()
            ids = ids[ids >= 0]
            records = [{k: reply40.pois[int(i)][k] for k in ('id', 'category', 'lat', 'lon')} for i in ids]
            size = len(json.dumps({'results': records}, separators=(',', ':'), ensure_ascii=False).encode())
            response_cache[key] = (set(map(int, ids)), size, len(ids))
        return response_cache[key]
    rows = []
    by_family = {f['family_id']: f for f in data['families']}
    for group, truth in zip(public['groups'], private['rows']):
        if truth['split'] not in ('selection', 'test'):
            continue
        family = by_family[truth['family_id']]
        for slot, spec in enumerate(family['evaluator_only']['sessions']):
            trace = {int(p['time_s']): p for p in data['traces'][spec['session_id']]}
            for method in METHODS:
                for event, primary_utility in zip(group['streams'][method][slot]['events'], truth['evaluator_sessions'][slot]['utility'][method]):
                    t = int(event['timestamp_s'])
                    assert t == primary_utility['t']
                    point = trace[t]
                    state, _ = rn.nearest(point['lat'], point['lon'])
                    refs = [set(map(int, ids[ids >= 0])) for ids in reference.signatures[state] if np.any(ids >= 0)]
                    qstates = [rn.nearest(q['lat'], q['lon'])[0] for q in event['candidates']]
                    for depth in DEPTHS:
                        replies = [response(s, depth) for s in qstates]
                        union = set().union(*(r[0] for r in replies))
                        recall = float(np.mean([len(ids & union)/len(ids) for ids in refs])) if refs else None
                        if depth == 10:
                            assert recall == primary_utility['recall5'] or (
                                recall is not None and primary_utility['recall5'] is not None and np.isclose(recall, primary_utility['recall5']))
                        rows.append({'family_id': truth['family_id'], 'split': truth['split'], 'slot': slot,
                            'method': method, 'L': depth, 't': t, 'recall5': recall, 'nonempty_categories': len(refs),
                            'reply_bytes': sum(r[1] for r in replies), 'returned_records': sum(r[2] for r in replies)})
        print('Frozen native reply depths evaluated', truth['family_id'], flush=True)
    results = {}
    for method in METHODS:
        results[method] = {}
        for split in ('selection', 'test'):
            results[method][split] = {}
            for depth in DEPTHS:
                base = [r for r in rows if r['method'] == method and r['split'] == split and r['L'] == depth]
                results[method][split][str(depth)] = {phase: aggregate([r for r in base if start <= r['t'] <= end])
                    for phase, (start, end) in PHASES.items()}
    choices = {}
    for method in METHODS[1:]:
        selected = results[method]['selection']
        baseline_bytes = selected['10']['all_0_600']['mean_reply_bytes_per_event']
        eligible = []
        gates = {}
        for depth in DEPTHS:
            metrics = selected[str(depth)]['all_0_600']
            ratio = metrics['mean_reply_bytes_per_event']/baseline_bytes
            passed = metrics['family_macro_recall5'] >= .9 and metrics['median_session_recall5'] >= .9 and ratio <= 2.
            gates[str(depth)] = {'passes': passed, 'mean_reply_byte_ratio_to_L10': ratio,
                                'family_macro_recall5': metrics['family_macro_recall5'], 'median_session_recall5': metrics['median_session_recall5']}
            if passed:
                eligible.append(depth)
        chosen = min(eligible) if eligible else None
        choices[method] = {'chosen_L': chosen, 'selection_gates': gates,
            'test_at_chosen_L': results[method]['test'][str(chosen)] if chosen is not None else None}
    compressed_save(OUT/'utility_rows.json.gz', {'schema': 'native-frozen-reply-utility-rows-v1', 'rows': rows})
    save(OUT/'results.json', {'schema': 'native-fixed-Q-retrieval-depth-readout-v1',
        'protocol_sha256': sha(OUT/'protocol.json'), 'source_sha256': declaration['source_sha256'],
        'diagnostic_source_sha256': {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
            'benchmark/public_poi_context.py': sha(ROOT/'benchmark/public_poi_context.py')},
        'row_artifact_sha256': sha(OUT/'utility_rows.json.gz'), 'results': results, 'selection': choices,
        'resources': {'native_sha256': sha(net_path), 'catalogue_sha256': rn.catalogue_sha256,
            'reference_sha256': reference.sha256, 'reply10_sha256': reply10.sha256, 'reply40_sha256': reply40.sha256,
            'original_L10_prefix_parity': True},
        'no_new_emissions': True, 'primary_attacker_unchanged': True,
        'scope': declaration['status'], 'cost_scope': declaration['reply_cost']})
    print('Saved frozen-Q reply depth diagnostic', OUT, flush=True)


if __name__ == '__main__':
    main()
