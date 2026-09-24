"""Deterministic service/cost frontier with invertible public-query supplements.

Replays existing stochastic outputs exactly. No new private reads, RNG runs or
SUMO journeys. Every A/B/C record is scored; the old learned attacks transfer
through the explicit public inverse instead of treating constants as secrets.
"""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
from benchmark.engines.public_supplement import append_public_view, remove_public_view
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from evaluation.public_cover import fit_public_cover
from evaluation.lane_travel import LanePoiService
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.research_loop_cases import public_view, recall_pair, mean_optional, passes_recall_gate, SCENARIOS

BASE = ROOT/'artifacts/benchmarks/research_loop'
DATA = ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'
SCREEN = BASE/'iteration18_expanded_screening.json'
ATTACKS = BASE/'iteration18_expanded_attacks.json'
PUBLIC = BASE/'iteration22_public_cover.json'
OUT = BASE/'iteration27_public_supplement.json'
PARENTS = ('response_paced', 'response_paced_slack03')
COUNTS = (1, 2, 3)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def category_recall(reference, reply, states, truth):
    result = {}
    for ci, name in enumerate(reference.categories):
        target = set(reference.signatures[truth, ci])-{-1}
        found = set(reply.signatures[states, ci].ravel())-{-1}
        result[name] = len(target & found)/len(target) if target else None
    return result


def calculate(verbose=False):
    rn, _, reference, belief, metadata = load()
    service = LanePoiService(rn, list(reference.pois), k=10)
    reply = PublicPoiContext(service, CACHE/'poi10.npz')
    response = ResponseAwareAnchorModel(belief, reply)
    # All plans are deployment-public; construct before opening target labels.
    plans = {str(b): fit_public_cover(rn, response, reply, b) for b in COUNTS}
    controls = {str(5+b): fit_public_cover(rn, response, reply, 5+b) for b in COUNTS}
    for plan in [*plans.values(), *controls.values()]:
        states = [int(rn.nearest(*c)[0]) for c in plan['coordinates']]
        plan['server_access_states'] = states
        plan['reply_slots_L10_per_event'] = int(np.sum(reply.signatures[states] >= 0))
        # Independent coordinate API check at every new public query.
        for c, state in zip(plan['coordinates'], states):
            for ci, category in enumerate(reference.categories):
                expected = [reply.pois[int(i)]['id'] for i in reply.signatures[state, ci] if i >= 0]
                assert service.query(tuple(c), category) == expected
    data = json.loads(DATA.read_text()); screen = json.loads(SCREEN.read_text())
    attacks = json.loads(ATTACKS.read_text()); public = json.loads(PUBLIC.read_text())
    assert screen['provenance']['dataset_sha256'] == attacks['dataset_sha256'] == sha(DATA)
    assert attacks['screen_sha256'] == sha(SCREEN)
    assert screen['provenance']['resources_sha256'] == metadata['resource_sha256']
    records = [r for r in data['records'] if r['scenario'] in SCENARIOS]
    sources = {r['session_id']: r for r in public['sources']}
    assert len(sources) == 102 and len(records) == 173
    truth_by_session = {}
    for sid, ex in sources.items():
        trace = data['traces'][sid]
        actual = [int(rn.nearest(trace[i]['lat'], trace[i]['lon'])[0]) for i in ex['clock_indices']]
        assert actual == ex['reference_states']
        truth_by_session[sid] = dict(zip(ex['clock_indices'], actual))
    fixed_rows, fixed_sessions, fixed_summaries = [], [], []
    for key, plan in controls.items():
        k = int(key); states = plan['server_access_states']
        table = {state: recall_pair(reference, reply, states, state) for state in sorted({v for d in truth_by_session.values() for v in d.values()})}
        lookup = {sid: {i: table[state] for i, state in values.items()} for sid, values in truth_by_session.items()}
        for sid, utilities in lookup.items():
            fixed_sessions.append({'k': k, 'session_id': sid, 'family_id': sources[sid]['family_id'],
                'recall_L10': mean_optional([u['10'] for u in utilities.values()]), 'events': len(utilities)})
        for r in records:
            utilities = [lookup[sid][i] for sid, indices in zip(r['session_ids'], r['observed_indices']) for i in indices]
            fixed_rows.append({'k': k, 'record_id': r['record_id'], 'case_id': r['case_id'], 'family_id': r['family_id'],
                'recall': {L: mean_optional([u[L] for u in utilities]) for L in ('5', '10')},
                'events': len(utilities), 'eligible_events': sum(u['10'] is not None for u in utilities)})
        for case in sorted({r['case_id'] for r in records}):
            subset = [r for r in fixed_rows if r['k'] == k and r['case_id'] == case]
            value = mean_optional([r['recall']['10'] for r in subset])
            fixed_summaries.append({'k': k, 'case_id': case, 'recall_L10': value, 'pass_90pct': passes_recall_gate(value),
                'families': len({r['family_id'] for r in subset})})
    old_k8 = next(p for p in public['plans'] if p['k'] == 8)
    assert controls['8']['coordinates'] == [tuple(c) for c in old_k8['coordinates']]
    assert [r for r in fixed_rows if r['k'] == 8] == [r for r in public['case_rows'] if r['k'] == 8]
    attack_rows = {(r['record_id'], r['rep'], r['method']): r for r in attacks['rows']}
    attack_summaries = {(r['method'], r['case_id']): r for r in attacks['summaries']}
    executions, rows, manifests = [], [], []
    for item in screen['shards']:
        path = BASE/'expanded_screening'/item['file']; assert sha(path) == item['sha256']
        manifests.append({'file': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        shard = json.loads(gzip.decompress(path.read_bytes()))
        lookup = {}
        for parent in shard['executions']:
            if parent['method'] not in PARENTS:
                continue
            sid, rep = parent['session_id'], parent['rep']; clock = parent['clock_indices']
            assert clock == sources[sid]['clock_indices']
            original = {'events': [parent['events'][str(i)] for i in clock]}
            server_states = {i: [int(rn.nearest(c['lat'], c['lon'])[0]) for c in parent['events'][str(i)]['candidates']] for i in clock}
            for count in COUNTS:
                plan = plans[str(count)]; method = parent['method']+f'+public{count}'
                hybrid = append_public_view(original, plan['coordinates'])
                recovered = remove_public_view(hybrid, plan['coordinates']); assert recovered == original
                utility, category, items = {}, {}, {}
                for i in clock:
                    old = server_states[i]; states = old+plan['server_access_states']; truth = truth_by_session[sid][i]
                    before = recall_pair(reference, reply, old, truth)
                    assert before == parent['utility_by_index'][str(i)]
                    utility[str(i)] = recall_pair(reference, reply, states, truth)
                    for L in ('5', '10'):
                        assert (before[L] is None and utility[str(i)][L] is None) or utility[str(i)][L] >= before[L]-1e-12
                    category[str(i)] = category_recall(reference, reply, states, truth)
                    items[str(i)] = int(np.sum(reply.signatures[states] >= 0))
                    assert items[str(i)] == parent['reply_items_L10_by_index'][str(i)]+plan['reply_slots_L10_per_event']
                ex = {'method': method, 'parent_method': parent['method'], 'supplement': count, 'total_k': 5+count,
                    'family_id': parent['family_id'], 'session_id': sid, 'rep': rep, 'clock_indices': clock,
                    'parent_execution_sha256': digest(parent), 'parent_public_sha256': digest(original),
                    'hybrid_public_sha256': digest(hybrid), 'recovered_public_sha256': digest(recovered),
                    'utility_by_index': utility, 'category_recall_L10_by_index': category, 'reply_slots_by_index': items,
                    'budget_bound': parent['budget_bound'], 'parent_anchor_sha256': parent['evaluator_anchor_sha256'],
                    'whole_session_recall_L10': mean_optional([u['10'] for u in utility.values()])}
                executions.append(ex); lookup[sid, rep, method] = ex
        for parent_row in shard['rows']:
            if parent_row['method'] not in PARENTS:
                continue
            record = next(r for r in records if r['record_id'] == parent_row['record_id'])
            inherited = attack_rows[parent_row['record_id'], parent_row['rep'], parent_row['method']]
            for count in COUNTS:
                method = parent_row['method']+f'+public{count}'; utilities, categories, view_hashes = [], [], []
                for slot, (sid, indices) in enumerate(zip(record['session_ids'], record['observed_indices'])):
                    ex = lookup[sid, parent_row['rep'], method]
                    original = parent_row['public_views'][slot]
                    hybrid = append_public_view(original, plans[str(count)]['coordinates'])
                    assert remove_public_view(hybrid, plans[str(count)]['coordinates']) == original
                    view_hashes.append(digest(hybrid))
                    utilities.extend(ex['utility_by_index'][str(i)] for i in indices)
                    categories.extend(ex['category_recall_L10_by_index'][str(i)] for i in indices)
                rows.append({'method': method, 'parent_method': parent_row['method'], 'supplement': count,
                    'total_k': 5+count, 'record_id': record['record_id'], 'case_id': record['case_id'],
                    'family_id': record['family_id'], 'rep': parent_row['rep'], 'hybrid_view_sha256': view_hashes,
                    'inherited_attack_errors_sha256': digest(inherited['errors']),
                    'composition_bound': parent_row['composition_bound'],
                    'recall': {L: mean_optional([u[L] for u in utilities]) for L in ('5', '10')},
                    'eligible_events': sum(u['10'] is not None for u in utilities),
                    'empty_reference_events': sum(u['10'] is None for u in utilities),
                    'category_L10': {c: mean_optional([u[c] for u in categories]) for c in reference.categories}})
        if verbose:
            print('Completed supplemented family', shard['family_id'], flush=True)
    summaries = []
    for method in sorted({r['method'] for r in rows}):
        for case in sorted({r['case_id'] for r in rows}):
            subset = [r for r in rows if r['method'] == method and r['case_id'] == case]
            parent = subset[0]['parent_method']; learned = attack_summaries[parent, case]
            value = mean_optional([r['recall']['10'] for r in subset]); k = subset[0]['total_k']
            fixed = next(s for s in fixed_summaries if s['k'] == k and s['case_id'] == case)
            summaries.append({'method': method, 'parent_method': parent, 'supplement': subset[0]['supplement'], 'total_k': k,
                'case_id': case, 'family_count': len({r['family_id'] for r in subset}),
                'record_count': len({r['record_id'] for r in subset}), 'recall_L10': value,
                'pass_90pct': passes_recall_gate(value), 'same_K_fixed_control_recall_L10': fixed['recall_L10'],
                'family_L10': {f: mean_optional([r['recall']['10'] for r in subset if r['family_id'] == f]) for f in sorted({r['family_id'] for r in subset})},
                'category_L10': {c: mean_optional([r['category_L10'][c] for r in subset]) for c in reference.categories},
                'inherited_attack_selection': learned['selected_attack'],
                'inherited_privacy_metrics': {key: value for key, value in learned['metrics'].items() if key != 'recall_L10'},
                'privacy_transfer': 'remove public suffix and run unchanged parent attacker; coordinate information equivalent'})
    source_paths = (Path(__file__), DATA, SCREEN, ATTACKS, PUBLIC,
        BASE/'iteration24_site_density_components.json', ROOT/'benchmark/engines/public_supplement.py',
        ROOT/'evaluation/public_cover.py', ROOT/'experiments/research_loop_cases.py')
    return {'schema': 'public-supplement-expanded-frontier-v1', 'resources': metadata,
        'scope': 'deterministic augmentation of exposed expanded-development transcripts; no fresh RNG, SUMO or confirmation; elementary service/privacy equivalence at extra query cost',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in source_paths}, 'parent_shards': manifests,
        'new_stochastic_full_session_executions': 0, 'deterministic_hybrid_session_evaluations': len(executions),
        'new_fixed_control_session_evaluations': len(sources)*2, 'replayed_K8_fixed_control_sessions': len(sources),
        'service_contract': {'adaptive_k': 5, 'supplement_counts': COUNTS, 'reply_L': 10, 'reference_k': 5,
                             'cache_control_still_required': True, 'network_bytes_and_latency_measured': False},
        'public_supplement_plans': plans, 'matched_fixed_plans': controls,
        'fixed_control_rows': fixed_rows, 'fixed_control_sessions': fixed_sessions, 'fixed_control_summaries': fixed_summaries,
        'executions': executions, 'case_rows': rows, 'summaries': summaries,
        'raw_positive_control': [{k: v for k, v in r.items() if k in ('case_id','family_count','record_count','selected_attack','metrics')}
                                 for r in attacks['summaries'] if r['method'] == 'raw']}


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    result = calculate(verbose=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for method in sorted({r['method'] for r in result['summaries']}):
        summaries = [r for r in result['summaries'] if r['method'] == method]
        sessions = [r for r in result['executions'] if r['method'] == method]
        rare = next(r for r in summaries if r['case_id'] == 'S1.C')
        print(method, 'gates', sum(r['pass_90pct'] for r in summaries), 'S1.C', rare['recall_L10'],
              'whole', mean_optional([s['whole_session_recall_L10'] for s in sessions]), flush=True)


if __name__ == '__main__':
    main()
