"""Independent arithmetic/source/causal-cache audit of the matched development run.

Does not invoke the runner's generation, aggregation or selection helpers.
The v1 readout is immutable. A derived v2 fixes a legacy aggregate denominator
for cold/warm subsets; it changes no Q, model, score selection or observations.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from benchmark.public_poi_context import PublicPoiContext
from data.lane_states import build_lane_states
from evaluation.candidate_future_attack import CandidateFutureAttack, coordinates
from evaluation.lane_travel import LanePoiService

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/benchmarks/jisa_native_anchor_ablation_20261006_v1'
DATA = ROOT/'artifacts/datasets/future_controlled_20261005_v2/dataset.json.gz'
REPLY = ROOT/'artifacts/benchmarks/future_native_depth_20261005_v1/public_reply40.npz'
METHODS = ('raw', 'rem_epoch8', 'planar_epoch8')
PHASES = {'all_0_600': (0, 600), 'early_0_180': (0, 180),
          'middle_200_380': (200, 380), 'tail_400_600': (400, 600)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_text())


def check_fields(expected, actual):
    if isinstance(expected, dict):
        for name, value in expected.items():
            assert name in actual, name
            check_fields(value, actual[name])
    elif isinstance(expected, list):
        assert len(expected) == len(actual)
        for a, b in zip(expected, actual): check_fields(a, b)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        assert np.isclose(expected, actual, rtol=1e-10, atol=1e-10), (expected, actual)
    else:
        assert expected == actual, (expected, actual)


def summarize(rows):
    """Scope-aware counts; unrepresented slots are never reference failures."""
    groups = {}
    for row in rows:
        groups.setdefault((row['family_id'], row['slot']), []).append(row)
    sessions = []
    for (family, slot), values in sorted(groups.items()):
        defined = [r['recall5'] for r in values if r['recall5'] is not None]
        if defined: sessions.append({'family_id': family, 'slot': slot, 'recall5': float(np.mean(defined))})
    family_values = {}
    for family in sorted({r['family_id'] for r in rows}):
        defined = [r['recall5'] for r in rows if r['family_id'] == family and r['recall5'] is not None]
        family_values[family] = float(np.mean(defined)) if defined else None
    means = [v for v in family_values.values() if v is not None]
    count = sum(r['recall5'] is not None for r in rows)
    return {'family_macro_recall5': float(np.mean(means)), 'family_values': family_values,
        'minimum_family_recall5': min(means), 'maximum_family_recall5': max(means),
        'median_session_recall5': float(np.median([s['recall5'] for s in sessions])),
        'minimum_session': min(sessions, key=lambda s: s['recall5']),
        'reference_defined_windows': count, 'total_windows': len(rows), 'reference_coverage': count/len(rows),
        'represented_session_count': len(groups), 'undefined_session_count': len(groups)-len(sessions),
        'mean_cached_poi_count': float(np.mean([r['cached_poi_count'] for r in rows])),
        'maximum_cached_poi_count': max(r['cached_poi_count'] for r in rows),
        'reply_bytes_total': sum(r['reply_bytes'] for r in rows), 'request_count': sum(r['requests'] for r in rows),
        'session_values': sessions}


def metrics(rows, probabilities):
    truth = np.array([r['choice_index'] for r in rows]); p = np.array(probabilities)
    assert p.shape == (len(rows), 2) and np.all(np.isfinite(p)) and np.all(p >= 0)
    assert np.allclose(p.sum(axis=1), 1.)
    prediction = np.argmax(p, axis=1); correct = prediction == truth
    errors, chances = [], []
    for row, predicted in zip(rows, prediction):
        dest = np.array([c['destination_xy'] for c in row['public_context']['choices']])
        distances = np.linalg.norm(dest-np.array(row['destination_xy']), axis=1)
        errors.append(float(distances[predicted])); chances.append(float(np.mean(distances)))
    return {'exact_candidate_edge_accuracy': float(np.mean(correct)),
        'balanced_accuracy': float(np.mean([np.mean(correct[truth == i]) for i in (0, 1)])),
        'brier': float(np.mean((p[:, 1]-truth)**2)),
        'log_loss': float(np.mean(-np.log(np.maximum(p[np.arange(len(rows)), truth], 1e-12)))),
        'destination_mae_m': float(np.mean(errors)), 'destination_hit100': float(np.mean(np.array(errors) <= 100)),
        'uniform_destination_mae_m': float(np.mean(chances)),
        'routine_accuracy': float(np.mean([v for v, r in zip(correct, rows) if r['destination_role'] == 'routine'])),
        'rare_accuracy': float(np.mean([v for v, r in zip(correct, rows) if r['destination_role'] == 'rare'])),
        'n': len(rows), 'family_count': len({r['family_id'] for r in rows})}


def attack_rows(public, private, method, stage):
    result = []
    for group, truth in zip(public['groups'], private['rows']):
        for slot in (6, 7):
            target = truth['evaluator_sessions'][slot]
            prefix = [e for e in group['streams'][method][slot]['events']
                      if e['timestamp_s'] <= group['public_clocks'][stage+'_t']]
            result.append({'family_id': truth['family_id'], 'split': truth['split'],
                'query_slot': slot, 'query': {'events': prefix}, 'histories': group['streams'][method][:6],
                'public_context': group['public_context'], 'choice_index': target['choice_index'],
                'destination_role': target['destination_role'], 'destination_xy': target['destination_xy'],
                'actual_next_edge': target['next_edge_labels'][stage+'_t']})
    return result


def verify(out, validation_output=None):
    protocol = load(out/'protocol.json'); result = load(out/'results.json')
    assert sha(out/'protocol.json') == (out/'protocol.sha256').read_text().strip()
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
        snapshot = out/'source_snapshot'/name
        if snapshot.exists(): assert sha(snapshot) == digest
    for name, digest in protocol['preserve_old_evidence_sha256'].items():
        assert sha(ROOT/name) == digest, 'Old evidence was altered: '+name
    for name, digest in result['file_sha256'].items(): assert sha(out/name) == digest
    cfg = protocol['configuration']; budget = cfg['budget']
    check_fields({'session_slots': 8, 'horizon': 12, 'total_effective_epsilon_per_m': .23,
        'unit_epsilon_per_m': .00125, 'nominal_session_budget_per_m': .03,
        'effective_session_cap_per_m': .02875, 'max_units_per_session': 23,
        'read_interval_s': 60., 'start_s': 0., 'end_s': 12000.}, budget)
    check_fields({'K': 5, 'reply_depth_L': 20, 'planner_reply_depth_L': 10,
                  'reference_top_k': 5, 'theta_m': 200., 'utility_slack': .03}, cfg)
    data, public, private = load(DATA), load(out/'public_transcripts.json.gz'), load(out/'private_accounting.json.gz')
    assert len(data['families']) == len(public['groups']) == len(private['rows']) == 24
    assert private['private_keys_exported'] is False
    families = {f['family_id']: f for f in data['families']}
    assert set(families) == {f for group in protocol['splits'].values() for f in group}
    rn = build_lane_states(ROOT/data['network']['compressed_path'], spacing_m=40.)
    pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')} for p in
        load(ROOT/'artifacts/benchmarks/research_loop/resources.json')['pois_used']]
    reply = PublicPoiContext(LanePoiService(rn, pois, k=40), REPLY)
    assert rn.catalogue_sha256 == result['resources']['catalogue']['sha256']
    assert reply.sha256 == result['resources']['reply40_sha256'] and len(reply.pois) == 418
    ledger_checks = 0; event_count = 0
    for group, truth in zip(public['groups'], private['rows']):
        family = families[truth['family_id']]
        assert group['public_context'] == family['public_context']
        assert truth['family_id'] in protocol['splits'][truth['split']]
        for slot, spec in enumerate(family['evaluator_only']['sessions']):
            target = truth['evaluator_sessions'][slot]
            assert target['slot'] == slot and target['choice_index'] == spec['choice_index']
            assert target['destination_role'] == spec['destination_role']
            trace = data['traces'][spec['session_id']]
            end = next(p for p in trace if p['time_s'] == 600)
            endpoint = coordinates({'events': [{'event_id': 'e', 'timestamp_s': 0.,
                'candidates': [{'candidate_id': 'q', 'lat': end['lat'], 'lon': end['lon']}]}]})[0][0][0]
            check_fields(endpoint.tolist(), target['destination_xy'])
            if slot >= 6:
                for stage in ('shared_fork_t', 'turn_visible_t'):
                    cut = family['clocks'][stage]
                    actual_edge = next(p['edge_id'] for p in trace if p['time_s'] > cut and
                        not p['edge_id'].startswith(':') and (stage != 'shared_fork_t' or p['edge_id'] != family['fork_edge']))
                    assert target['next_edge_labels'][stage] == actual_edge
        for method in METHODS:
            assert len(group['streams'][method]) == 8
            for slot, spec in enumerate(family['evaluator_only']['sessions']):
                assert spec['depart_s'] == slot*1500.
                expected = sorted(set(range(0, 601, 20)) | ({family['clocks']['shared_fork_t'], family['clocks']['turn_visible_t']} if slot >= 6 else set()))
                events = group['streams'][method][slot]['events']
                assert [e['timestamp_s'] for e in events] == expected
                coordinates({'events': events})  # Fails closed on private feature fields.
                assert all(len(e['candidates']) == (1 if method == 'raw' else 5) for e in events)
                event_count += len(events)
                if method == 'raw':
                    trace = {int(p['time_s']): p for p in data['traces'][spec['session_id']]}
                    for event in events:
                        source = trace[int(event['timestamp_s'])]
                        assert (event['candidates'][0]['lat'], event['candidates'][0]['lon']) == (source['lat'], source['lon'])
                    continue
                state = truth['evaluator_sessions'][slot]['ledger'][method]
                costs = state['ledger']; assert len(costs) == len(events) == len(state['states'])
                spent = 0; read_times = []
                for e, values, lane_states in zip(events, costs, state['states']):
                    cost = values['cost_units']; spent += cost
                    assert values['spent_units'] == spent <= 23
                    if values['private_read']:
                        read_times.append(e['timestamp_s'])
                        assert cost == (1 if len(read_times) == 1 or values['branch'] == 'reuse' else 2)
                    else: assert cost == 0
                    assert [(q['lat'], q['lon']) for q in e['candidates']] == [rn.latlon(s) for s in lane_states]
                    ledger_checks += 1
                assert read_times == state['supplier_times_s'] and len(read_times) == state['private_reads'] <= 11
                assert read_times[0] == 0 and all(b-a >= 60 for a, b in zip(read_times, read_times[1:]))
                check_fields(spent*.00125, state['spent_per_m']); assert state['spent_per_m'] <= .02875+1e-12
                assert state['anchor_primitive'] == ('pr_sm_rem' if method == 'rem_epoch8' else 'predictive_planar_laplace')
            if method != 'raw':
                check_fields(sum(s['ledger'][method]['spent_per_m'] for s in truth['evaluator_sessions']), truth['epoch_accounting'][method]['spent_per_m'])
                assert truth['epoch_accounting'][method]['spent_per_m'] <= .23+1e-12
                check_fields(.23, truth['epoch_accounting'][method]['reserved_cap_per_m'])

    utility_rows = load(out/'utility_rows.json.gz')['rows']; wire_rows = load(out/'wire_rows.json.gz')['rows']
    lookup = {(r['family_id'], r['slot'], r['method'], r['event_id'], r['policy']): r for r in utility_rows}
    wire = {(r['family_id'], r['slot'], r['method'], r['event_id']): r for r in wire_rows}
    assert len(lookup) == len(utility_rows) == 2*event_count
    assert len(wire) == len(wire_rows) == event_count
    verified_rows = []
    for group, truth in zip(public['groups'], private['rows']):
        family = families[truth['family_id']]; accumulated = {m: set() for m in METHODS}
        last_time = {m: -1. for m in METHODS}
        for slot, spec in enumerate(family['evaluator_only']['sessions']):
            trace = {int(p['time_s']): p for p in data['traces'][spec['session_id']]}
            for method in METHODS:
                for event in group['streams'][method][slot]['events']:
                    t = int(event['timestamp_s']); absolute = slot*1500.+t
                    assert last_time[method] < absolute < 12000.; last_time[method] = absolute
                    replies = []; response_bytes = 0; request_bytes = 0
                    for q in event['candidates']:
                        qs, _ = rn.nearest(q['lat'], q['lon'])
                        ids = reply.signatures[reply.access[qs], :, :20].ravel(); ids = list(map(int, ids[ids >= 0]))
                        replies.append(ids)
                        records = [{k: reply.pois[i][k] for k in ('id', 'category', 'lat', 'lon')} for i in ids]
                        response_bytes += len(json.dumps({'results': records}, separators=(',', ':'), ensure_ascii=False).encode())
                        payload = {'timestamp_s': absolute, 'lat': q['lat'], 'lon': q['lon'], 'categories': list(reply.categories), 'L': 20}
                        request_bytes += len(json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode())
                    current = {i for ids in replies for i in ids}; accumulated[method].update(current)
                    expected_wire = {'reply_poi_indices_by_Q': replies, 'requests': len(replies),
                        'request_bytes': request_bytes, 'reply_bytes': response_bytes, 'absolute_public_time_s': absolute, 't': t}
                    check_fields(expected_wire, wire[truth['family_id'], slot, method, event['event_id']])
                    gps = trace[t]; state, _ = rn.nearest(gps['lat'], gps['lon'])
                    refs = [set(map(int, ids[ids >= 0])) for ids in reply.signatures[state, :, :5] if np.any(ids >= 0)]
                    for cache_policy, ids in (('current_only', current), ('versioned_epoch8', accumulated[method])):
                        actual = lookup[truth['family_id'], slot, method, event['event_id'], cache_policy]
                        expected = {'cached_poi_ids': sorted(ids), 'cached_poi_count': len(ids),
                            'nonempty_reference_categories': len(refs), 'absolute_public_time_s': absolute,
                            'recall5': float(np.mean([len(ref & ids)/len(ref) for ref in refs])) if refs else None,
                            'requests': len(replies), 'request_bytes': request_bytes, 'reply_bytes': response_bytes}
                        check_fields(expected, actual); assert len(ids) <= 418
                        verified_rows.append(actual)
                    # Same Q/reply cost for both local policy answers; cached IDs only affect local candidates.

    corrected = {}; denominator_changes = []
    for method in METHODS:
        corrected[method] = {}
        for split in protocol['splits']:
            corrected[method][split] = {}
            for cache_policy in ('current_only', 'versioned_epoch8'):
                rows = [r for r in verified_rows if r['method'] == method and r['split'] == split and r['policy'] == cache_policy]
                groups = {name: [r for r in rows if a <= r['t'] <= b] for name, (a, b) in PHASES.items()}
                groups.update(cold_first_session=[r for r in rows if r['slot'] == 0], warm_later_sessions=[r for r in rows if r['slot'] > 0])
                corrected[method][split][cache_policy] = {}
                for name, scope in groups.items():
                    derived = summarize(scope); original = result['utility'][method][split][cache_policy][name]
                    compare = {k: v for k, v in derived.items() if k != 'represented_session_count' and
                               (k != 'undefined_session_count' or name in PHASES)}
                    check_fields(compare, original)
                    if derived['undefined_session_count'] != original['undefined_session_count']:
                        denominator_changes.append({'method': method, 'split': split, 'policy': cache_policy, 'scope': name,
                            'v1_undefined_session_count': original['undefined_session_count'],
                            'corrected_undefined_session_count': derived['undefined_session_count'],
                            'represented_session_count': derived['represented_session_count']})
                    if name == 'all_0_600':
                        derived['request_bytes_total'] = sum(r['request_bytes'] for r in scope)
                        check_fields(derived['request_bytes_total'], original['request_bytes_total'])
                    corrected[method][split][cache_policy][name] = derived

    selections = load(out/'attack_selection.json'); prediction_rows = load(out/'attack_predictions.json.gz')['rows']
    prediction_lookup = {(p['attack_key'], p['family_id'], p['slot']): p for p in prediction_rows}
    assert len(prediction_lookup) == len(prediction_rows) == 12*12
    attack_checks = 0
    for name, selection in selections.items():
        method, stage, task = name.split('--'); history = task == 'S6_history_destination'
        rows = attack_rows(public, private, method, stage)
        train = [r for r in rows if r['split'] == 'train']; validation = [r for r in rows if r['split'] == 'selection']; test = [r for r in rows if r['split'] == 'test']
        stored_models = {}
        for label, record in selection['models'].items():
            assert sha(out/record['path']) == record['sha256']
            with (out/record['path']).open('rb') as stream: stored_models[label] = pickle.load(stream)
        model, negative = stored_models['model'], stored_models['permutation']
        assert model.use_history == negative.use_history == history
        refit = CandidateFutureAttack(train, use_history=history)
        refit_negative = CandidateFutureAttack(train, use_history=history, permutation=True)
        validation_banks = [model.predict(r['query'], r['public_context'], r['histories']) for r in validation]
        for row, bank, stored in zip(validation, validation_banks, selection['selection_predictions']):
            check_fields({n: p.tolist() for n, p in bank.items()}, stored)
            again = refit.predict(row['query'], row['public_context'], row['histories'])
            assert np.array_equal(again['candidate_trees'], bank['candidate_trees'])
        scores = {n: metrics(validation, [p[n] for p in validation_banks]) for n in sorted(validation_banks[0])}
        check_fields(scores, selection['selection_scores'])
        chosen = min(scores, key=lambda n: (-scores[n]['balanced_accuracy'], scores[n]['log_loss'], n))
        assert chosen == selection['selected_attacker'] == result['attacks'][name]['selected_attacker']
        banks = []
        for row in test:
            actual = prediction_lookup[name, row['family_id'], row['query_slot']]
            bank = model.predict(row['query'], row['public_context'], row['histories'])
            permuted = negative.predict(row['query'], row['public_context'], row['histories'])['candidate_trees']
            check_fields({n: p.tolist() for n, p in bank.items()}, actual['banks'])
            check_fields(permuted.tolist(), actual['permuted_candidate_trees'])
            assert np.array_equal(refit.predict(row['query'], row['public_context'], row['histories'])['candidate_trees'], bank['candidate_trees'])
            assert np.array_equal(refit_negative.predict(row['query'], row['public_context'], row['histories'])['candidate_trees'], permuted)
            banks.append(bank); attack_checks += len(bank)+1
        output = result['attacks'][name]
        check_fields(metrics(test, [p[chosen] for p in banks]), output['test'])
        check_fields({n: metrics(test, [p[n] for p in banks]) for n in sorted(banks[0])}, output['bank_test_descriptive_only'])
        check_fields(metrics(test, [prediction_lookup[name, r['family_id'], r['query_slot']]['permuted_candidate_trees'] for r in test]), output['permuted_training_control'])
        for family in protocol['splits']['test']:
            check_fields(metrics([r for r in test if r['family_id'] == family], [b[chosen] for r, b in zip(test, banks) if r['family_id'] == family]), output['test_family_metrics'][family])
    # Genuine raw signal at the public turn clock; before-fork chance remains ambiguity.
    for task in ('S5_next_edge', 'S6_history_destination'):
        assert result['attacks'][f'raw--turn_visible--{task}']['test']['exact_candidate_edge_accuracy'] == 1.
        assert result['attacks'][f'raw--shared_fork--{task}']['test']['exact_candidate_edge_accuracy'] == .5

    derived = {'schema': 'jisa-matched-anchor-scope-corrected-readout-v2',
        'source_results_sha256': sha(out/'results.json'), 'source_utility_rows_sha256': sha(out/'utility_rows.json.gz'),
        'calculation_source_sha256': sha(Path(__file__)), 'protocol_sha256': sha(out/'protocol.json'),
        'reason': 'legacy aggregate undefined_session_count assumed eight slots for cold/warm subsets; use actual represented sessions. Independently recomputed means/minimum/coverage/cost, unchanged Q/attacks/selection',
        'configuration': result['configuration'], 'status': result['status'], 'attacks': result['attacks'],
        'utility': corrected, 'denominator_changes': denominator_changes, 'no_emission_or_attacker_change': True}
    derived_path = out/'derived_readout_v2.json'
    if derived_path.exists(): assert load(derived_path) == derived
    else: derived_path.write_text(json.dumps(derived, indent=2, allow_nan=False)+'\n')
    record = {'schema': 'jisa-matched-independent-validation-v1', 'validated_utc': datetime.now(timezone.utc).isoformat(),
        'source_results_sha256': sha(out/'results.json'), 'derived_readout_v2_sha256': sha(derived_path),
        'verifier_sha256': sha(Path(__file__)), 'event_count': event_count,
        'utility_rows_checked': len(verified_rows), 'ledger_steps_checked': ledger_checks,
        'attacker_probability_rows_checked': attack_checks, 'checkpoint_refits_checked': 24,
        'cold_warm_denominators_corrected': len(denominator_changes), 'old_source_outputs_unchanged': True,
        'same_cap_map_clock_K_L': True, 'dynamic_service_evaluated': False, 'confirmation': False}
    target = Path(validation_output) if validation_output else out/'validation.json'
    if target.exists():
        print('All independent checks passed; existing validation retained:', target)
    else:
        target.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        print('All independent checks passed; validation saved:', target)
    print(json.dumps({k: v for k, v in record.items() if k not in ('validated_utc', 'verifier_sha256')}, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--validation-output', type=Path)
    args = parser.parse_args(); verify(args.output, args.validation_output)


if __name__ == '__main__':
    main()
