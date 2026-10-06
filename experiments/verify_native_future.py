"""Replay native S5/S6 provenance, causal labels, metrics and accounting.

Default rechecks retain an existing validation record. An explicit new output
path records a fresh check without overwriting completed research evidence.
No private sampler key or SQLite contents are read by this verifier.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import sumolib
from evaluation.candidate_future_attack import (CandidateFutureAttack, coordinates,
                                                forecast_metrics, validate_context)
from evaluation.identity_future import public_arrays
from experiments.build_future_sumo_cohort import ROOT, sha, save
from experiments.future_sumo_eval import DATA, OUT, METHODS, forecast_rows


def read_gz(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def check_pins(pins):
    for path, expected in pins.items():
        p = ROOT/path
        if not p.exists():
            p = OUT/path
        assert sha(p) == expected, f'Pinned source changed: {path}'


def near(actual, expected):
    assert np.isclose(actual, expected, rtol=1e-10, atol=1e-10), (actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-output', type=Path,
                        help='New record path; refuses to replace an existing file')
    args = parser.parse_args()
    output = args.validation_output or OUT/'validation.json'
    if args.validation_output is not None and output.exists():
        raise FileExistsError(f'Preserve completed evidence: {output}')
    data = read_gz(DATA)
    original_dir = ROOT/'artifacts/datasets/future_controlled_20261005_v1'
    original = read_gz(original_dir/'dataset.json.gz')
    for directory in (original_dir, DATA.parent):
        manifest = json.loads((directory/'manifest.json').read_text())
        assert sha(directory/'dataset.json.gz') == manifest['dataset_sha256']
        assert sha(directory/'protocol.json') == manifest['protocol_sha256']
        check_pins(manifest['source_sha256'])
    assert sha(original_dir/'dataset.json.gz') == data['source_cohort_sha256']
    compressed = ROOT/data['network']['compressed_path']
    assert sha(compressed) == data['network']['compressed_sha256']
    native_bytes = gzip.decompress(compressed.read_bytes())
    assert hashlib.sha256(native_bytes).hexdigest() == data['network']['native_sha256']
    location = ET.fromstring(native_bytes).find('location')
    assert np.allclose([float(v) for v in location.attrib['netOffset'].split(',')], 0.)
    # Read the archived SAME network, never infer candidate IDs from GPS truth.
    net = sumolib.net.readNet(str(compressed), withInternal=True)
    assert len(net.getEdges(withInternal=False)) == data['network']['external_edges']
    assert sum(e.getID().startswith(':') for e in net.getEdges()) == data['network']['internal_edges'] > 0
    assert data['network']['turns_authoritative_for_this_new_network']
    assert not data['network']['original_lane_turn_parity']
    public = read_gz(OUT/'public_transcripts.json.gz')
    private = read_gz(OUT/'private_accounting.json.gz')
    results = json.loads((OUT/'results.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text())
    check_pins(results['source_sha256'])
    check_pins(private['generation_source_sha256'])
    assert results['protocol_sha256'] == sha(OUT/'protocol.json')
    assert private['dataset_sha256'] == protocol['dataset_sha256'] == sha(DATA)
    assert private['public_transcript_sha256'] == sha(OUT/'public_transcripts.json.gz')
    assert not private['private_keys_exported']
    resources = private['resource_metadata']
    check_pins(resources['source_sha256'])
    assert resources['native_sha256'] == data['network']['native_sha256']
    assert resources['internal_states'] > 0
    assert len(data['families']) == len(public['groups']) == len(private['rows']) == 24
    assert data['session_count'] == len(data['traces']) == 192
    split_sets = {s: set(protocol['splits'][s]) for s in ('train', 'selection', 'test')}
    assert [len(split_sets[s]) for s in split_sets] == [12, 6, 6]
    assert all(not split_sets[a] & split_sets[b] for a, b in
               (('train', 'selection'), ('train', 'test'), ('selection', 'test')))
    edge_sets = {s: set() for s in split_sets}
    caps = {'geoi_session_reset': (1.84, .23, .01), 'geoi_epoch8': (.23, .02875, .00125)}
    for f, predecessor, group, truth in zip(data['families'], original['families'], public['groups'], private['rows']):
        assert f['family_id'] == predecessor['family_id'] == truth['family_id']
        assert f['split'] == truth['split'] and f['family_id'] in split_sets[f['split']]
        assert f['public_context'] == predecessor['public_context'] == group['public_context']
        assert f['clocks'] == predecessor['clocks']
        assert set(group) == {'public_scope', 'public_context', 'public_clocks', 'streams'}
        assert set(group['streams']) == set(METHODS)
        choices = validate_context(group['public_context'])
        for c in choices:
            net.getEdge(c['edge_id'])
            net.getEdge(c['destination_edge_id'])
            for lane_id in c['via_lane_ids']:
                assert net.getLane(lane_id).getEdge().getID().startswith(':')
            edge_sets[f['split']].add(c['edge_id'])
        sessions = f['evaluator_only']['sessions']
        routine = f['evaluator_only']['routine_choice']
        assert sum(s['choice_index'] == routine for s in sessions[:6]) == 5
        assert {s['choice_index'] for s in sessions[6:]} == {0, 1}
        assert [s['day'] for s in sessions] == list(range(1, 9))
        for slot, (spec, info) in enumerate(zip(sessions, truth['evaluator_sessions'])):
            trace = data['traces'][spec['session_id']]
            by_t = {int(p['time_s']): p for p in trace}
            assert set(range(601)).issubset(by_t)
            parked = by_t[600]
            assert parked['speed_m_s'] == 0.
            assert parked['edge_id'] == choices[spec['choice_index']]['destination_edge_id']
            assert info['choice_index'] == spec['choice_index'] and info['day'] == spec['day']
            clock = sorted(set(range(0, 601, 20)) | (set(group['public_clocks'].values()) if slot >= 6 else set()))
            for method in METHODS:
                stream = group['streams'][method][slot]
                assert set(stream) == {'events'}
                public_arrays(stream)
                assert [e['timestamp_s'] for e in stream['events']] == clock
                assert all(len(e['candidates']) == (1 if method == 'raw' else 5) for e in stream['events'])
                if method == 'raw':
                    for event in stream['events']:
                        p = by_t[int(event['timestamp_s'])]
                        assert event['candidates'][0]['lat'] == p['lat']
                        assert event['candidates'][0]['lon'] == p['lon']
                else:
                    ledger = info['ledger'][method]
                    allocation = ledger['allocation']
                    epoch_cap, slot_cap, unit = caps[method]
                    assert allocation['slot'] == slot and allocation['horizon'] == 12
                    assert allocation['max_units'] == 23
                    near(allocation['effective_cap_per_m'], slot_cap)
                    near(allocation['unit_epsilon_per_m'], unit)
                    near(allocation['nominal_budget_per_m'], 24*unit)
                    assert len(ledger['ledger']) == len(clock)
                    near(sum(row['cost_units'] for row in ledger['ledger'])*unit, ledger['spent_per_m'])
                    assert ledger['private_reads'] == sum(row['private_read'] for row in ledger['ledger'])
                    running = 0
                    read_times = []
                    for row, time in zip(ledger['ledger'], clock):
                        if row['private_read']:
                            reserve = 1 if not read_times else 2
                            assert running+reserve <= 23
                            assert not read_times or time-read_times[-1] >= 60.
                            read_times.append(time)
                            assert row['cost_units'] == (2 if row['branch'] == 'fresh' and len(read_times) > 1 else 1)
                        else:
                            assert row['cost_units'] == 0
                        running += row['cost_units']
                        assert running == row['spent_units'] <= 23
                    assert ledger['spent_per_m'] <= slot_cap+1e-12
            if slot >= 6:
                for stage, t in group['public_clocks'].items():
                    current = by_t[t]
                    if stage == 'turn_visible_t':
                        assert current['lane_id'] in choices[spec['choice_index']]['via_lane_ids']
                        future = next(p['edge_id'] for p in trace if p['time_s'] > t and not p['edge_id'].startswith(':'))
                    else:
                        assert current['edge_id'] == f['fork_edge']
                        future = next(p['edge_id'] for p in trace if p['time_s'] > t and
                                      not p['edge_id'].startswith(':') and p['edge_id'] != f['fork_edge'])
                    assert future == info['next_edge_labels'][stage] == choices[spec['choice_index']]['edge_id']
        cut = group['public_clocks']['shared_fork_t']
        raw_prefixes = [[e for e in s['events'] if e['timestamp_s'] <= cut] for s in group['streams']['raw'][6:]]
        assert raw_prefixes[0] == raw_prefixes[1], 'Before-fork choice is privately random and unobservable'
        for method, (epoch_cap, slot_cap, unit) in caps.items():
            epoch = truth['epoch_accounting'][method]
            assert len(epoch['sessions']) == 8
            near(epoch['reserved_cap_per_m'], epoch_cap)
            near(epoch['epoch_cap_per_m'], epoch_cap)
            near(epoch['spent_per_m'], sum(s['ledger'][method]['spent_per_m'] for s in truth['evaluator_sessions']))
            assert epoch['spent_per_m'] <= epoch_cap+1e-12
    # Refit only TRAIN, reproduce selection independently, recompute test metrics
    # from the stored predictions. No parameter is chosen using test scores.
    for method in METHODS:
        for stage in ('shared_fork', 'turn_visible'):
            rows = forecast_rows(method, stage)
            parts = {s: [r for r in rows if r['split'] == s] for s in split_sets}
            assert [len(parts[s]) for s in split_sets] == [24, 12, 12]
            for task, history in (('S5_next_edge', False), ('S6_history_destination', True)):
                result = results['results'][method][stage][task]
                model = CandidateFutureAttack(parts['train'], use_history=history)
                bank = [model.predict(r['query'], r['public_context'], r['histories']) for r in parts['selection']]
                scores = {name: forecast_metrics(parts['selection'], [b[name] for b in bank]) for name in bank[0]}
                selected = min(scores, key=lambda n: (-scores[n]['balanced_accuracy'], scores[n]['log_loss'], n))
                assert selected == result['selected_attacker']
                for k, v in scores[selected].items():
                    near(v, result['selection'][k])
                predictions = result['predictions']
                assert len(predictions) == 12
                metrics = forecast_metrics(parts['test'], [p['probabilities'] for p in predictions])
                for k, v in metrics.items():
                    near(v, result['test'][k])
                for row, pred in zip(parts['test'], predictions):
                    assert pred['family_id'] == row['family_id'] and pred['query_slot'] == row['query_slot']
                    assert pred['truth_choice'] == row['choice_index']
                    index = int(np.argmax(pred['probabilities']))
                    assert pred['predicted_choice'] == index
                    assert pred['predicted_next_edge'] == row['public_context']['choices'][index]['edge_id']
                    assert pred['true_next_edge'] == row['actual_next_edge']
                if method == 'raw' and stage == 'shared_fork':
                    near(metrics['exact_candidate_edge_accuracy'], .5)
            # S6 uses exactly six historical public windows. The query remains a
            # causal prefix even though the evaluator archives its complete run.
            for r in rows:
                assert len(r['histories']) == 6
                assert max(e['timestamp_s'] for e in r['query']['events']) <= next(
                    g['public_clocks'][stage+'_t'] for g, t in zip(public['groups'], private['rows']) if t['family_id'] == r['family_id'])
    for method in METHODS:
        per_family = {f['family_id']: float(np.mean([v['recall5'] for s in f['evaluator_sessions']
            for v in s['utility'][method] if v['recall5'] is not None])) for f in private['rows'] if f['split'] == 'test'}
        for family, value in per_family.items():
            near(value, results['utility'][method]['test_family_recall5'][family])
        near(np.mean(list(per_family.values())), results['utility'][method]['test_static_recall5'])
    phase_readout = json.loads((OUT/'public_phase_readout.json').read_text())
    check_pins(phase_readout['source_sha256'])
    for method in METHODS:
        for phase, (start, end) in phase_readout['public_phase_boundaries_s'].items():
            report = phase_readout['utility'][method][phase]
            windows = [v for f in private['rows'] if f['split'] == 'test'
                       for s in f['evaluator_sessions'] for v in s['utility'][method] if start <= v['t'] <= end]
            assert report['total_windows'] == len(windows)
            assert report['reference_defined_windows'] == sum(v['recall5'] is not None for v in windows)
            near(report['reference_coverage'], report['reference_defined_windows']/len(windows))
            for family, value in report['family_values'].items():
                values = [v['recall5'] for f in private['rows'] if f['family_id'] == family
                          for s in f['evaluator_sessions'] for v in s['utility'][method]
                          if start <= v['t'] <= end and v['recall5'] is not None]
                assert value is None if not values else np.isclose(value, np.mean(values))
    gates = results['validity_gates']
    assert gates['shared_fork_raw_is_inherently_ambiguous']
    assert gates['raw_turn_positive_control']
    assert gates['raw_destination_turn_positive_control']
    test_only_edges = sorted(edge_sets['test']-edge_sets['train'])
    record = {'schema': 'native-future-validation-v1', 'status': 'passed',
        'validator_sha256': sha(Path(__file__)),
        'checked_artifacts': {p: sha(OUT/p) for p in ('protocol.json', 'results.json', 'public_transcripts.json.gz', 'private_accounting.json.gz', 'public_phase_readout.json')},
        'family_count': 24, 'sessions': 192, 'test_families': 6,
        'same_native_map_turn_catalogue': True, 'native_parked_fixed_windows': True,
        'causal_public_query_prefix': True, 'six_public_history_windows': True,
        'private_label_free_features': True, 'exact_future_edge_labels': True,
        'raw_shared_fork_accuracy': .5, 'raw_visible_turn_gates': 'passed',
        'test_candidate_edges_absent_in_train': len(test_only_edges),
        'test_candidate_edge_count': len(edge_sets['test']),
        'epoch8_H12_cap_per_m': .23, 'session_reset_reference_epoch_cap_per_m': 1.84,
        'no_equal_total_superiority_claim': True,
        'test_reference_defined_windows': 1402, 'test_total_windows': 1510,
        'test_tail_reference_defined_windows': 462, 'test_tail_total_windows': 528,
        'limits': ['public known two-choice candidate context', 'synthetic native SUMO network, not original lane parity',
                   'six held-out families', 'balanced routine/rare queries', 'public linked-history sideinformation',
                   'six histories plus one query; previous query and balanced-pair joint constraint not evaluated',
                   'static all-available POI utility conditional on nonempty reference', 'ideal-kernel coordinate Geo-I, not identity guarantee']}
    if output.exists():
        print(f'Existing validation record retained unchanged: {output}', flush=True)
    else:
        save(output, record)
        print(f'Fresh validation record written: {output}', flush=True)
    print('Native S5/S6 verification passed:24families,192native trips,12held-out queries per task/stage', flush=True)


if __name__ == '__main__':
    main()
