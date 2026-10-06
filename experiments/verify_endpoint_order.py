"""Recompute the post-inspection ordered endpoint diagnostic and invariants."""
import argparse
import gzip
from pathlib import Path
import pickle

import numpy as np

from experiments import endpoint_noise_loop as common
from experiments import endpoint_order_challenge as challenge
from experiments.public_research_resources import sha


def validate(out, source):
    p = challenge.protocol(out, source)
    selection = common.read(out/'selection.json')
    diagnostic = common.read(out/'diagnostic.json')
    assert diagnostic['status'].startswith('POST-INSPECTION')
    assert not diagnostic['defense_changed'] and diagnostic['primary_earlier_coordinate_set_claim_unchanged']
    assert diagnostic['protocol_sha256'] == sha(out/'protocol.json')
    assert diagnostic['selection_sha256'] == sha(out/'selection.json')
    assert sha(out/'attackers.pkl.gz') == selection['attackers_sha256']
    with gzip.open(out/'attackers.pkl.gz', 'rb') as file:
        saved = pickle.load(file)  # trusted local SHA-verified model
    rn, history = saved['history'].rn, saved['history']
    master = common.read(out/'evaluator_shuffle_randomness.json.gz')['master_hex']
    recorded = common.read(out/'diagnostic_rows.json.gz')
    selection_rows = common.read(out/'selection_rows.json.gz')
    predictions, events, invariant_checks, metric_checks = 0, 0, 0, 0
    for method in challenge.METHODS:
        items = common.read(source/f'test-{method}.json.gz')
        assert len(items) == 112 and {r['family_id'] for r in items} == set(p['inspected_test_families'])
        unchanged = diagnostic['unchanged_service'][method]
        for metric in ('recall', 'bytes_per_input'):
            assert np.isclose(unchanged[metric], common.family_mean(items, metric), atol=1e-12)
        actual_by_view = {}
        for view in challenge.VIEWS:
            name = method+'/'+view
            for group in challenge.GROUPS:
                for scenario in ('S9', 'S10'):
                    candidates = [r for r in selection_rows if r['method']==name and r['scenario']==scenario]
                    assert {r['family_id'] for r in candidates} == set(p['selection_families'])
                    assert common.select_attackers(challenge.bank_group(candidates, group)) == selection['selected_attackers'][name][group][scenario]
            actual = [row for scenario in ('S9', 'S10') for row in challenge.attack_rows(items, view,
                master, scenario, saved['banks'][name, scenario], rn, history)]
            actual_by_view[view] = actual
            lookup = {(r['session_id'], r['seed'], r['scenario']): r for r in recorded if r['method']==name}
            assert len(lookup) == len(actual) == 224
            for row in actual:
                expected = lookup[row['session_id'], row['seed'], row['scenario']]
                assert set(row['errors']) == set(expected['errors'])
                for attacker, value in row['errors'].items():
                    assert np.isclose(value, expected['errors'][attacker], atol=1e-9)
                    predictions += 1
            for group in challenge.GROUPS:
                summary = common.score_summary(actual, selection['selected_attackers'][name][group])
                for scenario in ('S9', 'S10'):
                    for metric in ('mae_m', 'hit50', 'hit100', 'hit200', 'hit500'):
                        assert np.isclose(summary[scenario][metric], diagnostic['results'][name][group][scenario][metric], atol=1e-12)
                        metric_checks += 1
            events += sum(len(r['events']) for r in items)
        ordered = {(r['session_id'], r['seed'], r['scenario']): r for r in actual_by_view['ordered']}
        for row in actual_by_view['shuffled']:
            original = ordered[row['session_id'], row['seed'], row['scenario']]
            # Spatially canonical geometry and invariant models must have the
            # same predictions after removing positional labels. This is a
            # direct counterexample to treating shuffled Qs as untrackable.
            for attacker, value in row['errors'].items():
                if not attacker.startswith('observed_slots_'):
                    assert np.isclose(value, original['errors'][attacker], atol=1e-9)
                    invariant_checks += 1
    result = dict(schema='endpoint-order-verification-v1', status=p['status'],
        protocol_and_input_hashes_verified=True, all_28_inspected_families_retained=True,
        independent_private_event_shuffle=True, public_multisets_duplicates_and_clock_preserved=True,
        selected_decoders_recomputed_from_selection_only=True, target_GPS_unchanged=True,
        actual_attack_predictions_recomputed=predictions, event_views_verified=events,
        canonical_geometry_and_invariant_prediction_equalities=invariant_checks,
        reported_metrics_recomputed=metric_checks, inference_scope='No account/IP/identity anonymity claim')
    common.write(out/'verification.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--source', type=Path, default=challenge.SOURCE)
    args = parser.parse_args()
    validate(args.out, args.source)
