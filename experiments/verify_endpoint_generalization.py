"""Independent endpoint generalization verification, including linked estimates."""
import argparse
import gzip
import hashlib
from pathlib import Path
import pickle

import numpy as np

from experiments import endpoint_noise_loop as common
from experiments.endpoint_generalization import LOCKED, linked_rows, session_seed
from experiments.public_research_resources import sha
from experiments.verify_endpoint_noise import validate as validate_primary


def validate(out, cache):
    # Independent existing verifier recalculates service, bytes, private caps,
    # all single-trip bank predictions, target coordinates and clock schemas.
    validate_primary(out, cache, depth=True)
    verified = common.read(out/'verification.json')
    p, selection, heldout = [common.read(out/name) for name in ('protocol.json', 'selection.json', 'heldout.json')]
    assert p['test_families'] == [f'family-{i}' for i in range(1205, 1233)]
    assert heldout['complete_predeclared_test'] and heldout['defense_not_reselected']
    assert p['fixed_defense'] == LOCKED and heldout['selected_defense'] == LOCKED
    assert not set(p['test_families']) & set(p['previously_examined_test_families'])
    assert sha(out/'evaluator_randomness.json.gz') == p['private_randomness_sha256']
    master = common.read(out/'evaluator_randomness.json.gz')['master_hex']
    model_path = out/'attackers.pkl'
    content = model_path.read_bytes() if model_path.exists() else gzip.decompress((out/'attackers.pkl.gz').read_bytes())
    assert hashlib.sha256(content).hexdigest() == selection['attackers_sha256']
    saved = pickle.loads(content)  # trusted, verified local model
    rn, history = saved['history'].rn, saved['history']
    recorded_rows = common.read(out/'heldout_linked_rows.json.gz')
    count, unique_streams, method_seeds = 0, set(), {}
    for method, summary in heldout['rows'].items():
        executions = common.read(out/f'test-{method}.json.gz')
        assert len(executions) == 28*2*2
        method_seeds[method] = {}
        for row in executions:
            resolved = session_seed(master, row['session_id'], row['rep'])
            assert row['rng_seed_evaluator_only'] == resolved
            assert row['seed'] == row['rep']
            unique_streams.add(resolved)
            method_seeds[method][row['session_id'], row['rep']] = resolved
            for event in row['events']:
                assert 'rng_seed_evaluator_only' not in event and 'master_hex' not in event
        for scenario in ('S9', 'S10'):
            actual = linked_rows(executions, scenario, saved['banks'][method, scenario],
                                 saved['linked'][method, scenario], rn, history)
            recorded = [r for r in recorded_rows if r['method']==method and r['scenario']==scenario]
            assert len(actual) == len(recorded) == len(executions)
            lookup = {(r['family_id'], r['session_id'], r['seed']): r for r in recorded}
            for row in actual:
                expected = lookup[row['family_id'], row['session_id'], row['seed']]
                assert np.isclose(row['endpoint_pair_spread_m'], expected['endpoint_pair_spread_m'], atol=1e-12)
                assert set(row['errors']) == set(expected['errors'])
                for name, value in row['errors'].items():
                    assert np.isclose(value, expected['errors'][name], atol=1e-9)
                    count += 1
            selected = selection['linked_attackers'][method][scenario]
            for key in ('mae_m', 'hit50', 'hit100', 'hit200', 'hit500'):
                name = selected['mae' if key=='mae_m' else key]
                values = [dict(r, v=r['errors'][name] if key=='mae_m'
                               else float(r['errors'][name] <= int(key[3:]))) for r in actual]
                assert np.isclose(summary['linked_secondary'][scenario][key],
                                  common.family_mean(values, 'v'), atol=1e-12)
    assert len(unique_streams) == 112
    assert all(value == next(iter(method_seeds.values())) for value in method_seeds.values())
    assert p['linked_composition_cap_per_m'] == .115
    verified.update(schema='endpoint-generalization-verification-v3',
        all_28_predeclared_families_retained=True, defense_not_reselected=True,
        independent_session_rep_substreams=112, matched_method_pairing_only=True,
        linked_attack_predictions_recomputed=count, linked_targets_remain_individual_GPS=True,
        linked_access_requires_account_linkability=True, linked_two_session_cap_per_m=.115)
    common.write(out/'verification.json', verified)
    print(verified)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--cache', type=Path, default=Path('/private/tmp/trajectory-research-20261005-public-map'))
    args = parser.parse_args()
    validate(args.out, args.cache)
