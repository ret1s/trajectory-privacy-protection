"""Report fixed control and expanded decoder generalization without fallback.

Every decoder remains the one selected before this runner opened test labels.
No per-test oracle minimum, fallback attacker or defense reselection is used.
"""
import argparse
from pathlib import Path

import numpy as np

from experiments import endpoint_noise_loop as common
from experiments import endpoint_order_challenge as challenge
from experiments.public_research_resources import sha


def family_errors(rows, method, scenario, attacker):
    rows = [r for r in rows if r['method']==method and r['scenario']==scenario]
    return {family: float(np.mean([r['errors'][attacker] for r in rows if r['family_id']==family]))
            for family in sorted({r['family_id'] for r in rows})}


def readout(out, source):
    p = challenge.protocol(out, source)
    selection = common.read(out/'selection.json')
    development = common.read(out/'development_summary.json')
    diagnostic = common.read(out/'diagnostic.json')
    test_rows = common.read(out/'diagnostic_rows.json.gz')
    selection_rows = common.read(out/'selection_rows.json.gz')
    original = common.read(source/'heldout.json')
    failures, matching = {}, {}
    for method in challenge.METHODS:
        for view in challenge.VIEWS:
            name = method+'/'+view
            matching[name] = {}
            for scenario in ('S9', 'S10'):
                invariant = diagnostic['results'][name]['invariant'][scenario]
                # The retained control must recover the original fixed bank
                # results: the intervention cannot improve set-only features.
                for metric in ('mae_m', 'hit50', 'hit100', 'hit200', 'hit500'):
                    assert np.isclose(invariant[metric], original['rows'][method][scenario][metric], atol=1e-12)
                matching[name][scenario] = True
                chosen = {group: selection['selected_attackers'][name][group][scenario]['mae']
                          for group in ('full', 'invariant')}
                dev = {group: family_errors(selection_rows, name, scenario, attacker) for group, attacker in chosen.items()}
                test = {group: family_errors(test_rows, name, scenario, attacker) for group, attacker in chosen.items()}
                families = sorted(test['full'])
                delta = np.asarray([test['full'][f]-test['invariant'][f] for f in families])
                rng = np.random.default_rng(20261005097)
                boot = delta[rng.integers(len(families), size=(3000, len(families)))].mean(axis=1)
                failures[name+'/'+scenario] = dict(selected_mae_decoders=chosen,
                    selection_mae_m={group: development[name][group][scenario]['mae_m'] for group in chosen},
                    inspected_test_mae_m={group: diagnostic['results'][name][group][scenario]['mae_m'] for group in chosen},
                    selection_family_mae_m=dev, inspected_test_family_mae_m=test,
                    inspected_test_full_minus_invariant_mae_m=dict(delta=float(delta.mean()),
                        bootstrap95_low=float(np.quantile(boot, .025)), bootstrap95_high=float(np.quantile(boot, .975)),
                        family_deltas=dict(zip(families, delta.tolist()))),
                    interpretation='These are fixed selected decoders, not test-based fallback. A lower selection '
                        'MAE can generalize to a higher inspected-test MAE; adding candidate attacks cannot reduce '
                        'an optimal attacker\'s information, but this finite selection rule can select a weaker decoder.')
    common.write(out/'readout.json', dict(schema='endpoint-order-readout-v1', status=p['status'],
        diagnostic_sha256=sha(out/'diagnostic.json'), selection_sha256=sha(out/'selection.json'),
        original_coordinate_set_bank_recovered=matching,
        fixed_control_and_expanded_decoder_generalization=failures,
        no_test_based_fallback_or_oracle_minimum=True,
        claim='Private shuffle removes explicit slot-position labels. It shows no Endpoint20 MAE benefit in '
            'this diagnostic, and geometry association survives exactly. Broad attacker-robust superiority is '
            'not established; fixed-bank and expanded-bank estimates must both be reported.'))
    print('Readout retains original fixed bank and all expanded selection/generalization outcomes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--source', type=Path, default=challenge.SOURCE)
    args = parser.parse_args()
    readout(args.out, args.source)
