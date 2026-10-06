"""Retain robust and old fixed endpoint selector outcomes without fallback."""
import argparse
from pathlib import Path

import numpy as np

from experiments import endpoint_noise_loop as common
from experiments import endpoint_robust_selection_20261006 as run
from experiments.endpoint_noise_readout import paired_interval
from experiments.endpoint_order_readout import family_errors
from experiments.public_research_resources import sha


def readout(out, source, order_source):
    p = run.protocol(out, source, order_source)
    selection = run.frozen_selection(out)
    diagnostic = common.read(out/'diagnostic.json')
    stats = common.read(out/'candidate_statistics.json.gz')
    test_rows = common.read(out/'diagnostic_rows.json.gz')
    old = common.read(order_source/'diagnostic.json')  # only after frozen new selection and scoring
    gaps, restored = {}, {}
    for method in p['methods']:
        for control, group in (('old_two_family_invariant', 'invariant'), ('old_two_family_expanded', 'full')):
            for scenario in p['scenarios']:
                for metric in ('mae_m', 'hit50', 'hit100', 'hit200', 'hit500'):
                    assert np.isclose(diagnostic['rows'][method][control][scenario][metric],
                        old['results'][method+'/shuffled'][group][scenario][metric], atol=1e-12)
        restored[method] = True
        for scenario in p['scenarios']:
            key = method+'/'+scenario
            gaps[key] = {}
            for selector in run.SELECTORS:
                name = selection['selected_attackers'][method][selector][scenario]['mae']
                candidate = stats[key][name]['mae']
                test = family_errors(test_rows, method, scenario, name)
                gaps[key][selector] = dict(selected_mae_decoder=name,
                    crossfit10_mean_mae_m=candidate['mean'], crossfit10_se_m=candidate['se'],
                    crossfit10_upper_objective_m=candidate['objective'], validation_family_mae_m=candidate['family_values'],
                    inspected_test_mean_mae_m=float(np.mean(list(test.values()))), inspected_test_family_mae_m=test,
                    inspected_test_minus_validation_mean_m=float(np.mean(list(test.values()))-candidate['mean']),
                    interpretation='Descriptive gap between different family populations and 7/8-family validation '
                        'fits versus the final 8-family model; not a paired causal or fresh-generalization estimate.')
    augmented, selectors = [], {'selected_attackers': {}}
    for method in p['methods']:
        for selector in run.SELECTORS:
            name = method+'/'+selector
            selectors['selected_attackers'][name] = selection['selected_attackers'][method][selector]
            augmented.extend(dict(row, method=name) for row in test_rows if row['method']==method)
    uncertainty = {}
    for scenario in p['scenarios']:
        for metric in ('mae_m', 'hit100', 'hit500'):
            uncertainty[f'{scenario}/robust/Endpoint20-plainL20/{metric}'] = paired_interval(augmented, selectors,
                'scale025_L20/robust_crossfit', 'scale100_L20/robust_crossfit', scenario, metric)
            for method in p['methods']:
                for old_selector in run.SELECTORS[1:]:
                    uncertainty[f'{scenario}/{method}/robust-{old_selector}/{metric}'] = paired_interval(
                        augmented, selectors, method+'/robust_crossfit', method+'/'+old_selector, scenario, metric)
    common.write(out/'readout.json', dict(schema='endpoint-robust-readout-v1', status=p['status'],
        diagnostic_sha256=sha(out/'diagnostic.json'), selection_sha256=sha(out/'selection.json'),
        results=diagnostic['rows'], original_controls_recovered=restored,
        selection_test_generalization=gaps, paired_family_uncertainty=uncertainty,
        uncertainty_scope='All previously inspected families, exploratory bootstrap; no multiplicity correction; '
            'cross-fit SE penalty is not a confidence interval because fits overlap',
        no_test_based_fallback=True, defense_changed=False,
        claim='One robust selector diagnostic only; outcomes including failures retained. '
            'Cannot establish universal attacker strength or Geo-I superiority.'))
    print('Readout retains cross-fit selector, both original controls and every generalization gap', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--source', type=Path, default=run.SOURCE)
    parser.add_argument('--order-source', type=Path, default=run.ORDER_SOURCE)
    args = parser.parse_args()
    readout(args.out, args.source, args.order_source)
