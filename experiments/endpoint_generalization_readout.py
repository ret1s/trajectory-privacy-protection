"""Paired 28-family readout for the locked endpoint generalization study."""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import beta

from experiments import endpoint_noise_loop as common
from experiments.endpoint_noise_readout import paired_interval
from experiments.endpoint_generalization import LOCKED
from experiments.public_research_resources import sha


def proportions_interval(successes, total):
    """Exact binomial caution for the independent-family 'any hit' indicator."""
    if not 0 <= successes <= total or total < 1:
        raise ValueError('Positive number of families and valid successes required')
    return dict(families=total, families_with_any_hit=successes,
        observed=successes/total,
        exact95_low=0. if successes==0 else float(beta.ppf(.025, successes, total-successes+1)),
        exact95_high=1. if successes==total else float(beta.ppf(.975, successes+1, total-successes)),
        one_sided95_zero_success_upper=1-.05**(1/total) if successes==0 else None,
        scope='Family-level Bernoulli model for this same-city generator only; '
              'not a universal attacker or real-user guarantee')


def readout(out):
    p, selection, heldout = [common.read(out/name) for name in ('protocol.json', 'selection.json', 'heldout.json')]
    assert heldout['selection_sha256'] == sha(out/'selection.json')
    assert heldout['selected_defense'] == LOCKED and heldout['defense_not_reselected']
    rows = common.read(out/'heldout_attack_rows.json.gz')
    linked = common.read(out/'heldout_linked_rows.json.gz')
    comparisons, linked_comparisons, family_cautions = {}, {}, {}
    for scenario in ('S9', 'S10'):
        for comparator in ('raw', 'scale100_L10', 'scale100_L20'):
            for key in ('mae_m', 'hit100', 'hit500'):
                comparisons[f'{scenario}/{LOCKED}-{comparator}/{key}'] = paired_interval(
                    rows, selection, LOCKED, comparator, scenario, key)
                linked_comparisons[f'{scenario}/{LOCKED}-{comparator}/{key}'] = paired_interval(
                    linked, dict(selected_attackers=selection['linked_attackers']),
                    LOCKED, comparator, scenario, key)
        for access, data, choices in [('single', rows, selection['selected_attackers']),
                                      ('linked_two_trip', linked, selection['linked_attackers'])]:
            for method in ('scale100_L20', LOCKED):
                attacker = choices[method][scenario]['hit100']
                selected = [r for r in data if r['scenario']==scenario and r['method']==method]
                families = sorted({r['family_id'] for r in selected})
                success = sum(any(r['errors'][attacker] <= 100. for r in selected
                                  if r['family_id']==family) for family in families)
                family_cautions[f'{access}/{scenario}/{method}/hit100'] = proportions_interval(success, len(families))
    utility_intervals = {}
    candidate_runs = common.read(out/f'test-{LOCKED}.json.gz')
    by_candidate = {(r['family_id'], r['session_id'], r['seed']): r for r in candidate_runs}
    rng = np.random.default_rng(2026100593)
    for comparator in ('scale100_L10', 'scale100_L20'):
        controls = common.read(out/f'test-{comparator}.json.gz')
        by_control = {(r['family_id'], r['session_id'], r['seed']): r for r in controls}
        assert set(by_control) == set(by_candidate)
        families = sorted({k[0] for k in by_candidate})
        for key in ('recall', 'bytes_per_input'):
            differences = np.asarray([np.mean([by_candidate[k][key]-by_control[k][key]
                for k in by_candidate if k[0]==family]) for family in families])
            boot = differences[rng.integers(len(families), size=(3000, len(families)))].mean(axis=1)
            utility_intervals[f'{LOCKED}-{comparator}/{key}'] = dict(delta=float(differences.mean()),
                bootstrap95_low=float(np.quantile(boot, .025)), bootstrap95_high=float(np.quantile(boot, .975)),
                families=len(families))
    common.write(out/'readout.json', dict(schema='endpoint-generalization-readout-v3',
        heldout_sha256=sha(out/'heldout.json'), protocol_sha256=sha(out/'protocol.json'),
        selection_sha256=sha(out/'selection.json'), selected_defense=LOCKED,
        results=heldout['rows'], paired_family_uncertainty=comparisons,
        linked_secondary_uncertainty=linked_comparisons,
        family_hit100_cautions=family_cautions, utility_and_byte_uncertainty=utility_intervals,
        primary='Single-trip S10 versus matched scale100_L20, defense fixed before this study',
        scope=p['scope']))
    print('Readout stored; 28-family paired primary and linked secondary intervals')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    readout(parser.parse_args().out)
