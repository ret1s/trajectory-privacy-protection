"""Paired family uncertainty and finite-sample caution for endpoint development."""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments import endpoint_noise_loop as common
from experiments.public_research_resources import sha


def paired_interval(rows, selection, candidate, comparator, scenario, key):
    maps = {}
    for method in (candidate, comparator):
        attacker = selection['selected_attackers'][method][scenario]['mae' if key=='mae_m' else key]
        maps[method] = {(r['family_id'], r['session_id'], r['seed']):
            (r['errors'][attacker] if key=='mae_m' else float(r['errors'][attacker] <= int(key[3:])))
            for r in rows if r['method']==method and r['scenario']==scenario}
    assert set(maps[candidate]) == set(maps[comparator])
    families = sorted({k[0] for k in maps[candidate]})
    deltas = np.asarray([np.mean([maps[candidate][k]-maps[comparator][k]
        for k in maps[candidate] if k[0]==f]) for f in families])
    rng = np.random.default_rng(2026100581)
    boot = deltas[rng.integers(len(families), size=(3000, len(families)))].mean(axis=1)
    return {'delta': float(deltas.mean()), 'bootstrap95_low': float(np.quantile(boot, .025)),
        'bootstrap95_high': float(np.quantile(boot, .975)), 'family_deltas': dict(zip(families, deltas.tolist())),
        'families': len(families), 'scope': 'Paired family bootstrap; very small fixed same-generator sample, '
            'no multiple-comparison correction; degenerate zero-Hit CI is not absolute privacy proof'}


def readout(out):
    selection = common.read(out/'selection.json')
    heldout = common.read(out/'heldout.json')
    rows = common.read(out/'heldout_attack_rows.json.gz')
    candidate = heldout['selected_defense']
    if candidate is None:
        raise ValueError('No defense satisfied the development utility gate')
    depth = 'L'+candidate.split('_L')[-1] if '_L' in candidate else None
    comparators = ['raw', 'scale100_L10', 'delay60_L10', 'scale100_'+depth] if depth else ['raw', 'plain', 'delay60']
    comparators = [m for m in dict.fromkeys(comparators) if m != candidate and m in heldout['rows']]
    uncertainty = {}
    zero_cautions = {}
    for scenario in ('S9', 'S10'):
        attacker = selection['selected_attackers'][candidate][scenario]['hit100']
        selected_rows = [r for r in rows if r['method']==candidate and r['scenario']==scenario]
        families = sorted({r['family_id'] for r in selected_rows})
        family_any_hit = [any(r['errors'][attacker] <= 100. for r in selected_rows
                             if r['family_id']==family) for family in families]
        zero_cautions[scenario] = {'families': len(families), 'endpoint_predictions': len(selected_rows),
            'families_with_any_hit100': sum(family_any_hit),
            'one_sided95_zero_success_upper': 1-.05**(1/len(families)) if not any(family_any_hit) else None,
            'interpretation': 'If families were independent Bernoulli draws from this generator, '
                'zero observed family successes still allow this upper probability of any Hit100 in one family. '
                'Seeds are not independent real people; model-based caution, not a guarantee for every attacker.'}
        for comparator in comparators:
            for key in ('mae_m', 'hit100', 'hit500'):
                uncertainty[f'{scenario}/{candidate}-{comparator}/{key}'] = paired_interval(
                    rows, selection, candidate, comparator, scenario, key)
    traffic = {m: heldout['rows'][candidate].get('bytes_per_input', 0.)/
                  heldout['rows'][m]['bytes_per_input'] for m in comparators
               if heldout['rows'][m].get('bytes_per_input')}
    payload = dict(schema='endpoint-noise-readout-v1', heldout_sha256=sha(out/'heldout.json'),
        selected_defense=candidate, results=heldout['rows'], paired_family_uncertainty=uncertainty,
        zero_hit100_cautions=zero_cautions, traffic_ratio=traffic,
        claim='Development tradeoff only; finite attack bank and few families cannot establish universal superiority')
    common.write(out/'readout.json', payload)
    print('Readout', candidate, json.dumps(zero_cautions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    readout(parser.parse_args().out)
