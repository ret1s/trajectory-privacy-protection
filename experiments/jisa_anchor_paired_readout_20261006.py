"""Descriptive family-paired uncertainty for the already-inspected native pilot.

Added after reading pilot point estimates. No model/attacker/metric selection,
new emissions or fresh-confirmation claim; every requested contrast is retained.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/benchmarks/jisa_native_anchor_ablation_20261006_v1'


def paired(left, right):
    families = sorted(set(left) & set(right))
    valid = [f for f in families if left[f] is not None and right[f] is not None]
    if not valid: raise ValueError('No defined family pairs')
    delta = np.array([left[f]-right[f] for f in valid])
    indices = np.random.default_rng(20261006023).integers(0, len(delta), size=(10000, len(delta)))
    means = delta[indices].mean(axis=1)
    return {'family_ids': valid, 'family_count': len(valid),
            'excluded_undefined_family_pairs': len(families)-len(valid),
            'family_REM_minus_Planar': dict(zip(valid, delta.tolist())),
            'mean_REM_minus_Planar': float(delta.mean()),
            'percentile95_family_bootstrap': np.quantile(means, [.025, .975]).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT); args = parser.parse_args()
    source = args.output/'derived_readout_v2.json'; result = json.loads(source.read_text())
    contrasts = {}
    for policy in ('current_only', 'versioned_epoch8'):
        for scope in ('all_0_600', 'tail_400_600', 'cold_first_session', 'warm_later_sessions'):
            left = result['utility']['rem_epoch8']['test'][policy][scope]['family_values']
            right = result['utility']['planar_epoch8']['test'][policy][scope]['family_values']
            contrasts[f'utility--{policy}--{scope}'] = paired(left, right)
    for stage in ('shared_fork', 'turn_visible'):
        for task in ('S5_next_edge', 'S6_history_destination'):
            left = result['attacks'][f'rem_epoch8--{stage}--{task}']['test_family_metrics']
            right = result['attacks'][f'planar_epoch8--{stage}--{task}']['test_family_metrics']
            for metric in ('exact_candidate_edge_accuracy', 'destination_mae_m', 'destination_hit100'):
                contrasts[f'attack--{stage}--{task}--{metric}'] = paired(
                    {f: v[metric] for f, v in left.items()}, {f: v[metric] for f, v in right.items()})
    record = {'schema': 'jisa-matched-descriptive-paired-readout-v1',
        'source_derived_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'calculation_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'posthoc descriptive family bootstrap after pilot point-estimate inspection; six native test families, one secret draw per subject/session; conditional on same synthetic generator/map',
        'selection_use': False, 'confirmation': False, 'replicates': 10000,
        'seed': 20261006023, 'multiple_comparison_adjustment': 'none; not confirmatory significance claims',
        'sign': 'REM minus Planar; higher recall/MAE and lower attacker success have different meanings; no universal winner selected',
        'contrasts': contrasts}
    path = args.output/'paired_differences.json'
    if path.exists(): raise FileExistsError('Preserve completed paired diagnostic')
    path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    for name, values in contrasts.items():
        print(name, values['mean_REM_minus_Planar'], values['percentile95_family_bootstrap'])


if __name__ == '__main__':
    main()
