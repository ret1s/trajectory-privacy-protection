"""Fixed public-phase utility and family accounting readout; no model tuning."""
import gzip
import json
from pathlib import Path
import numpy as np
from experiments.future_sumo_eval import OUT, METHODS
from experiments.build_future_sumo_cohort import sha, save, ROOT


def main():
    private = json.loads(gzip.decompress((OUT/'private_accounting.json.gz').read_bytes()))
    rows = [r for r in private['rows'] if r['split'] == 'test']
    # Equal public thirds, fixed without using branch, budget exhaustion or score.
    phases = {'all_0_600': (0, 600), 'early_0_180': (0, 180),
              'middle_200_380': (200, 380), 'tail_400_600': (400, 600)}
    utility = {}
    for method in METHODS:
        utility[method] = {}
        for phase, (start, end) in phases.items():
            per_family = {}
            per_session = []
            valid_windows = total_windows = missing_sessions = 0
            for r in rows:
                family_values = []
                for s in r['evaluator_sessions']:
                    windows = [v for v in s['utility'][method] if start <= v['t'] <= end]
                    values = [v['recall5'] for v in windows if v['recall5'] is not None]
                    total_windows += len(windows)
                    valid_windows += len(values)
                    family_values.extend(values)
                    if values:
                        per_session.append(float(np.mean(values)))
                    else:
                        missing_sessions += 1
                per_family[r['family_id']] = float(np.mean(family_values)) if family_values else None
            values = [v for v in per_family.values() if v is not None]
            utility[method][phase] = {'family_macro_recall5': float(np.mean(values)),
                'family_min_recall5': min(values), 'family_max_recall5': max(values),
                'session_min_recall5': min(per_session), 'family_values': per_family,
                'session_count': len(per_session), 'family_count': len(values),
                'undefined_session_count': missing_sessions,
                'reference_defined_windows': valid_windows, 'total_windows': total_windows,
                'reference_coverage': valid_windows/total_windows}
    accounting = {}
    for method in METHODS[1:]:
        spends = {r['family_id']: r['epoch_accounting'][method]['spent_per_m'] for r in rows}
        reads = {r['family_id']: sum(s['ledger'][method]['private_reads'] for s in r['evaluator_sessions']) for r in rows}
        accounting[method] = {'test_family_spend_per_m': spends, 'mean_spend_per_m': float(np.mean(list(spends.values()))),
            'min_spend_per_m': min(spends.values()), 'max_spend_per_m': max(spends.values()),
            'test_family_GPS_reads': reads, 'min_GPS_reads': min(reads.values()), 'max_GPS_reads': max(reads.values())}
    save(OUT/'public_phase_readout.json', {'schema': 'native-future-public-phase-readout-v1',
        'source_sha256': {'private_accounting.json.gz': sha(OUT/'private_accounting.json.gz'),
                          str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))},
        'public_phase_boundaries_s': phases, 'utility': utility, 'accounting': accounting,
        'scope': 'descriptive, fixed public-clock phases; static six-category all-available POI coverage; '
                 'Recall conditional on at least one reachable reference POI; undefined windows explicitly counted; no model or attacker selected here',
        'attacker_limit': 'six histories plus one query prefix; previous query and joint balanced-pair constraint not evaluated; not full linked-eight-trip adversary'})
    print('Native future fixed public-phase readout written', flush=True)


if __name__ == '__main__':
    main()
