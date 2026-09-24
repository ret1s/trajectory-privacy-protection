"""Exploratory paired family summaries; not a new tuning/selection procedure."""
import json
from pathlib import Path
import numpy as np
from experiments.research_loop_resources import ROOT, sha

BASE = ROOT/'artifacts/benchmarks/research_loop'
SOURCE = BASE/'iteration18_expanded_attacks.json'
OUT = BASE/'iteration18_paired_comparisons.json'
PAIRS = (('response_paced', 'filter_paced', 'reply_objective'),
         ('response_paced', 'response_progress', 'pacing'),
         ('response_paced_slack03', 'response_paced', 'planning_slack'))


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    source = json.loads(SOURCE.read_text())
    lookup = {(r['method'], r['case_id']): r for r in source['summaries']}
    rows = []
    for candidate, control, component in PAIRS:
        for case in sorted({r['case_id'] for r in source['summaries']}):
            a, b = lookup[candidate, case]['family_metrics'], lookup[control, case]['family_metrics']
            assert set(a) == set(b)
            families = sorted(a)
            # Same draws across metrics keep their paired dependence visible.
            rng = np.random.default_rng(20260924)
            draws = rng.integers(0, len(families), (10000, len(families)))
            metrics = {}
            for key in a[families[0]]:
                difference = np.array([a[f][key]-b[f][key] for f in families])
                interval = np.quantile(difference[draws].mean(axis=1), [.025, .975])
                metrics[key] = {'mean_delta': float(difference.mean()),
                    'family_bootstrap_percentile_95': interval.tolist(),
                    'family_deltas': dict(zip(families, difference.tolist()))}
            rows.append({'candidate': candidate, 'control': control, 'component': component,
                         'case_id': case, 'families': len(families), 'metrics': metrics})
    OUT.write_text(json.dumps({'scope': 'exploratory paired development; 10,000 family-cluster bootstrap draws; no multiplicity correction; not independent confirmation or population guarantee',
        'delta_definition': 'candidate minus control; higher Recall and MAE favor defender; lower attacker Hit favors defender',
        'source_sha256': sha(SOURCE), 'code_sha256': sha(Path(__file__)),
        'draws': 10000, 'seed': 20260924, 'rows': rows}, indent=2, allow_nan=False)+'\n')
    for row in rows:
        if row['case_id'] in ('S1.C', 'S3.A', 'S9.C', 'S10.C'):
            print(row['component'], row['case_id'],
                  {k: (round(v['mean_delta'], 4), [round(x, 4) for x in v['family_bootstrap_percentile_95']])
                   for k, v in row['metrics'].items() if k in ('mae_m', 'hit100', 'recall_L10')})


if __name__ == '__main__':
    main()
