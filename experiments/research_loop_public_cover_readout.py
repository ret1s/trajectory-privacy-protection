"""Paired family-level utility differences against GPS-independent K5 control."""
import json
from pathlib import Path
import numpy as np
from experiments.research_loop_resources import ROOT, sha

BASE = ROOT/'artifacts/benchmarks/research_loop'
PUBLIC = BASE/'iteration22_public_cover.json'
ADAPTIVE = BASE/'iteration18_expanded_screening.json'
OUT = BASE/'iteration22_public_cover_comparisons.json'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    fixed = json.loads(PUBLIC.read_text()); adaptive = json.loads(ADAPTIVE.read_text())
    rows = []
    for s in adaptive['summaries']:
        if s['method'] not in ('response_paced', 'response_paced_slack03'):
            continue
        subset = [r for r in fixed['case_rows'] if r['k'] == 5 and r['case_id'] == s['case_id']]
        control = {f: float(np.mean([r['recall']['10'] for r in subset if r['family_id'] == f]))
                   for f in sorted({r['family_id'] for r in subset})}
        assert set(control) == set(s['family_L10'])
        delta = np.array([s['family_L10'][f]-control[f] for f in control])
        draws = np.random.default_rng(20260924).integers(0, len(delta), (10000, len(delta)))
        rows.append({'method': s['method'], 'case_id': s['case_id'], 'families': len(delta),
            'adaptive_minus_public_K5_recall': float(delta.mean()),
            'family_deltas': dict(zip(control, delta.tolist())),
            'family_bootstrap_percentile_95': np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()})
    OUT.write_text(json.dumps({'scope': 'exposed development, paired family utility; no multiplicity adjustment or confirmation',
        'draws': 10000, 'seed': 20260924, 'rows': rows,
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (PUBLIC, ADAPTIVE, Path(__file__))}},
        indent=2, allow_nan=False)+'\n')
    for r in rows:
        if r['method'] == 'response_paced_slack03':
            print(r['case_id'], round(r['adaptive_minus_public_K5_recall']*100, 2),
                  [round(v*100, 2) for v in r['family_bootstrap_percentile_95']])


if __name__ == '__main__':
    main()
