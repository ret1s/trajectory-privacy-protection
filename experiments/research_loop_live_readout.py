"""Compact, source-backed service comparisons; family-cluster uncertainty."""
import json
from pathlib import Path
import numpy as np
from experiments.research_loop_live_service import BASE, OUT as SOURCE
from experiments.research_loop_resources import ROOT, sha

OUT = BASE/'iteration28_readout.json'


def calculate():
    result = json.loads(SOURCE.read_text())
    nominal = result['protocol']['nominal_availability_probability']
    sessions = []
    for manifest in result['shards']:
        path = ROOT/manifest['file']; assert sha(path) == manifest['sha256']
        shard = json.loads(path.read_text())
        if shard['probability'] == nominal:
            sessions.extend(shard['sessions'])
    summaries = [s for s in result['summaries'] if s['probability'] == nominal]
    variants = sorted({(s['method'], s['mode']) for s in summaries})
    table = []
    for method, mode in variants:
        cases = [s for s in summaries if (s['method'], s['mode']) == (method, mode)]
        ex = [s for s in sessions if (s['method'], s['mode']) == (method, mode)]
        events = sum(s['events'] for s in ex)
        table.append({'method': method, 'mode': mode,
            'case_mean_recall': float(np.mean([s['recall'] for s in cases])),
            'session_mean_recall': float(np.mean([s['recall'] for s in ex])),
            'case_gates_passed': sum(s['pass_90pct'] for s in cases),
            'S1_C_recall': next(s['recall'] for s in cases if s['case_id'] == 'S1.C'),
            'coordinate_queries_per_event': sum(s['communication']['coordinate_queries'] for s in ex)/events,
            'response_body_bytes_per_event': sum(s['communication']['response_body_bytes'] for s in ex)/events,
            'request_json_bytes_per_event': sum(s['communication']['request_json_bytes'] for s in ex)/events})
    cases = sorted({s['case_id'] for s in summaries})
    families = sorted({f for s in summaries for f in s['family_recall']})
    def family_matrix(method, mode):
        cells = {s['case_id']: s['family_recall'] for s in summaries if (s['method'], s['mode']) == (method, mode)}
        return np.array([[cells[c].get(f, np.nan) for f in families] for c in cases])
    target = family_matrix('response_paced_slack03', 'epoch_cache')
    comparisons = []
    # Draw clusters once and use identical draws for paired contrasts.
    rng = np.random.default_rng(24092899)
    sampled = rng.integers(len(families), size=(10000, len(families)))
    for method, mode in [('response_paced_slack03', 'fresh'), ('response_paced', 'epoch_cache'),
                         ('fixed_K5', 'epoch_cache'), ('fixed_K12', 'epoch_cache')]:
        control = family_matrix(method, mode)
        assert np.array_equal(np.isnan(target), np.isnan(control))
        delta = target-control
        boot = np.nanmean(delta[:, sampled], axis=2).mean(axis=0)
        assert np.all(np.isfinite(boot))
        comparisons.append({'target': 'response_paced_slack03/epoch_cache', 'control': f'{method}/{mode}',
            'case_mean_delta': float(np.nanmean(delta, axis=1).mean()),
            'family_bootstrap_ci95': np.quantile(boot, [.025, .975]).tolist(),
            'families': len(families), 'draws': len(sampled), 'multiplicity_adjusted': False,
            'scope': 'paired exploratory development contrast; availability worlds averaged within family, not independent samples'})
    return {'schema': 'live-service-readout-v1', 'source_sha256': sha(SOURCE),
            'source_code_sha256': sha(Path(__file__)), 'nominal_probability': nominal,
            'table': table, 'comparisons': comparisons,
            'working_configuration': 'response_paced_slack03/epoch_cache',
            'selection_basis': 'highest nominal mean case Recall among the evaluated K5 paced variants; same .23 cap. This is a service-priority development choice, not a Pareto or six-paper winner.',
            'privacy': 'coordinate transcript unchanged by client caching; use parent attacks, including known endpoint counterexamples. No privacy gain is attributed to cache.'}


if __name__ == '__main__':
    result = calculate()
    if OUT.exists():
        assert json.loads(OUT.read_text()) == result
    else:
        OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
