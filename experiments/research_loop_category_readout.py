"""Preserve all frozen variants, sensitivities, costs and paired differences."""
import json
from pathlib import Path
import numpy as np
from experiments.research_loop_resources import ROOT, sha

BASE = ROOT/'artifacts/benchmarks/research_loop'
OUT = BASE/'iteration30_readout.json'


def aggregate_cost(rows, mode):
    costs = [r['costs'][mode] for r in rows]
    events = sum(r['events'] for r in costs)
    return {k+'_per_event': sum(r[k] for r in costs)/events for k in costs[0] if k != 'events'}


def calculate():
    new = json.loads((BASE/'iteration29_category_cover.json').read_text())
    confirm = json.loads((BASE/'iteration30_category_confirmation.json').read_text())
    old = json.loads((BASE/'iteration28_live_service.json').read_text())
    table = []
    for split, data in [('expanded_development', new), ('new_family_check', confirm)]:
        for method in sorted({s['method'] for s in data['summaries']}):
            for p in (.5, .8, .95):
                subset = [s for s in data['summaries'] if s['method'] == method and s['probability'] == p]
                table.append({'split': split, 'method': method, 'probability': p,
                              'case_mean_recall': float(np.mean([s['recall'] for s in subset])),
                              'case_gates_passed': sum(s['pass_90pct'] for s in subset),
                              'case_count': len(subset), 'minimum_case_recall': min(s['recall'] for s in subset),
                              'S1_C_recall': next(s['recall'] for s in subset if s['case_id'] == 'S1.C')})
    costs = []
    for method in sorted({s['method'] for s in confirm['sessions']}):
        rows = [s for s in confirm['sessions'] if s['method'] == method and s['probability'] == .8]
        for mode in ('every_event', 'epoch_refresh'):
            costs.append({'method': method, 'mode': mode, **aggregate_cost(rows, mode)})
    cases = sorted({s['case_id'] for s in new['summaries']})
    families = sorted({f for s in new['summaries'] for f in s['family_recall']})
    def matrix(data, method, mode=None):
        rows = {s['case_id']: s['family_recall'] for s in data['summaries']
                if s['probability'] == .8 and s['method'] == method and (mode is None or s['mode'] == mode)}
        return np.array([[rows[c].get(f, np.nan) for f in families] for c in cases])
    target = matrix(new, 'public_category_budget30')
    sampled = np.random.default_rng(24093099).integers(len(families), size=(10000, len(families)))
    comparisons = []
    for method in ('response_paced_slack03', 'fixed_K5', 'fixed_K12'):
        control = matrix(old, method, 'epoch_cache')
        assert np.array_equal(np.isnan(target), np.isnan(control))
        delta = target-control
        bootstrap = np.nanmean(delta[:, sampled], axis=2).mean(axis=0)
        s1 = cases.index('S1.C')
        comparisons.append({'target': 'public_category_budget30', 'control': method+'/epoch_cache',
                            'case_mean_delta': float(np.nanmean(delta, axis=1).mean()),
                            'family_bootstrap_ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
                            'S1_C_delta': float(np.nanmean(delta[s1])),
                            'S1_C_family_bootstrap_ci95': np.quantile(np.nanmean(delta[s1, sampled], axis=1), [.025, .975]).tolist(),
                            'draws': 10000, 'family_count': len(families), 'multiplicity_adjusted': False,
                            'scope': 'Exploratory development; equal category request count vs K5, not equal distinct coordinates or bytes'})
    return {'schema': 'category-cover-readout-v1', 'table': table, 'new_family_costs': costs,
            'expanded_comparisons': comparisons,
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
                BASE/'iteration29_category_cover.json', BASE/'iteration30_category_confirmation.json',
                BASE/'iteration28_live_service.json', Path(__file__)]},
            'decision': 'Keep both operating points. Budget30 passes nominal 15/15 on expanded and new families, but only 11/15 at p=.95 on new families. Wider67 achieves 100% on evaluated cases at higher bytes. Neither replaces the K5-coordinate method without specifying interface/cost requirements.',
            'privacy': 'Public plan is independent of GPS conditional on region/catalogue/clock/server state. Zero additional coordinate information is not zero attack success or protection of timing/identity.',
            'confirmation': 'Four new families and three new availability seeds after plan freeze; same city and generator. Old adaptive defender not run on the new families; do not compare its expanded scores with new-family scores.'}


if __name__ == '__main__':
    result = calculate()
    if OUT.exists(): assert json.loads(OUT.read_text()) == result
    else: OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['expanded_comparisons'], indent=2))
