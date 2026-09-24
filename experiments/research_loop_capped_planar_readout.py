"""All-case development readout; retain regressions and weak-attacker caveat."""
from pathlib import Path
import json
import numpy as np
from experiments.research_loop_resources import ROOT, sha

BASE = ROOT/'artifacts/benchmarks/research_loop'
SOURCES = (BASE/'iteration25_capped_service_cases.json', BASE/'iteration26_planar_anchor_cases.json')
FIRST_SOURCES = (BASE/'iteration16_first_query_attack.json', BASE/'iteration26_first_query_attack.json',
                 BASE/'iteration15_response_cases.json')
OUT = BASE/'iteration25_26_readout.json'
METHODS = ('raw', 'response_paced', 'response_paced_slack03', 'capped90_paced',
           'capped90_paced_slack03', 'planar_paced', 'planar_paced_slack03')


def calculate():
    inputs = [json.loads(p.read_text()) for p in SOURCES]
    for key in ('dataset_sha256', 'resources', 'reference_depth', 'optimized_reply_depth', 'K', 'tight_session_cap'):
        assert inputs[0][key] == inputs[1][key]
    by_method, sessions, rows = {}, {}, {}
    for data in inputs:
        for method in {s['method'] for s in data['summaries']}:
            subset = [s for s in data['summaries'] if s['method'] == method]
            source_sessions = [e for e in data['executions'] if e['method'] == method]
            source_rows = [r for r in data['rows'] if r['method'] == method]
            if method in by_method:
                assert subset == by_method[method] and source_sessions == sessions[method] and source_rows == rows[method]
            by_method[method], sessions[method], rows[method] = subset, source_sessions, source_rows
    cases = sorted({r['case_id'] for r in inputs[0]['rows']}, key=lambda x: (int(x.split('.')[0][1:]), x))
    eligibility = []
    for case in cases:
        subset = [r for r in rows['raw'] if r['split'] == 'development_validation' and r['case_id'] == case]
        eligibility.append({'case_id': case, 'families': sorted({r['family_id'] for r in subset}),
            'unique_records': len({r['record_id'] for r in subset}), 'record_RNG_pairs': len(subset),
            'note': 'RNG repetitions and multiple targets are not independent route families'})
    summaries, comparisons = [], []
    for method in METHODS:
        ex = [e for e in sessions[method] if e['split'] == 'development_validation']
        summaries.append({'method': method, 'case_gates_passed': sum(r['pass_90pct_case_recall_L10'] for r in by_method[method]),
            'case_gates_total': 15, 'validation_families': len({e['family_id'] for e in ex}),
            'validation_source_sessions': len({e['session_id'] for e in ex}),
            'validation_whole_session_L10_mean': float(np.mean([e['whole_session_recall']['10'] for e in ex])),
            'validation_whole_session_L10_min': min(e['whole_session_recall']['10'] for e in ex),
            'all_case_metrics': by_method[method]})
        if method == 'raw' or method.startswith('response_'):
            continue
        control = 'response_paced_slack03' if method.endswith('slack03') else 'response_paced'
        for case in cases:
            a = [r for r in rows[method] if r['split'] == 'development_validation' and r['case_id'] == case]
            b = [r for r in rows[control] if r['split'] == 'development_validation' and r['case_id'] == case]
            families = sorted({r['family_id'] for r in a})
            delta = {f: float(np.mean([r['recall']['10'] for r in a if r['family_id'] == f])-
                              np.mean([r['recall']['10'] for r in b if r['family_id'] == f])) for f in families}
            comparisons.append({'method': method, 'control': control, 'case_id': case,
                                'mean_family_recall_delta': float(np.mean(list(delta.values()))), 'family_deltas': delta})
    old_probe, new_probe, old_case_source = [json.loads(p.read_text()) for p in FIRST_SOURCES]
    old_first = {(e['session_id'], e['rep']): next(iter(e['events'].values())) for e in old_case_source['executions'] if e['method'] == 'response_paced'}
    current_first = {(e['session_id'], e['rep']): next(iter(e['events'].values())) for e in inputs[1]['executions'] if e['method'] == 'response_paced'}
    assert old_first == current_first
    first_summaries = []
    for method, selected, examples in (
            ('response_paced', old_probe['selection'], [r for r in old_probe['core_rows'] if r['method'] == 'response_paced']),
            ('planar_paced_and_slack_same_first_query', new_probe['summary']['selection'], new_probe['core_rows'])):
        assert {(r['session_id_evaluator_only'], r['rep']) for r in examples} == set(current_first)
        val = [r for r in examples if r['split'] == 'development_validation']
        first_summaries.append({'method': method, 'selection': selected,
            'mae_m': float(np.mean([r['errors'][selected['mae']][0] for r in val])),
            'hits': {str(rad): float(np.mean([r['errors'][selected[f'hit{rad}']][0] <= rad for r in val])) for rad in (50,100,200,500)},
            'validation_families': len({r['family_id'] for r in val}),
            'validation_source_sessions': len({r['session_id_evaluator_only'] for r in val}),
            'validation_record_RNG_pairs': len(val)})
    return {'scope': 'exposed core development; 2 validation families overall but S9.B and S10.A each have only 1 eligible family; no independent confirmation; new matched learned attacks limited to planar FIRST-query probe; runtime omitted due to overlapping jobs',
            'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (*SOURCES, *FIRST_SOURCES, Path(__file__))},
            'new_full_session_executions': sum(d['new_executions'] for d in inputs),
            'unique_replayed_control_executions': 198, 'case_validation_eligibility': eligibility,
            'summaries': summaries, 'paired_utility_comparisons': comparisons,
            'first_query_probe_scope': 'separate full-session origin challenge, NOT masked S9.A/B/C scores; mechanism-matched fit and auxiliary selection; raw exact first coordinate has zero error',
            'first_query_comparison': first_summaries}


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    result = calculate()
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for s in result['summaries']:
        print(s['method'], s['case_gates_passed'], s['validation_whole_session_L10_mean'])


if __name__ == '__main__':
    main()
