"""Repair only undefined-reference aggregation; preserve all defender outputs.

The original run emitted NaN for zero eligible POI categories. The existing
retrieval_frontier contract uses None and reports empty references separately.
No defender is rerun and no eligible reference is changed by its output.
"""
import json
from pathlib import Path
from experiments.research_loop_resources import ROOT, load, sha
from experiments.research_loop_cases import mean_optional, select_and_summarize

SOURCE = ROOT/'artifacts/benchmarks/research_loop/iteration10_cases.json'
OUT = ROOT/'artifacts/benchmarks/research_loop/iteration10_cases_checked.json'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, service, context, belief, metadata = load()
    data = json.loads((ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json').read_text())
    result = json.loads(SOURCE.read_text())
    executions = {}
    for ex in result['executions']:
        trace = data['traces'][ex['session_id']]
        for index, utility in ex['utility_by_index'].items():
            p = trace[int(index)]
            state, _ = rn.nearest(p['lat'], p['lon'])
            if not (context.signatures[state] >= 0).any():
                utility.update({'5': None, '10': None})
            else:
                assert all(v is not None and 0 <= v <= 1 for v in utility.values())
        ex['whole_session_recall'] = {L: mean_optional([u[L] for u in ex['utility_by_index'].values()]) for L in ('5', '10')}
        ex['eligible_events'] = sum(u['5'] is not None for u in ex['utility_by_index'].values())
        ex['empty_reference_events'] = len(ex['clock_indices'])-ex['eligible_events']
        executions[ex['session_id'], ex['rep'], ex['method']] = ex
    by_id = {r['record_id']: r for r in data['records']}
    for row in result['rows']:
        record = by_id[row['record_id']]
        utility = [executions[sid, row['rep'], row['method']]['utility_by_index'][str(i)]
                   for sid, ids in zip(record['session_ids'], record['observed_indices']) for i in ids]
        row['eligible_events'] = sum(u['5'] is not None for u in utility)
        row['empty_reference_events'] = len(utility)-row['eligible_events']
        row['recall'] = {L: mean_optional([u[L] for u in utility]) for L in ('5', '10')}
    result['summaries'] = select_and_summarize(result['rows'])
    result['aggregation_repair'] = {
        'reason': 'NaN from zero eligible categories; restore existing null/empty-reference convention',
        'original_artifact_sha256': sha(SOURCE), 'repair_code_sha256': sha(Path(__file__)),
        'original_generation_source': 'artifacts/benchmarks/research_loop/sources/iteration10_cases_v1.py',
        'current_runner_sha256': sha(ROOT/'experiments/research_loop_cases.py'),
        'existing_metric_source': 'evaluation/retrieval_frontier.py',
        'defender_outputs_or_privacy_errors_changed': False,
        'eligibility_depends_only_on_reference': True}
    result['schema'] = 'persistent-exact-case-development-v1-aggregation-checked'
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Corrected', len(result['executions']), 'executions;', len(result['rows']), 'case rows')
    for method in ('switching_core', 'filter_progress', 'filter_switching_progress'):
        rows = [r for r in result['summaries'] if r['method'] == method]
        print(method, '90% gates passed', sum(r['pass_90pct_case_recall_L10'] for r in rows), '/', len(rows))
        for r in rows:
            print(r['case_id'], round(r['recall']['10'], 4), r['reference_empty_events_per_repetition'])


if __name__ == '__main__':
    main()
