"""Inspect a bounded scenario example; no pretend protected transcript.

python -m experiments.inspect_scenario_suite S5.C --evaluator
"""
import argparse
import json
from pathlib import Path

from data.scenario_suite.records import device_view, reference_queries

DEFAULT=Path(__file__).resolve().parents[1]/'artifacts/datasets/urban_scenarios_v1/dataset.json'


def inspect_case(data, case, evaluator=False, limit=5):
    if limit<1:
        raise ValueError('limit must be positive')
    spec=next((c for c in data['catalogue'] if c['case_id']==case),None)
    if spec is None:
        raise ValueError(f'Unknown case {case}; see catalogue')
    r=next((r for r in data['records'] if r['case_id']==case),None)
    result={'case':spec,'warning':'LOCAL INPUT, NOT PROTECTED OR ATTACKER-READY DATA'}
    if r is None:
        result['rejections']=[x for x in data['rejections'] if x['case_id']==case]
        return result
    result['device_streams']=[list(device_view(r,data['traces'],i))[:limit] for i in range(len(r['session_ids']))]
    result['total_events']=[len(idx) for idx in r['observed_indices']]
    if r['scenario']=='S7':
        result['query_only_reference_control']=reference_queries(r,data['traces'])[:limit]
    if evaluator:
        result['evaluator_only']={'record':r,'targets_and_full_traces':'Resolve label indices in dataset.json traces',
                                   'sessions':[s for f in data['families'] for s in f['sessions'] if s['session_id'] in r['session_ids']]}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case')
    p.add_argument('--dataset',type=Path,default=DEFAULT)
    p.add_argument('--evaluator',action='store_true')
    p.add_argument('--limit',type=int,default=5)
    args=p.parse_args()
    print(json.dumps(inspect_case(json.loads(args.dataset.read_text()),args.case,args.evaluator,args.limit),ensure_ascii=False,indent=2))
