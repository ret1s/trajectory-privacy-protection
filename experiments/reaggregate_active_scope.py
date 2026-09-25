"""Explicit A/C-only S10 reporting view over unchanged measured runs.

No model/attacker fitting or test generation. Keeps original results intact.
The scope decision follows the B diagnosis and user instruction; intervals
describe the revised subset, not a new prospective confirmation.
"""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np

from evaluation.report_scope import PRIORITY_CASES, ALL_CASES, suffixes, scenario_macro
from experiments.research_loop_resources import ROOT, sha
from experiments.run_live_paper_comparison import write_json

OUT = ROOT/'artifacts/benchmarks/active_scope_ac_v2'
HIST = ROOT/'artifacts/benchmarks/live_paper_comparison_v1'
END = ROOT/'artifacts/benchmarks/endpoint_calendar_expanded_v1'
DATA = {'illustration':'urban_fresh_v2','development':'research_loop_expanded_v1',
        'new_groups':'research_loop_confirmation_v1','expanded':'endpoint_holdout_expanded_v1'}


def read(p):return json.loads(p.read_text())


def inventory(name):
    d=read(ROOT/f'artifacts/datasets/{name}/dataset.json')
    active=[r for r in d['records'] if r['case_id'] in ALL_CASES]
    priority=[r for r in active if r['case_id'] in PRIORITY_CASES]
    sessions=sorted({sid for r in priority for sid in r['session_ids']})
    return {'families':len(d['families']),'completed_trips':len(d['traces']),
            'stored_records':len(d['records']),'active_records':len(active),
            'excluded_records':len(d['records'])-len(active),'priority_records':len(priority),
            'priority_source_sessions':len(sessions),'priority_session_ids':sessions,
            'endpoint_records':sum(r['scenario'] in ('S9','S10') for r in priority)}


def cost_summary(rows, sessions):
    rows=[r for r in rows if r['session_id'] in sessions]
    result={'cost_completed_sessions':len({r['session_id'] for r in rows})}
    for metric in ('request_bytes','response_bytes','emissions','generation_ms','category_queries','coordinates','distinct_coordinates'):
        nested=defaultdict(lambda:defaultdict(list))
        for r in rows:
            if r.get(metric) is not None:
                nested[r['family_id']][r['session_id']].append(r[metric]/r['service_events'])
        result[metric+'_per_service_event']=float(np.mean([np.mean([np.mean(v) for v in ss.values()])
                                                         for ss in nested.values()])) if nested else None
    return result


def paired_interval(by_case, draws=3000):
    families=sorted(set().union(*(set(v) for v in by_case.values())))
    matrix=np.array([[v.get(f,np.nan) for f in families] for v in by_case.values()])
    present=np.isfinite(matrix)
    rng=np.random.default_rng(20260926)
    weights=rng.multinomial(len(families),np.ones(len(families))/len(families),size=draws)
    den=weights@present.T
    valid=np.all(den>0,axis=1)
    values=((weights[valid]@np.nan_to_num(matrix).T)/den[valid]).mean(axis=1)
    return {'delta':float(np.nanmean(matrix,axis=1).mean()),'ci95':np.quantile(values,[.025,.975]).tolist(),
            'families':len(families),'draws':draws,'valid_draws':int(valid.sum()),
            'multiple_comparisons_adjusted':False,'scope':'revised-scope descriptive paired family bootstrap'}


def main():
    inputs=[ROOT/'evaluation/report_scope.py',Path(__file__),
            ROOT/'artifacts/benchmarks/s10b_diagnostic_v1/readout.json',
            *[ROOT/f'artifacts/datasets/{name}/dataset.json' for name in DATA.values()],
            *[HIST/n for n in ('readout.json','verification.json','service_development.json','service_new_groups.json')],
            *[END/n for n in ('readout.json','verification.json','holdout_service.json','calendar_ablation.json')]]
    provenance={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    inventory_by_split={split:inventory(name) for split,name in DATA.items()}
    historical=read(HIST/'readout.json');endpoint=read(END/'readout.json')
    assert read(HIST/'verification.json')['readout_sha256']==sha(HIST/'readout.json')
    assert read(END/'verification.json')['readout_sha256']==sha(END/'readout.json')
    hs=[r for r in historical['summaries'] if r['case_id'] in PRIORITY_CASES]
    ha=[]
    for split in ('development','new_groups'):
        costs=read(HIST/f'service_{split}.json')['costs']
        sessions=set(inventory_by_split[split]['priority_session_ids'])
        for original in (a for a in historical['aggregates'] if a['split']==split):
            m=original['method'];rows=[r for r in hs if r['method']==m and r['split']==split]
            a={**original,**cost_summary([r for r in costs if r['method']==m],sessions),'cases':len(rows),
               'attempted_records':sum(r['attempted_records'] for r in rows),
               'completed_privacy_records':sum(r['privacy_records'] for r in rows)}
            a['completed_cost_sessions']=a['cost_completed_sessions']
            for q in (.5,.8,.95):
                a[f'recall_{q}']=scenario_macro({r['case_id']:r[f'recall_{q}'] for r in rows})
                a[f'gates_{q}']=sum(r[f'gate_{q}'] for r in rows)
            for metric in ('mae_m','hit50','hit100','hit200'):
                a[metric]=scenario_macro({r['case_id']:r[metric] for r in rows})
            ha.append(a)
    es=[r for r in endpoint['summaries'] if r['case_id'] in PRIORITY_CASES]
    ecost=read(END/'holdout_service.json')['costs'];ea=[]
    for original in endpoint['aggregates']:
        m=original['method'];a=dict(original)
        a.update(cost_summary([r for r in ecost if r['method']==m],set(inventory_by_split['expanded']['priority_session_ids'])))
        for q in (.5,.8,.95):
            vals={c:v for c,v in a[f'case_recall_{q}'].items() if c in PRIORITY_CASES}
            a[f'case_recall_{q}']=vals;a[f'recall_{q}']=scenario_macro(vals)
            a[f'gates_{q}']=sum(v is not None and v>=.9-1e-12 for v in vals.values())
        for sc in ('S9','S10'):
            rows=[r for r in es if r['method']==m and r['case_id'].startswith(sc+'.')]
            a[sc]={k:float(np.mean([r[k] for r in rows if r[k] is not None]))
                   if any(r[k] is not None for r in rows) else None for k in original[sc]}
        ea.append(a)
    comparisons=[];index={(r['method'],r['case_id']):r for r in es}
    for target in ('calendar30','calendar67'):
        for control in ('dls','rdg','transprotect_markov','semantic_poi','fake_queries'):
            for sc in ('S9','S10'):
                for metric in ('hit100','mae_m'):
                    by_case={}
                    for suffix in suffixes(sc):
                        c=sc+'.'+suffix
                        a,b=(index[m,c]['family_metrics'][metric] for m in (target,control))
                        by_case[c]={f:a[f]-b[f] for f in a.keys()&b.keys()}
                    comparisons.append({'target':target,'control':control,'scenario':sc,'metric':metric,
                                        **paired_interval(by_case)})
    ablation_rows=[r for r in read(END/'calendar_ablation.json')['rows'] if r['session_id'] in set(inventory_by_split['expanded']['priority_session_ids'])]
    ablation={}
    for method in ('calendar30','calendar67'):
        nested=defaultdict(lambda:defaultdict(list))
        for r in ablation_rows:
            if r['method']!=method:continue
            a,b=r['active_epoch'],r['public_calendar']
            nested[r['family_id']][r['session_id']].append((b['request_bytes']+b['response_bytes'])/(a['request_bytes']+a['response_bytes']))
        ablation[method]=float(np.mean([np.mean([np.mean(v) for v in sessions.values()]) for sessions in nested.values()]))
    result={'schema':'active-scope-ac-v2','date':'2026-09-25','sources':provenance,
            'active_priority_cases':PRIORITY_CASES,'active_all_cases':ALL_CASES,
            'scope_change':'S10.B excluded by user instruction after duplicate-task/raw-control diagnosis; S10 keeps A/C with original IDs. Frozen data/results remain archived.',
            'aggregation':'equal cases within each scenario, equal weight per scenario; missing offline heads marked partial',
            'cost_policy':'only source sessions used by retained cases; keep complete original session clocks, service-event denominator and all emitted traffic including full calendar hour; no transcript regenerated',
            'not_new_confirmation':True,'inventory':inventory_by_split,
            'ablation':{'calendar_to_active_epoch_byte_ratio':ablation,'equal_utility_event_checks':sum(r['service_events'] for r in ablation_rows)},
            'historical':{'summaries':hs,'aggregates':ha},
            'endpoint':{'summaries':es,'aggregates':ea,'comparisons':comparisons}}
    write_json(OUT/'readout.json',result)
    for split,i in inventory_by_split.items():print(split,{k:v for k,v in i.items() if k!='priority_session_ids'})
    for a in ea:
        if a['method'] in ('calendar30','calendar67'):
            print(a['method'],a['S10'],a['recall_0.8'],a['gates_0.95'],a['request_bytes_per_service_event']+a['response_bytes_per_service_event'])


if __name__=='__main__':main()
