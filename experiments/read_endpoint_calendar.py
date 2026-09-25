"""Reconcile endpoint and service evidence, including missing/failing cases."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

from experiments.endpoint_calendar_study import OUT,DATA,METHODS,CASES,gzread,family_mean,protocol
from experiments.research_loop_resources import ROOT,sha
from experiments.run_live_paper_comparison import write_json


def family_values(rows,metric):
    nested=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r.get(metric) is not None:nested[r['family_id']][r['record_id']].append(r[metric])
    return {f:float(np.mean([np.mean(x) for x in records.values()])) for f,records in nested.items()}


def main():
    p=protocol();privacy=gzread(OUT/'holdout_privacy.json.gz');service=json.loads((OUT/'holdout_service.json').read_text())
    summaries=[];aggregates=[];comparisons=[]
    for m in METHODS:
        for c in CASES:
            rr=[r for r in privacy['rows'] if r['method']==m and r['case_id']==c];good=[r for r in rr if r['status']=='ok']
            s={'method':m,'case_id':c,'attempted_records':len({r['record_id'] for r in rr}),
               'valid_record_runs':len(good),'attempted_record_runs':len(rr),'valid_families':len({r['family_id'] for r in good}),
               'generation_failed_runs':sum(r['status']=='failed' for r in rr),
               'no_attacker_runs':sum(r['status']=='no_calibrated_attacker' for r in rr),'family_metrics':{}}
            for metric in ('mae_m','median_m','p90_m','hit50','hit100','hit200','hit500','log_gain_bits','nll_bits','entropy_bits','credible90_cells','credible90_coverage'):
                s[metric]=family_mean(good,metric);s['family_metrics'][metric]=family_values(good,metric)
            for q in p['probabilities']:
                values=[r for r in service['rows'] if r['method']==m and r['case_id']==c and r['probability']==q]
                s[f'recall_{q}']=family_mean(values,'recall')
            summaries.append(s)
        item={'method':m,'family_count':len({r['family_id'] for r in service['rows'] if r['method']==m})}
        for q in p['probabilities']:
            all_cases=sorted({r['case_id'] for r in service['rows']})
            recalls={c:family_mean([r for r in service['rows'] if r['method']==m and r['case_id']==c and r['probability']==q],'recall') for c in all_cases}
            item[f'recall_{q}']=float(np.mean([v for v in recalls.values() if v is not None]))
            item[f'gates_{q}']=sum(v is not None and v>=.9-1e-12 for v in recalls.values());item[f'case_recall_{q}']=recalls
        costs=[r for r in service['costs'] if r['method']==m]
        item['cost_completed_sessions']=len({r['session_id'] for r in costs})
        for metric in ('request_bytes','response_bytes','emissions'):
            ratios=[dict(family_id=r['family_id'],record_id=r['session_id'],value=r[metric]/r['service_events']) for r in costs]
            item[metric+'_per_service_event']=family_mean(ratios,'value')
        item['outside_interval_events']=sum(r.get('outside_interval_events',0) for r in costs)
        for scenario in ('S9','S10'):
            rows=[s for s in summaries if s['method']==m and s['case_id'].startswith(scenario+'.')]
            item[scenario]={key:float(np.mean([s[key] for s in rows if s[key] is not None])) if any(s[key] is not None for s in rows) else None for key in ('hit100','hit200','mae_m','log_gain_bits','credible90_coverage')}
        aggregates.append(item)
    # Keep uncertainty per endpoint task; do not merge different privacy heads.
    for target in ('calendar30','calendar67'):
        for control in ('dls','rdg','transprotect_markov','semantic_poi','fake_queries'):
            for scenario in ('S9','S10'):
                for metric in ('hit100','mae_m','log_gain_bits'):
                    by_case={}
                    for c in (scenario+'.'+v for v in 'ABC'):
                        a=next(s for s in summaries if s['method']==target and s['case_id']==c)['family_metrics'][metric]
                        b=next(s for s in summaries if s['method']==control and s['case_id']==c)['family_metrics'][metric]
                        by_case[c]={f:a[f]-b[f] for f in a.keys()&b.keys()}
                    families=sorted(set().union(*(v.keys() for v in by_case.values())))
                    point=np.mean([np.mean(list(v.values())) for v in by_case.values() if v]);rng=np.random.default_rng(20260926);draws=[]
                    for _ in range(3000):
                        sample=rng.choice(families,len(families),replace=True)
                        vals=[np.mean([v[f] for f in sample if f in v]) for v in by_case.values() if any(f in v for f in sample)]
                        draws.append(np.mean(vals))
                    comparisons.append({'target':target,'control':control,'scenario':scenario,'metric':metric,
                                        'delta':float(point),'ci95':np.quantile(draws,[.025,.975]).tolist(),
                                        'families':len(families),'draws':3000,'multiple_comparisons_adjusted':False})
    result={'schema':'endpoint-calendar-readout-v1','protocol_sha256':sha(OUT/'protocol.json'),
        'dataset_sha256':sha(DATA/'dataset.json'),'source_sha256':{str((OUT/f).relative_to(ROOT)):sha(OUT/f) for f in ('selection.json','holdout_privacy.json.gz','holdout_service.json','construction_amendment.json')},
        'readout_code_sha256':sha(Path(__file__)),'summaries':summaries,'aggregates':aggregates,'comparisons':comparisons}
    write_json(OUT/'readout.json',result)
    with (OUT/'case_results.csv').open('w',newline='') as f:
        fields=[k for k in summaries[0] if k!='family_metrics'];writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(summaries)
    for a in aggregates:print(a['method'],a['S9'],a['S10'],'recall',a['recall_0.8'],flush=True)


if __name__=='__main__':main()
