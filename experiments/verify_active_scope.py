"""Verify revised denominators and reconcile published cells to measured rows."""
from collections import defaultdict
import gzip
import json
from statistics import mean
import numpy as np

from experiments.reaggregate_active_scope import OUT, HIST, END, ROOT, DATA
from experiments.research_loop_resources import sha
from experiments.run_live_paper_comparison import write_json


def read(p):return json.loads(p.read_text())


def grouped(rows, metric, entity='record_id'):
    nested=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r.get(metric) is not None:nested[r['family_id']][r[entity]].append(r[metric])
    return mean(mean(mean(v) for v in group.values()) for group in nested.values()) if nested else None


def equal(a,b):
    if a is None or b is None:assert a is b,(a,b)
    else:assert abs(a-b)<1e-8,(a,b)


def main():
    d=read(OUT/'readout.json');checks=0
    for p,h in d['sources'].items():assert sha(ROOT/p)==h,p
    active=set(d['active_priority_cases'])
    assert len(active)==14 and 'S10.B' not in active
    assert len(d['active_all_cases'])==29
    for split,name in DATA.items():
        ds=read(ROOT/f'artifacts/datasets/{name}/dataset.json');inv=d['inventory'][split]
        selected=[r for r in ds['records'] if r['case_id'] in active]
        assert inv['priority_records']==len(selected)
        assert set(inv['priority_session_ids'])=={s for r in selected for s in r['session_ids']}
        assert inv['stored_records']-inv['active_records']==sum(r['case_id']=='S10.B' for r in ds['records'])
    hp=json.loads(gzip.decompress((HIST/'privacy.json.gz').read_bytes()))['rows']
    ep=json.loads(gzip.decompress((END/'holdout_privacy.json.gz').read_bytes()))['rows']
    eservice=read(END/'holdout_service.json')
    for kind,measured in [('historical',hp),('endpoint',ep)]:
        index=defaultdict(list)
        for r in measured:
            if r['status']=='ok':index[(r.get('split'),r['method'],r['case_id'])].append(r)
        for s in d[kind]['summaries']:
            assert s['case_id'] in active
            key=(s.get('split') if kind=='historical' else 'holdout',s['method'],s['case_id'])
            rows=index[key]
            for metric in ('hit100','mae_m'):
                equal(s[metric],grouped(rows,metric));checks+=1
    for kind in ('historical','endpoint'):
        for split in (('development','new_groups') if kind=='historical' else ('expanded',)):
            service=read(HIST/f'service_{split}.json') if kind=='historical' else eservice
            retained=set(d['inventory'][split]['priority_session_ids'])
            for a in d[kind]['aggregates']:
                if kind=='historical' and a['split']!=split:continue
                for probability in (.5,.8,.95):
                    case_values={}
                    for case in active:
                        rows=[r for r in service['rows'] if r['method']==a['method'] and r['probability']==probability and r['case_id']==case]
                        case_values[case]=grouped(rows,'recall')
                    # Independent explicit 3/3/3/3/2 hierarchy; never divide by 14.
                    expected=mean(mean(v for c,v in case_values.items() if c.startswith(sc+'.') and v is not None)
                                  for sc in ('S1','S2','S3','S9','S10'))
                    equal(a[f'recall_{probability}'],expected)
                    assert a[f'gates_{probability}']==sum(v is not None and v>=.9-1e-12 for v in case_values.values())
                    checks+=2
                costs=[r for r in service['costs'] if r['method']==a['method'] and r['session_id'] in retained]
                for metric in ('request_bytes','response_bytes','generation_ms'):
                    ratios=[{**r,'ratio':r[metric]/r['service_events']} for r in costs if r.get(metric) is not None]
                    equal(a[metric+'_per_service_event'],grouped(ratios,'ratio','session_id'));checks+=1
    idx={(r['method'],r['case_id']):r for r in d['endpoint']['summaries']}
    for a in d['endpoint']['aggregates']:
        for sc,suffixes in [('S9','ABC'),('S10','AC')]:
            for metric in ('hit100','hit200','mae_m'):
                values=[idx[a['method'],sc+'.'+c][metric] for c in suffixes]
                expected=mean(v for v in values if v is not None) if any(v is not None for v in values) else None
                equal(a[sc][metric],expected);checks+=1
    for b in d['endpoint']['comparisons']:
        deltas=[]
        for c in ('ABC' if b['scenario']=='S9' else 'AC'):
            aa,bb=(idx[m,b['scenario']+'.'+c]['family_metrics'][b['metric']] for m in (b['target'],b['control']))
            deltas.append(mean(aa[f]-bb[f] for f in aa.keys()&bb.keys()))
        equal(b['delta'],mean(deltas));assert b['valid_draws']==3000;checks+=1
    write_json(OUT/'verification.json',{'status':'passed','readout_sha256':sha(OUT/'readout.json'),
        'verifier_sha256':sha(ROOT/'experiments/verify_active_scope.py'),'numerical_checks':checks,
        'privacy_cells_recomputed_from_measured_rows':2*(len(d['historical']['summaries'])+len(d['endpoint']['summaries'])),
        'case_exclusion_consistent':True,'counts_reconciled_to_original_datasets':True,
        'recall_recomputed_from_original_service_rows':True,'costs_recomputed_on_retained_sessions':True,
        'endpoint_intervals_recomputed_for_current_cases':True,'historical_benchmark_sources_preserved':True,
        'scope':'Post-diagnosis A/C-only reporting view; no retraining, regenerated transcript, new cohort or broad novelty claim.'})
    print('Verified current scope:',checks,'numerical checks',flush=True)


if __name__=='__main__':main()
