"""Train/select the common attack bank, then read the named comparator results."""
from collections import defaultdict
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from evaluation.live_comparison_attacks import public_view,features_and_geometry,LearnedAttack
from evaluation.live_comparison_endpoint_attacks import extrapolations
from benchmark.paper_comparators import PublicHistory
from experiments.run_live_paper_comparison import (OUT,METHODS,LABELS,data_path,protocol,write_json,shard_path)
from experiments.research_loop_resources import ROOT,load,sha
from experiments.research_loop_cases import target_xy,SCENARIOS,mean_optional

CASES=[f'S{s}.{c}' for s in (1,2,3,9,10) for c in 'ABC']
RADII=(50,100,200,500)


def family_mean(rows,field):
    records=defaultdict(list)
    for r in rows:
        if r[field] is not None:records[r['family_id'],r['record_id']].append(r[field])
    groups=defaultdict(list)
    for (family,_),values in records.items():groups[family].append(float(np.mean(values)))
    fm={f:float(np.mean(v)) for f,v in sorted(groups.items())}
    return mean_optional(list(fm.values())),fm


def input_rows(split,method,rn,history):
    data=json.loads(data_path(split).read_text())
    records=[r for r in data['records'] if r['scenario'] in SCENARIOS]
    cache={}
    for record in records:
        for rep in (0,1):
            views=[];valid=True
            for slot,(sid,indices) in enumerate(zip(record['session_ids'],record['observed_indices'])):
                if (sid,rep) not in cache:
                    cache[sid,rep]=json.loads(gzip.decompress(shard_path(split,method,sid,rep).read_bytes()))
                ex=cache[sid,rep]
                if ex['status']!='ok':valid=False;break
                events,target_slots=public_view(ex,indices)
                x,bank=features_and_geometry(events,record['scenario'],rn,history)
                if record['scenario'] in ('S9','S10'):
                    bank.update(extrapolations(events,record['scenario'],rn,history))
                if record['scenario'] in ('S1','S3'):
                    x=x[target_slots];bank={a:v[target_slots] for a,v in bank.items()}
                y=target_xy(record,slot,data['traces'][sid],rn)
                assert x.shape[0]==y.shape[0]
                views.append((x,bank,y))
            row={k:record[k] for k in ('record_id','family_id','case_id','scenario')}
            row.update(method=method,rep=rep,split=split,status='ok' if valid else 'failed')
            if valid:
                row.update(x=np.concatenate([v[0] for v in views]),y=np.concatenate([v[2] for v in views]),
                           bank={a:np.concatenate([v[1][a] for v in views]) for a in views[0][1]})
            yield row


def task(row):return 'current' if row['scenario'] in ('S1','S3') else row['scenario']


def predictions(row,learned,rn,history):
    bank=dict(row['bank'])
    if learned is not None:bank.update(learned.predict(row['x']))
    bank['public_prior']=np.repeat(np.average(rn.xy,axis=0,weights=history.q)[None,:],len(row['y']),axis=0)
    bank.update({a+'_road':rn.xy[rn.tree.query(p)[1]] for a,p in list(bank.items())})
    if row['case_id'] in ('S9.C','S10.C'):
        bank.update({a+'_joint':np.repeat(p.mean(axis=0,keepdims=True),len(p),axis=0) for a,p in list(bank.items())})
    return bank


def score_bank(row,learned,rn,history):
    pred=predictions(row,learned,rn,history)
    errors={a:np.linalg.norm(x-row['y'],axis=1).tolist() for a,x in pred.items()}
    return {k:v for k,v in row.items() if k not in ('x','y','bank')}|{'errors':errors}


def attack_selection(rows,case):
    candidates=[r for r in rows if r['case_id']==case]
    scope='same_case'
    if not candidates:
        candidates=[r for r in rows if r['scenario']==case.split('.')[0]];scope='scenario_fallback'
    if not candidates:return None
    names=set.intersection(*(set(r['errors']) for r in candidates))
    selected={};scores={}
    for metric in ('mae',*[f'hit{r}' for r in RADII]):
        values={}
        for name in sorted(names):
            rr=[dict(family_id=r['family_id'],record_id=r['record_id'],value=float(np.mean(r['errors'][name])) if metric=='mae' else float(np.mean(np.asarray(r['errors'][name])<=int(metric[3:])))) for r in candidates]
            values[name]=family_mean(rr,'value')[0]
        selected[metric]=min(values,key=lambda a:(values[a] if metric=='mae' else -values[a],a))
        scores[metric]=values
    return {'selected':selected,'scope':scope,'record_count':len({r['record_id'] for r in candidates}),
            'family_count':len({r['family_id'] for r in candidates}),'validation_scores':scores}


def attacks():
    p=protocol();dst=OUT/'privacy.json.gz'
    if dst.exists():raise FileExistsError(dst)
    rn,*_=load();aux=json.loads(data_path('auxiliary').read_text())
    train_ids={s['session_id'] for f in aux['families'] if f['family_id'] in p['fit_families'] for s in f['sessions']}
    training=[[rn.nearest(v['lat'],v['lon'])[0] for v in aux['traces'][sid][::20]] for sid in sorted(train_ids)]
    history=PublicHistory(rn,training)
    selections={};fitting=[];results=[]
    for method in METHODS:
        auxiliary=list(input_rows('auxiliary',method,rn,history))
        learners={}
        for name in ('current','S2','S9','S10'):
            train=[r for r in auxiliary if r['status']=='ok' and r['family_id'] in p['fit_families'] and task(r)==name]
            if train:
                x=np.concatenate([r['x'] for r in train]);y=np.concatenate([r['y'] for r in train])
                learners[name]=LearnedAttack(x,y)
            else:learners[name]=None
            fitting.append({'method':method,'task':name,'families':sorted({r['family_id'] for r in train}),
                            'records':sorted({r['record_id'] for r in train}),'points':sum(len(r['y']) for r in train)})
        validation=[score_bank(r,learners[task(r)],rn,history) for r in auxiliary
                    if r['status']=='ok' and r['family_id'] in p['attack_selection_families']]
        for case in CASES:selections[method+'/'+case]=attack_selection(validation,case)
        for split in ('development','new_groups'):
            for row in input_rows(split,method,rn,history):
                if row['status']!='ok':results.append(row);continue
                scored=score_bank(row,learners[task(row)],rn,history)
                sel=selections[method+'/'+row['case_id']]
                if sel is None:
                    # No usable validation output: a fixed public prior, never test selection.
                    chosen={m:'public_prior' for m in ('mae',*[f'hit{r}' for r in RADII])}
                else:chosen=sel['selected']
                assert all(a in scored['errors'] for a in chosen.values())
                e=np.array(scored['errors'][chosen['mae']])
                scored.update(selected=chosen,mae_m=float(e.mean()),median_m=float(np.median(e)),p90_m=float(np.quantile(e,.9)),
                    **{f'hit{r}':float(np.mean(np.array(scored['errors'][chosen[f'hit{r}']])<=r)) for r in RADII})
                results.append(scored)
        print('Attack fit/select/evaluate',method,'done',flush=True)
    result={'schema':'common-live-paper-attacks-v2','protocol_sha256':sha(OUT/'protocol.json'),
            'attacker_protocol_sha256':sha(OUT/'attacker_protocol.json'),
            'analysis_code_sha256':sha(Path(__file__)),'fit':fitting,'selection':selections,'rows':results,
            'scope':'Finite auxiliary-selected attackers; offline AnotherMe and execution failures remain explicit'}
    dst.write_bytes(gzip.compress(json.dumps(result,separators=(',',':'),allow_nan=False).encode(),mtime=0))


def readout():
    p=protocol();privacy=json.loads(gzip.decompress((OUT/'privacy.json.gz').read_bytes()))
    summaries=[];aggregates=[];bootstrap=[];cost_rows=[]
    for split in ('development','new_groups'):
        service=json.loads((OUT/f'service_{split}.json').read_text())
        for method in METHODS:
            for case in CASES:
                rr=[r for r in privacy['rows'] if r['split']==split and r['method']==method and r['case_id']==case]
                good=[r for r in rr if r['status']=='ok']
                item={'split':split,'method':method,'case_id':case,'privacy_records':len({r['record_id'] for r in good}),
                      'attempted_records':len({r['record_id'] for r in rr}),'privacy_family_count':len({r['family_id'] for r in good}),
                      'failed_privacy_runs':len(rr)-len(good),'privacy_family_metrics':{}}
                for field in ('mae_m','median_m','p90_m',*[f'hit{r}' for r in RADII]):
                    value,fm=family_mean(good,field);item[field]=value;item['privacy_family_metrics'][field]=fm
                for prob in p['probabilities']:
                    rows=[r for r in service['rows'] if r['method']==method and r['case_id']==case and r['probability']==prob]
                    value,fm=family_mean(rows,'recall')
                    item[f'recall_{prob}']=value;item[f'utility_families_{prob}']=fm
                    item[f'gate_{prob}']=value is not None and value>=.9-1e-12
                summaries.append(item)
            costs=[r for r in service['costs'] if r['method']==method]
            by_session=defaultdict(list)
            for r in costs:by_session[r['family_id'],r['session_id']].append(r)
            item={'split':split,'method':method,'completed_cost_sessions':len(by_session),
                  'online':method!='anotherme_offline'}
            for metric in ('request_bytes','response_bytes','category_queries','emissions','coordinates','distinct_coordinates','generation_ms'):
                by_family=defaultdict(list)
                for (family,sid),values in by_session.items():
                    by_family[family].append(np.mean([r[metric]/r['service_events'] for r in values]))
                item[metric+'_per_service_event']=mean_optional([float(np.mean(v)) for v in by_family.values()])
            cost_rows.append(item)
            rows=[s for s in summaries if s['split']==split and s['method']==method]
            a={**item,'label':LABELS[method],'cases':len(rows),'attempted_records':sum(s['attempted_records'] for s in rows),
               'completed_privacy_records':sum(s['privacy_records'] for s in rows),
               'minimum_privacy_families':min(s['privacy_family_count'] for s in rows)}
            for metric in ('mae_m',*[f'hit{r}' for r in RADII]):a[metric]=mean_optional([s[metric] for s in rows])
            for prob in p['probabilities']:
                a[f'recall_{prob}']=mean_optional([s[f'recall_{prob}'] for s in rows])
                a[f'gates_{prob}']=sum(s[f'gate_{prob}'] for s in rows)
            aggregates.append(a)
        # Paired case/family bootstrap. No differing-dataset subtraction.
        for target in ('ours30','ours67'):
            for control in ('dls','rdg','transprotect_markov','semantic_poi','fake_queries'):
                deltas=defaultdict(list)
                for case in CASES:
                    a=next(s for s in summaries if (s['split'],s['method'],s['case_id'])==(split,target,case))
                    b=next(s for s in summaries if (s['split'],s['method'],s['case_id'])==(split,control,case))
                    for metric,fa,fb in [('recall',a['utility_families_0.8'],b['utility_families_0.8']),
                                          ('hit100',a['privacy_family_metrics']['hit100'],b['privacy_family_metrics']['hit100'])]:
                        for f in set(fa)&set(fb):deltas[metric].append((case,f,fa[f]-fb[f]))
                for metric,values in deltas.items():
                    families=sorted({f for c,f,d in values});by_case=defaultdict(dict)
                    for c,f,d in values:by_case[c][f]=d
                    point=float(np.mean([np.mean(list(v.values())) for v in by_case.values()]))
                    rng=np.random.default_rng(20260925);draws=[]
                    for _ in range(5000):
                        sample=rng.choice(families,len(families),replace=True)
                        cs=[np.mean([v[f] for f in sample if f in v]) for v in by_case.values() if any(f in v for f in sample)]
                        draws.append(np.mean(cs))
                    bootstrap.append({'split':split,'target':target,'control':control,'metric':metric,
                        'delta':point,'ci95':np.quantile(draws,[.025,.975]).tolist(),'families':len(families),'draws':5000,
                        'scope':'exploratory paired family bootstrap, unadjusted, not a new confirmation'})
    result={'schema':'common-live-paper-readout-v1','protocol_sha256':sha(OUT/'protocol.json'),
            'source_sha256':{str((OUT/f).relative_to(ROOT)):sha(OUT/f) for f in ('privacy.json.gz','service_development.json','service_new_groups.json')},
            'analysis_code_sha256':sha(Path(__file__)),'summaries':summaries,'aggregates':aggregates,'bootstrap':bootstrap}
    write_json(OUT/'readout.json',result)
    import csv
    with (OUT/'case_results.csv').open('w',newline='') as f:
        fields=['split','method','case_id','attempted_records','privacy_records','mae_m','median_m','p90_m','hit50','hit100','hit200','recall_0.5','recall_0.8','recall_0.95']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(summaries)
    print('Readout ready',len(summaries),'case rows',flush=True)
    for a in aggregates:
        if a['split']=='new_groups':print(a['method'],a['recall_0.8'],a['hit100'],a['completed_privacy_records'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['attacks','readout']);a=parser.parse_args()
    attacks() if a.stage=='attacks' else readout()
