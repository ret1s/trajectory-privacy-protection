"""Retrospective paired ablations and cross-seed public endpoint attack challenge."""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import numpy as np
from experiments.report_boundary_audit import cut_public, utility, xy
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/benchmarks/contribution_stress'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def avg(x): return float(np.mean(x))

def paired_ablation(data, draws=10000):
    rows=data['rows']; summaries={(s['method'],s['case_id']):s for s in data['summaries']}
    families=sorted({r['family_id'] for r in rows}); cases=sorted({r['case_id'] for r in rows})
    methods=sorted({r['method'] for r in rows})
    metrics=['mae_m','hit100','recall_L5','recall_L10','step_ms','reply_bytes_L5']
    cells=defaultdict(list)
    for r in rows:
        s=summaries[r['method'],r['case_id']]
        cells[r['method'],r['case_id'],r['family_id']].append([
            avg(r['errors_by_attack'][s['selected_mae_attack']]),
            avg(np.array(r['errors_by_attack'][s['selected_hit_attack']])<=100),
            r['utility']['5']['poi_recall_at_5'],r['utility']['10']['poi_recall_at_5'],
            avg(r['step_ms']),r['utility']['5']['response_id_bytes_per_event']])
    values=np.full((len(methods),len(cases),len(families),len(metrics)),np.nan)
    for mi,m in enumerate(methods):
        for ci,c in enumerate(cases):
            for fi,f in enumerate(families):
                if (m,c,f) in cells: values[mi,ci,fi]=np.mean(cells[m,c,f],axis=0)
    assert np.array_equal(np.isnan(values),np.broadcast_to(np.isnan(values[:1]),values.shape))
    observed=np.nanmean(values,axis=2).mean(axis=1)
    rng=np.random.default_rng(240926); boot=[]
    while len(boot)<draws:
        ids=rng.integers(len(families),size=len(families))
        sample=values[:,:,ids,:]
        if np.any(np.all(np.isnan(sample),axis=2)): continue
        boot.append(np.nanmean(sample,axis=2).mean(axis=1))
    boot=np.array(boot); target=methods.index('switching_exchange'); contrasts=[]
    for mi,m in enumerate(methods):
        if mi==target:continue
        delta=boot[:,target]-boot[:,mi]
        contrasts.append({'baseline':m,'difference':'switching minus baseline','metrics':{
            metric:{'delta':float(observed[target,j]-observed[mi,j]),
                    'ci95':np.quantile(delta[:,j],[.025,.975]).tolist()} for j,metric in enumerate(metrics)}})
    return {'rows':len(rows),'clusters':families,'cases':cases,'bootstrap_draws':draws,
        'scope':'retrospective; frozen selected attackers; six clusters; no multiplicity correction',
        'means':{m:dict(zip(metrics,observed[i].tolist())) for i,m in enumerate(methods)},'contrasts':contrasts}

def endpoint_predictions(public,scenario,lat0):
    """No target, truth, evaluator state, route or private label input."""
    events=public['events']; times=np.array([e['timestamp_s'] for e in events])
    tracks=np.array([[xy(c['lat'],c['lon'],lat0) for c in e['candidates']] for e in events])
    streams={'mean':tracks.mean(axis=1),'median':np.median(tracks,axis=1)}
    streams.update({f'track_{i}':tracks[:,i] for i in range(tracks.shape[1])})
    first=scenario=='S9'; at=0 if first else -1; sign=-1 if first else 1
    result={'window_mean':tracks.mean(axis=(0,1))}
    for name,points in streams.items():
        result[name+'_boundary']=points[at]
        for n in (2,3,6):
            ids=np.arange(min(n,len(events))) if first else np.arange(max(0,len(events)-n),len(events))
            t=times[ids]-times[at]; design=np.column_stack([np.ones(len(t)),t])
            coeff=np.linalg.lstsq(design,points[ids],rcond=None)[0]
            for seconds in (30,60,120):
                result[f'{name}_ols{n}_{seconds}s']=np.array([1.,sign*seconds])@coeff
    return result

def cross_seed_select(rows):
    """Select only from other seeds; target errors consumed by evaluator only."""
    result=[]
    for seed in sorted({r['seed'] for r in rows}):
        train=[r for r in rows if r['seed']!=seed]; test=[r for r in rows if r['seed']==seed]
        attacks=sorted(train[0]['errors'])
        mae=min(attacks,key=lambda a:(avg([r['errors'][a] for r in train]),a))
        hit=min(attacks,key=lambda a:(-avg([r['errors'][a]<=100 for r in train]),
                                    avg([r['errors'][a] for r in train]),a))
        for r in test:
            result.append({**r,'selected_mae':mae,'selected_hit':hit,
                'selection_seeds':sorted({t['seed'] for t in train}),
                'mae_m':r['errors'][mae],'hit100':float(r['errors'][hit]<=100)})
    return result

def endpoint_challenge(data):
    lat0={m['seed']:m['projection_lat0'] for m in data['manifests']};groups=defaultdict(list)
    for r in data['rows']:
        if r['scenario'] not in ('S9','S10') or r['k']!=5 or r['status']!='ok':continue
        for drop in (0,2,4):
            public,ids=cut_public(r['public'],r['scenario'],drop)
            target=np.array(xy(*r['hidden_target'],lat0[r['seed']]))
            pred=endpoint_predictions(public,r['scenario'],lat0[r['seed']])
            key=(r['scenario'],r['method'],drop)
            groups[key].append({'seed':r['seed'],'record_id':r['record_id'],
                'errors':{a:float(np.linalg.norm(v-target)) for a,v in pred.items()},
                'recall_all_queries':utility(r,ids)['recall_all_original_queries']})
    rows=[];summaries=[]
    for (s,m,d),group in sorted(groups.items()):
        selected=cross_seed_select(group)
        assert len(selected)==12
        rows.extend({**r,'scenario':s,'method':m,'extra_cut_s':20*d} for r in selected)
        summaries.append({'scenario':s,'method':m,'extra_cut_s':20*d,'trips':len(selected),
            'mae_m':avg([r['mae_m'] for r in selected]),'hit100':avg([r['hit100'] for r in selected]),
            'recall_all_queries':avg([r['recall_all_queries'] for r in selected]),
            'hit_by_seed':{str(seed):avg([r['hit100'] for r in selected if r['seed']==seed]) for seed in (81,82,83)}})
    return {'scope':'new cross-seed attack runs on historical protected transcripts; not new defender generation',
        'excluded_rows':[{'method':r['method'],'scenario':r['scenario'],'seed':r['seed'],'record_id':r['record_id'],'status':r['status']} for r in data['rows'] if r['k']==5 and r['scenario'] in ('S9','S10') and r['status']!='ok'],
        'attack_count_by_candidates':{'K1':31,'K5':71},'rows':rows,'summaries':summaries}

def main():
    sources={name:ROOT/path for name,path in {
        'fresh':'artifacts/benchmarks/fresh_switching/confirmation.json',
        'paper':'artifacts/benchmarks/paper_benchmark/results.json'}.items()}
    result={'schema':'contribution-stress-v1','protocol_sha256':sha(ROOT/'docs/research/contribution_stress_protocol.md'),
        'source_sha256':{n:sha(p) for n,p in sources.items()},'code_sha256':sha(Path(__file__)),
        'helper_sha256':sha(ROOT/'experiments/report_boundary_audit.py'),
        'ablation':paired_ablation(json.loads(sources['fresh'].read_text())),
        'endpoints':endpoint_challenge(json.loads(sources['paper'].read_text()))}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result['ablation'],indent=2))
    for s in result['endpoints']['summaries']:
        if s['method'] in ('unprotected','br_private','br_fresh'):
            print(s)
if __name__=='__main__':main()
