"""Iteration 9: compare the latest switching-belief core at matched tight bound."""
from collections import defaultdict
import json,time
from pathlib import Path
import numpy as np
from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.quotient_cover import QuotientCoverLaneDummy
from benchmark.engines.filtered_cover import FilteredCoverLaneDummy
from benchmark.engines.progress_cover import ProgressCoverLaneDummy,FilteredProgressCoverLaneDummy
from benchmark.engines.slack_progress import FilteredSlackProgressCoverLaneDummy
from functools import partial
from benchmark.engines.switching_progress import SwitchingQuotientCoverLaneDummy,MatchedSwitchingProgressCoverLaneDummy
from benchmark.engines.matched_filter import MatchedFilteredProgressCoverLaneDummy,MatchedFilteredSlackProgressCoverLaneDummy
from core.boundary_release import BoundaryPolicy,BoundaryProtectedStream
from experiments.research_loop_resources import ROOT,CACHE,load,sha
from experiments.contribution_stress import endpoint_predictions
from experiments.rng_util import rng_from_key
OUT=ROOT/'artifacts/benchmarks/research_loop/iteration09_switching.json'
DATA=ROOT/'artifacts/datasets/research_loop_development_v1/dataset.json'
SPECS={'raw':(0,0,0),'core_H12':(12,0,0),'filter_progress':(12,0,0),'switching_core':(12,0,0),'filter_switching_progress':(12,0,0)}

class Raw:
    def reset(self):pass
    def protect_step(self,lat,lon,t):return ((lat,lon),)

def query_recall(context,states,truth_state):
    scores=[]
    for c in range(len(context.categories)):
        ref=set(context.signatures[truth_state,c]);ref.discard(-1)
        if not ref:continue
        got=set(context.signatures[states,c].ravel()) if states else set()
        got.discard(-1)
        # All found true top-5 items survive local exact reranking.
        scores.append(len(ref&got)/len(ref))
    return float(np.mean(scores)) if scores else None

def main():
    if OUT.exists():raise FileExistsError('Preserve completed evidence')
    rn,service,context,belief12,metadata=load()
    data=json.loads(DATA.read_text());models={};beliefs={12:belief12}
    _,inv,counts=np.unique(np.floor(rn.xy/120.).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    prior=1./counts[inv];prior/=prior.sum()
    beliefs[64]=PublicAnchorModel(rn,context,prior,epsilon_release=.24/128,epsilon_test=.24/128,
        cache_path=CACHE/'belief_H64.npz')
    for name,(h,head,delay) in SPECS.items():
        cls=(FilteredProgressCoverLaneDummy if name.startswith('filter_progress') else ProgressCoverLaneDummy if name.startswith('progress') else FilteredCoverLaneDummy if name.startswith('filter') else QuotientCoverLaneDummy)
        if name=='filter_progress':cls=MatchedFilteredProgressCoverLaneDummy
        if name=='switching_core':cls=SwitchingQuotientCoverLaneDummy
        if name=='filter_switching_progress':cls=MatchedSwitchingProgressCoverLaneDummy
        if name.startswith('filter_slack'):cls=partial(MatchedFilteredSlackProgressCoverLaneDummy,utility_slack=int(name[-2:])/100)
        models[name]=Raw() if h==0 else cls(rn,belief_model=beliefs[h],
            budget=.24,horizon=h,k=5,rng=np.random.default_rng(0))
    rows=[]
    for f in data['families']:
        for sess in f['sessions']:
            if sess['role'] not in ('base','repeat','access_endpoint'):continue
            trace=data['traces'][sess['session_id']]
            indices=list(range(0,len(trace),20))
            points=[trace[i] for i in indices];start=points[0]['time_s']
            for name,(h,head,delay) in SPECS.items():
                model=models[name]
                if h:
                    seeds=rng_from_key(sess['session_id'],schema='whole-session-boundary-v1').integers(0,2**63,size=2,dtype=np.int64)
                    model.anchor_rng,model.dummy_rng=(np.random.default_rng(int(s)) for s in seeds)
                stream=BoundaryProtectedStream(model,BoundaryPolicy(head,delay),session_start_s=start)
                events=[];recalls=[];deliveries=[];generated_times=[];actual_delays=[]
                begin=time.perf_counter()
                for p in points:
                    old=stream.generated
                    released=stream.ingest(p['time_s'],p['lat'],p['lon'])
                    if stream.generated>old:generated_times.append(p['time_s'])
                    actual_state,_=rn.nearest(p['lat'],p['lon'])
                    for e in released:
                        source_t=generated_times[len(events)]
                        actual_delays.append(p['time_s']-source_t)
                        events.append(e.to_dict())
                    states=[rn.nearest(c.lat,c.lon)[0] for e in released for c in e.candidates]
                    recall=query_recall(context,states,actual_state)
                    if recall is not None:recalls.append(recall)
                    deliveries.append(bool(released))
                stream.close(trace[-1]['time_s'])
                elapsed=(time.perf_counter()-begin)*1000
                assert stream.generated==len(events)+stream.cancelled
                public={'events':events,'public_parameters':{'head_s':head,'delay_s':delay,'nominal_horizon':h,'budget_filter':name.startswith('filter'),'tight_cap':.23 if h else None,'service_progress':'progress' in name or 'slack' in name}}
                errors={};predictions={}
                for scenario,target in [('S9',trace[0]),('S10',trace[-1])]:
                    pred=endpoint_predictions(public,scenario,rn.proj.lat0) if events else {'prior':rn.xy.mean(axis=0)}
                    # This is a modest road-snap extension, NOT a road-transition attack.
                    pred.update({a+'_road':rn.xy[int(rn.tree.query(p)[1])] for a,p in list(pred.items())})
                    true=np.array(rn.point_xy(target['lat'],target['lon']))
                    errors[scenario]={a:float(np.linalg.norm(p-true)) for a,p in pred.items()}
                    predictions[scenario]={a:p.tolist() for a,p in pred.items()}
                row={'family_id':f['family_id'],'split':f['split'],'session_id':sess['session_id'],
                    'role':sess['role'],'method':name,'input_events':len(points),'public':public,
                    'recall_all_current_queries':float(np.mean(recalls)),'delivery_fraction':float(np.mean(deliveries)),
                    'latency_mean_s':float(np.mean(actual_delays)) if actual_delays else None,
                    'budget_bound':model.spent_bound if h else None,'privacy_reads':sum(v['private_read'] for v in model.evaluator_ledger) if name.startswith('filter') else min(h,stream.generated) if h else None,
                    'postprocessing_only_events':sum(not v['private_read'] for v in model.evaluator_ledger) if name.startswith('filter') else max(0,stream.generated-h) if h else None,
                    'evaluator_ledger':model.evaluator_ledger if name.startswith('filter') else None,
                    'model_step_ms':model.step_ms if h else [],'accounting':stream.evaluator_summary(),'evaluation_and_generation_ms':elapsed,
                    'errors':errors,'predictions':predictions}
                rows.append(row)
                print(sess['session_id'],name,'recall',round(row['recall_all_current_queries'],3),'events',len(events),flush=True)
    summaries=[];selection={}
    for method in SPECS:
        train=[r for r in rows if r['method']==method and r['split']=='development_train']
        val=[r for r in rows if r['method']==method and r['split']=='development_validation']
        for scenario in ('S9','S10'):
            names=sorted(train[0]['errors'][scenario])
            mae=min(names,key=lambda a:(np.mean([r['errors'][scenario][a] for r in train]),a))
            chosen={'mae':mae}
            for radius in (50,100,200,500):
                chosen[f'hit{radius}']=min(names,key=lambda a:(-np.mean([r['errors'][scenario][a]<=radius for r in train]),np.mean([r['errors'][scenario][a] for r in train]),a))
            selection[method+'/'+scenario]=chosen
            summaries.append({'method':method,'scenario':scenario,'validation_sessions':len(val),'validation_families':2,
                'selected_mae_m':float(np.mean([r['errors'][scenario][mae] for r in val])),
                'hits':{str(q):float(np.mean([r['errors'][scenario][chosen[f'hit{q}']]<=q for r in val])) for q in (50,100,200,500)},
                'recall_all_current_queries':float(np.mean([r['recall_all_current_queries'] for r in val])),
                'delivery_fraction':float(np.mean([r['delivery_fraction'] for r in val])),
                'latency_mean_s':float(np.mean([r['latency_mean_s'] for r in val if r['latency_mean_s'] is not None]))})
    result={'schema':'whole-session-matched-switching-development-v1','resources':metadata,'dataset_sha256':sha(DATA),
        'code_sha256':sha(Path(__file__)),'scope':'new full-session defender runs; finite attack bank; development only; NOT all A/B/C or independent confirmation',
        'source_sha256':{p:sha(ROOT/p) for p in ('core/boundary_release.py','benchmark/engines/quotient_cover.py','benchmark/engines/filtered_cover.py','benchmark/engines/progress_cover.py','benchmark/engines/slack_progress.py','benchmark/engines/matched_filter.py','benchmark/engines/switching_progress.py','benchmark/engines/switching_cover.py','benchmark/switching_belief.py','docs/research/predictive_filter_argument.md','experiments/contribution_stress.py')},
        'per_session_budget':.24,'repeated_pair_composition_bound':.48,'rows':rows,'selection':selection,'summaries':summaries}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
