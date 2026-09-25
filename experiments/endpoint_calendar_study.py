"""Endpoint attack upgrade plus a calendar-driven public POI client.

Historical datasets are used only for fit/selection. Eight predeclared new
families are evaluated once after selection is sealed. Original evidence is
never overwritten; no defense parameters are chosen on the held-out result.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from benchmark.scheduled_category_client import ScheduledCategoryClient
from benchmark.paper_comparators import PublicHistory,generate
from core.demo_protocol import TrajectoryPoint
from evaluation.live_comparison_attacks import public_view,features_and_geometry,LearnedAttack
from evaluation.live_comparison_endpoint_attacks import extrapolations
from evaluation.road_endpoint_attack import RoadEndpointAttack
from evaluation.live_poi import RankedRoadPois,LivePointService,AvailabilityWorld,score_returned
from experiments.build_endpoint_holdout import OUT,DATA
from experiments.research_loop_resources import ROOT,CACHE,load,sha
from experiments.research_loop_cases import target_xy,mean_optional
from experiments.research_loop_category_confirmation import sources
from experiments.run_live_paper_comparison import resources,DATASETS,write_json,score_execution,encoded
from experiments.rng_util import rng_from_key

OLD=ROOT/'artifacts/benchmarks/live_paper_comparison_v1'
METHODS=('raw','dls','rdg','transprotect_markov','semantic_poi','fake_queries','anotherme_offline','ours30','ours67','calendar30','calendar67')
CASES=tuple(f'S{i}.{c}' for i in (9,10) for c in 'ABC')
FIT={f'family-{i}' for i in (501,502,*range(701,709))}
SELECT={f'family-{i}' for i in (601,602,*range(709,713),*range(901,905))}


def gzwrite(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),mtime=0))


def gzread(path):return json.loads(gzip.decompress(path.read_bytes()))


def protocol():
    p=json.loads((OUT/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items():
        if path=='experiments/build_endpoint_holdout.py' and (OUT/'construction_amendment.json').exists():
            amendment=json.loads((OUT/'construction_amendment.json').read_text())
            assert amendment['original_builder_sha256']==digest
            assert amendment['protocol_sha256']==sha(OUT/'protocol.json')
            digest=amendment['revised_builder_sha256']
        assert sha(ROOT/path)==digest,path
    return p


def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Protocol already sealed')
    files=['benchmark/scheduled_category_client.py','evaluation/road_endpoint_attack.py',
        'evaluation/live_comparison_attacks.py','evaluation/live_comparison_endpoint_attacks.py',
        'benchmark/paper_comparators.py','experiments/build_endpoint_holdout.py',
        'data/scenario_suite/mobility.py','data/scenario_suite/records.py',
        'data/scenario_suite_v2/design.py','data/scenario_suite_v2/records.py',
        'artifacts/benchmarks/research_loop/iteration29_plans.json',
        'artifacts/benchmarks/live_paper_comparison_v1/protocol.json']
    p={'schema':'endpoint-calendar-protocol-v1','created':'2026-09-25','methods':METHODS,
       'source_sha256':{f:sha(ROOT/f) for f in files},'holdout_builder_sha256':sha(ROOT/'experiments/build_endpoint_holdout.py'),
       'fit_families':sorted(FIT),'selection_families':sorted(SELECT),'holdout_seeds':list(range(1101,1109)),
       'cases':CASES,'world_seeds':[4101,4102,4103],'probabilities':[.5,.8,.95],'mechanism_reps':[0,1],
       'calendar':{'start_s':0,'end_s':3600,'epoch_s':60,'active_start_range_s':[300,900],
           'activation':'public subscription chosen before trips; no on-demand activation or cancellation',
           'local_queries':'read current epoch cache only; never trigger network traffic'},
       'attack':'old geometric/learned bank plus road-distance endpoint posterior; same selection rule for all methods',
       'new_attack_grid':{'cell_m':100,'A_C_horizon_s':[30,60,90],'B_horizon_s':[60,180,360,600],
           'prior':'90% uniform road cells + 10% uniform bounding grid','posterior_prior_mix':.1,
           'uncertainty_m':'max(80,0.35*speed*horizon)','lane_changes':'public adjacent lanes within 25 m'},
       'selection_rule':'case/family balanced; minimize MAE or maximize hit with MAE tie-break; posterior minimize NLL',
       'endpoints':'first/last simulated FCD, not inferred real homes',
       'primary_source':'https://doi.org/10.1145/3548606.3560616',
       'scope':'new-family same-city check; not six original-paper reproduction or real-world confirmation'}
    write_json(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    print('Protocol sealed before generating or inspecting new families',flush=True)


@lru_cache(maxsize=4)
def dataset(split):
    path=DATA/'dataset.json' if split=='holdout' else ROOT/f'artifacts/datasets/{DATASETS[split]}/dataset.json'
    return json.loads(path.read_text())


@lru_cache(maxsize=2)
def plan_for(method):
    plans=json.loads((ROOT/'artifacts/benchmarks/research_loop/iteration29_plans.json').read_text())['plans']
    return plans['public_category_budget30' if method.endswith('30') else 'public_category_full_cover']


def execution(split,method,sid,rep):
    if method.startswith('calendar'):
        plan=plan_for(method);queries=[[q['category_index'],*q['coordinate']] for q in plan['queries']]
        return {'status':'ok','events':[{'timestamp_s':t,'queries':queries} for t in range(0,3600,60)]}
    base=OUT if split=='holdout' else OLD
    return gzread(base/'transcripts'/split/method/f'{sid}_r{rep}.json.gz')


def generate_holdout():
    p=protocol();rn,service,history,shared,setup=resources();data=dataset('holdout');ss,_=sources(data,rn)
    # Include all five scenarios in the common session clock, even though this
    # study's attack target is S9/S10. Avoid resetting models at each case.
    for method in METHODS:
        if method.startswith('calendar'):continue
        for source in ss:
            sid=source['session_id'];ids=source['clock_indices']
            points=tuple(TrajectoryPoint(t,*rn.latlon(s)) for t,s in zip(source['timestamps_s'],source['reference_states']))
            for rep in p['mechanism_reps']:
                path=OUT/'transcripts/holdout'/method/f'{sid}_r{rep}.json.gz'
                if path.exists():continue
                seed=int(rng_from_key(method,sid,rep,schema='endpoint-calendar-holdout-v1').integers(0,2**31))
                try:
                    if method.startswith('ours'):
                        q=[[x['category_index'],*x['coordinate']] for x in plan_for(method)['queries']]
                        ex={'events':[{'timestamp_s':t,'queries':q} for t in source['timestamps_s']],
                            'service_event_positions':dict(enumerate(range(len(ids)))),'generation_ms':None}
                    else:ex=generate(method,points,rn,history,shared,seed)
                    for e in ex['events']:
                        xy=e.get('coordinates') or [q[1:] for q in e['queries']]
                        e['server_states']=[rn.nearest(*v)[0] for v in xy]
                    ex['status']='ok'
                except (ValueError,RuntimeError,IndexError) as error:ex={'status':'failed','error':f'{type(error).__name__}: {error}'}
                ex.update(method=method,session_id=sid,rep=rep,seed=seed,clock_indices=ids,protocol_sha256=sha(OUT/'protocol.json'))
                gzwrite(path,ex)
        print('Generated holdout',method,flush=True)


def history_for(rn):
    d=dataset('auxiliary');sids={s['session_id'] for f in d['families'] if f['family_id'] in ('family-501','family-502') for s in f['sessions']}
    return PublicHistory(rn,[[rn.nearest(p['lat'],p['lon'])[0] for p in d['traces'][sid][::20]] for sid in sorted(sids)])


def case_inputs(split,method,rn,history):
    d=dataset(split)
    for record in d['records']:
        if record['case_id'] not in CASES:continue
        for rep in (0,1):
            views=[];valid=True
            for slot,(sid,indices) in enumerate(zip(record['session_ids'],record['observed_indices'])):
                ex=execution(split,method,sid,rep)
                if ex['status']!='ok':valid=False;break
                if method.startswith('calendar'):
                    events=[{'timestamp_s':e['timestamp_s'],'coordinates':[q[1:] for q in e['queries']]} for e in ex['events']]
                else:events,_=public_view(ex,indices)
                x,bank=features_and_geometry(events,record['scenario'],rn,history)
                bank.update(extrapolations(events,record['scenario'],rn,history))
                views.append({'events':events,'x':x,'bank':bank,'truth':target_xy(record,slot,d['traces'][sid],rn)})
            yield {k:record[k] for k in ('record_id','family_id','case_id','scenario')}|{
                'method':method,'rep':rep,'split':split,'status':'ok' if valid else 'failed','views':views if valid else []}


def family_mean(rows,key):
    records=defaultdict(list)
    for row in rows:
        if row.get(key) is not None:records[row['family_id'],row['record_id']].append(row[key])
    groups=defaultdict(list)
    for (family,_),values in records.items():groups[family].append(np.mean(values))
    return mean_optional([float(np.mean(v)) for v in groups.values()])


def score_row(row,learner,attacker,posterior_cache):
    if row['status']!='ok':return {k:v for k,v in row.items() if k!='views'}
    banks=[];densities=[];truth=np.concatenate([v['truth'] for v in row['views']])
    for view in row['views']:
        bank=dict(view['bank'])
        if learner is not None:bank.update(learner.predict(view['x']))
        bank.update({name+'_road':attacker.rn.xy[attacker.rn.tree.query(xy)[1]] for name,xy in list(bank.items())})
        key=row['case_id'],hashlib.sha256(json.dumps(view['events'],separators=(',',':')).encode()).hexdigest()
        if key not in posterior_cache:
            if len(posterior_cache)>64:posterior_cache.clear()
            posterior_cache[key]=attacker.posteriors(view['events'],row['case_id'])
        density=posterior_cache[key];densities.append(density)
        for name,p in density.items():
            for decoder,xy in attacker.estimates(p).items():bank[name+'_'+decoder]=xy[None,:]
        banks.append(bank)
    pred={a:np.concatenate([b[a] for b in banks]) for a in banks[0]}
    if row['case_id'].endswith('.C'):
        pred.update({a+'_joint_mean':np.repeat(xy.mean(axis=0,keepdims=True),len(truth),axis=0) for a,xy in list(pred.items())})
        for name in densities[0]:
            p=attacker.joint([d[name] for d in densities])
            for decoder,xy in attacker.estimates(p).items():pred[name+'_joint_'+decoder]=np.repeat(xy[None,:],len(truth),axis=0)
    metrics={}
    for name in densities[0]:
        values=[attacker.distribution_metrics(d[name],view['truth']) for d,view in zip(densities,row['views'])]
        metrics[name]={k:float(np.mean([v[k] for v in values])) for k in values[0]}
        if row['case_id'].endswith('.C'):
            metrics[name+'_joint']=attacker.distribution_metrics(attacker.joint([d[name] for d in densities]),truth)
    return {k:v for k,v in row.items() if k!='views'}|{'target_xy_evaluator_only':truth.tolist(),
        'errors':{a:np.linalg.norm(xy-truth,axis=1).tolist() for a,xy in pred.items()},'posterior_metrics':metrics}


def select_case(rows):
    good=[r for r in rows if r['status']=='ok']
    if not good:return None
    names=sorted(set.intersection(*(set(r['errors']) for r in good)));selected={}
    def mean(values):return family_mean([dict(r,value=v) for r,v in zip(good,values)],'value')
    maes={a:mean([float(np.mean(r['errors'][a])) for r in good]) for a in names}
    selected['mae']=min(names,key=lambda a:(maes[a],a))
    for radius in (50,100,200,500):
        scores={a:mean([float(np.mean(np.array(r['errors'][a])<=radius)) for r in good]) for a in names}
        selected[f'hit{radius}']=min(names,key=lambda a:(-scores[a],maes[a],a))
    names=sorted(good[0]['posterior_metrics'])
    selected['posterior']=min(names,key=lambda a:(mean([r['posterior_metrics'][a]['nll_bits'] for r in good]),a))
    return selected


def fit_select():
    protocol();dst=OUT/'selection.json'
    if dst.exists():raise FileExistsError('Selection already sealed')
    rn,*_=load();history=history_for(rn);attacker=RoadEndpointAttack(rn,history)
    learners={};rows=[];selections={};fitting=[]
    for method in METHODS:
        inputs=[r for split in ('auxiliary','development','new_groups') for r in case_inputs(split,method,rn,history)]
        for scenario in ('S9','S10'):
            fit=[r for r in inputs if r['family_id'] in FIT and r['scenario']==scenario and r['status']=='ok']
            views=[v for r in fit for v in r['views']]
            learners[method,scenario]=LearnedAttack(np.concatenate([v['x'] for v in views]),np.concatenate([v['truth'] for v in views])) if views else None
            fitting.append({'method':method,'scenario':scenario,'families':sorted({r['family_id'] for r in fit})})
        cache={}
        scored=[score_row(r,learners[method,r['scenario']],attacker,cache) for r in inputs if r['family_id'] in SELECT]
        rows.extend(scored)
        for case in CASES:selections[method+'/'+case]=select_case([r for r in scored if r['case_id']==case])
        print('Selected',method,flush=True)
    models=CACHE/'endpoint_calendar_attackers.pkl'
    with models.open('wb') as f:pickle.dump(learners,f)
    gzwrite(OUT/'selection_rows.json.gz',rows)
    write_json(dst,{'protocol_sha256':sha(OUT/'protocol.json'),'runner_sha256':sha(Path(__file__)),
        'model_path':str(models.relative_to(ROOT)),'model_sha256':sha(models),'fit':fitting,
        'selected':selections,'selection_rows_sha256':sha(OUT/'selection_rows.json.gz'),
        'input_dataset_sha256':{s:sha(ROOT/f'artifacts/datasets/{DATASETS[s]}/dataset.json') for s in ('auxiliary','development','new_groups')},
        'holdout_not_read_by_fit_select':True})


def evaluate_attacks():
    protocol();selection=json.loads((OUT/'selection.json').read_text());dst=OUT/'holdout_privacy.json.gz'
    if dst.exists():raise FileExistsError('Retain the first held-out result')
    assert selection['runner_sha256']==sha(Path(__file__))
    models=ROOT/selection['model_path'];assert sha(models)==selection['model_sha256']
    with models.open('rb') as f:learners=pickle.load(f) # locally created and verified
    rn,*_=load();history=history_for(rn);attacker=RoadEndpointAttack(rn,history);rows=[]
    for method in METHODS:
        cache={}
        for row in case_inputs('holdout',method,rn,history):
            r=score_row(row,learners[method,row['scenario']],attacker,cache)
            chosen=selection['selected'][method+'/'+row['case_id']]
            if r['status']=='ok' and chosen is not None:
                r['selected']=chosen;errors=np.array(r['errors'][chosen['mae']])
                r.update(mae_m=float(errors.mean()),median_m=float(np.median(errors)),p90_m=float(np.quantile(errors,.9)),
                    **{f'hit{k}':float(np.mean(np.array(r['errors'][chosen[f'hit{k}']])<=k)) for k in (50,100,200,500)},
                    **r['posterior_metrics'][chosen['posterior']])
            elif r['status']=='ok':r['status']='no_calibrated_attacker'
            rows.append(r)
        print('Evaluated endpoints',method,flush=True)
    gzwrite(dst,{'rows':rows,'selection_sha256':sha(OUT/'selection.json'),'dataset_sha256':sha(DATA/'dataset.json'),
                'runner_sha256':sha(Path(__file__))})


def scheduled_service(method,source,server,ranking,rn,phase):
    c=ScheduledCategoryClient(plan_for(method),ranking.n)
    states={tuple(q['coordinate']):rn.nearest(*q['coordinate'])[0] for q in plan_for(method)['queries']}
    clocks=source['timestamps_s'];utility={};request_bytes=response_bytes=0
    for t in range(0,3600,60):
        result=c.tick(t,lambda q:server.query(states[q['coordinate']],q['epoch'])[q['category_index']])
        queries=[[q['category_index'],*q['coordinate']] for q in result['requests']]
        request_bytes+=encoded({'time_s':t,'epoch':result['epoch'],'queries':queries})
        response_bytes+=encoded({'epoch':result['epoch'],'results':result['replies']})
        for i,clock in enumerate(clocks):
            now=phase+clock
            if t<=now<t+60:
                known=c.local_snapshot(now);available=server.world.at_epoch(result['epoch'])
                state=source['reference_states'][i]
                utility[source['clock_indices'][i]]=score_returned(ranking.top(state,available,5),ranking.top(state,known,5),available)
    # An out-of-subscription service request is a failure, never silently cut.
    for i in source['clock_indices']:utility.setdefault(i,{'recall':0.,'outside_public_interval':True})
    return utility,{'request_bytes':request_bytes,'response_bytes':response_bytes,'service_events':len(clocks),
                    'emissions':60,'outside_interval_events':sum(v.get('outside_public_interval',False) for v in utility.values())}


def evaluate_service():
    p=protocol();dst=OUT/'holdout_service.json'
    if dst.exists():raise FileExistsError(dst)
    rn,service,*_=load();ranking=RankedRoadPois(service,CACHE/'live_poi_full_rank_v1.npy')
    data=dataset('holdout');ss,records=sources(data,rn);rows=[];costs=[]
    # All five scenario cases retain a utility check, including S1/S2/S3.
    phases={s['session_id']:int(rng_from_key(s['session_id'],schema='public-window-private-phase-v1').integers(300,901)) for s in ss}
    for method in METHODS:
        for prob in p['probabilities']:
            for seed in p['world_seeds']:
                server=LivePointService(ranking,AvailabilityWorld(ranking.n,seed,prob,60),10);lookup={}
                for source in ss:
                    sid=source['session_id'];phase=phases[sid]
                    for rep in (0,1):
                        ex=execution('holdout',method,sid,rep)
                        if ex['status']!='ok':continue
                        if method.startswith('calendar'):result=scheduled_service(method,source,server,ranking,rn,phase)
                        else:
                            # Same absolute service epochs for every method.
                            ex={**ex,'events':[dict(e,timestamp_s=e['timestamp_s']+phase) for e in ex['events']]}
                            result=score_execution(ex,source,server,ranking)
                        utility,count=result;lookup[sid,rep]=utility
                        if prob==.8:costs.append(dict(method=method,rep=rep,world_seed=seed,session_id=sid,family_id=source['family_id'],**count))
                for record in records:
                    for rep in (0,1):
                        values=[];valid=True
                        for sid,ids in zip(record['session_ids'],record['observed_indices']):
                            if (sid,rep) in lookup:values.extend(lookup[sid,rep][i]['recall'] for i in ids)
                            else:
                                valid=False;trace=data['traces'][sid]
                                for i in ids:
                                    epoch=int((phases[sid]+trace[i]['time_s']-trace[0]['time_s'])//60)
                                    ref=ranking.top(rn.nearest(trace[i]['lat'],trace[i]['lon'])[0],server.world.at_epoch(epoch),5)
                                    values.append(0. if any(ref) else None)
                        rows.append({k:record[k] for k in ('record_id','family_id','case_id')}|{
                            'method':method,'rep':rep,'world_seed':seed,'probability':prob,'status':'ok' if valid else 'failed',
                            'recall':mean_optional(values),'eligible_events':sum(v is not None for v in values)})
        print('Service',method,flush=True)
    write_json(dst,{'rows':rows,'costs':costs,'phases_evaluator_only':phases,'dataset_sha256':sha(DATA/'dataset.json'),
                   'protocol_sha256':sha(OUT/'protocol.json'),'runner_sha256':sha(Path(__file__))})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','generate','select','attacks','service']);a=p.parse_args()
    {'prepare':prepare,'generate':generate_holdout,'select':fit_select,'attacks':evaluate_attacks,'service':evaluate_service}[a.stage]()
