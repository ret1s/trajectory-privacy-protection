"""Independent raw-row verification of fresh-family switching evaluation."""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import dijkstra
from sklearn.ensemble import ExtraTreesRegressor

from core.demo_protocol import TrajectoryPoint
from evaluation.expanded_shadow import TREE_PARAMS
from evaluation.lane_travel import matrix
from experiments.run_coverage_frontier import prepare, DEPTHS
from experiments.run_fresh_switching import generate, OUTPUT, METHODS, PARENT, PATHS, fresh_records
from experiments.publish_fresh_scenarios import FOLDER
from experiments.run_contextual_lane import estimators
from experiments.run_service_cover import ROOT,read,write,sha
from experiments.verify_service_cover import shadow_features,near
from experiments.verify_prior_factors import loss_oracle
from experiments.rng_util import rng_from_key


def equal(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): equal(x,y)
    elif isinstance(a,(float,np.floating)):
        near(a,b)
    else: assert a==b,(a,b)


def empirical(model,x,prefix):
    tx,ty=np.asarray(model['x']),np.asarray(model['y'])
    mean,scale=np.asarray(model['mean']),np.asarray(model['scale'])
    order=np.argsort(cdist((x-mean)/scale,(tx-mean)/scale),axis=1,kind='stable')
    result={prefix+f'knn_{k}':ty[order[:,:k]].mean(axis=1) for k in (1,5,15)}
    for n in (15,45):
        decisions=[loss_oracle(ty[ids[:n]]) for ids in order]
        for key in ('mae_action','hit_action'):
            result[prefix+f'shadow_loss_{key}_{n}']=np.array([d[key] for d in decisions])
        if n==45: result[prefix+'shadow_mean_45']=np.array([d['mean'] for d in decisions])
    if prefix=='':
        for n in (1,5,15): result[f'shadow_knn_{n}']=result.pop(f'knn_{n}')
    return result


def service_oracle(service):
    @lru_cache(maxsize=None)
    def replies_at(lat,lon,category,depth):
        distances=service.distances((lat,lon))
        return sorted([pid for pid in distances if service.by_id[pid]['category']==category],
                      key=lambda p:(distances[p],p))[:depth]

    def evaluate(public,points,depth):
        rows=[]
        for event,point in zip(public['events'],points):
            lat,lon=point['lat'],point['lon']
            for c in service.categories:
                ref=replies_at(lat,lon,c,5)
                replies=[replies_at(q['lat'],q['lon'],c,depth) for q in event['candidates']]
                pool=set(p for reply in replies for p in reply)
                distance=service.distances((lat,lon))
                returned=sorted([p for p in pool if p in distance],key=lambda p:(distance[p],p))[:5]
                complete=len(returned)==len(ref) if ref else None
                gap=(max(0.,sum(distance[p] for p in returned)/len(returned)-
                            sum(distance[p] for p in ref)/len(ref)) if complete else None)
                rows.append(dict(event_id=event['event_id'],category=c,
                    recall=len(set(ref)&set(returned))/len(ref) if ref else None,
                    complete=complete,extra_distance_m=gap,request_count=len(replies),
                    reply_items=sum(len(r) for r in replies),
                    response_id_json_bytes=len(json.dumps(replies,separators=(',',':')).encode())))
        valid=[r for r in rows if r['recall'] is not None]
        extra=[r['extra_distance_m'] for r in valid if r['extra_distance_m'] is not None]
        return dict(poi_rows=rows,poi_recall_at_5=np.mean([r['recall'] for r in valid]),
            poi_complete_rate=np.mean([r['complete'] for r in valid]),
            poi_extra_distance_m=float(np.mean(extra)) if extra else None,
            evaluable_queries=len(valid),empty_references=len(rows)-len(valid),
            requests_per_event=sum(r['request_count'] for r in rows)/len(points),
            reply_items_per_event=sum(r['reply_items'] for r in rows)/len(points),
            response_id_bytes_per_event=sum(r['response_id_json_bytes'] for r in rows)/len(points))
    return evaluate


def verify(output=OUTPUT,check_only=False):
    budget=.24
    output=Path(output)
    phases={p:read(output/f'{p}.json') for p in ('training','validation','confirmation')}
    selection=read(output/'selection.json')
    auxiliary,rn,prior,belief,service,deeper,provenance=prepare(budget)
    provenance.update(source_sha256={p:sha(ROOT/p) for p in sorted(set(provenance['source_sha256'])|set(PATHS))},
        methods=METHODS, parent_training_sha256=sha(PARENT/'training.json'),
        scope='same_city_new_family_confirmation; S1-S3_only; internal_ablations_not_SOTA')
    for payload in (*phases.values(),selection):
        for k,v in provenance.items():
            if not k.endswith('_ms'): equal(v,payload[k])
        if payload.get('phase')!='training':
            assert payload['fresh_registry_sha256']==sha(FOLDER/'registry.json')
            assert payload['fresh_verification_sha256']==sha(FOLDER/'verification.json')
            assert payload['fresh_dataset_content_sha256']==read(FOLDER/'registry.json')['content_sha256']
    for p,h in provenance['source_sha256'].items(): assert sha(ROOT/p)==h,p
    assert selection['training_sha256']==sha(output/'training.json')
    assert selection['validation_sha256']==sha(output/'validation.json')
    train=phases['training']; lookup={r['record_id']:r for r in train['records']}
    classifiers={}; counts=defaultdict(int)
    for method in METHODS:
        models=train['shadow_models'][method]
        for kind,model in models.items():
            group=[r for r in train['rows'] if r['method']==method and
                   (kind=='expanded' or not r['case_id'].startswith('AUX'))]
            x=np.concatenate([shadow_features(r['public'],rn) for r in group])
            y=np.concatenate([np.array([rn.point_xy(p['lat'],p['lon'])
                                       for p in lookup[r['record_id']]['points']]) for r in group])
            near(x,model['x']);near(y,model['y']);near(x.mean(axis=0),model['mean'])
            scale=x.std(axis=0);scale[scale<1e-12]=1;near(scale,model['scale'])
            equal(model['provenance'],{'families':sorted({r['family_id'] for r in group}),
                'row_keys':[[r['record_id'],r['replicate']] for r in group],'holdout_used':False})
            assert len(x)==(360 if kind=='core' else 3233)
            assert len({r['family_id'] for r in group})==(2 if kind=='core' else 66)
            counts['training_matrices']+=1
        m=models['expanded'];x,y=np.asarray(m['x']),np.asarray(m['y'])
        z=(x-m['mean'])/m['scale']
        info=train['forests'][method];path=ROOT/info['path'];assert sha(path)==info['sha256']
        classifiers[method]={}
        with np.load(path,allow_pickle=False) as arrays:
            for mode in ('direct','residual'):
                center=x[:,:10].reshape(-1,5,2).mean(axis=1)*1000
                forest=ExtraTreesRegressor(**TREE_PARAMS).fit(z,y if mode=='direct' else y-center)
                classifiers[method][mode]=forest
                for i,e in enumerate(forest.estimators_):
                    for key,value in [('left',e.tree_.children_left),('right',e.tree_.children_right),
                        ('feature',e.tree_.feature),('threshold',e.tree_.threshold),('value',e.tree_.value[:,:,0])]:
                        assert np.array_equal(arrays[f'{mode}/{i}/{key}'],value)
                        counts['forest_arrays']+=1
                counts['refit_forests']+=1
    oracle=service_oracle(service)
    transitions=defaultdict(set)
    for phase,payload in phases.items():
        parent=read(PARENT/'training.json')
        assert payload['records']==(parent['records'] if phase=='training' else fresh_records(phase)[0])
        records={r['record_id']:r for r in payload['records']}
        expected={(rid,m,rep) for rid,r in records.items() for m in METHODS
                  for rep in ((1,) if r['case_id'].startswith('AUX') else (1,2,3))}
        keys=[(r['record_id'],r['method'],r['replicate']) for r in payload['rows']]
        assert len(keys)==len(set(keys)) and set(keys)==expected
        families={r['family_id'] for r in records.values()}
        if phase!='training': assert families.isdisjoint({r['family_id'] for r in train['records']})
        assert payload['training_sha256']==(None if phase=='training' else sha(output/'training.json'))
        assert payload['selection_sha256']==(sha(output/'selection.json') if phase=='confirmation' else None)
        paired={}; replays=set()
        old={(r['record_id'],r['replicate'],r['method']):r for r in parent['rows']}
        for index,row in enumerate(payload['rows']):
            record=records[row['record_id']];method=row['method'];public=row['public']
            points=record['points'];n=len(points)
            assert all(row[k]==record[k] for k in ('case_id','family_id','split'))
            aux=record['case_id'].startswith('AUX')
            seed=int((rng_from_key(record['record_id'],schema='expanded-shadow-v1') if aux else
                rng_from_key(record['record_id'],5,row['replicate'],schema='service-cover-row-v1')).integers(0,2**31))
            assert row['rng_seed']==seed
            assert set(public)=={'mechanism','output_kind','public_parameters','events'}
            assert public['output_kind']=='dummy_only' and public['public_parameters']['budget_per_m']==budget
            assert n==len(public['events'])==len(row['evaluator_states'])==len(row['evaluator_anchors'])
            assert n==len(row['step_ms']) and min(row['step_ms'])>=0
            near(row['spent_bound'],budget/24+(min(n,12)-1)*budget/12)
            key=record['record_id'],row['replicate']
            if key in paired: assert paired[key]==row['evaluator_anchors']
            paired[key]=row['evaluator_anchors']
            for t,(event,p,states) in enumerate(zip(public['events'],points,row['evaluator_states'])):
                assert set(event)=={'event_id','timestamp_s','candidates'}
                assert event['timestamp_s']==p['timestamp_s'] and len(event['candidates'])==5
                assert all(set(q)=={'candidate_id','lat','lon'} for q in event['candidates'])
                assert [q['candidate_id'] for q in event['candidates']]==[f'candidate_{j:04d}' for j in range(5)]
                assert [rn.latlon(s) for s in states]==[(q['lat'],q['lon']) for q in event['candidates']]
                if t:
                    dt=event['timestamp_s']-public['events'][t-1]['timestamp_s']
                    for a,b in zip(row['evaluator_states'][t-1],states): transitions[a,dt].add(b)
                    counts['motion_transitions']+=5
            for objective in row['evaluator_objective']:
                if 'objective_history' in objective:
                    hist=objective['objective_history'];assert 1<=len(hist)<=4
                    assert all(b>a+1e-12 for a,b in zip(hist,hist[1:]))
                    near(hist[0],sum(objective['greedy_gains']));near(hist[-1],objective['value'])
                    counts['objective_checks']+=1
            if phase=='training' and method!='switching_exchange':
                equal(row,old[record['record_id'],row['replicate'],method])
                counts['historical_training_parity']+=1
            if phase!='training':
                x=shadow_features(public,rn);small=train['shadow_models'][method]['core']
                large=train['shadow_models'][method]['expanded']
                predictions=estimators(public,rn,prior,record['scenario'])
                predictions.update(empirical(small,x,''));predictions.update(empirical(large,x,'expanded_'))
                z=(x-large['mean'])/large['scale']
                for mode,forest in classifiers[method].items():
                    pred=forest.predict(z)
                    if mode=='residual':pred+=x[:,:10].reshape(-1,5,2).mean(axis=1)*1000
                    predictions['expanded_tree_'+mode]=pred
                assert set(predictions)==set(row['predictions'])==set(row['errors_by_attack'])
                truth=np.array([rn.point_xy(p['lat'],p['lon']) for p in points])
                for name,pred in predictions.items():
                    near(pred,row['predictions'][name])
                    # Strict event scoring of persisted public-only estimates;
                    # tiny circle-construction roundoff is not a relaxed hit radius.
                    error=np.sqrt(np.square(np.asarray(row['predictions'][name])-truth).sum(axis=1))
                    near(error,row['errors_by_attack'][name]);counts['point_predictions']+=n
                for depth in DEPTHS:
                    expected=oracle(public,points,depth)
                    equal(expected,row['utility'][str(depth)])
                    counts['category_queries']+=len(expected['poi_rows'])
                a,b=row['utility']['5'],row['utility']['10']
                assert a['poi_recall_at_5']<=b['poi_recall_at_5']+1e-12
                assert a['response_id_bytes_per_event']<=b['response_id_bytes_per_event']
                counts['scored_runs']+=1
                replay_key=method,record['case_id']
                if row['replicate']==1 and replay_key not in replays:
                    pts=tuple(TrajectoryPoint(**p) for p in points)
                    replay=generate(method,pts,rn,belief,seed)
                    for key in ('public','evaluator_anchors','evaluator_states','evaluator_objective','spent_bound'):
                        equal(replay[key],row[key])
                    short=generate(method,pts[:max(1,n//2)],rn,belief,seed)
                    equal(short['public']['events'],public['events'][:max(1,n//2)])
                    replays.add(replay_key);counts['full_replays']+=1;counts['prefix_replays']+=1
            counts['runs']+=1;counts['events']+=n
            if (index+1)%100==0:print(f'Verify B={budget:.2f} {phase}: {index+1}/{len(payload["rows"])}',flush=True)
        if phase=='training':continue
        for summary in payload['summaries']:
            group=[r for r in payload['rows'] if r['method']==summary['method'] and r['case_id']==summary['case_id']]
            assert len(group)==summary['rows']
            assert len({r['family_id'] for r in group})==summary['families']
            assert summary['families'] in (5,6)
            for name in summary['mae_by_attack']:
                near(summary['mae_by_attack'][name],np.mean([np.mean(r['errors_by_attack'][name]) for r in group]))
                near(summary['hit_by_attack'][name],np.mean([np.mean(np.asarray(r['errors_by_attack'][name])<=100) for r in group]))
            near(summary['envelope_hit100'],max(summary['hit_by_attack'].values()))
            near(summary['envelope_mae_m'],min(summary['mae_by_attack'].values()))
            near(summary['step_mean_ms'],np.mean([np.mean(r['step_ms']) for r in group]))
            near(summary['step_p95_ms'],np.percentile(np.concatenate([r['step_ms'] for r in group]),95))
            for depth in map(str,DEPTHS):
                for key,value in summary['utility'][depth].items():
                    if key=='by_category':
                        for category,score in value.items():
                            near(score,np.mean([np.mean([q['recall'] for q in r['utility'][depth]['poi_rows']
                                if q['category']==category and q['recall'] is not None]) for r in group
                                if any(q['category']==category and q['recall'] is not None for q in r['utility'][depth]['poi_rows'])]))
                    else:near(value,np.mean([r['utility'][depth][key] for r in group]))
            val=next(s for s in phases['validation']['summaries'] if s['method']==summary['method'] and s['case_id']==summary['case_id'])
            choice={'mae':min(val['mae_by_attack'],key=lambda a:(val['mae_by_attack'][a],a)),
                    'hit':min(val['hit_by_attack'],key=lambda a:(-val['hit_by_attack'][a],a))}
            assert selection['attackers'][summary['method']+'/'+summary['case_id']]==choice
            if phase=='confirmation':
                assert summary['selected_hit_attack']==choice['hit'] and summary['selected_mae_attack']==choice['mae']
                near(summary['selected_hit100'],summary['hit_by_attack'][choice['hit']])
                near(summary['selected_mae_m'],summary['mae_by_attack'][choice['mae']])
    for depth in map(str,DEPTHS):
        candidates=[]
        for method in METHODS:
            group=[s for s in phases['validation']['summaries'] if s['method']==method]
            candidates.append(dict(method=method,min_case_recall=min(s['utility'][depth]['poi_recall_at_5'] for s in group),
                                   macro_hit100=float(np.mean([s['envelope_hit100'] for s in group]))))
        feasible=[c for c in candidates if round(c['min_case_recall'],12)>=.9]
        chosen=min(feasible,key=lambda c:(round(c['macro_hit100'],12),c['method'])) if feasible else None
        equal(selection['method_selection_by_depth'][depth],dict(candidates=candidates,utility_feasible=bool(feasible),chosen=chosen))
    travel=matrix(rn,time=True)
    for (start,dt),ends in transitions.items():
        distances=dijkstra(travel,directed=True,indices=start,limit=dt+1e-8)
        assert all(np.isfinite(distances[end]) for end in ends)
    result={'verified':True,'budget':budget,'counts':dict(counts),
        'phase_sha256':{p:sha(output/f'{p}.json') for p in phases},
        'selection_sha256':sha(output/'selection.json'),
        'verifier_sha256':sha(Path(__file__)),
        'scope':'all metrics independently reconstructed; full/prefix replay subset; fresh same-city families only'}
    if not check_only:write(output/'verification.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--check-only',action='store_true',help='Recheck without replacing the sealed receipt')
    args=p.parse_args();verify(args.output,args.check_only)
