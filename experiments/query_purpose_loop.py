"""S7 development loop: fixed-cover retrieval and private multi-purpose ranking.

Run with --out /private/tmp/... first. Source GPS and original evidence never
change. Group splits are internal development audits, not fresh confirmation.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import gzip
import hashlib
import json
import platform

import numpy as np

from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.query_purpose import (QueryPurpose, QuerySpec, MultiPurposeRoadRanking,
                                     PurposeIndependentCoverClient)
from evaluation.live_poi import AvailabilityWorld, LivePointService
from evaluation.query_intent import wire_features, selected_attack
from experiments.public_research_resources import load_public_research_resources
from experiments.rng_util import rng_from_key

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'
SOURCE_FILES=[DATA,Path(__file__),ROOT/'benchmark/query_purpose.py',ROOT/'evaluation/query_intent.py',
              ROOT/'benchmark/engines/paced_slack.py',ROOT/'core/mechanisms.py']
DEPTHS=(10,20,40,80)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean_by_family(rows,field,**filters):
    grouped=defaultdict(list)
    for row in rows:
        if all(row.get(k)==v for k,v in filters.items()) and row[field] is not None:
            grouped[row['family_id']].append(row[field])
    return {'value':float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None,
            'families':len(grouped),'eligible_rows':sum(len(v) for v in grouped.values()),
            'family_values':{f:float(np.mean(v)) for f,v in sorted(grouped.items())}}


def evaluate_attacks(records,categories,purposes,task):
    rows=[]
    for method in ('explicit_metadata_control','category_cover_purpose_visible','full_cover'):
        subset=[r for r in records if r['method']==method]
        split={s:[r for r in subset if r['split']==s] for s in ('fit','selection','test')}
        x=[np.array([r['features'] for r in split[s]]) for s in split]
        y=[np.array([r[task] for r in split[s]]) for s in split]
        classes=sorted(set(v for label in y for v in label.tolist()))
        result=selected_attack(x[0],y[0],x[1],y[1],x[2],y[2],classes)
        rows.append({'method':method,'task':task,'chance_balanced':1./len(classes),
                     'split_counts':{s:len(v) for s,v in split.items()},**result})
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resources',type=Path,default=Path('/private/tmp/trajectory-research-20261005-public-map'))
    args=parser.parse_args();out=args.out
    if (out/'readout.json').exists():raise FileExistsError('Completed evidence is immutable; choose new output')
    out.mkdir(parents=True,exist_ok=True)
    hashes={str(p.relative_to(ROOT)):sha(p) for p in SOURCE_FILES}
    data=json.loads(DATA.read_text())
    families=sorted(f['family_id'] for f in data['families'])
    assert len(families)==12
    splits={f:('fit' if i<6 else 'selection' if i<9 else 'test') for i,f in enumerate(families)}
    records=sorted((r for r in data['records'] if r['case_id']=='S7.A'),key=lambda r:r['record_id'])
    assert len(records)==36
    # Freeze source selection, depths and decision rule before any scores.
    protocol={'schema':'query-purpose-loop-v1','created':'2026-10-05',
        'scope':'Internal development on unchanged SUMO S7 records and reconstructed public network',
        'split_by_family':splits,'records':[r['record_id'] for r in records],
        'purposes':[p.value for p in QueryPurpose],'radius_m':1000.,
        'private_destination':'Actual trace endpoint, local evaluator/device only',
        'response_depth_grid':list(DEPTHS),'selection_rule':
            'Smallest L with selection mean Recall >= .90 and each purpose mean >= .80; otherwise largest L marked gate_failed',
        'counterfactuals':'Every category x purpose on exactly the same protected stream and schedule',
        'correlated_intent':'Original synthetic intent labels, evaluated separately using public geometry and payload',
        'coordinate_policy':'Current GeoI-Slack protected coordinates; purpose never changes them',
        'attacker_bank':['knn1','knn5','knn15','ExtraTrees_leaf1','ExtraTrees_leaf3','logistic'],
        'attacker_selection':'Max balanced accuracy then macro-F1 on family-disjoint selection only',
        'traffic':'Request + response JSON with public OSM IDs/lat/lon/category, excludes HTTP/TLS',
        'source_sha256':hashes,'python':platform.python_version()}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rn,_,_,_,belief,ranking,metadata=load_public_research_resources(args.resources)
    local=MultiPurposeRoadRanking(ranking)
    model=PacedSlackProgressLaneDummy(rn,belief_model=belief,k=5,budget=.24,horizon=12,
                                      utility_slack=.03,rng=np.random.default_rng(31005))
    world=AvailabilityWorld(ranking.n,seed=31005,probability=.8)
    servers={L:LivePointService(ranking,world,response_l=L) for L in DEPTHS}
    utility=[];attack=[];correlated=[];streams=[];map_errors=[]
    categories=list(ranking.categories);purposes=[p.value for p in QueryPurpose]
    for record in records:
        sid=record['session_ids'][0];trace=data['traces'][sid]
        indices=record['observed_indices'][0];family=record['family_id'];split=splits[family]
        seeds=rng_from_key(record['record_id'],schema='s7-query-purpose-loop-v1').integers(0,2**63,2)
        model.anchor_rng,model.dummy_rng=[np.random.default_rng(int(s)) for s in seeds]
        model.reset();clients={L:PurposeIndependentCoverClient(categories,ranking.n,response_l=L) for L in DEPTHS}
        start=trace[indices[0]]['time_s'];destination=rn.nearest(trace[-1]['lat'],trace[-1]['lon'])[0]
        public=[]
        for ordinal,index in enumerate(indices):
            point=trace[index];t=point['time_s']-start
            coords=model.protect_step(point['lat'],point['lon'],t)
            state,error=rn.nearest(point['lat'],point['lon']);map_errors.append(error)
            available=world.at_epoch(world.epoch(t))
            deep_results={}
            for L,client in clients.items():
                def server(request):
                    assert request['categories']==categories
                    return servers[L].query(rn.nearest(*request['coordinate'])[0],request['epoch'])
                result=client.step(t,coords,server);deep_results[L]=result
                wire_replies=[[[{k:ranking.pois[v][k] for k in ('id','lat','lon','category')} for v in cat]
                               for cat in reply] for reply in result['replies']]
                traffic=len(json.dumps({'requests':result['requests'],'replies':wire_replies},
                                       separators=(',',':'),sort_keys=True).encode())
                for category in categories:
                    for purpose in QueryPurpose:
                        query=QuerySpec(purpose,category,radius_m=1000. if purpose==QueryPurpose.WITHIN_RADIUS else None,
                                        destination_state=destination if purpose==QueryPurpose.MIN_DETOUR else None)
                        reference=local.top(state,available,query);returned=local.top(state,result['known'],query)
                        recall=len(set(reference)&set(returned))/len(reference) if reference else None
                        utility.append({'family_id':family,'split':split,'record_id':record['record_id'],
                            'time_s':t,'L':L,'purpose':purpose.value,'category':category,'recall':recall,
                            'reference':reference,'returned':returned,'traffic_bytes':traffic,
                            'unavailable_returned':sum(not available[v] for v in returned)})
            if ordinal in {0,len(indices)//2,len(indices)-1}:
                base=deep_results[10]
                replies=[[[{k:ranking.pois[v][k] for k in ('id','lat','lon','category')} for v in cat]
                          for cat in reply] for reply in base['replies']]
                # Identical retrieval for all counterfactual private purposes.
                wirehash=hashlib.sha256(json.dumps((base['requests'],replies),sort_keys=True).encode()).hexdigest()
                for ci,category in enumerate(categories):
                    for pi,purpose in enumerate(QueryPurpose):
                        for method in ('explicit_metadata_control','category_cover_purpose_visible','full_cover'):
                            requests=[dict(q) for q in base['requests']]
                            if method!='full_cover':
                                for q in requests:q['purpose']=purpose.value
                            if method=='explicit_metadata_control':
                                for q in requests:q['category']=category;q['categories']=[category]
                            feature=wire_features(requests,replies,categories,purposes).tolist()
                            attack.append({'method':method,'family_id':family,'split':split,
                                'record_id':record['record_id'],'time_s':t,'category_label':ci,
                                'purpose_label':pi,'joint_label':ci*len(purposes)+pi,'features':feature,
                                'counterfactual_full_cover_wire_sha256':wirehash})
                for method in ('explicit_metadata_control','full_cover'):
                    requests=[dict(q) for q in base['requests']]
                    if method=='explicit_metadata_control':
                        for q in requests:q['purpose']='nearest_distance';q['categories']=record['labels']['true_queries']
                    correlated.append({'method':method,'family_id':family,'split':split,
                        'intent_label':record['labels']['intent'],'features':wire_features(requests,replies,categories,purposes).tolist()})
            public.append({'timestamp_s':t,'coordinates':[list(c) for c in coords]})
        streams.append({'record_id_evaluator_only':record['record_id'],'family_id_evaluator_only':family,
                        'events':public,'spent_bound':model.spent_bound})
        print('S7 generated/scored',record['record_id'],len(utility),'utility rows',flush=True)
    all_depths=[]
    for L in DEPTHS:
        by_purpose={p:mean_by_family(utility,'recall',split='selection',L=L,purpose=p)['value'] for p in purposes}
        score=mean_by_family(utility,'recall',split='selection',L=L)
        all_depths.append({'L':L,'selection_recall':score,'by_purpose':by_purpose,
            'passed':score['value'] is not None and score['value']>=.90-1e-12 and
                     all(v is not None and v>=.80-1e-12 for v in by_purpose.values())})
    selected=next((r['L'] for r in all_depths if r['passed']),DEPTHS[-1])
    selection={'selected_L':selected,'gate_passed':any(r['passed'] for r in all_depths),
               'all_development_depths':all_depths,'test_used_for_selection':False}
    (out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    attacks=[]
    for task in ('category_label','purpose_label','joint_label'):
        attacks.extend(evaluate_attacks(attack,categories,purposes,task))
    correlated_results=[]
    for method in ('explicit_metadata_control','full_cover'):
        subset=[r for r in correlated if r['method']==method]
        splits2={s:[r for r in subset if r['split']==s] for s in ('fit','selection','test')}
        xs=[np.array([r['features'] for r in splits2[s]]) for s in splits2]
        ys=[np.array([r['intent_label'] for r in splits2[s]]) for s in splits2]
        labels=sorted(set(v for a in ys for v in a.tolist()))
        correlated_results.append({'method':method,'chance_balanced':1/len(labels),
            **selected_attack(xs[0],ys[0],xs[1],ys[1],xs[2],ys[2],labels)})
    # Label permutation is a diagnostic control, never used to fit the defense.
    full=[r for r in attack if r['method']=='full_cover'];grouped=defaultdict(list)
    for r in full:grouped[(r['record_id'],r['time_s'])].append(r)
    noninterference=all(len({tuple(r['features']) for r in group})==1 for group in grouped.values())
    assert noninterference and all(r['unavailable_returned']==0 for r in utility)
    test={p:mean_by_family(utility,'recall',split='test',L=selected,purpose=p) for p in purposes}
    traffic=mean_by_family(utility,'traffic_bytes',split='test',L=selected)
    readout={'schema':'query-purpose-loop-readout-v1','scope':protocol['scope'],
        'resources':metadata,'selection':selection,'test_recall_by_purpose':test,
        'test_recall_all':mean_by_family(utility,'recall',split='test',L=selected),
        'test_traffic_bytes_per_event':traffic,'test_L10_recall':mean_by_family(utility,'recall',split='test',L=10),
        'counterfactual_attacks':attacks,'correlated_synthetic_intent_attacks':correlated_results,
        'wire_noninterference_checked':noninterference,'map_access_error_m':
            {'median':float(np.median(map_errors)),'p95':float(np.percentile(map_errors,95)),'max':float(max(map_errors))},
        'unavailable_returned_items':0,'row_counts':{'utility':len(utility),'counterfactual':len(attack),'correlated':len(correlated)},
        'limitations':['Internal family-disjoint development audit, not untouched external holdout',
            'Reconstructed turns and constant speed: fastest and nearest may coincide in this map',
            'No price/rating/opening-hour metadata: those purposes unsupported, not fabricated',
            'Purpose-independent request does not hide intent correlated with trajectory or activation',
            'Counterfactual metadata-positive-control is not a paper-method reproduction'],
        'source_hashes_unchanged':all(sha(ROOT/name)==h for name,h in hashes.items())}
    assert readout['source_hashes_unchanged']
    (out/'readout.json').write_text(json.dumps(readout,indent=2,allow_nan=False)+'\n')
    for name,value in [('utility_rows',utility),('attack_rows',attack),('correlated_rows',correlated),('public_streams',streams)]:
        (out/(name+'.json.gz')).write_bytes(gzip.compress(json.dumps(value,separators=(',',':')).encode(),mtime=0))
    print(json.dumps({'selected_L':selected,'gate':selection['gate_passed'],'test':readout['test_recall_all']['value'],
                      'traffic':traffic['value'],'wire_equal':noninterference}),flush=True)


if __name__=='__main__':main()
