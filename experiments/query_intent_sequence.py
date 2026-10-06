"""S7 sequence audit: same protected stream for different private query histories.

Uses original S7.C diagnostic templates. The first category is identical, so an
attacker must exploit later explicit content. Purpose augmentations are synthetic.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json

import numpy as np

from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.query_purpose import (QueryPurpose,QuerySpec,MultiPurposeRoadRanking,
                                     PurposeIndependentCoverClient)
from evaluation.live_poi import AvailabilityWorld,LivePointService
from evaluation.query_intent import wire_features,selected_attack
from experiments.public_research_resources import load_public_research_resources,sha
from experiments.query_purpose_loop import ROOT,DATA,DEPTHS,mean_by_family
from experiments.rng_util import rng_from_key


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resources',type=Path,default=Path('/private/tmp/trajectory-research-20261005-public-map'))
    args=parser.parse_args();out=args.out
    if (out/'readout.json').exists():raise FileExistsError('Preserve completed audit')
    out.mkdir(parents=True,exist_ok=True)
    data=json.loads(DATA.read_text());families=sorted(f['family_id'] for f in data['families'])
    split={f:('fit' if i<6 else 'selection' if i<9 else 'test') for i,f in enumerate(families)}
    records=[r for r in data['records'] if r['case_id']=='S7.C'];assert len(records)==36
    source_paths=[DATA,Path(__file__),ROOT/'benchmark/query_purpose.py',ROOT/'evaluation/query_intent.py',
                  ROOT/'experiments/query_purpose_loop.py',ROOT/'experiments/public_research_resources.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in source_paths}
    protocol={'schema':'s7-sequence-audit-v1','created':'2026-10-05','source_sha256':hashes,
        'split_by_family':split,'records':[r['record_id'] for r in records],
        'depths':list(DEPTHS),'selection_rule':'Smallest L with mean selection Recall >= .90; otherwise max L, gate_failed',
        'seed_key':'session_id + observed_indices, no intent/category/purpose/record_id',
        'scope':'Synthetic S7.C query templates, internal development; not human-intent evidence',
        'purpose_augmentation':{'first_query':'nearest_distance', 'later_medical':'within_radius_1000m',
                               'later_shopping':'fastest_travel','later_road_trip':'minimum_detour'},
        'attacker_observation':'Whole request/reply prefix to t=0,20,40; no evaluator truth'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rn,_,_,_,belief,ranking,metadata=load_public_research_resources(args.resources)
    model=PacedSlackProgressLaneDummy(rn,belief_model=belief,k=5,budget=.24,horizon=12,
        utility_slack=.03,rng=np.random.default_rng(31006))
    local=MultiPurposeRoadRanking(ranking);world=AvailabilityWorld(ranking.n,31006,.8)
    servers={L:LivePointService(ranking,world,response_l=L) for L in DEPTHS}
    utility=[];attacks=[];witnesses={};public_streams=[]
    purposes=[p.value for p in QueryPurpose];categories=list(ranking.categories)
    for record in records:
        sid=record['session_ids'][0];trace=data['traces'][sid];indices=record['observed_indices'][0]
        seeds=rng_from_key(sid,tuple(indices),schema='s7-shared-private-intent-v1').integers(0,2**63,2)
        model.anchor_rng,model.dummy_rng=[np.random.default_rng(int(s)) for s in seeds];model.reset()
        clients={L:PurposeIndependentCoverClient(categories,ranking.n,response_l=L) for L in DEPTHS}
        start=trace[indices[0]]['time_s'];destination=rn.nearest(trace[-1]['lat'],trace[-1]['lon'])[0]
        histories={L:{method:([],[]) for method in ('explicit_metadata_control','full_cover')} for L in DEPTHS}
        events=[]
        for j,index in enumerate(indices):
            point=trace[index];t=point['time_s']-start;coords=model.protect_step(point['lat'],point['lon'],t)
            category=record['labels']['true_queries'][j];intent=record['labels']['intent']
            purpose=QueryPurpose.NEAREST if j==0 else {'medical_visit':QueryPurpose.WITHIN_RADIUS,
                     'routine_shopping':QueryPurpose.FASTEST,'road_trip':QueryPurpose.MIN_DETOUR}[intent]
            query=QuerySpec(purpose,category,radius_m=1000. if purpose==QueryPurpose.WITHIN_RADIUS else None,
                destination_state=destination if purpose==QueryPurpose.MIN_DETOUR else None)
            state=rn.nearest(point['lat'],point['lon'])[0];available=world.at_epoch(world.epoch(t))
            reference=local.top(state,available,query)
            for L,client in clients.items():
                result=client.step(t,coords,lambda q:servers[L].query(rn.nearest(*q['coordinate'])[0],q['epoch']))
                returned=local.top(state,result['known'],query)
                utility.append({'family_id':record['family_id'],'split':split[record['family_id']],
                    'record_id':record['record_id'],'time_s':t,'L':L,'purpose':purpose.value,
                    'reference':reference,'returned':returned,
                    'recall':len(set(reference)&set(returned))/len(reference) if reference else None})
                if L in DEPTHS:
                    reply=[[[{k:ranking.pois[v][k] for k in ('id','lat','lon','category')} for v in cat]
                            for cat in point_reply] for point_reply in result['replies']]
                    for method,(requests_history,replies_history) in histories[L].items():
                        requests=[dict(q) for q in result['requests']]
                        if method=='explicit_metadata_control':
                            for q in requests:
                                q['categories']=[category];q['purpose']=purpose.value
                                if query.radius_m is not None:q['radius_m']=query.radius_m
                                if query.destination_state is not None:q['destination']=rn.latlon(query.destination_state)
                        requests_history.extend(requests);replies_history.extend(reply)
                        attacks.append({'method':method,'family_id':record['family_id'],
                            'split':split[record['family_id']],'record_id':record['record_id'],
                            'prefix_time_s':t,'L':L,'label':intent,
                            'features':wire_features(requests_history,replies_history,categories,purposes).tolist()})
                    wire=histories[L]['full_cover']
                    digest=hashlib.sha256(json.dumps(wire,sort_keys=True).encode()).hexdigest()
                    witnesses.setdefault((sid,tuple(indices),t,L),set()).add(digest)
            events.append({'timestamp_s':t,'coordinates':[list(c) for c in coords]})
        public_streams.append({'session_id_evaluator_only':sid,'record_id_evaluator_only':record['record_id'],
                              'events':events,'spent_bound':model.spent_bound})
    depths=[{'L':L,**mean_by_family(utility,'recall',split='selection',L=L)} for L in DEPTHS]
    selected=next((r['L'] for r in depths if r['value'] is not None and r['value']>=.90-1e-12),DEPTHS[-1])
    result=[]
    for time in sorted(set(r['prefix_time_s'] for r in attacks)):
        for method in ('explicit_metadata_control','full_cover'):
            rows=[r for r in attacks if r['prefix_time_s']==time and r['method']==method and r['L']==selected]
            sets=[[r for r in rows if r['split']==s] for s in ('fit','selection','test')]
            xs=[np.array([r['features'] for r in group]) for group in sets]
            ys=[np.array([r['label'] for r in group]) for group in sets]
            result.append({'prefix_time_s':time,'method':method,'L':selected,'chance_balanced':1/3,
                **selected_attack(xs[0],ys[0],xs[1],ys[1],xs[2],ys[2],sorted(set(ys[0])))})
    assert all(len(v)==1 for v in witnesses.values())
    assert all(sha(ROOT/n)==h for n,h in hashes.items())
    readout={'schema':'s7-sequence-readout-v1','resources':metadata,
        'scope':protocol['scope'],'selected_L':selected,'selection_depths':depths,
        'test_recall':mean_by_family(utility,'recall',split='test',L=selected),
        'intent_attacks_by_public_prefix':result,'wire_prefix_noninterference':True,
        'counterfactual_prefixes_checked':len(witnesses),'source_hashes_unchanged':True,
        'limitations':['Same GPS with synthetic templates checks conditional content privacy only',
            'No claim about intent correlated with route/account/session activation',
            'Source SUMO network reconstructed; turns unverified; constant speed makes fastest coincide with distance']}
    (out/'readout.json').write_text(json.dumps(readout,indent=2)+'\n')
    for name,value in [('attack_rows',attacks),('utility_rows',utility),('public_streams',public_streams)]:
        (out/(name+'.json.gz')).write_bytes(gzip.compress(json.dumps(value,separators=(',',':')).encode(),mtime=0))
    print(json.dumps({'L':selected,'recall':readout['test_recall']['value'],'wire_equal':True,
                     'attacks':[(r['method'],r['prefix_time_s'],r['test']['balanced_accuracy']) for r in result]}))


if __name__=='__main__':main()
