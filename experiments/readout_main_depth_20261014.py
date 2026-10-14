"""Descriptive replication readout; no defense selection/tuning on TEST."""
from pathlib import Path
import argparse
import json

import numpy as np

from benchmark.whole_path_chi_square import whole_path_bound
from experiments import main_depth_study_20261014 as study


def analyze(output=study.OUT):
    output=Path(output)
    study.validate_sources(output)
    families=study.load_families(output)
    test=[f for f in families if f['split']=='test']
    primary=study.CONFIG['primary']
    purposes=[p.value for p in study.QueryPurpose]
    draws=np.random.default_rng(study.CONFIG['bootstrap_seed']).integers(0,len(test),(10000,len(test)))
    def means(method,cache='session',lo=0,hi=600):
        rows=[]
        for family in test:
            row={p:float(np.mean([s[cache][p]['recall5'] for t in family['trips'] for s in t['utility'][method]
                if lo<=s['t']<=hi and s[cache][p]['recall5'] is not None])) for p in purposes}
            row['macro']=float(np.mean(list(row.values())))
            rows.append(row)
        return rows
    base=means(primary)
    comparisons=[]
    for method in study.METHODS:
        other=means(method)
        metrics={}
        for purpose in purposes+['macro']:
            delta=np.array([a[purpose]-b[purpose] for a,b in zip(base,other)])
            metrics[purpose]=dict(delta=float(delta.mean()),paired95=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist())
        comparisons.append(dict(control=method,metrics=metrics))
    absolute={p:np.quantile(np.array([r[p] for r in base])[draws].mean(axis=1),[.025,.975]).tolist() for p in purposes+['macro']}
    coverage={m:dict(initial=float(np.mean([len(t['retrieved_POI_indices'][m][0]['session']) for f in test for t in f['trips']])),
        final=float(np.mean([len(t['retrieved_POI_indices'][m][-1]['session']) for f in test for t in f['trips']]))) for m in study.METHODS}
    periods=[dict(window=[lo,hi],means={m:float(np.mean([r['macro'] for r in means(m,lo=lo,hi=hi)])) for m in study.METHODS}) for lo,hi in ((0,40),(60,180),(200,600))]
    sample=next(f for f in families if f['split']=='train')['trips'][0]
    walkthrough=[]
    for tick in (0,1,2,3,6,9,15,21,30):
        t=tick*20;cert=sample['Q_certificates'][primary][tick]
        walkthrough.append(dict(t=t,selection=tick%3==0,GeoI_read=sample['GeoI_ledger'][tick]['private_read'],
            GeoI_branch=sample['GeoI_ledger'][tick]['branch'],GPS_state=sample['truth'][tick]['state'],
            Z_local=sample['GeoI_anchors'][tick],Q_states_public=sample['sampled_states'][primary][tick],
            public_library_size=cert['public_library_size'],epsilon_Q=cert['epsilon_Q_upper'],
            budget_reserved=cert['reserved_prefix_epsilon_Q'],
            received_current=len(sample['retrieved_POI_indices'][primary][tick]['current']),
            cached_POIs=len(sample['retrieved_POI_indices'][primary][tick]['session'])))
    rn,reference,reply,ids=study.public.resources(study.WORK/'public_resources')
    from benchmark.public_poi_context import PublicPoiContext
    from evaluation.lane_travel import LanePoiService
    pois=[{k:v for k,v in p.items() if k not in ('vertex','access_offset_m')} for p in study.read(study.ROOT/study.public.RESOURCES)['pois_used']]
    reply30=PublicPoiContext(LanePoiService(rn,pois,k=30),study.WORK/'public_resources'/'reply30.npz')
    utility=study.Utility(reference,reply)
    fixed_wire_depth=[];stale=[]
    data=study.read(study.DATA)
    for family in test:
        trip_scores=[];stale_scores={m:[] for m in study.METHODS}
        for trip in family['trips']:
            trace={int(p['time_s']):p for p in data['traces'][trip['native_session_id']]}
            destination=rn.nearest(trace[600]['lat'],trace[600]['lon'])[0]
            known30=np.zeros(len(reply.pois),bool)
            for tick,truth in enumerate(trip['truth']):
                for coordinate in trip['events'][primary][tick]['coordinates']:
                    values=reply30.query_indices(rn.nearest(*coordinate)[0]).ravel()
                    known30[values[values>=0]]=True
                actual60=np.zeros(len(reply.pois),bool)
                actual60[trip['retrieved_POI_indices'][primary][tick]['session']]=True
                assert np.all(~known30|actual60)
                value30=utility.score(truth['state'],destination,known30)
                trip_scores.append({p:value30[p]['recall5'] for p in purposes})
                stale_point=trace[(int(truth['t'])//60)*60]
                stale_state=rn.nearest(stale_point['lat'],stale_point['lon'])[0]
                stale_refs=utility.refs(stale_state,destination)
                actual_refs=utility.refs(truth['state'],destination)
                for method in study.METHODS:
                    received=set(trip['retrieved_POI_indices'][method][tick]['session'])
                    row={}
                    for purpose in purposes:
                        values=[]
                        for (true_full,true_ref),(local_full,_) in zip(actual_refs[purpose],stale_refs[purpose]):
                            answer=[i for i in local_full if i in received][:5]
                            if true_ref:values.append(len(set(true_ref)&set(answer))/len(true_ref))
                        row[purpose]=float(np.mean(values)) if values else None
                    stale_scores[method].append(row)
        row={p:float(np.mean([s[p] for s in trip_scores if s[p] is not None])) for p in purposes}
        row['macro']=float(np.mean(list(row.values())));fixed_wire_depth.append(row)
        family_stale={}
        for method,values in stale_scores.items():
            row={p:float(np.mean([s[p] for s in values if s[p] is not None])) for p in purposes}
            row['macro']=float(np.mean(list(row.values())));family_stale[method]=row
        stale.append(family_stale)
    same_wire={}
    for purpose in purposes+['macro']:
        delta=np.array([a[purpose]-b[purpose] for a,b in zip(base,fixed_wire_depth)])
        same_wire[purpose]=dict(L30_mean=float(np.mean([r[purpose] for r in fixed_wire_depth])),
            L60_minus_L30=float(delta.mean()),paired95=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist())
    stale_readout={}
    for method in study.METHODS:
        metrics={}
        for purpose in purposes+['macro']:
            values=np.array([row[method][purpose] for row in stale])
            delta=values-np.array([row['baseline'][purpose] for row in stale])
            metrics[purpose]=dict(mean=float(values.mean()),paired_delta=float(delta.mean()),
                paired95=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist())
        stale_readout[method]=metrics
    payload=dict(schema='targeted-cache-replication-readout-v1',primary=primary,
        scope='new randomized defense draws on REUSED native trajectories; NOT untouched trajectory generalization',
        results_sha256=study.sha(output/'results.json'),comparisons=comparisons,primary_absolute95=absolute,
        periods=periods,POI_coverage=coverage,training_sample_walkthrough=walkthrough,same_Q_L30_counterfactual=same_wire,
        same_Q_counterfactual_scope='fixed L60-planned Q, truncate provider depth to30; NOT an end-to-end L30 policy',
        stale_local_GPS_60s=stale_readout,local_GPS_scope='primary metric uses fresh native fix every20s; sensitivity holds last60s local fix; no new network or privacy budget',
        exact_whole_path_certificates={str(g):whole_path_bound(g,64,infinite=True) for g in (0,.5,1,2)},
        prior_results_replaced=False,no_defense_tuning_after_test_readout=True)
    p=output/'release_readout.json'
    if p.exists():assert study.read(p)==payload
    else:study.save(p,payload)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('exact_whole_path_certificates','training_sample_walkthrough')},indent=2))
    return payload

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=study.OUT)
    analyze(parser.parse_args().output)
