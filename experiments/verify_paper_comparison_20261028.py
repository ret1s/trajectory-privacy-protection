"""Independently replay public requests/cache and recompute saved evaluations.

No defense generation, attacker fitting or TEST selection. Tests the transport
metadata correction separately from unchanged proposed Q and service results.
"""
from collections import Counter
import json
import time

import numpy as np
from benchmark.engines.transprotect import TargetTravelCostUtility
from benchmark.public_segment_scores import local_sorted_pois
from benchmark.query_purpose import MultiPurposeRoadRanking,QuerySpec,QueryPurpose
from experiments import paper_comparison_20261028 as study


def verify():
    protocol=study.validate_sources();families=study.families();result=study.read(study.OUT/'results.json')
    assert result['protocol_sha256']==study.sha(study.OUT/'protocol.json')
    assert result['generation_sha256']==study.sha(study.OUT/'generation.json')
    assert Counter(f['split'] for f in families)==study.CONFIG['splits']
    original={f['family_id']:f for f in study.sealed.load_families(study.sealed.OUT)}
    rn,ref,reply,_=study.resources();ranking=MultiPurposeRoadRanking(ref,cache_limit=4096)
    metadata=study.read(study.OUT/'resources.json');all_pois=np.ones(len(reply.pois),bool)
    targets=TargetTravelCostUtility(rn,target_indices=metadata['Eq13_target_indices'],target_prior=metadata['Eq13_target_prior'])
    assert targets.disconnection_penalty==metadata['Eq13_unreachable_penalty']
    refs={};checked=0;queries=0;proposed_prefixes=0;native_count=0
    for family in families:
        src=original[family['family_id']];assert family['split']==src['split']
        assert len(family['trips'])==2*study.CONFIG['draws_by_split'][family['split']]
        for trip,old in zip(family['trips'],src['trips']):
            assert trip['session_id']==old['session_id'] and trip['truth']==old['truth']
            for method in study.METHODS:
                if trip['status'][method]!='ok':continue
                events=trip['events'][method]
                assert all(set(e)=={'timestamp_s','coordinates'} for e in events)
                assert all(0<=e['timestamp_s']<=600 for e in events)
                if method=='geoi_bq':assert events==old['events']['cache_target_g1']
                if method=='geo_i_Z_k1':assert [e['coordinates'][0] for e in events]==old['GeoI_anchors']
                costs=dict(requests=0,packets=0,request_bytes=0,reply_bytes=0)
                seen=np.zeros(len(reply.pois),bool);epoch_seen=seen.copy();current=seen.copy();epoch=None;cursor=0
                for tick,truth in enumerate(trip['truth']):
                    while cursor<len(events) and events[cursor]['timestamp_s']<=truth['t']:
                        e=events[cursor];new_epoch=int(e['timestamp_s']//60)
                        if new_epoch!=epoch:epoch_seen[:]=False;epoch=new_epoch
                        current[:]=False;costs['packets']+=1
                        for coordinate in e['coordinates']:
                            request=dict(schema='all_category_cover_v1',timestamp_s=e['timestamp_s'],
                                coordinate=coordinate,categories=list(sorted(reply.categories)),response_l=60,epoch=epoch)
                            response=[list(map(int,a[a>=0])) for a in reply.query_indices(rn.nearest(*coordinate)[0])]
                            for ids in response:current[ids]=True
                            costs['requests']+=1;costs['request_bytes']+=len(json.dumps(request,separators=(',',':')).encode())
                            costs['reply_bytes']+=len(json.dumps(response,separators=(',',':')).encode())
                        epoch_seen|=current;seen|=current;cursor+=1
                    masks={'current':current,'known':epoch_seen,'session':seen}
                    for cache,mask in masks.items():
                        assert np.array_equal(mask,study.unpacked(trip['masks'][method][tick][cache],len(reply.pois)))
                        reference_key=(truth['state'],trip['truth'][-1]['state'])
                        if reference_key not in refs:
                            by_purpose={}
                            for purpose in QueryPurpose:
                                sets=[]
                                for category in ranking.categories:
                                    spec=QuerySpec(purpose,category,k=5,radius_m=1000. if purpose==QueryPurpose.WITHIN_RADIUS else None,
                                        destination_state=reference_key[1] if purpose==QueryPurpose.MIN_DETOUR else None)
                                    sets.append(local_sorted_pois(ranking,reference_key[0],all_pois,spec)[:5])
                                by_purpose[purpose.value]=sets
                            refs[reference_key]=by_purpose
                        for purpose,sets in refs[reference_key].items():
                            values=[sum(bool(mask[i]) for i in s)/len(s) for s in sets if s]
                            expected=float(np.mean(values)) if values else None
                            actual=trip['utility'][method][tick][cache][purpose]['recall5']
                            assert actual==expected,(family['family_id'],method,truth['t'],cache,purpose)
                            checked+=1
                    if method=='geoi_bq':
                        assert trip['utility'][method][tick]==old['utility']['cache_target_g1'][tick];proposed_prefixes+=1
                assert cursor==len(events) and costs==trip['cost'][method]
                queries+=costs['requests']
                if all(len(e['coordinates'])==1 for e in events):
                    by_time={e['timestamp_s']:e['coordinates'][0] for e in events}
                    real=[t['state'] for t in trip['truth']]
                    released=[rn.nearest(*by_time[float(t['t'])])[0] for t in trip['truth']]
                    value=float(np.mean(np.abs(targets.cost_to_targets[real]-targets.cost_to_targets[released])@targets.target_prior))
                    assert np.isclose(value,trip['native'][method]['Eq13'],rtol=0,atol=1e-9);native_count+=1
        print('Verified public/cache',family['family_id'],flush=True)
    recomputed=study.summarize(families)
    for key,value in recomputed.items():assert value==result[key],key
    family_privacy=0
    for row in result['attackers']:
        if row['status']!='ok':continue
        method,scenario=row['method'],row['scenario']
        selection=study.read(study.OUT/'attacker_selection'/(method+'--'+scenario+'.json'))
        assert selection['fit_split']=='train' and selection['selection_split']=='selection'
        assert selection['model_sha256']==study.sha(study.OUT/'attacker_models'/(method+'--'+scenario+'.pkl.gz'))
        assert selection['selected']==row['selected_attacker']
        predictions=study.read(study.OUT/'attacker_predictions'/(method+'--'+scenario+'.json.gz'))
        assert len(predictions)==120 and {x['family_id'] for x in predictions}=={f['family_id'] for f in families if f['split']=='test'}
        for fm in row['family_metrics']:
            group=[p for p in predictions if p['family_id']==fm['family_id']]
            assert len(group)==6
            mae=float(np.mean([np.mean(x['errors'][selection['selected']['mae']]) for x in group]))
            hit=float(np.mean([np.mean(np.asarray(x['errors'][selection['selected']['hit100']])<=100) for x in group]))
            assert mae==fm['MAE_m'] and hit==fm['Hit100'];family_privacy+=1
        for key in ('MAE_m','Hit100'):
            values=np.array([f[key] for f in row['family_metrics']]);indices=np.random.default_rng(study.CONFIG['bootstrap_seed']).integers(0,len(values),(10000,len(values)))
            assert float(values.mean())==row[key]
            assert np.quantile(values[indices].mean(axis=1),[.025,.975]).tolist()==row[key+'_CI95']
        if method=='raw':assert row['Hit100']>=.99 and row['MAE_m']<1.,('Raw alignment',scenario,row['Hit100'],row['MAE_m'])
    paths=[p for p in study.OUT.rglob('*') if p.is_file() and 'source_snapshot' not in p.parts]
    receipt=dict(schema='matched-paper-comparison-verification-v1',status='pass',
        protocol_sha256=study.sha(study.OUT/'protocol.json'),results_sha256=study.sha(study.OUT/'results.json'),
        unchanged_main_pins=True,source_and_input_pins=True,proposed_Q_and_utility_replay_checks=proposed_prefixes,
        utility_cells_independently_recomputed=checked,point_queries_independently_counted=queries,
        singleton_Eq13_recomputed=native_count,privacy_family_metrics_recomputed=family_privacy,
        private_labels_absent_from_public_events=True,request_depth_metadata_normalized_to_actual60=True,
        coverage_all_arms_retained=True,numerical_sampler_certified=False,
        files_sha256={str(p.relative_to(study.OUT)):study.sha(p) for p in sorted(paths) if p.name!='validation.json'})
    study.save(study.OUT/'validation.json',receipt)
    print('PASS:',checked,'utility cells;',proposed_prefixes,'unchanged proposed prefixes;',queries,'point queries',flush=True)


if __name__=='__main__':verify()
