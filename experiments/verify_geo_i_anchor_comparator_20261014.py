"""Independently rebuild bare-anchor coverage and selected-attacker readout."""
from pathlib import Path
import json,numpy as np
from experiments import main_depth_study_20261014 as study
from experiments.geo_i_anchor_comparator_20261014 import OUT,METHOD


def verify():
    protocol=study.read(OUT/'protocol.json');results=study.read(OUT/'results.json')
    assert protocol['source_sha256']==study.sha(OUT/'source_snapshot'/'comparator.py')
    assert protocol['source_generation_sha256']==study.sha(study.OUT/'generation.json')
    families=study.load_families(study.OUT)
    rn,reference,reply,_=study.public.resources(study.WORK/'public_resources')
    utility=study.Utility(reference,reply);data=study.read(study.DATA)
    test=[f for f in families if f['split']=='test'];purposes=[p.value for p in study.QueryPurpose]
    family_means=[];recomputed=0;costs=[]
    for f in test:
        rows=[]
        for trip in f['trips']:
            trace={int(p['time_s']):p for p in data['traces'][trip['native_session_id']]}
            destination=rn.nearest(trace[600]['lat'],trace[600]['lon'])[0]
            known=np.zeros(len(reply.pois),bool)
            requests=replies=0
            for truth,anchor in zip(trip['truth'],trip['GeoI_anchors']):
                state=rn.nearest(*anchor)[0];values=reply.query_indices(state)
                known[values[values>=0]]=True
                scored=utility.score(truth['state'],destination,known)
                rows.append(scored);recomputed+=1
                t=truth['t'];request=dict(schema='all_category_cover_v1',timestamp_s=float(t),coordinate=list(anchor),
                    categories=list(reply.categories),response_l=60,epoch=int(t//60))
                response=[list(map(int,row[row>=0])) for row in values]
                requests+=len(json.dumps(request,separators=(',',':')).encode())
                replies+=len(json.dumps(response,separators=(',',':')).encode())
            costs.append((requests,replies))
        metrics={p:float(np.mean([r[p]['recall5'] for r in rows if r[p]['recall5'] is not None])) for p in purposes}
        metrics['macro']=float(np.mean(list(metrics.values())));family_means.append(metrics)
    for p in purposes+['macro']:
        assert abs(results['metrics'][p]['bare_mean']-np.mean([row[p] for row in family_means]))<1e-14
    assert results['cost']['requests']==31
    assert abs(results['cost']['request_bytes']-np.mean([a for a,b in costs]))<1e-8
    assert abs(results['cost']['reply_bytes']-np.mean([b for a,b in costs]))<1e-8
    test_ids={f['family_id'] for f in test}
    for row in results['attackers']:
        stem=METHOD+'--'+row['scenario'];selection=study.read(OUT/'attacker_selection'/(stem+'.json'))
        assert selection['model_sha256']==study.sha(OUT/'attacker_models'/(stem+'.pkl.gz'))
        assert selection['fit_split']=='train' and selection['selection_split']=='selection' and selection['before_test_predictions']
        assert row['selected_attacker']==selection['selected']
        predictions=study.read(OUT/'attacker_predictions'/(stem+'.json.gz'))
        assert len(predictions)==120 and {p['family_id'] for p in predictions}==test_ids
        mae=[];hit=[]
        for fid in sorted(test_ids):
            group=[p for p in predictions if p['family_id']==fid]
            mae.append(np.mean([np.mean(p['errors'][selection['selected']['mae']]) for p in group]))
            hit.append(np.mean([np.mean(np.array(p['errors'][selection['selected']['hit100']])<=100) for p in group]))
        assert abs(np.mean(mae)-row['MAE_m'])<1e-10 and abs(np.mean(hit)-row['Hit100'])<1e-14
    receipt=dict(status='pass',protocol_sha256=study.sha(OUT/'protocol.json'),results_sha256=study.sha(OUT/'results.json'),
        verifier_sha256=study.sha(Path(__file__)),actual_coordinate_utilities_recomputed=recomputed,
        source_main_audit_sha256=study.sha(study.OUT/'independent_validation.json'),post_hoc_controlled_comparator=True,
        untouched_cohort_claimed=False)
    p=OUT/'independent_validation.json'
    if p.exists():assert study.read(p)==receipt
    else:study.save(p,receipt)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':verify()
