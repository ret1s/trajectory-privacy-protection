"""Read-only independent metrics/eligibility audit of the S4--S8 supplement."""
import argparse
from collections import Counter,defaultdict
import json

import numpy as np
from sklearn.metrics import roc_auc_score,f1_score

from experiments import remaining_scenarios_20261109 as study


def summary(rows,values):
    bins=defaultdict(list)
    for r,v in zip(rows,values):bins[r['group']].append(float(v))
    groups=sorted(bins);a=np.array([np.mean(bins[g]) for g in groups])
    indices=np.random.default_rng(20261109).integers(0,len(a),(10000,len(a)))
    return dict(mean=float(np.mean(a)),ci95=np.quantile(np.mean(a[indices],axis=1),[.025,.975]).tolist(),
        groups=len(a),records=len(rows),group_values={g:float(v) for g,v in zip(groups,a)})


def verify(write=False):
    protocol=study.validate_sources();result=study.read(study.OUT/'results.json')
    assert result['protocol_sha256']==study.sha(study.OUT/'protocol.json')
    assert result['full_coverage'] and not result['failures']
    assert len(result['tasks'])==result['expected_tasks']==66
    for n,h in study.read(study.OUT/'output_manifest.json')['files_sha256'].items():
        assert study.sha(study.OUT/n)==h,n
    families,native,by_id=study.datasets();block_map=study.blocks(families)
    test_families={f['family_id'] for f in families if f['split']=='test'}
    checked=0;prefixes=0;queries=0
    for task in result['tasks']:
        name=task['task'];rows=study.read(study.OUT/'predictions'/(name+'.json.gz'))
        assert study.read(study.OUT/'tasks'/(name+'.json'))==task
        assert len(rows)==(176 if name.startswith('S7') else 120),name
        if not name.startswith('S7'):
            chosen=study.read(study.OUT/'selections'/(name+'.json'))
            assert chosen['selected']==task['selected'] and chosen['scores']==task['selection']
            assert study.sha(study.OUT/'models'/(name+'.pkl.gz'))==chosen['model_sha256']
        if name.startswith(('S4','S5','S6')):
            y=np.array([r['label'] for r in rows]);p=np.array([r['probabilities'][task['selected']] for r in rows])
            assert p.shape==(120,2) and np.allclose(p.sum(axis=1),1) and np.isfinite(p).all()
            threshold=float(task['selected'].rsplit('@',1)[1]) if '@' in task['selected'] else None
            pred=(p[:,1]>=threshold).astype(int) if threshold is not None else p.argmax(axis=1)
            assert pred.tolist()==[r['selected_prediction'] for r in rows]
            correct=pred==y;loss=-np.log(np.maximum(1e-12,p[np.arange(120),y]))
            for metric,values in [('accuracy',correct),('balanced_accuracy',correct),('log_loss',loss),('brier',(p[:,1]-y)**2)]:
                assert summary(rows,values)==task['metrics'][metric],(name,metric)
            assert float(roc_auc_score(y,p[:,1]))==task['metrics']['roc_auc']
            assert float(f1_score(y,pred,average='macro',zero_division=0))==task['metrics']['macro_f1']
            group_auc={g:float(roc_auc_score(y[[i for i,r in enumerate(rows) if r['group']==g]],
                p[[i for i,r in enumerate(rows) if r['group']==g],1])) for g in sorted({r['group'] for r in rows})}
            assert summary(rows,[group_auc[r['group']] for r in rows])==task['metrics']['group_mean_roc_auc']
        if name.startswith('S4'):
            for r in rows:
                a,b=r['source_families'];assert a in test_families and b in test_families
                assert int(a==b)==r['label']
                assert block_map[a][0]==block_map[b][0]==r['group']
            assert len({r['group'] for r in rows})==10
        if name.startswith(('S5','S6')):
            parts=name.split('--');expected=study.future_rows(families,native,by_id,parts[2],parts[0],parts[1])['test']
            assert {r['family_id'] for r in rows}==test_families
            for r,e in zip(rows,expected):
                assert r['cut_s']==e['cut_s'] and r['observed_packets']==e['observed_packets']
                assert r['label']==e['label'] and r['edge_id']==e['edge_id']
                choice=by_id[r['family_id']]['public_context']['choices'][r['selected_prediction']]
                error=float(np.linalg.norm(np.array(choice['destination_xy'])-r['destination_xy']))
                assert error==r['selected_destination_error_m']
                assert r['cut_s'] < (r['first_target_edge_s'] if parts[0]=='S5' else r['destination_arrival_s'])
                prefixes+=1;queries+=r['observed_coordinates']
            errors=np.array([r['selected_destination_error_m'] for r in rows])
            assert summary(rows,errors)==task['destination_MAE_m']
            assert summary(rows,errors<=100)==task['destination_Hit100']
            if parts[1]=='before_fork' and parts[2]=='raw':
                assert task['metrics']['accuracy']['mean']==.5,'Shared-prefix Raw must have no future branch signal'
        if name.startswith('S7'):
            assert all(r['equal_wire'] and len(set(r['protected_hashes']))==1 and len(set(r['explicit_hashes']))==24 for r in rows)
            test=[r for r in rows if r['split']=='test']
            for k in task['metrics']:
                assert summary(test,[r[k] for r in test])==task['metrics'][k]
            assert task['metrics']['purpose_vulnerability']['mean']==.25
            assert np.isclose(task['metrics']['category_vulnerability']['mean'],1/6)
            assert np.isclose(task['metrics']['joint_vulnerability']['mean'],1/24)
        if name.startswith('S8'):
            _,view,method=name.split('--')
            expected=study.joint_rows(families,native,by_id,method,view)['test']
            assert {r['family_id'] for r in rows}==test_families
            for r,e in zip(rows,expected):
                for key in ('target_xy','cut_s','group','partner_session_id','observed_packets','observed_coordinates'):
                    assert r[key]==e[key],(name,key)
                for candidate,estimate in r['estimates'].items():
                    error=float(np.linalg.norm(np.array(estimate)-r['target_xy']))
                    assert np.isclose(error,r['errors'][candidate],rtol=0,atol=1e-9)
                prefixes+=1;queries+=r['observed_coordinates']
            for metric,key in [('MAE_m','mae'),('Hit100','hit100')]:
                error=np.array([r['errors'][task['selected'][key]] for r in rows])
                assert summary(rows,error if key=='mae' else error<=100)==task['metrics'][metric]
                base_task=next(t for t in result['tasks'] if t['task']=='S8--target_only--'+method)
                base=study.read(study.OUT/'predictions'/('S8--target_only--'+method+'.json.gz'))
                assert [r['session_id'] for r in base]==[r['session_id'] for r in rows]
                base_error=np.array([r['errors'][base_task['selected'][key]] for r in base])
                delta=error-base_error if key=='mae' else (error<=100).astype(float)-(base_error<=100).astype(float)
                assert summary(rows,delta)==task['paired_vs_target_only'][metric]
            if view=='raw_partner':
                assert task['metrics']['Hit100']['mean']==1 and task['metrics']['MAE_m']['mean']<1e-6
        checked+=1
    receipt=dict(schema='remaining-scenarios-validation-v1',status='pass',
        source_and_input_hashes_verified=True,core_release_pins_verified=108,
        tasks_recomputed=checked,test_prefixes_rechecked=prefixes,
        observed_coordinates_recounted=queries,protected_outputs_regenerated=False,
        S4_identity_labels_available=False,S8_simultaneous_native_pairs=False,
        result_sha256=study.sha(study.OUT/'results.json'),protocol_sha256=study.sha(study.OUT/'protocol.json'))
    path=study.OUT/'validation.json'
    if write:study.save(path,receipt)
    else:assert study.read(path)==receipt
    print('PASS:',checked,'tasks; strict future labels; independent metrics/CI; frozen outputs',flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true')
    verify(parser.parse_args().write)
