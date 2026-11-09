"""Post-V1 S8 attacker repair, with frozen original outputs and fitted models."""
import argparse
from collections import defaultdict
import gzip
import pickle

import numpy as np

from evaluation.correlated_channel_fusion import fusion_estimates
from experiments import remaining_scenarios_20261109 as base

ROOT=base.ROOT
OUT=ROOT/'artifacts/benchmarks/s8_fusion_20261109_v2'


def freeze():
    base.validate_sources()
    if OUT.exists():raise FileExistsError('Keep completed evidence')
    assert base.read(base.OUT/'validation.json')['status']=='pass'
    names=['evaluation/correlated_channel_fusion.py','tests/test_correlated_channel_fusion.py',
        'experiments/s8_fusion_20261109.py','docs/research/2026-11-09_s8_fusion_protocol.md']
    protocol=dict(schema='s8-fusion-protocol-v2',date='2026-11-09',
        source_sha256={n:base.sha(ROOT/n) for n in names},
        inputs_sha256={str((base.OUT/n).relative_to(ROOT)):base.sha(base.OUT/n)
            for n in ('protocol.json','results.json','validation.json','output_manifest.json')},
        post_v1_diagnostic=True,prior_test_inspected=True,new_defense_samples=False,
        rules='V1 candidates plus fixed public fusion; SELECTION mean+/-SE; TEST10-block CI10000',
        methods=list(base.METHODS),views=base.CONFIG['joint_views'])
    base.save(OUT/'protocol.json',protocol)
    for n in names:
        p=OUT/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    print('Frozen V2 fusion bank before scoring; V1 retained',flush=True)


def validate_sources():
    base.validate_sources();p=base.read(OUT/'protocol.json')
    for n,h in p['source_sha256'].items():assert base.sha(ROOT/n)==base.sha(OUT/'source_snapshot'/n)==h,n
    for n,h in p['inputs_sha256'].items():assert base.sha(ROOT/n)==h,n
    for n,h in base.read(base.OUT/'output_manifest.json')['files_sha256'].items():assert base.sha(base.OUT/n)==h,n
    return p


def tapes(families,by_id,method,view,split):
    family_map={f['family_id']:f for f in families};mapping=base.blocks(families);result=[]
    for f in families:
        if f['split']!=split:continue
        cut=20*(by_id[f['family_id']]['clocks']['shared_fork_t']//20)
        for trip in f['trips']:
            other=family_map[mapping[f['family_id']][1]] if view=='unrelated_partner' else f
            partner=next(t for t in other['trips'] if t['draw']==trip['draw'] and
                t['native_session_id'].split('_day')[-1]!=trip['native_session_id'].split('_day')[-1])
            result.append(fusion_estimates(trip['events'][method],partner['events']['raw' if view=='raw_partner' else method],cut))
    return result


def candidates(rows,model,base_model,fusion):
    x=np.array([r['features'] for r in rows])
    result={'learned_'+k:v for k,v in model.predict(x).items()}
    result.update({'target_only_'+k:v for k,v in base_model.predict(np.array([r['target_features'] for r in rows])).items()})
    for k in rows[0]['geometry']:result[k]=np.array([r['geometry'][k] for r in rows])
    if fusion:
        for k in fusion[0]:result[k]=np.array([f[k] for f in fusion])
    return result


def run():
    validate_sources()
    if (OUT/'results.json').exists():raise FileExistsError('Keep completed evidence')
    families,native,by_id=base.datasets();results=[]
    for method in base.METHODS:
        target_records=None;target_selected=None
        for view in base.CONFIG['joint_views']:
            name='S8--'+view+'--'+method
            with gzip.open(base.OUT/'models'/(name+'.pkl.gz'),'rb') as stream:model,base_model=pickle.load(stream)
            sets=base.joint_rows(families,native,by_id,method,view)
            predictions={}
            for split in ('selection','test'):
                if split=='test':
                    selected,scores=base.regression_selection(sets['selection'],predictions['selection'])
                    base.save(OUT/'selections'/(name+'.json'),dict(selected=selected,scores=scores))
                fusion=[] if view=='target_only' else tapes(families,by_id,method,view,split)
                predictions[split]=candidates(sets[split],model,base_model,fusion)
                target=np.array([r['target_xy'] for r in sets[split]])
                records=[{**{k:v for k,v in r.items() if k not in ('features','target_features','geometry')},
                    'estimates':{n:p[i].tolist() for n,p in predictions[split].items()},
                    'errors':{n:float(np.linalg.norm(p[i]-target[i])) for n,p in predictions[split].items()}}
                    for i,r in enumerate(sets[split])]
                base.save(OUT/(split+'_predictions')/(name+'.json.gz'),records)
            if view=='target_only':target_records=records;target_selected=selected
            assert [r['session_id'] for r in records]==[r['session_id'] for r in target_records]
            metrics={};delta={}
            for metric,key in [('MAE_m','mae'),('Hit100','hit100')]:
                a=np.array([r['errors'][selected[key]] for r in records])
                b=np.array([r['errors'][target_selected[key]] for r in target_records])
                metrics[metric]=base.summary(records,a if key=='mae' else a<=100)
                diff=a-b if key=='mae' else (a<=100).astype(float)-(b<=100).astype(float)
                delta[metric]=base.summary(records,diff)
            result=dict(task=name,status='ok',method=method,view=view,selected=selected,
                metrics=metrics,paired_vs_target_only=delta,
                mean_observed_packets=float(np.mean([r['observed_packets'] for r in records])),
                mean_observed_coordinates=float(np.mean([r['observed_coordinates'] for r in records])))
            base.save(OUT/'tasks'/(name+'.json'),result);results.append(result)
            print('Scored V2',name,flush=True)
    validate_sources()
    base.save(OUT/'results.json',dict(schema='s8-fusion-results-v2',tasks=results,completed_tasks=len(results),
        full_coverage=len(results)==24,failures=[],post_v1_diagnostic=True,
        protocol_sha256=base.sha(OUT/'protocol.json'),source_v1_sha256=base.sha(base.OUT/'results.json')))
    base.save(OUT/'output_manifest.json',dict(files_sha256={str(p.relative_to(OUT)):base.sha(p)
        for p in sorted(OUT.rglob('*')) if p.is_file() and 'source_snapshot' not in p.parts and p.name!='output_manifest.json'}))


def verify(write=False):
    validate_sources();r=base.read(OUT/'results.json');assert r['full_coverage'] and len(r['tasks'])==24
    for n,h in base.read(OUT/'output_manifest.json')['files_sha256'].items():assert base.sha(OUT/n)==h,n
    old={t['task']:t for t in base.read(base.OUT/'results.json')['tasks']}
    from experiments.verify_remaining_scenarios_20261109 import summary
    for task in r['tasks']:
        name=task['task'];selection_rows=base.read(OUT/'selection_predictions'/(name+'.json.gz'))
        predictions={n:np.array([r['estimates'][n] for r in selection_rows]) for n in selection_rows[0]['estimates']}
        selected,scores=base.regression_selection(selection_rows,predictions)
        assert selected==task['selected']
        assert base.read(OUT/'selections'/(name+'.json'))==dict(selected=selected,scores=scores)
        rows=base.read(OUT/'test_predictions'/(name+'.json.gz'));assert len(rows)==120
        target_task=next(t for t in r['tasks'] if t['task']=='S8--target_only--'+task['method'])
        target_rows=base.read(OUT/'test_predictions'/(target_task['task']+'.json.gz'))
        assert [r['session_id'] for r in rows]==[r['session_id'] for r in target_rows]
        for row in selection_rows+rows:
            for n,p in row['estimates'].items():
                assert np.isclose(np.linalg.norm(np.array(p)-row['target_xy']),row['errors'][n],rtol=0,atol=1e-9)
        for metric,key in [('MAE_m','mae'),('Hit100','hit100')]:
            errors=np.array([row['errors'][selected[key]] for row in rows])
            before=np.array([row['errors'][target_task['selected'][key]] for row in target_rows])
            assert summary(rows,errors if key=='mae' else errors<=100)==task['metrics'][metric]
            diff=errors-before if key=='mae' else (errors<=100).astype(float)-(before<=100).astype(float)
            assert summary(rows,diff)==task['paired_vs_target_only'][metric]
            if task['view']=='target_only':assert task['metrics'][metric]==old[name]['metrics'][metric]
        if task['view']=='raw_partner':assert task['metrics']['Hit100']['mean']==1 and task['metrics']['MAE_m']['mean']<1e-6
    receipt=dict(schema='s8-fusion-verification-v2',status='pass',tasks_recomputed=24,
        selection_replayed=True,core_outputs_changed=False,new_defense_samples=False,
        results_sha256=base.sha(OUT/'results.json'),protocol_sha256=base.sha(OUT/'protocol.json'))
    if write:base.save(OUT/'validation.json',receipt)
    else:assert base.read(OUT/'validation.json')==receipt
    print('PASS:V2 selectors and24 tasks; V1 outputs retained',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','run','verify'))
    parser.add_argument('--write',action='store_true');args=parser.parse_args()
    if args.stage=='verify':verify(args.write)
    else:globals()[args.stage]()
