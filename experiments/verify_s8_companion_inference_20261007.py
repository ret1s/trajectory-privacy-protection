"""Independent frozen-S8 causal features, fit/selection and arithmetic checker.

Does not import the production S8 runner/scorer. Rebuilds observer prefixes
from pinned tapes and independently computes features/group aggregation.
The common historical regression-bank implementation is source-pinned.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation.identity_future import RegressorBank

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/benchmarks/s8_companion_inference_20261007_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path)
    if path.exists():
        raise FileExistsError('Independent evidence is write-once')
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def same(left,right):
    if isinstance(left,dict):
        assert set(left)==set(right)
        for k in left:same(left[k],right[k])
    elif isinstance(left,list):
        assert len(left)==len(right)
        for a,b in zip(left,right):same(a,b)
    elif isinstance(left,(int,float)) and not isinstance(left,bool):
        assert np.isclose(left,right,rtol=1e-12,atol=1e-9),(left,right)
    else:assert left==right,(left,right)


def projected(points):
    points=np.asarray(points,dtype=float)
    c=np.pi/180*6371000.
    return np.c_[(points[:,1]-116.32)*c*np.cos(np.radians(40.)),(points[:,0]-40.)*c]


def clip(tape,offset,cut):
    assert set(tape)=={'events'}
    result=[]
    for event in tape['events']:
        time=float(event['timestamp_s'])+offset
        if time>cut:break
        assert set(event)=={'event_id','timestamp_s','candidates'}
        assert all(set(c)=={'candidate_id','lat','lon'} for c in event['candidates'])
        result.append((time,projected([[c['lat'],c['lon']] for c in event['candidates']])))
    assert all(b[0]>a[0] for a,b in zip(result,result[1:]))
    return result


def summary(prefix):
    assert prefix
    centres=np.array([p.mean(axis=0) for _,p in prefix])
    spreads=np.array([p.std(axis=0) for _,p in prefix])
    values=np.column_stack((centres,spreads))/1000.
    span=prefix[-1][0]-prefix[0][0]
    velocity=(centres[-1]-centres[0])/max(1.,span)
    x=np.concatenate((values[0],values[-1],values.mean(axis=0),values.std(axis=0),
                      np.quantile(values,[.25,.5,.75],axis=0).ravel(),velocity,
                      np.array([span/60.,np.log1p(len(prefix))])))
    assert len(x)==32
    return x


def inputs(target,partner,cut,joint):
    assert target[-1][0]<=cut
    a=summary(target)
    if not joint:return a
    if not partner:return np.r_[a,np.zeros(32),0.,0.,0.]
    assert partner[-1][0]<=cut
    return np.r_[a,summary(partner),(cut-target[-1][0])/60.,(cut-partner[-1][0])/60.,1.]


def location(prefix,cut,velocity=False):
    xy=np.array([p.mean(axis=0) for _,p in prefix])
    answer=xy[-1].copy()
    if velocity and len(prefix)>1:
        answer+=(cut-prefix[-1][0])*(xy[-1]-xy[0])/max(1.,prefix[-1][0]-prefix[0][0])
    return answer


def grouped(rows,prediction):
    errors=np.linalg.norm(np.array([r['truth'] for r in rows])-prediction,axis=1)
    cells=defaultdict(list)
    for row,e in zip(rows,errors):cells[(row['family_id'],row['case_id'])].append(float(e))
    families={}
    for family in sorted({f for f,_ in cells}):
        values=[]
        for f,case in sorted(cells):
            if f==family:
                v=np.array(cells[(f,case)])
                values.append(dict(case_id=case,mae_m=float(v.mean()),
                                   hit100=float(np.mean(v<=100)),events=len(v)))
        families[family]=dict(mae_m=float(np.mean([v['mae_m'] for v in values])),
            hit100=float(np.mean([v['hit100'] for v in values])),cases=values)
    return dict(family_macro_mae_m=float(np.mean([v['mae_m'] for v in families.values()])),
        family_macro_hit100=float(np.mean([v['hit100'] for v in families.values()])),
        family_values=families,events=len(rows),case_family_cells=len(cells),
        partner_missing_events=sum(not bool(r['partner']) for r in rows),
        event_pooled_mae_m=float(errors.mean()),event_pooled_hit100=float(np.mean(errors<=100)))


def source_contract(out):
    out=Path(out);p=read(out/'protocol.json')
    assert p['schema']=='historical-frozen-S8-location-diagnostic-v1'
    for f,h in p['source_sha256'].items():assert sha(ROOT/f)==sha(out/'source_snapshot'/f)==h
    groups=[set(v) for v in p['splits'].values()]
    assert all(groups) and not any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3))
    assert len(p['pairs'])==22
    pairhash=hashlib.sha256(json.dumps(p['pairs'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert pairhash==p['inventory_sha256']
    return p


def build_rows(p,view):
    data=read(ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json')
    path=ROOT/'artifacts/benchmarks/identity_future_20261005'
    meta=read(path/'linkage_results.json');public=read(path/'linkage_public_transcripts.json')['transcripts']
    assert meta['dataset_sha256']==sha(ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json')
    assert meta['public_transcripts_sha256']==sha(path/'linkage_public_transcripts.json')
    tapes={m['session_id']:t for m,t in zip(meta['evaluator_sessions'],public)}
    records={r['record_id']:r for r in data['records']}
    rows=[]
    for pair in p['pairs']:
        record=records[pair['record_id']]
        assert record['case_id'] in ('S8.A','S8.B') and record['labels']['declared_companions'] is True
        assert record['session_ids']==[pair['target_sid'],pair['partner_sid']]
        assert record['family_id']==pair['family_id']
        assert pair['family_id'] in p['splits'][pair['split']]
        starts=[data['traces'][sid][0]['time_s'] for sid in record['session_ids']]
        assert [s-min(starts) for s in starts]==[pair['target_offset_s'],pair['partner_offset_s']]
        pool=p['splits'][pair['split']]
        other=pool[(pool.index(pair['family_id'])+1)%len(pool)]
        unrelated=next(m['session_id'] for m in meta['evaluator_sessions'] if m['family_id']==other and m['role']==pair['public_role'])
        assert unrelated==pair['unrelated_sid']
        target=tapes[pair['target_sid']]
        for i,event in enumerate(target['raw']['events']):
            cut=event['timestamp_s']+pair['target_offset_s']
            raw=view=='raw_target_positive_control'
            a=clip(target['raw' if raw else 'geoi_slack_reconstructed'],pair['target_offset_s'],cut)
            b=[]
            if view.startswith('joint_'):
                sid=pair['unrelated_sid'] if 'unrelated' in view else pair['partner_sid']
                method='raw' if view=='joint_public_raw_partner' else 'geoi_slack_reconstructed'
                b=clip(tapes[sid][method],pair['partner_offset_s'],cut)
            c=event['candidates'][0]
            source=data['traces'][pair['target_sid']][i*60]
            assert (c['lat'],c['lon'])==(source['lat'],source['lon'])
            assert event['timestamp_s']==source['time_s']-starts[0]
            rows.append(dict(record_id=pair['record_id'],family_id=pair['family_id'],case_id=pair['case_id'],
                split=pair['split'],target_event_index=i,public_cut_s=cut,partner_visible=bool(b),
                target=a,partner=b,truth=projected([[c['lat'],c['lon']]])[0],
                x=inputs(a,b,cut,view.startswith('joint_'))))
    return rows


def estimates(bank,rows,view,target_bank):
    values=bank.predict(np.array([r['x'] for r in rows]))
    values['public_target_last']=np.array([location(r['target'],r['public_cut_s']) for r in rows])
    values['public_target_velocity']=np.array([location(r['target'],r['public_cut_s'],True) for r in rows])
    if view.startswith('joint_'):
        prior=target_bank.predict(np.array([summary(r['target']) for r in rows]))
        values.update({f'ignore_partner_{n}':y for n,y in prior.items()})
        values['public_partner_last']=np.array([location(r['partner'] or r['target'],r['public_cut_s']) for r in rows])
        values['public_partner_velocity']=np.array([location(r['partner'] or r['target'],r['public_cut_s'],True) for r in rows])
    return values


def declare(out=OUT):
    out=Path(out);p=source_contract(out)
    assert not (out/'attacker_selection.json').exists() and not (out/'readout.json').exists()
    value=dict(schema='independent-S8-verification-protocol-v1',protocol_sha256=sha(out/'protocol.json'),
        verifier_sha256=sha(__file__),declared_before_attacker_selection_and_test_scoring=True,
        independent_features_clock_and_family_aggregation=True,
        shared_source_pinned_regression_library='evaluation/identity_future.py')
    write(out/'verification_protocol.json',value)
    return value


def verify(out=OUT):
    out=Path(out);p=source_contract(out);v=read(out/'verification_protocol.json')
    assert v['protocol_sha256']==sha(out/'protocol.json') and v['verifier_sha256']==sha(__file__)
    assert v['declared_before_attacker_selection_and_test_scoring'] is True
    selected=read(out/'attacker_selection.json')
    assert selected['protocol_sha256']==sha(out/'protocol.json') and selected['test_predictions_not_opened'] is True
    actual=read(out/'readout.json');saved=read(out/'predictions.json')
    assert actual['selection_sha256']==sha(out/'attacker_selection.json')
    assert actual['predictions_sha256']==sha(out/'predictions.json')
    banks={};results={};predictions=[]
    for view in p['views']:
        rows=build_rows(p,view)
        train=[r for r in rows if r['split']=='train'];select=[r for r in rows if r['split']=='selection']
        test=[r for r in rows if r['split']=='test']
        bank=RegressorBank([r['x'] for r in train],[r['truth'] for r in train],p['seed']);banks[view]=bank
        candidates=estimates(bank,select,view,banks.get('target_only'))
        metrics={n:grouped(select,y) for n,y in candidates.items()}
        name=min(metrics,key=lambda n:(metrics[n]['family_macro_mae_m'],n))
        same(dict(selected=name,selection_bank=metrics),selected['selection'][view])
        candidates=estimates(bank,test,view,banks.get('target_only'))
        results[view]=dict(selected_attacker=name,test=grouped(test,candidates[name]),
            bank_test_descriptive_only={n:grouped(test,y) for n,y in candidates.items()})
        for row,y in zip(test,candidates[name]):
            predictions.append({k:row[k] for k in ('record_id','family_id','case_id','target_event_index',
                'public_cut_s','partner_visible')}|dict(view=view,
                    evaluator_truth_xy_m=row['truth'].tolist(),prediction_xy_m=y.tolist()))
    same(results,actual['results']);same(dict(evaluator_only=True,rows=predictions),saved)
    assert results['raw_target_positive_control']['test']['family_macro_mae_m']<=1e-9
    value=dict(schema='independent-S8-companion-validation-v1',status='pass',
        protocol_sha256=sha(out/'protocol.json'),verification_protocol_sha256=sha(out/'verification_protocol.json'),
        readout_sha256=sha(out/'readout.json'),verifier_sha256=sha(__file__),
        exact_clock_and_tape_ancestry=True,causal_public_features_independently_rebuilt=True,
        train_only_fits_selection_only_choices_and_predictions_reproduced=True,
        equal_family_case_denominators_recomputed=True,test_families=3,pairs=22,
        prediction_rows=len(predictions),no_group_privacy_or_current_Epoch8_claim=True)
    source_contract(out)
    if (out/'validation.json').exists():same(value,read(out/'validation.json'))
    else:write(out/'validation.json',value)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare',action='store_true');parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    print(json.dumps(declare(args.output) if args.declare else verify(args.output),indent=2))
