"""Independent fixed-Q prefix/cost/conditional summary adversarial checks."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.query_purpose import QueryPurpose
from experiments import verify_qplanner_response_depth_20261006 as audit


def fixture():
    pois=tuple(dict(id=f'p{i:02d}é',category='food',lat=float(i),lon=0.) for i in range(60))
    signatures=np.array([[list(range(20,40))+list(range(20))+list(range(40,60))],[list(range(60))]])
    full=SimpleNamespace(pois=pois,categories=('food',),query_indices=lambda state:signatures[int(state)])
    rn=SimpleNamespace(nearest=lambda lat,lon:(int(lat),0.))
    class Ranking:
        n=60;categories=('food',)
        def top(self,state,mask,spec):
            if spec.purpose==QueryPurpose.WITHIN_RADIUS:return []
            return [i for i in range(60) if mask[i]][:spec.k]
    local=Ranking();family={'evaluator_only':{'sessions':[{'depart_s':s*1500} for s in range(8)]}}
    bundle={'public':{'streams':{'raw':[],audit.METHOD:[]}},'evaluator_only':{
        'family_id':'a','split':'selection','draw':1,'sessions':[{'ledger':{audit.METHOD:{'anchors':[s],
            'ledger':[s],'step_ms':[.1]}}} for s in range(8)]},'utility':[],'wire':[]}
    baseline={p:dict(recall5=None if p=='within_radius' else 0.,completion=None if p=='within_radius' else 1.,
        reference_category_count=0 if p=='within_radius' else 1,all_category_count=1,
        overlap_total=0,reference_poi_total=0 if p=='within_radius' else 5) for p in audit.PURPOSES}
    for slot,spec in enumerate(family['evaluator_only']['sessions']):
        raw=[];events=[]
        for j,t in enumerate((0,600)):
            e=f'e{j}';raw.append(dict(event_id=e,timestamp_s=float(t),candidates=[dict(lat=1.,lon=0.)]))
            events.append(dict(event_id=e,timestamp_s=float(t),candidates=[dict(lat=0.,lon=0.)]*5))
            fields=dict(family_id='a',split='selection',draw=1,slot=slot,method=audit.METHOD,event_id=e,t=t)
            ids=list(range(20,40));records=[{k:pois[i][k] for k in ('id','category','lat','lon')} for i in ids]
            payload=dict(timestamp_s=spec['depart_s']+t,lat=0.,lon=0.,categories=['food'],L=20)
            bundle['wire'].append(dict(fields,requests=5,request_bytes=5*len(json.dumps(payload,separators=(',',':')).encode()),
                reply_bytes=5*len(json.dumps({'results':records},separators=(',',':'),ensure_ascii=False).encode()),reply_poi_ids_by_Q=[ids]*5))
            for policy in ('current','static_epoch_cache'):
                bundle['utility'].append(dict(fields,cache=policy,purposes=deepcopy(baseline),available_count=20))
        bundle['public']['streams']['raw'].append({'events':raw})
        bundle['public']['streams'][audit.METHOD].append({'events':events})
    return bundle,family,rn,full,local


def test_independent_replay_prefix_cost_integer_clock_and_causal_monotonicity():
    bundle,family,rn,full,local=fixture();before=deepcopy(bundle)
    result=audit.replay_expected(bundle,family,rn,full,local)
    assert bundle==before
    assert len(result['wire'])==64 and len(result['utility'])==128
    assert result['frozen_controls']['Q_not_regenerated'] and result['frozen_controls']['private_reads_not_performed']
    for policy in ('current','static_epoch_cache'):
        rows=[r for r in result['utility'] if r['slot']==0 and r['t']==0 and r['cache']==policy]
        assert [r['purposes']['nearest_distance']['recall5'] for r in rows]==[0.,1.,1.,1.]
        assert all(r['purposes']['within_radius']['recall5'] is None for r in rows)
    assert all(isinstance(r['t'],int) for r in result['wire'])
    assert result['wire'][0]['reply_bytes']<result['wire'][1]['reply_bytes']


@pytest.mark.parametrize('fault',['Q','recall','cost','missing','duplicate','clock'])
def test_independent_replay_rejects_material_baseline_faults(fault):
    bundle,family,rn,full,local=fixture()
    if fault=='Q':bundle['public']['streams'][audit.METHOD][0]['events'][0]['candidates'][0]['lat']=1.
    elif fault=='recall':bundle['utility'][0]['purposes']['nearest_distance']['recall5']=.1
    elif fault=='cost':bundle['wire'][0]['reply_bytes']+=1
    elif fault=='missing':bundle['utility'].pop()
    elif fault=='duplicate':bundle['wire'].append(deepcopy(bundle['wire'][0]))
    else:bundle['public']['streams'][audit.METHOD][0]['events'][0]['timestamp_s']=1.
    with pytest.raises((AssertionError,KeyError)):audit.replay_expected(bundle,family,rn,full,local)


@pytest.mark.parametrize('fault',['recall','completion','defined','reference'])
def test_monotonic_check_includes_completion_and_na_not_just_recall(fault):
    value={p:dict(recall5=.4,completion=1.,reference_category_count=1,all_category_count=1,
                  reference_poi_total=5,overlap_total=2) for p in audit.PURPOSES};other=deepcopy(value)
    key={'recall':'recall5','completion':'completion','defined':'recall5','reference':'reference_poi_total'}[fault]
    other[audit.PURPOSES[0]][key]=None if fault=='defined' else 0.
    with pytest.raises(AssertionError):audit.monotonic(value,other)


def summary_fixture():
    out={}
    for L,gain,ratio in ((20,0.,1.),(30,.019,1.4),(40,.025,2.),(60,.04,3.)):
        p={name:dict(family_values={'a':.7+gain,'b':.9+gain},defined_windows=4,total_windows=5,
            defined_categories=4,total_categories=5) for name in audit.PURPOSES}
        p['equal_purpose_macro']={'family_values':{'a':.7+gain,'b':.9+gain}}
        out[str(L)]={'selection':{'current':{'all':p},'cost':{'requests':10,'reply_bytes':1000*ratio}}}
    return out


def test_independent_selector_smallest_fixed_gate_and_none():
    result=audit.independent_selection(summary_fixture())
    assert result['selected_depth']==40 and result['candidates'][0]['eligible'] is False
    assert result['candidates'][2]['gates']['reply_byte_ratio_at_most_2_5'] is False
    value=summary_fixture();value['40']['selection']['cost']['reply_bytes']=2501
    assert audit.independent_selection(value)['selected_depth'] is None


@pytest.mark.parametrize('fault',['NA','coverage','requests','empty'])
def test_independent_selector_rejects_missing_pairs_or_changed_coverage(fault):
    value=summary_fixture();row=value['40']['selection']
    if fault=='NA':row['current']['all']['equal_purpose_macro']['family_values']['a']=None
    elif fault=='coverage':row['current']['all']['within_radius']['defined_windows']+=1
    elif fault=='requests':row['cost']['requests']+=1
    else:row['current']['all']['nearest_distance']['family_values']={}
    with pytest.raises(AssertionError):audit.independent_selection(value)


def test_summary_equal_family_nested_draw_and_whole_na():
    bundle,family,rn,full,local=fixture();result=audit.replay_expected(bundle,family,rn,full,local)
    rows=[];wires=[]
    for fam,values in (('a',(.1,.2,.3)),('b',(.9,))):
        for draw,value in enumerate(values,1):
            for row in result['utility']:
                record=deepcopy(row);record.update(family_id=fam,draw=draw)
                for p in audit.PURPOSES:
                    if p!='within_radius':record['purposes'][p]['recall5']=value
                rows.append(record)
            for row in result['wire']:wires.append(dict(row,family_id=fam,draw=draw))
    summary=audit.independent_summary(rows,wires,['selection'])['20']['selection']['current']['all']
    assert summary['nearest_distance']['family_mean']==pytest.approx(.55)
    assert summary['within_radius']['family_mean'] is None
    assert summary['equal_purpose_macro']['family_mean']==pytest.approx(.55)
