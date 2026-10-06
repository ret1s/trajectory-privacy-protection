import copy
import numpy as np
import pytest
from evaluation.identity_future import (prefix_features,pair_features,public_arrays,
    check_group_split,classification_metrics,location_metrics,ClassifierBank,RegressorBank)


def public(offset=0.,k=5):
    return {'events':[{'event_id':f'e{i}','timestamp_s':60.*i,
        'candidates':[{'candidate_id':f'q{j}','lat':40.+offset+i*.001+j*.00001,
                       'lon':116.32+j*.00002} for j in range(k)]} for i in range(3)]}


def test_private_truth_cannot_enter_features():
    base=public()
    for level,key,value in [('window','person_id','p1'),('event','source_timestamp_s',0.),
                            ('candidate','is_real',True),('candidate','vertex',123)]:
        item=copy.deepcopy(base)
        target=item if level=='window' else item['events'][0] if level=='event' else item['events'][0]['candidates'][0]
        target[key]=value
        with pytest.raises(ValueError):prefix_features(item)


def test_candidate_order_and_pair_direction_are_invariant():
    a,b=public(),public(.001)
    reordered=copy.deepcopy(a)
    for e in reordered['events']:e['candidates'].reverse()
    assert np.allclose(prefix_features(a),prefix_features(reordered))
    assert np.array_equal(pair_features(a,b),pair_features(b,a))


def test_relative_clock_and_variable_candidate_count():
    a=public(k=1);b=copy.deepcopy(a)
    for e in b['events']:e['timestamp_s']+=3600.
    assert np.array_equal(prefix_features(a),prefix_features(b))
    assert prefix_features(a).shape==prefix_features(public(k=5)).shape


def test_bad_public_clocks_and_coordinates_rejected():
    a=public();a['events'][1]['timestamp_s']=a['events'][0]['timestamp_s']
    with pytest.raises(ValueError):public_arrays(a)
    a=public();a['events'][0]['candidates'][0]['lat']=float('nan')
    with pytest.raises(ValueError):public_arrays(a)
    with pytest.raises(ValueError):public_arrays({'events':[]})


def test_group_disjointness_includes_every_linked_family():
    assert check_group_split(['f1','f2'],['f3'],['f4'])
    with pytest.raises(ValueError):check_group_split(['f1'],['f2'],['f1'])
    with pytest.raises(ValueError):check_group_split([],['f2'],['f3'])


def test_imbalanced_linkage_requires_balanced_metrics():
    truth=np.array([0]*9+[1]);pred=np.zeros(10,dtype=int)
    m=classification_metrics(truth,pred,np.ones(10)*.1)
    assert m['accuracy']==.9 and m['balanced_accuracy']==.5 and m['roc_auc']==.5


def test_location_error_uses_metres_and_declared_radii():
    m=location_metrics([[0.,0.],[0.,0.]],[[30.,40.],[300.,400.]])
    assert m['mae_m']==275. and m['hit50']==.5 and m['hit100']==.5 and m['hit500']==1.


def test_attack_scaler_is_frozen_after_training():
    x=np.arange(40,dtype=float).reshape(20,2);y=np.arange(20)%2
    bank=ClassifierBank(x,y,seed=42);old=bank.scale.mean_.copy()
    pred=bank.predict(x+100000.);prob=bank.probability(x+100000.)
    assert np.array_equal(bank.scale.mean_,old)
    assert all(len(v)==20 for v in pred.values())
    assert all(np.logical_and(v>=0,v<=1).all() for v in prob.values())
    reg=RegressorBank(x,np.c_[x[:,0],x[:,1]],seed=42)
    assert all(p.shape==(20,2) for p in reg.predict(x).values())


def test_public_track_shape_matching_ignores_track_names_and_pair_order():
    from experiments.identity_future_refine import shape_features
    a,b=public(),public(.002)
    reordered=copy.deepcopy(a)
    for event in reordered['events']:
        event['candidates'].reverse()
        for c in event['candidates']:c['candidate_id']='arbitrary_public_name'
    assert np.allclose(shape_features(a,b),shape_features(reordered,b))
    assert np.allclose(shape_features(a,b),shape_features(b,a))


def test_fixed_prefix_cut_withholds_future_event_from_attack_features():
    from experiments.identity_future_prefix_cut import cut_rows
    trace=[{'lat':40.+i*.00001,'lon':116.32} for i in range(61)]
    data={'records':[{'record_id':'r','session_ids':['s'],'observed_indices':[[0,20,40]]}],
          'traces':{'s':trace}}
    row={'record_id':'r','public':public(),'method':'raw','family_id':'f','split':'test','rep':0}
    a,rejected=cut_rows([row],data)
    assert not rejected and len(a[0]['public']['events'])==2
    assert a[0]['evaluator_only']['target_index']==40
    before=prefix_features(a[0]['public'])
    changed=copy.deepcopy(row)
    changed['public']['events'][-1]['candidates'][0]['lat']=41.
    b,_=cut_rows([changed],data)
    assert np.array_equal(before,prefix_features(b[0]['public']))
