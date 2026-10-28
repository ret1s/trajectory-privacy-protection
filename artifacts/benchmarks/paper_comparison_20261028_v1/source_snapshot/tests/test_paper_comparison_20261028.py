"""Meaningful observation-window, data-boundary and accounting checks."""
import numpy as np
import pytest
from experiments import paper_comparison_20261028 as study


def event(t,point=(0.,0.)):
    return dict(timestamp_s=t,coordinates=[list(point)],real=True,Z=[12,34])


def test_public_time_window_keeps_all_fake_events_and_strips_private_fields():
    events=[event(t) for t in (520,540,545,555,560,580,599,600)]
    view=study.public_events(events,540,600)
    assert [e['timestamp_s'] for e in view]==[540,545,555,560,580,599,600]
    assert all(set(e)=={'timestamp_s','coordinates'} for e in view)
    assert events[0]['real'] is True


def test_invalid_or_duplicate_public_clock_rejected():
    with pytest.raises(ValueError):study.public_events([event(0),event(0)])
    with pytest.raises(ValueError):study.public_events([event(0,(float('nan'),1))])
    with pytest.raises(ValueError):study.public_events([event(0)],540,600)


def test_cache_bit_encoding_retains_non_byte_aligned_catalogue():
    mask=np.random.default_rng(5).integers(0,2,418).astype(bool)
    assert np.array_equal(study.unpacked(study.packed(mask),418),mask)


def fake_features(events,scenario,rn,history):
    rows=np.array([[e['timestamp_s']] for e in events])
    geometry={'clock_xy':np.c_[rows[:,0],rows[:,0]]}
    if scenario=='S2':return rows.mean(axis=0,keepdims=True),{k:v.mean(axis=0,keepdims=True) for k,v in geometry.items()}
    return rows,geometry


def test_s2_uses_time_window_not_last_four_packets(monkeypatch):
    captured=[]
    def feature(events,*args):captured.extend(e['timestamp_s'] for e in events);return fake_features(events,*args)
    monkeypatch.setattr(study,'features_and_geometry',feature)
    truth=[dict(t=t,xy=[t,0],state=0) for t in (520,540,560,580,600)]
    trip=dict(truth=truth,events={'m':[event(t) for t in (520,540,545,555,560,580,599,600)]})
    _,_,target=study.attack_sample(trip,'S2','m',None,None)
    assert captured==[540,545,555,560,580,599,600]
    assert target.tolist()==[[570.,0.]]


def test_s3_observes_extra_packets_but_targets_same_public_gps_grid(monkeypatch):
    monkeypatch.setattr(study,'features_and_geometry',fake_features)
    truth=[dict(t=t,xy=[t,0],state=0) for t in (0,20,40)]
    trip=dict(truth=truth,events={'m':[event(t) for t in (0,7,12,20,29,40)]})
    features,geometry,target=study.attack_sample(trip,'S3','m',None,None)
    assert features['shadow'][:,0].tolist()==[0,20,40]
    assert geometry['clock_xy'][:,0].tolist()==[0,20,40]
    assert target[:,0].tolist()==[0,20,40]


def test_replay_counts_fake_queries_and_applies_cache_causally():
    class RN:
        def nearest(self,lat,lon):return int(lat),0
    class Reply:
        categories=('c',);pois=(0,1,2);sha256='public'
        def query_indices(self,state):return [np.array([state])]
    class Utility:
        def score(self,state,destination,mask):return {'count':int(mask.sum())}
    events=[event(t,(s,0.)) for t,s in ((0,0),(7,1),(12,2),(20,0),(40,0))]
    truth=[dict(t=t,state=0) for t in (0,20,40)]
    result=study.replay(events,dict(session_id='toy',truth=truth),RN(),Reply(),Utility())
    assert result['cost']['requests']==result['cost']['packets']==5
    assert [r['session']['count'] for r in result['utility']]==[1,3,3]
    assert [r['current']['count'] for r in result['utility']]==[1,1,1]


def test_s1_uses_only_one_public_packet_per_service_target(monkeypatch):
    counts=[]
    def features(events,*args):counts.append(len(events));return fake_features(events,*args)
    monkeypatch.setattr(study,'features_and_geometry',features)
    trip=dict(truth=[dict(t=t,xy=[t,0],state=0) for t in (0,20,40)],
              events={'m':[event(t) for t in (0,7,20,29,40)]})
    x,_,_=study.attack_sample(trip,'S1','m',None,None)
    assert counts==[1,1,1] and len(x['shadow'])==3
