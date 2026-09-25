from copy import deepcopy
import numpy as np
import pytest
from benchmark.scheduled_category_client import ScheduledCategoryClient


PLAN={'queries':[{'category_index':0,'coordinate':[40.,116.]},{'category_index':1,'coordinate':[40.1,116.1]}]}


def test_public_schedule_is_identical_across_different_local_trip_boundaries():
    def run(reads):
        c=ScheduledCategoryClient(PLAN,4,end_s=180);wire=[]
        for t in (0,60,120):
            wire.append(c.tick(t,lambda q:[q['category_index'],q['epoch']%4]))
            for time in reads:
                if t<=time<t+60:c.local_snapshot(time)[:] = False
        return wire,c.local_snapshot(170)
    a,mask=run([5,25,75,100]);b,other=run([85,165]);empty,_=run([])
    assert a==b==empty;np.testing.assert_array_equal(mask,other)
    assert mask.any() # callers cannot mutate the cache through a snapshot


def test_missed_refresh_or_outside_subscription_is_not_silently_valid():
    c=ScheduledCategoryClient(PLAN,4,end_s=120)
    with pytest.raises(ValueError):c.local_snapshot(0)
    with pytest.raises(ValueError):c.tick(60,lambda _:[])
    c.tick(0,lambda _:[0])
    with pytest.raises(ValueError):c.local_snapshot(60)
    c.tick(60,lambda _:[1])
    with pytest.raises(ValueError):c.local_snapshot(120)
    with pytest.raises(ValueError):c.tick(120,lambda _:[])


def test_plan_is_frozen_and_only_public_fields_reach_callback():
    plan=deepcopy(PLAN);c=ScheduledCategoryClient(plan,4,end_s=60);plan['queries'][0]['coordinate'][0]=99
    seen=[]
    def query(q):seen.append(deepcopy(q));q['coordinate']=('mutated',);return [0]
    result=c.tick(0,query)
    assert result['requests'][0]['coordinate']==(40.,116.)
    assert all(set(q)=={'category_index','coordinate','epoch','time_s'} for q in seen)


def test_joint_posterior_removes_duplicated_prior_and_single_session_is_identity():
    from evaluation.road_endpoint_attack import RoadEndpointAttack
    a=object.__new__(RoadEndpointAttack);a.prior=np.array([.7,.2,.1])
    p=np.array([.5,.3,.2]);q=np.array([.2,.4,.4])
    np.testing.assert_allclose(a.joint([p]),p)
    np.testing.assert_allclose(a.joint([p,a.prior]),p)
    expected=p*q/a.prior;expected/=expected.sum()
    np.testing.assert_allclose(a.joint([p,q]),expected)


def test_road_distances_are_directed_and_reverse_changes_endpoint_search():
    import networkx as nx
    from core.road_network import RoadNetwork
    from evaluation.road_endpoint_attack import RoadEndpointAttack
    g=nx.DiGraph()
    for i in range(4):g.add_node(i,x=116.+i*.001,y=40.,edge_id=str(i),lane_pos_m=0.)
    g.add_edge(0,1,length=80.);g.add_edge(1,2,length=80.);g.add_edge(3,1,length=200.)
    rn=RoadNetwork(g);rn.lane_indices={str(i):[i] for i in range(4)}
    a=RoadEndpointAttack(rn,None)
    forward=dict(zip(*a.distances(1,False)));reverse=dict(zip(*a.distances(1,True)))
    assert forward=={1:0.,2:80.};assert reverse=={0:80.,1:0.,3:200.}
    assert np.isclose(a.prior.sum(),1.)
