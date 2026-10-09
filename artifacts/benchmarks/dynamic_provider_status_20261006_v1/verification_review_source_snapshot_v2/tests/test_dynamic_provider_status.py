"""Controlled dynamic-provider fixtures independent of study data/performance."""
from types import SimpleNamespace
import hashlib
import json

import networkx as nx
import numpy as np
import pytest

from benchmark.query_purpose import MultiPurposeRoadRanking,QueryPurpose,QuerySpec
from core.road_network import RoadNetwork
from experiments.dynamic_provider_status_20261006 import (
    CurrentStatusCache,DynamicLocalEvaluator,epoch_at,requests,score_cases,status_world,
    cache_context_contract,
)


def graph_fixture():
    graph=nx.DiGraph()
    for i in range(5):graph.add_node(i,x=116.+i/10000.,y=39.9)
    for a,b,d,s in [(0,1,100.,1.),(0,2,200.,20.),(1,3,600.,1.),(2,3,50.,10.),(3,0,300.,10.)]:
        graph.add_edge(a,b,length=d,speed=s)
    rn=RoadNetwork(graph)
    pois=tuple(dict(id=name,category='cafe',vertex=v) for name,v in [('a',1),('b',2),('c',4)])
    ranking=MultiPurposeRoadRanking(SimpleNamespace(rn=rn,pois=pois,categories=('cafe',)))
    return ranking,DynamicLocalEvaluator(ranking)


def test_workload_public_id_hash_stable_under_catalogue_order_without_gps_inputs():
    a=status_world(['a','b','c'],[0,25,26]);b=status_world(['c','a','b'],[26,0,25])
    assert a['client_world_or_seed_access'] is False
    for epoch in a['snapshots']:
        assert dict(zip(a['poi_ids'],a['snapshots'][epoch]))==dict(zip(b['poi_ids'],b['snapshots'][epoch]))
    assert a==status_world(['a','b','c'],[0,25,26])


def test_unavailable_filtered_and_unknown_is_not_an_unavailable_status():
    cache=CurrentStatusCache(3)
    current,cumulative=cache.receive(0,[0,1],[False,True])
    assert current[1].tolist()==[True,True,False]
    assert current[2].tolist()==[False,True,False]
    cases={'nearest_distance':[np.array([0,1,2])]};truth=np.array([False,True,True])
    scored=score_cases(cases,truth,*current)['nearest_distance']
    assert scored['recall5']==.5 and scored['returned_items']==1
    assert scored['current_known_unavailable_candidates']==1
    assert scored['retrieval_miss_reference_pois']==1 and scored['returned_unavailable_items']==0


def test_static_records_survive_but_dynamic_status_expires_at_public_epoch_boundary():
    cache=CurrentStatusCache(3);cache.receive(0,[0],[True])
    _,inside=cache.receive(59,[1],[True]);assert inside[1].tolist()==[True,True,False]
    _,after=cache.receive(60,[2],[True])
    assert after[0].tolist()==[True,True,True] and after[1].tolist()==[False,False,True]
    scores=score_cases({'nearest_distance':[np.array([0,1,2])]},[True]*3,*after)['nearest_distance']
    assert scores['recall5']==pytest.approx(1/3)
    assert scores['current_status_unknown_reference_pois']==2 and scores['current_status_unknown_candidates']==2
    assert not scores['returned_current_status_unknown_items']


def test_absolute_departure_prevents_relative_session_epoch_carry_bug():
    assert epoch_at(0)==0 and epoch_at(1500)==25 and epoch_at(1500+60)==26
    cache=CurrentStatusCache(2);cache.receive(600,[0],[True]);_,masks=cache.receive(1500,[1],[True])
    assert masks[0].tolist()==[True,True] and masks[1].tolist()==[False,True]
    with pytest.raises(ValueError):cache.receive(20,[0],[True])


def test_empty_reference_na_and_zero_candidates_nonempty_reference_zero():
    cases={'nearest_distance':[np.array([0,1])],'within_radius':[np.array([],int)]};empty=np.zeros(2,bool)
    value=score_cases(cases,[True,False],empty,empty,empty)
    assert value['nearest_distance']['recall5']==0. and value['nearest_distance']['reference_exists_zero_answer_categories']==1
    assert value['within_radius']['recall5'] is None and value['within_radius']['empty_reference_categories']==1
    none=score_cases(cases,empty,[True]*2,[True]*2,empty)
    assert all(v['recall5'] is None for v in none.values())


def test_full_current_oracle_exact_for_four_different_graph_queries():
    ranking,evaluator=graph_fixture();cases=evaluator.references(0,3);mask=np.array([True,True,False])
    assert cases['nearest_distance'][0].tolist()==[0,1]
    assert cases['fastest_travel'][0].tolist()==[1,0]
    assert cases['minimum_detour'][0].tolist()==[1,0]
    scores=score_cases(cases,mask,[True]*3,[True]*3,mask)
    assert set(scores)=={p.value for p in QueryPurpose}
    assert all(v['recall5']==1. and not v['returned_unavailable_items'] for v in scores.values())


def test_stale_full_catalogue_safe_unknown_and_unsafe_control_false_availability():
    cases={'nearest_distance':[np.array([0,1])]};truth=[False,True];static=[True,True];unknown=[False,False]
    safe=score_cases(cases,truth,static,unknown,[False,False])['nearest_distance']
    unsafe=score_cases(cases,truth,static,unknown,[True,False],unsafe_candidates=[True,False])['nearest_distance']
    assert safe['recall5']==0. and safe['current_status_unknown_reference_pois']==1
    assert unsafe['returned_unavailable_items']==unsafe['returned_current_status_unknown_items']==1
    assert unsafe['recall5']==0.


def test_top_static_reply_with_status_differs_from_available_first_contract():
    ranking,evaluator=graph_fixture()
    static_reply=[0];world=[False,True,False];mask=[True,False,False]
    scores=score_cases(evaluator.references(0,3),world,mask,mask,[False]*3)
    assert static_reply==[0] and scores['nearest_distance']['recall5']==0.
    assert ranking.top(0,world,QuerySpec(QueryPurpose.NEAREST,'cafe'))==[1]


def test_private_query_purpose_changes_local_answer_not_request_bytes_or_order():
    ranking,evaluator=graph_fixture();positions=[(39.9,116.+i/10000.) for i in range(5)]
    network=requests(1500,positions,('cafe',),20)
    answers=[]
    for purpose in QueryPurpose:
        query=QuerySpec(purpose,'cafe',radius_m=150 if purpose==QueryPurpose.WITHIN_RADIUS else None,
            destination_state=3 if purpose==QueryPurpose.MIN_DETOUR else None)
        answers.append(ranking.top(0,[True,True,False],query))
        assert requests(1500,positions,('cafe',),20)==network
    assert answers[0]!=answers[1] and answers[0]!=answers[2]
    assert all(set(r)=={'timestamp_s','lat','lon','categories','L','status_epoch','include_availability'} for r in network)


@pytest.mark.parametrize('fault',['backwards','duplicate_clock','conflicting_bits','nonbool','bad_id'])
def test_causal_cache_rejects_malformed_or_conflicting_statuses(fault):
    cache=CurrentStatusCache(2);cache.receive(60,[0],[True])
    with pytest.raises(ValueError):
        if fault=='backwards':cache.receive(0,[1],[True])
        elif fault=='duplicate_clock':cache.receive(60,[1],[True])
        elif fault=='conflicting_bits':cache.receive(80,[0],[False])
        elif fault=='nonbool':cache.receive(80,[1],[1])
        else:cache.receive(80,[2],[True])


def test_provider_known_current_true_bit_cannot_contradict_ground_truth():
    with pytest.raises(ValueError):score_cases({'nearest_distance':[np.array([0])]},[False],[True],[True],[True])


def test_invalid_status_reply_does_not_mutate_clock_or_invalidate_previous_epoch():
    cache=CurrentStatusCache(2);cache.receive(60,[0],[True])
    with pytest.raises(ValueError):cache.receive(120,[1,9],[True,False])
    assert cache.epoch==1 and cache.last_t==60
    assert cache.static.tolist()==cache.known.tolist()==cache.available.tolist()==[True,False]


@pytest.mark.parametrize('fault',[None,'metadata','access','signature'])
def test_public_cache_semantics_bound_before_workload_ids(tmp_path,fault):
    metadata=dict(schema='public-poi-context-v1',k=60,catalogue_sha256='native',categories=['cafe'],
                  pois=[dict(id='a',category='cafe')],builder_sha256='builder')
    encoded=json.dumps(metadata,sort_keys=True,separators=(',',':'))
    signatures=np.full((2,1,60),-1,dtype='<i4');signatures[:,0,0]=0;access=np.array([0,1],dtype='<i4')
    digest=hashlib.sha256(encoded.encode());digest.update(signatures.tobytes());digest.update(access.tobytes());pin=digest.hexdigest()
    if fault=='metadata':metadata['pois'][0]['id']='replaced';encoded=json.dumps(metadata,sort_keys=True,separators=(',',':'))
    elif fault=='access':access[0]=1
    elif fault=='signature':signatures[0,0,0]=-1
    cache=tmp_path/'public.npz';np.savez_compressed(cache,metadata=encoded,signatures=signatures,access=access)
    if fault is None:assert cache_context_contract(cache,pin,'native')==metadata
    else:
        with pytest.raises(AssertionError):cache_context_contract(cache,pin,'native')
