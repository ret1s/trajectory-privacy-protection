from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from core.road_network import RoadNetwork
from benchmark.query_purpose import (QueryPurpose, QuerySpec, MultiPurposeRoadRanking,
                                     PurposeIndependentCoverClient)


def fixture():
    graph = nx.DiGraph()
    for i in range(5): graph.add_node(i, x=116.+i/10000., y=39.9)
    for a,b,length,speed in [(0,1,100.,1.),(0,2,200.,20.),(1,3,600.,1.),
                             (2,3,50.,10.),(3,0,300.,10.)]:
        graph.add_edge(a,b,length=length,speed=speed)
    rn = RoadNetwork(graph)
    pois = tuple({'id':name,'category':'cafe','vertex':v} for name,v in [('a',1),('b',2),('c',4)])
    return MultiPurposeRoadRanking(SimpleNamespace(rn=rn,pois=pois,categories=('cafe',)))


def test_objectives_use_directed_costs_and_private_destination():
    ranking = fixture(); mask = np.ones(3,dtype=bool)
    assert ranking.top(0,mask,QuerySpec(QueryPurpose.NEAREST,'cafe',k=1)) == [0]
    assert ranking.top(0,mask,QuerySpec(QueryPurpose.FASTEST,'cafe',k=1)) == [1]
    assert ranking.top(0,mask,QuerySpec(QueryPurpose.WITHIN_RADIUS,'cafe',radius_m=150)) == [0]
    detour = QuerySpec(QueryPurpose.MIN_DETOUR,'cafe',destination_state=3)
    assert ranking.top(0,mask,detour) == [1,0]
    assert ranking.scores(0,detour).tolist()[:2] == [450.,0.]
    assert ranking.top(4,mask,detour) == []  # No route to private destination.
    assert ranking.top(0,np.zeros(3,dtype=bool),detour) == []


def test_local_candidates_cannot_include_unreceived_or_unreachable_pois():
    ranking = fixture()
    assert ranking.top(0,[False,True,True],QuerySpec(QueryPurpose.NEAREST,'cafe')) == [1]
    with pytest.raises(ValueError): ranking.top(0,[True],QuerySpec(QueryPurpose.NEAREST,'cafe'))
    with pytest.raises(ValueError): ranking.top(0,[True]*3,QuerySpec(QueryPurpose.NEAREST,'clinic'))
    for query in [QuerySpec(QueryPurpose.NEAREST,'cafe'),
                  QuerySpec(QueryPurpose.WITHIN_RADIUS,'cafe',radius_m=90)]:
        assert ranking.top(0,[True]*3,query) == ([0,1] if query.purpose==QueryPurpose.NEAREST else [])


def test_purpose_category_radius_and_destination_never_reach_wire():
    coordinates = [(39.9,116.0001),(39.9,116.0002)]
    queries = [QuerySpec(QueryPurpose.NEAREST,'cafe'), QuerySpec(QueryPurpose.FASTEST,'cafe'),
               QuerySpec(QueryPurpose.WITHIN_RADIUS,'cafe',radius_m=100),
               QuerySpec(QueryPurpose.MIN_DETOUR,'cafe',destination_state=3)]
    transcripts=[];choices=[]
    for query in queries:
        client=PurposeIndependentCoverClient(['cafe'],3,k=2)
        result=client.step(0,coordinates,lambda request:[[0,1]])
        choices.append(fixture().top(0,result['known'],query))
        transcripts.append((result['requests'],result['replies']))
        assert all(set(q)=={'schema','timestamp_s','coordinate','categories','response_l','epoch'}
                   for q in result['requests'])
    assert all(t==transcripts[0] for t in transcripts)
    assert choices[0]!=choices[1] and choices[0]!=choices[2]


def test_cover_schedule_and_epoch_cache_do_not_follow_local_queries():
    client=PurposeIndependentCoverClient(['cafe'],3,k=1)
    coords=[(39.9,116.)]
    a=client.step(0,coords,lambda request:[[0]])
    b=client.step(20,coords,lambda request:[[1]])
    c=client.step(60,coords,lambda request:[[2]])
    assert [len(r['requests']) for r in [a,b,c]]==[1,1,1]
    assert np.flatnonzero(b['known']).tolist()==[0,1]
    assert np.flatnonzero(c['known']).tolist()==[2]
    a['known'][:]=False
    with pytest.raises(ValueError):client.step(60,coords,lambda request:[[0]])
    with pytest.raises(ValueError):client.step(80,[],lambda request:[[0]])


def test_private_spec_and_public_schema_reject_invalid_inputs():
    for build in [lambda:QuerySpec('cheapest','cafe'),
                  lambda:QuerySpec(QueryPurpose.NEAREST,'cafe',k=0),
                  lambda:QuerySpec(QueryPurpose.WITHIN_RADIUS,'cafe',radius_m=float('nan')),
                  lambda:QuerySpec(QueryPurpose.MIN_DETOUR,'cafe'),
                  lambda:QuerySpec(QueryPurpose.NEAREST,'cafe',destination_state=1),
                  lambda:PurposeIndependentCoverClient(['cafe','cafe'],3),
                  lambda:PurposeIndependentCoverClient(['cafe'],0)]:
        with pytest.raises(ValueError):build()
