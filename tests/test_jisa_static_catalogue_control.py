"""Application-control contracts, independent of the protected sampler."""
import json

import networkx as nx
import numpy as np
import pytest

from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from core.road_network import RoadNetwork
from evaluation.static_catalogue_control import catalogue_payload, evaluate_local_purposes
from experiments.jisa_static_catalogue_control_20261006 import summarize


def fixture():
    graph = nx.DiGraph()
    for i in range(5): graph.add_node(i, x=i*.0001, y=0.)
    # Different nearest, fastest and detour objectives; node4 unreachable.
    graph.add_edge(0, 1, length=10., speed=1.)
    graph.add_edge(0, 2, length=20., speed=10.)
    graph.add_edge(1, 3, length=100., speed=1.)
    graph.add_edge(2, 3, length=10., speed=10.)
    rn = RoadNetwork(graph)
    class Ranking:
        categories = ('cafe', 'unreachable')
        pois = ({'id':'a','category':'cafe','vertex':1,'lat':0.,'lon':.0001},
                {'id':'b','category':'cafe','vertex':2,'lat':0.,'lon':.0002},
                {'id':'u','category':'unreachable','vertex':4,'lat':0.,'lon':.0004})
    ranking = Ranking(); ranking.rn = rn
    return MultiPurposeRoadRanking(ranking)


def test_full_catalogue_answers_same_four_purpose_reference_with_n_a():
    local = fixture()
    rows = evaluate_local_purposes(local, 0, {'bulk':range(3), 'distance_only':[0]},
                                  destination_state=3, radius_m=15., k=1)
    by = {(r['purpose'],r['category']):r for r in rows}
    assert by['nearest_distance','cafe']['reference'] == [0]
    assert by['fastest_travel','cafe']['reference'] == [1]
    assert by['within_radius','cafe']['reference'] == [0]
    assert by['minimum_detour','cafe']['reference'] == [1]
    assert by['fastest_travel','cafe']['recall']['distance_only'] == 0.
    assert all(r['recall']['bulk'] == (1. if r['reference'] else None) for r in rows)
    assert all(r['recall']['bulk'] is None for r in rows if r['category']=='unreachable')
    for row in rows:
        purpose=QueryPurpose(row['purpose'])
        q=QuerySpec(purpose,row['category'],k=1,radius_m=15. if purpose==QueryPurpose.WITHIN_RADIUS else None,
                    destination_state=3 if purpose==QueryPurpose.MIN_DETOUR else None)
        assert row['reference']==local.top(0,np.ones(3,dtype=bool),q)


def test_candidate_membership_deduplicates_and_rejects_unknown_states():
    local=fixture()
    a=evaluate_local_purposes(local,0,{'x':[0,0,1]},destination_state=3)
    b=evaluate_local_purposes(local,0,{'x':[1,0]},destination_state=3)
    assert a==b
    for bad in (-1,3,True,.5):
        with pytest.raises(ValueError,match='candidate IDs'):
            evaluate_local_purposes(local,0,{'x':[bad]},destination_state=3)


def test_payload_is_coordinate_free_public_request_exact_record_response():
    local=fixture()
    result=catalogue_payload(reversed(local.pois),epoch_id='public_epoch',public_start_s=0.)
    assert set(result['request'])=={'schema','catalogue_version','epoch_id','timestamp_s'}
    assert all(set(p)=={'id','category','lat','lon'} for p in result['response']['results'])
    assert [p['id'] for p in result['response']['results']]==['a','b','u']
    encode=lambda x:json.dumps(x,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    assert result['request_bytes']==len(encode(result['request']))
    assert result['response_bytes']==len(encode(result['response']))
    changed=[dict(p) for p in local.pois];changed[0]['lat']=.001
    assert catalogue_payload(changed,epoch_id='public_epoch',public_start_s=0.)['catalogue_version']!=result['catalogue_version']
    # Local GPS/purpose/destination are absent from the payload API entirely.
    with pytest.raises(TypeError):
        catalogue_payload(local.pois,epoch_id='public_epoch',public_start_s=0.,destination_state=3)


@pytest.mark.parametrize('bad', [[],[{'id':'a','category':'c','lat':float('nan'),'lon':0.}],
    [{'id':'a','category':'c','lat':91.,'lon':0.}],
    [{'id':'a','category':'c','lat':0.,'lon':0.}]*2])
def test_invalid_catalogue_rejected(bad):
    with pytest.raises(ValueError):catalogue_payload(bad,epoch_id='e',public_start_s=0.)


def test_scope_summary_uses_equal_sessions_then_families_and_keeps_empty():
    def row(f,s,value):return {'family_id':f,'slot':s,'reference':[0] if value is not None else [],'recall':{'x':value}}
    rows=[row('a',0,1.),row('a',0,1.),row('a',1,0.),row('b',0,1.),row('b',1,None)]
    result=summarize(rows,'x')
    assert result['family_values']=={'a':.5,'b':1.}
    assert result['family_macro_recall5']==.75
    assert result['represented_session_count']==4 and result['undefined_session_count']==1
    assert result['reference_defined_queries']==4 and result['empty_reference_queries']==1
    assert summarize([row('a',0,None)],'x')['family_macro_recall5'] is None
