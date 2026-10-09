import json
import numpy as np
import pytest
from scipy.sparse.csgraph import dijkstra
from benchmark.multi_purpose_retrieval import (
    PublicRetrievalPlan,FixedMultiPurposeCoverClient,PublicPurposePoiService,round_robin_unique)
from benchmark.public_poi_context import PublicPoiContext
from evaluation.lane_travel import LanePoiService,matrix
from tests.test_query_purpose import fixture
from benchmark.query_purpose import QuerySpec,QueryPurpose


def service():
    ranking=fixture();rn=ranking.rn
    rn.catalogue_sha256='toy-variable-speed-directed-graph'
    pois=[dict(p,lat=rn.latlon(p['vertex'])[0],lon=rn.latlon(p['vertex'])[1]) for p in ranking.pois]
    context=PublicPoiContext(LanePoiService(rn,pois,k=3))
    vertices=[p['vertex'] for p in context.pois]
    distance=dijkstra(matrix(rn).transpose().tocsr(),directed=True,indices=vertices)
    time=dijkstra(matrix(rn,time=True).transpose().tocsr(),directed=True,indices=vertices)
    return ranking,PublicPurposePoiService(rn,context,distance,time,radius_m=150.,destination_states=(3,))


def test_templates_add_fastest_and_detour_candidates_without_private_destination():
    _,s=service()
    assert s.query(0,'nearest_distance',1)==[[0]]
    assert s.query(0,'fastest_travel',1)==[[1]]
    assert s.query(0,'within_radius',3)==[[0]]
    assert s.query(0,'public_detour_bank',1)==[[1]]
    assert round_robin_unique([[1,2,3],[2,4]],3)==[1,2,4]


def test_all_private_questions_use_identical_fixed_four_type_wire_and_local_answers_differ():
    ranking,s=service()
    plan=PublicRetrievalPlan(tuple((p,1) for p in ('nearest_distance','fastest_travel','within_radius','public_detour_bank')),150.,(3,))
    wires=[];answers=[]
    for query in [QuerySpec(QueryPurpose.NEAREST,'cafe',k=1),
                  QuerySpec(QueryPurpose.FASTEST,'cafe',k=1),
                  QuerySpec(QueryPurpose.WITHIN_RADIUS,'cafe',radius_m=90),
                  QuerySpec(QueryPurpose.MIN_DETOUR,'cafe',destination_state=3)]:
        c=FixedMultiPurposeCoverClient(('cafe',),3,plan,k=1)
        result=c.step(0,[s.rn.latlon(0)],s.serve)
        wires.append(json.dumps(result['requests'],sort_keys=True))
        answers.append(ranking.top(0,result['current'],query))
        assert len(result['requests'])==4
        assert np.flatnonzero(result['current']).tolist()==[0,1]
    assert len(set(wires))==1 and answers==[[0],[1],[],[1,0]]


def test_bad_public_parameters_and_private_parameters_on_wire_are_rejected():
    _,s=service()
    for channels,radius,dest in [((('unknown',10),),1000.,()),
                                ((('nearest_distance',0),),1000.,()),
                                ((('public_detour_bank',10),),1000.,()),
                                ((('nearest_distance',10),),float('nan'),())]:
        with pytest.raises(ValueError):PublicRetrievalPlan(channels,radius,dest)
    plan=PublicRetrievalPlan((('nearest_distance',1),))
    request=plan.requests(0,s.rn.latlon(0),s.categories,0)[0]
    request['private_destination']=3
    with pytest.raises(ValueError):s.serve(request)
    c=FixedMultiPurposeCoverClient(('cafe',),3,plan,k=1)
    c.step(0,[s.rn.latlon(0)],s.serve)
    with pytest.raises(ValueError):c.step(0,[s.rn.latlon(0)],s.serve)
