"""The negative native result must not become a universal rejection."""
from types import SimpleNamespace
import networkx as nx
import numpy as np
from scipy.sparse.csgraph import dijkstra
from core.road_network import RoadNetwork
from benchmark.query_purpose import MultiPurposeRoadRanking,QuerySpec,QueryPurpose
from benchmark.public_poi_context import PublicPoiContext
from benchmark.multi_purpose_retrieval import PublicPurposePoiService
from evaluation.lane_travel import LanePoiService,matrix


def test_distinct_speed_objectives_can_improve_macro_recall_at_equal_record_ceiling():
    graph=nx.DiGraph()
    for i in range(4):graph.add_node(i,x=116.+i*.0001,y=39.9)
    for target,length,speed in [(1,100.,1.),(2,110.,1.),(3,200.,20.)]:
        graph.add_edge(0,target,length=length,speed=speed)
    rn=RoadNetwork(graph);rn.catalogue_sha256='variable-speed-purpose-counterexample'
    pois=[dict(id=f'p{i}',category='cafe',vertex=i,lat=rn.latlon(i)[0],lon=rn.latlon(i)[1]) for i in (1,2,3)]
    context=PublicPoiContext(LanePoiService(rn,pois,k=3))
    vertices=[p['vertex'] for p in context.pois]
    distance=dijkstra(matrix(rn).transpose().tocsr(),directed=True,indices=vertices)
    travel=dijkstra(matrix(rn,time=True).transpose().tocsr(),directed=True,indices=vertices)
    service=PublicPurposePoiService(rn,context,distance,travel,radius_m=1000.,destination_states=(3,))
    baseline=set(service.query(0,'nearest_distance',2)[0])
    diversified=set(service.query(0,'nearest_distance',1)[0]+service.query(0,'fastest_travel',1)[0])
    assert len(baseline)==len(diversified)==2
    ranking=MultiPurposeRoadRanking(SimpleNamespace(rn=rn,pois=context.pois,categories=context.categories))
    all_ids=np.ones(3,bool)
    reference=[ranking.top(0,all_ids,QuerySpec(p,'cafe',k=1))[0] for p in (QueryPurpose.NEAREST,QueryPurpose.FASTEST)]
    assert sum(p in baseline for p in reference)/2==.5
    assert sum(p in diversified for p in reference)/2==1.
