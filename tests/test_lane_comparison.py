"""Small independent contracts for the lane extension and enhanced-DLS."""
import json

import networkx as nx
import numpy as np
import pytest
from scipy.special import logsumexp

from benchmark.engines.enhanced_dls import EnhancedDLSGraph, distance_product_probabilities
from benchmark.engines.lane_budgeted import LaneBudgetedDummy
from core.demo_protocol import TrajectoryPoint
from core.mechanisms import RoadExponential
from core.road_network import RoadNetwork
from data.lane_states import coordinate_catalogue
from evaluation.lane_travel import SparseTravel, LanePoiService
from evaluation.scenario_metrics import PoiService
from experiments.run_lane_comparison import release


def road():
    g=nx.DiGraph(schema='sumo-lane-progress-v1',spacing_m=20.)
    for i in range(30):
        g.add_node(i,x=i*.0001,y=0.)
    for i in range(29):
        g.add_edge(i,i+1,length=10.,speed=5.)
        g.add_edge(i+1,i,length=10.,speed=5.)
    # Distinct lane states at the same public coordinate, including zero arcs.
    g.add_node(30,x=0.,y=0.)
    g.add_edge(0,30,length=0.,speed=5.)
    g.add_edge(30,0,length=0.,speed=5.)
    rn=RoadNetwork(g)
    rn.catalogue_sha256='public-test-catalogue'
    return rn


def test_coordinate_selection_deduplicates_without_changing_metric():
    rn=road()
    sites=coordinate_catalogue(rn)
    assert len(sites)==30 and len(rn)==31
    assert sites.graph is None and sites.proj is rn.proj
    assert sites.nearest(0.,0.)[0]==0
    assert np.array_equal(sites.xy,rn.xy[sites.state_indices])
    points=(TrajectoryPoint(0.,0.,.00051),)
    for name in ('dls','enhanced_dls','uniform_sets'):
        r=release(name,points,rn,np.ones(len(rn)),5,12,sites)
        cs=r['public']['events'][0]['candidates']
        assert len({(c['lat'],c['lon']) for c in cs})==5


def test_distance_product_is_weighted_sampling_not_farthest():
    xy=np.array([[0.,0.],[1.,0.],[3.,0.],[5.,0.]])
    p=distance_product_probabilities(xy,[1,2],[0,3])
    assert np.allclose(p,[4/10,6/10])
    assert np.all(p>0)  # not a greedy farthest-point indicator
    assert np.allclose(distance_product_probabilities(np.zeros((3,2)),[1,2],[0]),[.5,.5])


def test_enhanced_dls_redundancy_unique_ids_and_prefix():
    rn=coordinate_catalogue(road())
    def make():
        return EnhancedDLSGraph(rn,np.linspace(1,2,len(rn)),k=3,rng=np.random.default_rng(17))
    redundant=make().redundant_candidates(10)
    assert len(set(redundant))==6 and 10 not in redundant
    points=tuple(TrajectoryPoint(10.*i,0.,.00031+i*.0001) for i in range(4))
    whole=make().protect_run(points).to_attacker_dict()
    assert whole['events'][:2]==make().protect_run(points[:2]).to_attacker_dict()['events']
    assert whole['public_parameters']['spatial_selection']=='sequential_distance_product_sampling'
    with pytest.raises(ValueError):
        EnhancedDLSGraph(rn,np.ones(len(rn)),k=8)


def test_sparse_paths_keep_zero_edges_and_match_independent_networkx():
    rn=road()
    travel=SparseTravel(rn,cache_limit=2)
    for v in (0,30,4):
        expected=nx.single_source_dijkstra_path_length(rn.graph,v,cutoff=7.,weight=lambda a,b,d:d['length']/d['speed'])
        assert travel.reachable(v,7.)==expected
    assert len(travel.cache)==2
    assert 30 in SparseTravel(rn).reachable(0,0.)


def test_sparse_poi_query_matches_reference_implementation():
    rn=road()
    pois=[{'id':str(i),'lat':0.,'lon':i*.0001,'category':'cafe'} for i in range(1,28,3)]
    fast,reference=LanePoiService(rn,pois),PoiService(rn,pois)
    for i in (0,7,15,29):
        p=rn.latlon(i)
        assert fast.distances(p)==reference.distances(p)
        assert fast.query(p,'cafe')==reference.query(p,'cafe')


def test_lane_dummy_prefix_budget_and_no_private_read_after_horizon():
    rn=road()
    points=tuple(TrajectoryPoint(4.*i,.00002,.00005+i*.0001) for i in range(7))
    def make():
        return LaneBudgetedDummy(rn,horizon=4,rng=np.random.default_rng(33))
    model=make()
    full=model.protect_run(points).to_attacker_dict()
    assert full['events'][:3]==make().protect_run(points[:3]).to_attacker_dict()['events']
    stream=make()
    for p,e in zip(points,full['events']):
        actual=stream.protect_step(p.lat,p.lon,p.timestamp_s)
        assert list(actual)==[(c['lat'],c['lon']) for c in e['candidates']]
    assert stream.spent_bound==pytest.approx(.21)
    assert stream.protect_step(float('nan'),float('nan'),28.)==model.protect_step(70.,100.,28.)
    assert 'evaluator_states' not in json.dumps(full)
    assert 'last_anchor' not in json.dumps(full)
    assert 'spent_bound' not in json.dumps(full)
    with pytest.raises(ValueError):
        make().protect_step(float('nan'),0.,0.)
    for previous,current in zip(model.evaluator_states,model.evaluator_states[1:]):
        for a,b in zip(previous,current):
            assert nx.shortest_path_length(rn.graph,a,b,weight=lambda u,v,d:d['length']/d['speed'])<=4.+1e-9


def test_ideal_raw_anchor_bound_uses_continuous_input_distance():
    rn=road()
    mechanism=RoadExponential(.02,rn)
    x=np.array(rn.point_xy(.000031,.000049))
    y=np.array(rn.point_xy(.000031,.000051))
    _,a=mechanism._candidate_logits(x)
    _,b=mechanism._candidate_logits(y)
    ratio=(a-logsumexp(a))-(b-logsumexp(b))
    assert np.max(np.abs(ratio))<=.02*np.linalg.norm(x-y)+1e-12
    # A finite numerical check supports the formula, not an executable DP proof.


def test_projected_ablation_metadata_is_not_raw_input_claim():
    rn=road()
    p=(TrajectoryPoint(0.,0.,.000049),)
    raw=release('lane_br_raw',p,rn,np.ones(len(rn)),3,17)
    projected=release('lane_br_projected',p,rn,np.ones(len(rn)),3,17)
    assert raw['public']['public_parameters']['input_representation']=='raw_GPS_in_fixed_local_projection'
    assert projected['public']['public_parameters']['input_representation']=='nearest_lane_state_ablation'
