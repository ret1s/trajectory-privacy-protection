"""Private-input boundary, public-context semantics and controlled ablations."""
import json

import networkx as nx
import numpy as np
import pytest

from benchmark.engines.contextual_lane import ContextualLaneDummy
from benchmark.engines.lane_budgeted import LaneBudgetedDummy
from benchmark.public_poi_context import PublicPoiContext
from core.demo_protocol import TrajectoryPoint
from evaluation.lane_travel import LanePoiService
from tests.test_lane_comparison import road


def context(rn, path=None):
    pois = [{'id': str(i), 'lat': 0., 'lon': i * .0001,
             'category': 'cafe' if i % 2 else 'pharmacy'} for i in range(1, 28, 3)]
    service = LanePoiService(rn, pois, k=3)
    return PublicPoiContext(service, path), service


def test_zero_weights_exactly_replay_old_generator():
    rn = road()
    points = tuple(TrajectoryPoint(i * 4., .00002, .00005 + i * .0001) for i in range(7))
    old = LaneBudgetedDummy(rn, horizon=4, rng=np.random.default_rng(13))
    new = ContextualLaneDummy(rn, horizon=4, rng=np.random.default_rng(13))
    assert old.protect_run(points).to_attacker_dict()['events'] == new.protect_run(points).to_attacker_dict()['events']
    assert old.evaluator_states == new.evaluator_states
    assert old.spent_bound == new.spent_bound


def test_poi_context_matches_direct_service_and_cache(tmp_path):
    rn = road()
    c, s = context(rn, tmp_path/'poi.npz')
    for state in range(len(rn)):
        for j, category in enumerate(c.categories):
            actual = [c.pois[i]['id'] for i in c.query_indices(state)[j] if i >= 0]
            assert actual == s.query(rn.latlon(state), category)
    cached, _ = context(rn, tmp_path/'poi.npz')
    assert cached.sha256 == c.sha256
    assert np.array_equal(c.signatures, cached.signatures)


def test_marginal_cover_does_not_reward_repeated_answers():
    rn = road()
    c, _ = context(rn)
    w = c.reference_weights(rn.latlon(10))
    assert w.sum() == pytest.approx(1.)
    a = c.marginal_gain([10], w, [])
    b = c.marginal_gain([10], w, [10])
    assert a[0] == pytest.approx(1.) and b[0] == 0.
    assert np.all(c.marginal_gain(np.arange(len(rn)), w, []) <= 1 + 1e-12)


@pytest.mark.parametrize('route,cover', [(0., 0.), (.5, 0.), (0., 6.), (.5, 6.)])
def test_contextual_prefix_anchor_pairing_and_motion(route, cover):
    rn = road()
    c, _ = context(rn)
    points = tuple(TrajectoryPoint(i * 4., .00002, .00005 + i * .0001) for i in range(7))
    def make():
        return ContextualLaneDummy(rn, context=c, route_weight=route, coverage_weight=cover,
                                   horizon=4, rng=np.random.default_rng(17))
    model = make()
    full = model.protect_run(points).to_attacker_dict()
    assert make().protect_run(points[:3]).to_attacker_dict()['events'] == full['events'][:3]
    control = ContextualLaneDummy(rn, horizon=4, rng=np.random.default_rng(17))
    control.protect_run(points)
    assert model.evaluator_anchors == control.evaluator_anchors
    assert model.spent_bound == pytest.approx(.21)
    assert 'evaluator' not in json.dumps(full)
    same = make()
    same.protect_run(points)
    assert model.protect_step(float('nan'), float('nan'), 28.) == same.protect_step(10., 20., 28.)
    for a, b in zip(model.evaluator_states, model.evaluator_states[1:]):
        for u, v in zip(a, b):
            assert nx.shortest_path_length(rn.graph, u, v, weight=lambda i,j,d:d['length']/d['speed']) <= 4. + 1e-9


def test_postprocessing_uses_only_protected_anchor_not_raw_input():
    rn = road()
    c, _ = context(rn)
    def make():
        m = ContextualLaneDummy(rn, context=c, route_weight=.5, coverage_weight=6., rng=np.random.default_rng(1))
        m.anchor.perturb = lambda *args, **kwargs: rn.latlon(15)
        return m
    a, b = make(), make()
    for i in range(4):
        assert a.protect_step(0., 0., i*4.) == b.protect_step(0., .0029, i*4.)


def test_goal_distance_is_directed():
    rn = road()
    rn.graph[1][0]['length'] = 100.
    m = ContextualLaneDummy(rn, route_weight=.5)
    values = m.distances_to_goal(rn.xy[0])
    expected = nx.single_source_dijkstra_path_length(rn.graph.reverse(), 0, weight='length')
    assert values[1] == 100.
    assert all(values[i] == expected[i] for i in expected)
    assert values[1] != nx.shortest_path_length(rn.graph, 0, 1, weight='length')


def test_goal_cache_evicts_and_reset_drops_state():
    from core.road_network import RoadNetwork
    g = nx.DiGraph(schema='sumo-lane-progress-v1', spacing_m=20.)
    for i in range(40):
        g.add_node(i, x=i*.0001, y=0.)
    for i in range(39):
        g.add_edge(i, i+1, length=10., speed=5.)
        g.add_edge(i+1, i, length=10., speed=5.)
    rn = RoadNetwork(g)
    m = ContextualLaneDummy(rn, route_weight=.5)
    for i in range(40):
        m.distances_to_goal(rn.xy[i])
    assert len(m.goal_cache) == 32 and 0 not in m.goal_cache
    assert list(m.goal_cache) == list(range(8, 40))
    m.distances_to_goal(rn.xy[8])
    assert next(reversed(m.goal_cache)) == 8
    m.reset()
    assert not m.goal_cache and not m.evaluator_anchors


def test_stale_public_context_is_rejected(tmp_path):
    rn = road()
    c, s = context(rn, tmp_path/'poi.npz')
    s.k += 1
    with pytest.raises(ValueError, match='Stale public context'):
        PublicPoiContext(s, tmp_path/'poi.npz')
    assert c.k == 3


def test_rendered_scores_retain_every_case_and_variant():
    from experiments.render_contextual_lane import render
    from experiments.run_contextual_lane import CASES, METHODS
    payload = {'summaries': [
        {'split':'development_test','k':k,'case_id':case,'method':name,
         'poi_recall':.7653,'audit_max_hit100':.1429,'audit_min_mae_m':506.414}
        for k in (3, 5) for case in CASES for name in METHODS]}
    text = render(payload)
    assert text.count('76.53 & 14.29 & 506.4') == 40
    assert all(text.count(case) == 2 for case in CASES)


@pytest.mark.parametrize('params', [{'route_weight': 2.}, {'coverage_weight': -1.}, {'coverage_weight': 1.}])
def test_invalid_context_parameters(params):
    with pytest.raises(ValueError):
        ContextualLaneDummy(road(), **params)


def test_configuration_selection_ignores_one_ulp_quality_difference():
    from experiments.run_contextual_lane import choose_configuration
    candidates = [{'method':'baseline','min_case_recall':.7833333333333334,'macro_hit100':.19},
                  {'method':'combined','min_case_recall':.7833333333333333,'macro_hit100':.13}]
    result = choose_configuration(candidates)
    assert result['chosen']['method'] == 'combined' and not result['utility_feasible']
