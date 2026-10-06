import numpy as np
import pytest

from experiments.native_cost_diagnostic import strict_query_cost


def test_q_extension_averages_all_queries_without_secret_nearest_choice():
    score = strict_query_cost([0, 100], [[0, 100], [100, 0]], [.5, .5])
    assert score["status"] == "computed"
    assert score["per_query_delta_cost_m"] == [0, 100]
    assert score["mean_of_per_query_delta_cost_m"] == 50  # not min=0
    assert score["finite_public_prior_mass_per_query"] == [1, 1]


def test_unreachable_query_is_retained_and_invalidates_full_prior_event():
    score = strict_query_cost([0, 100], [[0, 100], [20, np.inf]], [.5, .5])
    assert score["status"] == "not_available"
    assert score["mean_of_per_query_delta_cost_m"] is None
    assert score["per_query_delta_cost_m"] == [0, None]
    assert score["finite_public_prior_mass_per_query"] == [1, .5]
    score = strict_query_cost([0, np.inf], [[0, np.inf]], [.5, .5])
    assert score["mean_of_per_query_delta_cost_m"] is None  # inf-inf is not 0


def test_zero_public_prior_does_not_make_an_irrelevant_target_required():
    score = strict_query_cost([0, np.inf], [[20, np.inf]], [1, 0])
    assert score["mean_of_per_query_delta_cost_m"] == 20


def test_directed_road_costs_differ_from_reverse_and_straight_line():
    import networkx as nx
    from scipy.sparse.csgraph import dijkstra
    from evaluation.lane_travel import matrix
    from types import SimpleNamespace
    g = nx.DiGraph()
    g.add_edge(0, 1, length=10, speed=8)
    g.add_edge(1, 2, length=20, speed=8)
    g.add_edge(2, 0, length=100, speed=8)
    class Network(SimpleNamespace):
        def __len__(self):
            return 3
    cost = dijkstra(matrix(Network(graph=g)).transpose(), directed=True, indices=[0, 2]).T
    assert cost[1].tolist() == [120, 20]
    assert cost[0].tolist() == [0, 30]
    score = strict_query_cost(cost[0], [cost[1]], [.5, .5])
    assert score["mean_of_per_query_delta_cost_m"] == 65


@pytest.mark.parametrize("prior", [[1, 1], [np.nan, 1], [-1, 2]])
def test_invalid_prior_is_not_normalized_after_seeing_private_rows(prior):
    with pytest.raises(ValueError):
        strict_query_cost([0, 1], [[1, 2]], prior)
