import networkx as nx
import numpy as np
import pytest
from evaluation.public_cover import fit_public_cover
from benchmark.engines.fair_cover import CoverageObjective, exchange_refine
from benchmark.engines.service_cover import greedy_cover
from tests.test_belief_lane import fixture


@pytest.mark.parametrize('k', [1, 3, 5])
def test_public_control_matches_unreduced_selector_and_stays_feasible(k):
    rn, context, model = fixture()
    result = fit_public_cover(rn, model, context, k)
    nodes = max(nx.strongly_connected_components(rn.graph), key=len)
    viable = np.array([i for i, n in enumerate(rn.node_ids) if n in nodes])
    weights = np.asarray(model.prior @ model.poi_weights).ravel()
    objective = CoverageObjective(context.signatures, context.access, weights)
    center = np.average(model.xy, axis=0, weights=model.prior)
    ties = lambda j, ids: (np.zeros(len(ids)), np.linalg.norm(rn.xy[ids]-center, axis=1))
    groups = [viable]*k
    selected, _ = greedy_cover(groups, objective.marginal, ties)
    selected, _ = exchange_refine(groups, selected, objective, ties, 3)
    assert result['states'] == selected
    assert result['public_expected_reference_coverage'] == objective.value(selected)
    assert all(nx.has_path(rn.graph, rn.node_ids[s], rn.node_ids[s]) for s in selected)
    assert result['private_reads'] == result['coordinate_privacy_bound'] == 0
