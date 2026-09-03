"""Equation-level checks for the clean-room TransProtect paper core."""

import networkx as nx
import numpy as np

from benchmark.contracts import ImplementationLevel, MethodUnavailableError
from benchmark.engines.transprotect import (
    ArrayProbabilityProvider,
    CausalTransitionProbabilityProvider,
    TargetTravelCostUtility,
    TransProtectEngine,
    expected_travel_cost_loss,
    restricted_laplace_probabilities,
    select_top_k_candidates,
    solve_geo_ind_lp,
    transprotect_weighted_scores,
)
from benchmark.methods.transprotect import (
    TRANSPROTECT_CARD,
    TransProtectAdaptation,
)
from core.demo_protocol import OutputKind, TrajectoryPoint
from core.road_network import RoadNetwork


def _line_network(n=4):
    graph = nx.DiGraph()
    for index in range(n):
        graph.add_node(index, y=39.9, x=116.3 + index * 0.001)
        if index:
            graph.add_edge(index - 1, index, length=100.0, travel_time=10.0)
            graph.add_edge(index, index - 1, length=100.0, travel_time=10.0)
    return RoadNetwork(graph)


def _line_costs(n=4):
    indices = np.arange(n, dtype=float)
    return np.abs(indices[:, None] - indices[None, :]) * 10.0


def test_equation_13_expected_travel_cost_loss():
    costs = _line_costs(3)
    losses = expected_travel_cost_loss(costs, [0.0, 0.5, 0.5], real_index=0)

    assert np.allclose(losses, [0.0, 10.0, 10.0])
    assert losses[0] == 0.0


def test_section_4_4_weighting_and_top_k_are_exact_and_stable():
    probabilities = [0.05, 0.60, 0.30, 0.05]
    utility_losses = [2.0, 4.0, 0.0, 2.0]
    scores = transprotect_weighted_scores(
        probabilities, utility_losses, alpha=8.0
    )

    assert np.isinf(scores[2])
    assert np.allclose(scores[[0, 1, 3]], [4.05, 2.60, 4.05])
    selected, selected_scores = select_top_k_candidates(
        probabilities, utility_losses, k=3, alpha=8.0
    )
    assert selected.tolist() == [2, 0, 3]
    assert np.isinf(selected_scores[0])


def test_candidate_restricted_laplace_matches_exponential_distance_weights():
    distances = np.asarray(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    probabilities = restricted_laplace_probabilities(
        0, [0, 2], distances, epsilon=0.5
    )
    expected = np.asarray([1.0, np.exp(-1.0)])
    expected /= expected.sum()

    assert np.allclose(probabilities, expected)
    assert np.isclose(probabilities.sum(), 1.0)


def test_fitted_local_probability_proxy_is_sparse_deterministic_and_labelled():
    first = CausalTransitionProbabilityProvider(5, smoothing=0.5).fit(
        [(0, 1, 2), (0, 1, 3), (4, 1, 2)]
    )
    second = CausalTransitionProbabilityProvider(5, smoothing=0.5).fit(
        [(0, 1, 2), (0, 1, 3), (4, 1, 2)]
    )

    assert np.allclose(first.predict_proba(()), second.predict_proba(()))
    assert np.allclose(first.predict_proba((0, 1)), second.predict_proba((0, 1)))
    after_one = first.predict_proba((0, 1))
    assert after_one[2] > after_one[0]
    assert after_one[2] > after_one[4]
    assert "not_transprotect_transformer" in first.implementation_name
    assert isinstance(first._transition_counts, dict)
    assert np.allclose(first.visit_prior([1, 2, 3]), [3 / 6, 2 / 6, 1 / 6])


def test_target_utility_scales_as_locations_by_targets_not_square():
    road_network = _line_network(400)
    utility = TargetTravelCostUtility(
        road_network,
        target_indices=[0, 199, 399],
        target_prior=[0.2, 0.3, 0.5],
        edge_weight="length",
    )

    assert utility.cost_to_targets.shape == (400, 3)
    assert utility.losses(17).shape == (400,)
    assert utility.losses(17)[17] == 0.0
    assert utility.implementation_name == "equation_13_with_explicit_lbs_targets"


def test_local_factory_fits_only_supplied_background_and_records_proxy_source():
    road_network = _line_network(6)
    training = [
        {"points": [road_network.latlon(index) for index in (0, 1, 2, 3)]},
        {"points": [road_network.latlon(index) for index in (1, 2, 4)]},
    ]
    method = TransProtectAdaptation.from_road_network(
        road_network,
        training_trajectories=training,
        candidate_k=3,
        target_count=2,
        epsilon=0.01,
        rng=np.random.default_rng(23),
    )
    real = tuple(
        TrajectoryPoint(step * 20, *road_network.latlon(index))
        for step, index in enumerate((3, 4, 5))
    )
    run = method.protect_run(real)
    parameters = dict(run.transcript.public_parameters)

    assert parameters["training_sequence_count"] == 2
    assert parameters["training_transition_count"] == 5
    assert "disjoint_sumo_background" in parameters["probability_provider"]
    assert "disjoint_background_visit_target_proxy" in parameters["utility_provider"]
    assert parameters["utility_target_selection"] == (
        "top_visit_count_from_disjoint_background"
    )
    assert parameters["epsilon_unit"] == "m^-1"
    assert parameters["probability_smoothing"] == 1e-6
    assert parameters["probability_backoff_weight"] == 0.1
    assert len(run.transcript.events) == len(real)


def test_lp_solution_satisfies_rows_and_bidirectional_geo_i_constraints():
    utility = np.asarray(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]]
    )
    distances = utility.copy()
    epsilon = 0.4
    matrix = solve_geo_ind_lp(utility, distances, epsilon)

    assert matrix.shape == (3, 3)
    assert np.all(matrix >= -1e-10)
    assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8)
    for i in range(3):
        for j in range(3):
            bound = np.exp(epsilon * distances[i, j])
            assert np.all(matrix[i] <= bound * matrix[j] + 1e-8)


class _RecordingProvider:
    implementation_name = "test_imported_predictions"

    def __init__(self, distributions):
        self.distributions = np.asarray(distributions, dtype=float)
        self.histories = []

    def predict_proba(self, history_indices):
        self.histories.append(tuple(history_indices))
        return self.distributions[len(history_indices)]


def test_engine_is_causal_deterministic_and_emits_protocol_safe_replacement():
    road_network = _line_network()
    distributions = [
        [0.10, 0.60, 0.20, 0.10],
        [0.10, 0.20, 0.60, 0.10],
        [0.10, 0.10, 0.20, 0.60],
    ]
    real = tuple(
        TrajectoryPoint(step * 20, *road_network.latlon(index))
        for step, index in enumerate((1, 2, 3))
    )

    first_provider = _RecordingProvider(distributions)
    first = TransProtectAdaptation(
        road_network,
        probability_provider=first_provider,
        travel_cost_matrix=_line_costs(),
        target_prior=[0.1, 0.2, 0.3, 0.4],
        candidate_k=3,
        alpha=10_000.0,
        epsilon=0.005,
        rng=np.random.default_rng(17),
    )
    first_run = first.protect_run(real)

    second = TransProtectEngine(
        road_network,
        probability_provider=ArrayProbabilityProvider(distributions),
        travel_cost_matrix=_line_costs(),
        target_prior=[0.1, 0.2, 0.3, 0.4],
        candidate_k=3,
        alpha=10_000.0,
        epsilon=0.005,
        rng=np.random.default_rng(17),
    )
    second_run = second.protect_run(real)

    assert first_provider.histories == [(), (1,), (1, 2,)]
    assert first.last_output_indices == second.last_output_indices
    assert len(first.last_output_utility_losses) == len(real)
    assert all(loss >= 0 for loss in first.last_output_utility_losses)
    assert all(
        real_index in candidates
        for real_index, candidates in zip((1, 2, 3), first.last_candidate_sets)
    )
    assert first_run.transcript.output_kind is OutputKind.REPLACEMENT_TRAJECTORY
    assert len(first_run.transcript.events) == len(real)
    assert first_run.truth.real_candidate_ids == ()
    public = first_run.to_attacker_dict()
    assert "real_trajectory" not in str(public)
    parameters = dict(first_run.transcript.public_parameters)
    assert parameters["probability_provider"] == "test_imported_predictions"
    assert parameters["reportable_as_reproduced_sota"] is False
    assert parameters["zero_loss_tie_policy"] == "force_real_membership_replace_last"
    assert second_run.transcript.output_kind is OutputKind.REPLACEMENT_TRAJECTORY


def test_local_tie_policy_forces_real_membership_and_is_auditable():
    road_network = _line_network()
    zero_costs = np.zeros((4, 4), dtype=float)
    provider = ArrayProbabilityProvider([[0.7, 0.1, 0.1, 0.1]])
    real = (TrajectoryPoint(0, *road_network.latlon(3)),)
    method = TransProtectAdaptation(
        road_network,
        probability_provider=provider,
        travel_cost_matrix=zero_costs,
        target_prior=[0.25] * 4,
        candidate_k=1,
        alpha=10_000.0,
        epsilon=0.005,
        rng=np.random.default_rng(7),
    )

    run = method.protect_run(real)

    assert method.last_candidate_sets == [(3,)]
    assert method.last_forced_real_membership == [True]
    assert method.last_output_indices == [3]
    parameters = dict(run.transcript.public_parameters)
    assert "forced_real_membership_events" not in parameters
    assert parameters["zero_loss_tie_policy"] == (
        "force_real_membership_replace_last"
    )


def test_method_card_fails_closed_until_model_and_table_parity_exist():
    assert TRANSPROTECT_CARD.implementation_level is ImplementationLevel.PAPER_ADAPTATION
    assert TRANSPROTECT_CARD.reportable_as_reproduced_sota is False
    assert set(TRANSPROTECT_CARD.missing_components) == {
        "paper-equivalent learned weights",
        "paper dataset split and reported-table parity",
        "VehiTrack end-to-end attack parity",
    }
    try:
        TransProtectAdaptation.require_faithful()
    except MethodUnavailableError as exc:
        message = str(exc)
    else:
        raise AssertionError("incomplete upstream evidence must fail closed")
    assert "paper-equivalent learned weights" in message
    assert "reported-table parity" in message
