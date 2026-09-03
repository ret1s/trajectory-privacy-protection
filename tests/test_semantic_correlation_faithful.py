"""Source-mapped tests for the partial semantic-correlation clean room."""

from __future__ import annotations

import json
import math

import networkx as nx
import numpy as np

from benchmark.engines.semantic_correlation import (
    BeijingGrid,
    CoordinateNormalization,
    DenseWeights,
    FeatureEmbeddingTables,
    HistoricalVisit,
    LSTMLayerWeights,
    PaperSpecificationGap,
    SemanticDummySelector,
    SemanticHierarchy,
    SemanticLocation,
    StackedLSTMSemanticNetwork,
    TransitionObservation,
    adjusted_transition_probability,
    anonymity_success_rate,
    conditional_transition_probabilities,
    cosine_similarity,
    dummy_effectiveness_rate,
    dynamic_threshold,
    encode_paper_features,
    estimate_transition_probabilities,
    historical_query_probabilities,
    paper_time_weight,
    rank_semantic_types,
    require_all_transitions,
    transition_set_l1_difference,
)
from benchmark.methods.semantic_correlation import SemanticCorrelationComparator
from core.demo_protocol import TrajectoryPoint
from core.road_network import RoadNetwork


def _lstm_weights(input_size: int, current_input_column: int = 0) -> LSTMLayerWeights:
    hidden = 1
    width = hidden + input_size
    zeros = np.zeros((hidden, width))
    candidate = zeros.copy()
    candidate[0, hidden + current_input_column] = 1.0
    biases = np.zeros(hidden)
    return LSTMLayerWeights(
        forget_kernel=zeros,
        input_kernel=zeros,
        candidate_kernel=candidate,
        output_kernel=zeros,
        forget_bias=biases,
        input_bias=biases,
        candidate_bias=biases,
        output_bias=biases,
    )


def _tiny_road_network() -> RoadNetwork:
    graph = nx.MultiDiGraph()
    for index in range(24):
        graph.add_node(index, y=39.90 + index * 0.0001, x=116.30 + index * 0.0001)
    for index in range(23):
        graph.add_edge(index, index + 1, length=15.0, highway="residential")
        graph.add_edge(index + 1, index, length=15.0, highway="residential")
    return RoadNetwork(graph)


def test_paper_grid_and_query_probability_equation():
    grid = BeijingGrid()
    first = grid.cell(39.8, 116.2)
    last = grid.cell(40.0, 116.5)
    assert (first.row, first.column, first.cell_id) == (0, 0, 0)
    assert (last.row, last.column, last.cell_id) == (99, 99, 9999)
    assert math.isclose(first.centroid_lat, 39.801)
    assert math.isclose(first.centroid_lon, 116.2015)
    assert historical_query_probabilities([1, 1, 2, 3]) == {1: 0.5, 2: 0.25, 3: 0.25}


def test_transition_matrix_and_conditioning_are_row_normalized():
    observations = [
        TransitionObservation("a", "b", 5),
        TransitionObservation("a", "b", 8),
        TransitionObservation("a", "c", 20),
        TransitionObservation("b", "c", 5),
    ]
    total = estimate_transition_probabilities(observations)
    assert total == {("a", "b"): 2 / 3, ("a", "c"): 1 / 3, ("b", "c"): 1.0}
    conditioned = conditional_transition_probabilities(observations, None, 10)
    assert conditioned == {("a", "b"): 1.0, ("b", "c"): 1.0}
    assert math.isclose(transition_set_l1_difference(total, conditioned), 2 / 3)


def test_paper_time_table_boundaries_and_unspecified_tail_fail_closed():
    expected = {
        0: 0.75,
        5: 0.75,
        5.01: 0.21,
        10: 0.21,
        25: 0.10,
        50: 0.05,
        95: 0.03,
        180: 0.02,
        335: 0.01,
    }
    for delta, weight in expected.items():
        assert paper_time_weight(delta) == weight
    assert math.isclose(adjusted_transition_probability(0.4, 5), 0.3)
    try:
        paper_time_weight(336)
        assert False, "unpublished tail behavior must not be guessed"
    except PaperSpecificationGap:
        pass


def test_dynamic_threshold_requires_paper_missing_decay_hook():
    try:
        dynamic_threshold(0.5, 10, None)
        assert False, "Gen(delta_t) must be explicit"
    except PaperSpecificationGap:
        pass
    value = dynamic_threshold(0.5, 10, lambda delta: math.exp(-delta / 10))
    assert math.isclose(value, 0.5 / math.e)


def test_paper_feature_dimensions_and_two_layer_forward_are_deterministic():
    embeddings = FeatureEmbeddingTables(
        weekday=np.arange(21, dtype=float).reshape(7, 3),
        time_slot=np.arange(288, dtype=float).reshape(48, 6),
        semantic=np.arange(40, dtype=float).reshape(4, 10),
    )
    features = encode_paper_features(
        [(39.9, 116.3), (39.91, 116.31)],
        weekdays=[0, 1],
        half_hour_slots=[2, 3],
        semantic_indices=[1, 2],
        normalization=CoordinateNormalization(39.9, 116.3, 0.01, 0.01),
        embeddings=embeddings,
    )
    assert features.shape == (2, 21)
    assert np.allclose(features[:, :2], [[0, 0], [1, 1]])

    layer1 = _lstm_weights(21, 0)
    layer2 = _lstm_weights(1, 0)
    attention = DenseWeights(np.asarray([[0.0, 1.0]]), np.zeros(1))
    identity = DenseWeights(np.ones((1, 1)), np.zeros(1))
    model = StackedLSTMSemanticNetwork((layer1, layer2), attention, identity, identity)
    first = model.forward(features, [0.0, 0.5])
    second = model.forward(features, [0.0, 0.5])
    assert np.array_equal(first.prediction, second.prediction)
    assert np.allclose(first.attention_weights, [0.5, 1 / (1 + math.exp(-0.5))])
    assert first.hidden_sequence.shape == (2, 1)


def test_semantic_ranking_requires_explicit_similarity_metric():
    embeddings = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.5, 0.5]}
    try:
        rank_semantic_types([1.0, 0.0], embeddings, 2, None)
        assert False, "the paper does not define a similarity function"
    except PaperSpecificationGap:
        pass
    assert rank_semantic_types([1.0, 0.0], embeddings, 2, cosine_similarity) == ("a", "c")


def test_dummy_selector_filters_scores_and_expands_semantic_siblings():
    locations = (
        SemanticLocation("real", 39.9, 116.3, "home"),
        SemanticLocation("school", 39.91, 116.31, "education"),
        SemanticLocation("office", 39.92, 116.32, "office"),
        SemanticLocation("cafe", 39.93, 116.33, "food"),
    )
    selector = SemanticDummySelector(
        locations,
        SemanticHierarchy(
            {"home": "residential", "education": "workday", "office": "workday", "food": "leisure"}
        ),
        transitions={
            ("history", "school"): 0.7,
            ("history", "office"): 0.6,
            ("history", "cafe"): 0.05,
        },
        thresholds_by_origin={"history": 0.1},
        threshold_decay=lambda _delta: 1.0,
        eligibility_rule=require_all_transitions,
    )
    result = selector.select(
        ["education"],
        real_location_id="real",
        query_timestamp_minutes=5,
        history=[HistoricalVisit("history", 0)],
        k=3,
    )
    assert result.used_sibling_expansion is True
    assert tuple(location.location_id for location in result.selected) == ("school", "office")
    assert np.allclose([item.score for item in result.ranked], [0.525, 0.45])


def test_native_asr_and_dummy_effectiveness_equations():
    assert math.isclose(anonymity_success_rate([0.5, 0.2, 0.34], [2, 4, 3]), 200 / 3)
    assert dummy_effectiveness_rate([True, False, True, True]) == 0.75


def test_local_road_constructor_emits_complete_truth_separated_track():
    road_network = _tiny_road_network()
    method = SemanticCorrelationComparator.from_road_network(
        road_network,
        k=4,
        mu=3,
        candidate_pool_size=16,
        rng=np.random.default_rng(17),
    )
    real = tuple(
        TrajectoryPoint(timestamp, *road_network.latlon(index))
        for timestamp, index in ((0, 5), (300, 7), (600, 9))
    )
    run = method.protect_run(real)
    assert run.transcript.output_kind.value == "real_plus_dummies"
    assert all(len(event.candidates) == 4 for event in run.transcript.events)
    assert len(run.truth.real_candidate_ids) == len(real)
    assert method.method_card.reportable_as_reproduced_sota is False
    assert method.method_card.implementation_level.value == "paper_adaptation"

    attacker = json.dumps(run.to_attacker_dict())
    evaluator = run.to_evaluator_dict()
    assert "real_candidate_ids" not in attacker
    assert "evaluator_truth" not in attacker
    truth = evaluator["truth"]
    assert isinstance(truth, dict)
    assert truth["real_candidate_ids"]
    event_id_sets = [set(candidate.candidate_id for candidate in event.candidates) for event in run.transcript.events]
    assert not (event_id_sets[0] & event_id_sets[1])
    parameters = dict(run.transcript.public_parameters)
    assert parameters["candidate_linkage"] == "event_local_unlinked_sets"
    assert parameters["threshold_policy"] == "k_feasible_order_statistic_adaptation"
    assert parameters["history_limit"] == 8
    assert parameters["transition_scale_m"] == 900.0

    repeated = SemanticCorrelationComparator.from_road_network(
        road_network,
        k=4,
        mu=3,
        candidate_pool_size=16,
        rng=np.random.default_rng(17),
    ).protect_run(real)
    assert repeated.to_attacker_dict() == run.to_attacker_dict()
    try:
        method.require_faithful()
        assert False, "the local semantic fallback must remain fail-closed"
    except RuntimeError:
        pass


def test_minimum_candidate_pool_still_emits_requested_k():
    road_network = _tiny_road_network()
    real = (
        TrajectoryPoint(0, *road_network.latlon(5)),
        TrajectoryPoint(60, *road_network.latlon(6)),
    )
    run = SemanticCorrelationComparator.from_road_network(
        road_network,
        k=4,
        candidate_pool_size=4,
        rng=np.random.default_rng(9),
    ).protect_run(real)

    assert all(len(event.candidates) == 4 for event in run.transcript.events)


def test_shared_vertex_catalog_blocks_exact_membership_fingerprint():
    road_network = _tiny_road_network()
    real = []
    for timestamp, vertex_index in ((0, 5), (60, 7), (120, 9)):
        x, y = road_network.xy[vertex_index]
        lat, lon = road_network.proj.to_latlon(x + 0.6, y + 0.4)
        real.append(TrajectoryPoint(timestamp, lat, lon))
    run = SemanticCorrelationComparator.from_road_network(
        road_network,
        k=4,
        candidate_pool_size=12,
        rng=np.random.default_rng(10),
    ).protect_run(tuple(real))

    for event in run.transcript.events:
        vertex_distances = [
            road_network.nearest(candidate.lat, candidate.lon)[1]
            for candidate in event.candidates
        ]
        assert max(vertex_distances) < 1e-3
