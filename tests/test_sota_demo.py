"""Smoke and contract tests for executable paper adaptations."""

import networkx as nx
import numpy as np

from core.road_network import RoadNetwork
from core.demo_protocol import OutputKind, TrajectoryPoint
from benchmark.contracts import ImplementationLevel
from benchmark.methods import (
    AnotherMeAdaptation,
    SemanticDummyAdaptation,
    TransProtectAdaptation,
)


def _grid_network(rows=3, cols=4):
    graph = nx.DiGraph()
    coordinates = []
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            lat, lon = 39.9 + row * 0.001, 116.3 + col * 0.001
            coordinates.append((lat, lon))
            graph.add_node(idx, y=lat, x=lon)
    for row in range(rows):
        for col in range(cols):
            idx = row * cols + col
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < rows and 0 <= cc < cols:
                    graph.add_edge(idx, rr * cols + cc, length=100.0)
    return RoadNetwork(graph), coordinates


def test_adaptation_classes_are_unambiguously_labelled():
    for cls in (
        TransProtectAdaptation,
        AnotherMeAdaptation,
        SemanticDummyAdaptation,
    ):
        assert cls.name.endswith("_adaptation")
        assert cls.method_card.implementation_level is ImplementationLevel.PAPER_ADAPTATION
        assert cls.method_card.reportable_as_reproduced_sota is False
        assert cls.method_card.missing_components


def test_transprotect_adaptation_returns_one_road_point_per_input():
    rn, coordinates = _grid_network()
    points, times = coordinates[:4], [0, 60, 120, 180]
    first = TransProtectAdaptation(rn, rng=np.random.default_rng(7)).protect_trajectory(
        points, times
    )
    second = TransProtectAdaptation(rn, rng=np.random.default_rng(7)).protect_trajectory(
        points, times
    )
    assert first == second, "a fixed RNG seed must reproduce the demo"
    assert len(first) == len(points)
    road_points = {rn.latlon(i) for i in range(len(rn))}
    assert all(point in road_points for point in first)


def test_anotherme_adaptation_relocates_a_whole_trajectory_to_roads():
    rn, coordinates = _grid_network()
    points = coordinates[:4]
    mechanism = AnotherMeAdaptation(
        rn,
        anchor_min_m=100.0,
        anchor_max_m=400.0,
        rng=np.random.default_rng(11),
    )
    protected = mechanism.protect_trajectory(points, [0, 60, 120, 180])
    assert len(protected) == len(points)
    assert set(mechanism.last_transform) == {
        "anchor_vertex_index",
        "rotation_degrees",
        "scale",
    }
    road_points = {rn.latlon(i) for i in range(len(rn))}
    assert all(point in road_points for point in protected)
    assert protected != points, "the smoke case should be visibly relocated"


def test_semantic_dummy_adaptation_keeps_truth_separate_from_public_candidates():
    rn, coordinates = _grid_network()
    points, times = coordinates[:3], [0, 60, 120]
    releases = SemanticDummyAdaptation(
        rn, k=4, rng=np.random.default_rng(19)
    ).protect_trajectory(
        points,
        times,
        poi_categories=["work", "work", "shop"],
        vertex_categories=["work"] * 8 + ["shop"] * 4,
    )
    assert len(releases) == len(points)
    for real, release in zip(points, releases):
        assert len(release.public_candidates) == 4
        assert release.public_candidates[release.real_index] == real
        assert len(set(release.candidate_vertex_indices)) == 4
        assert len(set(release.candidate_ids)) == 4
        assert release.metadata["implementation_level"] == "paper_adaptation"


def test_semantic_dummy_adaptation_rejects_impossible_k():
    rn, _ = _grid_network(rows=1, cols=3)
    try:
        SemanticDummyAdaptation(rn, k=4)
        raise AssertionError("k larger than the graph must be rejected")
    except ValueError:
        pass


def test_protocol_adapters_hide_semantic_truth_and_label_output_kind():
    rn, coordinates = _grid_network()
    real = tuple(
        TrajectoryPoint(i * 60, lat, lon)
        for i, (lat, lon) in enumerate(coordinates[:3])
    )

    replacement = TransProtectAdaptation(
        rn, rng=np.random.default_rng(3)
    ).protect_run(real)
    assert replacement.transcript.output_kind is OutputKind.REPLACEMENT_TRAJECTORY
    assert replacement.truth.real_candidate_ids == ()

    candidate_run = SemanticDummyAdaptation(
        rn, k=4, rng=np.random.default_rng(4)
    ).protect_run(real)
    assert candidate_run.transcript.output_kind is OutputKind.REAL_PLUS_DUMMIES
    public = candidate_run.to_attacker_dict()
    assert "real_index" not in str(public)
    assert "real_candidate_ids" not in str(public)
    assert len(candidate_run.truth.real_candidate_ids) == len(real)
    parameters = dict(candidate_run.transcript.public_parameters)
    assert parameters["implementation_level"] == "paper_adaptation"
    assert parameters["reportable_as_reproduced_sota"] is False
    assert "demo_only" not in parameters
