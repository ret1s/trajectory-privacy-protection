"""Tests for the executable Geo-I-anchor + dummy-only thesis candidate."""

import networkx as nx
import numpy as np

from core.demo_protocol import OutputKind, TrajectoryPoint
from core.road_network import RoadNetwork
from benchmark.methods import GeoIAnchoredDummyTrajectories


def _road_line(n=9):
    graph = nx.Graph()
    for i in range(n):
        graph.add_node(i, y=0.0, x=i * 0.00045)  # about 50 m spacing
        if i:
            graph.add_edge(i - 1, i)
    return RoadNetwork(graph)


def test_thesis_demo_returns_k_aligned_road_tracks():
    rn = _road_line()
    points = [rn.latlon(i) for i in (2, 3, 4, 5)]
    model = GeoIAnchoredDummyTrajectories(
        0.02,
        rn,
        k=3,
        offset_m=80.0,
        candidate_radius_m=120.0,
        rng=np.random.default_rng(7),
    )
    batch = model.protect_trajectory(points, times=[0, 20, 40, 60])

    assert len(batch.anchors) == len(points)
    assert len(batch.trajectories) == 3
    assert all(len(track) == len(points) for track in batch.trajectories)
    graph_points = {rn.latlon(i) for i in range(len(rn))}
    assert all(point in graph_points for track in batch.trajectories for point in track)


def test_thesis_demo_is_deterministic_for_same_seed():
    rn = _road_line()
    points = [rn.latlon(i) for i in (1, 2, 3)]

    def run_once():
        model = GeoIAnchoredDummyTrajectories(
            0.02,
            rn,
            k=2,
            rng=np.random.default_rng(19),
        )
        return model.protect_trajectory(points, times=[0, 30, 60])

    assert run_once() == run_once()


def test_postprocessing_stage_accepts_only_public_anchors():
    rn = _road_line()
    anchors = [rn.latlon(i) for i in (2, 3, 4)]
    model = GeoIAnchoredDummyTrajectories(
        0.02,
        rn,
        k=2,
        rng=np.random.default_rng(5),
    )
    batch = model.generate_from_anchors(anchors, times=[0, 15, 30])

    assert batch.anchors == tuple(anchors)
    assert len(batch.trajectories) == 2
    assert all(len(track) == len(anchors) for track in batch.trajectories)


def test_thesis_demo_rejects_invalid_shapes():
    rn = _road_line()
    try:
        GeoIAnchoredDummyTrajectories(0.02, rn, k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("k=0 must be rejected")

    model = GeoIAnchoredDummyTrajectories(0.02, rn)
    try:
        model.protect_trajectory([rn.latlon(0)], times=[0, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched points/times must be rejected")


def test_protocol_adapter_publishes_only_dummy_tracks():
    rn = _road_line()
    real = tuple(
        TrajectoryPoint(i * 30, *rn.latlon(node_idx))
        for i, node_idx in enumerate((2, 3, 4))
    )
    run = GeoIAnchoredDummyTrajectories(
        0.02,
        rn,
        k=3,
        rng=np.random.default_rng(31),
    ).protect_run(real)

    assert run.transcript.output_kind is OutputKind.DUMMY_ONLY
    assert run.truth.real_candidate_ids == ()
    assert all(len(event.candidates) == 3 for event in run.transcript.events)
    public = run.to_attacker_dict()
    assert "truth" not in public
    assert all(
        set(event) == {"event_id", "timestamp_s", "candidates"}
        for event in public["events"]
    )
    assert "real_trajectory" not in str(public).lower()
