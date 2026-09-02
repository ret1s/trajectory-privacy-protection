"""Dependency-light checks for SUMO-first SOTA demo orchestration."""

from contextlib import redirect_stderr
from io import StringIO
import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx

from benchmark.contracts import MethodUnavailableError
from core.demo_protocol import (
    TrajectoryPoint,
    make_dummy_only_run,
    make_real_plus_dummies_run,
    make_replacement_run,
)
from core.road_network import RoadNetwork
from experiments.run_dummy_benchmark import (
    DEFAULT_PREVIEW_OUTPUT,
    DEMO_SCHEMA,
    _display_bounds,
    _embedded_road_geojson,
    _ground_truth_label,
    _load_geolife_records,
    _load_sumo_record,
    _public_tracks,
    _write_map,
    _write_preview,
    parse_args,
    run,
)


class _FakeEvaluatorMetadata:
    def to_dict(self):
        return {
            "vehicle_id": "veh_7",
            "route_edges": ["edge_a", "edge_b"],
            "samples": [
                {
                    "timestamp_s": 0.0,
                    "speed_m_s": 5.0,
                    "edge_id": "edge_a",
                    "lane_id": "edge_a_0",
                }
            ],
        }


class _FakeProvenance:
    def to_dict(self):
        return {
            "scenario": "fixture",
            "sha256": {"osm": "abc", "fcd": "def"},
        }


class _FakeSumoRecord:
    record_id = "sumo/fixture/veh_7"
    points = ((39.98, 116.31), (39.981, 116.311))
    times = (0.0, 20.0)
    evaluator_only = _FakeEvaluatorMetadata()
    provenance = _FakeProvenance()

    def to_mechanism_input(self):
        return {
            "record_id": self.record_id,
            "points": [list(point) for point in self.points],
            "times": list(self.times),
        }


def test_runner_defaults_to_sumo_and_quick_does_not_change_source():
    args = parse_args(["--quick", "--no-map"])
    assert args.mobility_source == "sumo"
    assert args.max_points == 8
    assert args.preview_output == "outputs/dummy_benchmark_preview.png"
    assert args.preview_output == DEFAULT_PREVIEW_OUTPUT
    assert DEMO_SCHEMA == "msc-dummy-benchmark-v3"


def test_sumo_pilot_rejects_silently_ignored_record_count():
    error = StringIO()
    with redirect_stderr(error):
        try:
            parse_args(["--n-trajectories", "2"])
        except SystemExit as exc:
            exit_code = exc.code
        else:
            raise AssertionError("SUMO pilot should reject more than one record")
    assert exit_code == 2
    assert "SUMO pilot emits exactly one" in error.getvalue()


def test_faithful_sota_gate_fails_before_loading_graph_or_mobility():
    args = parse_args(["--quick", "--no-map", "--require-faithful-sota"])
    try:
        run(args)
    except MethodUnavailableError as exc:
        message = str(exc)
    else:
        raise AssertionError("unavailable faithful SOTA implementations must fail closed")
    assert "TransProtect" in message
    assert "paper_adaptation" in message


def test_sumo_loader_keeps_privileged_mobility_fields_out_of_model_input():
    args = parse_args(
        [
            "--max-points",
            "6",
            "--interval-s",
            "15",
            "--sumo-route-seed",
            "11",
            "--sumo-simulation-seed",
            "12",
            "--sumo-min-trip-distance-m",
            "900",
            "--sumo-demand-end-s",
            "100",
            "--sumo-simulation-end-s",
            "200",
        ]
    )
    with patch(
        "experiments.run_dummy_benchmark.run_sumo_smoke_demo",
        return_value=_FakeSumoRecord(),
    ) as mocked:
        records, mobility = _load_sumo_record(args)

    config = mocked.call_args.kwargs["config"]
    assert config.route_seed == 11
    assert config.simulation_seed == 12
    assert config.min_trip_distance_m == 900
    assert config.demand_end_s == 100
    assert config.simulation_end_s == 200
    assert config.max_points == 6
    assert config.resample_interval_s == 15

    mechanism_input = records[0]
    assert "vehicle_id" not in mechanism_input
    assert "route_edges" not in mechanism_input
    assert "speed_m_s" not in str(mechanism_input)
    assert mobility["source"] == "sumo"
    assert mobility["sumo_runs"][0]["sha256"]["osm"] == "abc"
    private = mobility["evaluator_only"][0]
    assert private["vehicle_id"] == "veh_7"
    assert private["route_edges"] == ["edge_a", "edge_b"]
    assert private["samples"][0]["lane_id"] == "edge_a_0"


def test_explicit_geolife_mode_fails_instead_of_falling_back():
    args = SimpleNamespace(
        n_trajectories=1,
        max_points=8,
        interval_s=20,
    )
    with patch("experiments.run_dummy_benchmark.load_trajectories", return_value=[]):
        try:
            _load_geolife_records(args)
        except FileNotFoundError as exc:
            message = str(exc)
        else:
            raise AssertionError("missing GeoLife must fail without fallback")
    assert "no synthetic fallback" in message


def _tiny_map_fixture():
    graph = nx.MultiDiGraph()
    graph.add_node(0, y=39.9800, x=116.3100)
    graph.add_node(1, y=39.9810, x=116.3110)
    graph.add_node(2, y=39.9820, x=116.3120)
    graph.add_edge(0, 1, highway="primary")
    graph.add_edge(1, 0, highway="primary")
    graph.add_edge(1, 2, highway="residential")
    graph.add_edge(2, 1, highway="residential")
    rn = RoadNetwork(graph)
    real = (
        TrajectoryPoint(0, 39.9800, 116.3100),
        TrajectoryPoint(20, 39.9810, 116.3110),
        TrajectoryPoint(40, 39.9820, 116.3120),
    )
    replacement = (
        TrajectoryPoint(0, 39.9801, 116.3100),
        TrajectoryPoint(20, 39.9811, 116.3110),
        TrajectoryPoint(40, 39.9821, 116.3120),
    )
    dummy = (
        TrajectoryPoint(0, 39.9800, 116.3101),
        TrajectoryPoint(20, 39.9810, 116.3111),
        TrajectoryPoint(40, 39.9820, 116.3121),
    )
    runs = [
        (
            make_replacement_run(
                "transprotect_adaptation", real, replacement
            ),
            "fixture",
            0,
        ),
        (
            make_replacement_run("anotherme_adaptation", real, replacement),
            "fixture",
            0,
        ),
        (
            make_real_plus_dummies_run(
                "semantic_dummy_adaptation",
                real,
                {"candidate_0017": real, "candidate_0003": dummy},
                "candidate_0017",
            ),
            "fixture",
            0,
        ),
        (
            make_dummy_only_run(
                "geo_i_anchored_dummy",
                real,
                {"candidate_0000": dummy},
            ),
            "fixture",
            0,
        ),
    ]
    record = {"points": [(point.lat, point.lon) for point in real]}
    return rn, record, runs


def test_embedded_road_overlay_is_clipped_and_deduplicated():
    rn, record, runs = _tiny_map_fixture()
    bounds = _display_bounds(rn, record, runs, minimum_padding_m=25)
    roads = _embedded_road_geojson(rn, bounds)

    assert roads["major"]["geometry"]["type"] == "MultiLineString"
    assert roads["minor"]["geometry"]["type"] == "MultiLineString"
    # Reverse directed edges describe the same physical lines and appear once.
    assert len(roads["major"]["geometry"]["coordinates"]) == 1
    assert len(roads["minor"]["geometry"]["coordinates"]) == 1
    (south, west), (north, east) = bounds
    for feature in roads.values():
        for line in feature["geometry"]["coordinates"]:
            for lon, lat in line:
                assert west <= lon <= east
                assert south <= lat <= north


def test_semantic_candidate_tracks_use_opaque_stable_public_ids():
    _rn, _record, runs = _tiny_map_fixture()
    semantic_run = runs[2][0]
    tracks = _public_tracks(semantic_run)

    assert set(tracks) == {"candidate_0017", "candidate_0003"}
    assert all(len(track) == 3 for track in tracks.values())
    assert "real" not in " ".join(tracks)


def test_html_map_embeds_roads_and_does_not_enable_online_tiles_by_default():
    rn, record, runs = _tiny_map_fixture()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "map.html")
        _write_map(path, record, runs, rn, mobility_source="sumo")
        with open(path, "r", encoding="utf-8") as handle:
            rendered = handle.read()

    assert "local_pinned_graph" in rendered
    assert "Local road network" in rendered
    assert "preferCanvas" in rendered
    assert "Online OpenStreetMap tiles (optional)" in rendered
    def layer_is_enabled(label, variable_prefix):
        encoded_label = json.dumps(label)
        mapping_start = rendered.index(encoded_label)
        variable_start = rendered.index(variable_prefix, mapping_start)
        variable_end = rendered.index(",", variable_start)
        variable = rendered[variable_start:variable_end].strip()
        return f"{variable}.addTo(" in rendered

    assert not layer_is_enabled(
        "Online OpenStreetMap tiles (optional)", "tile_layer_"
    )
    assert layer_is_enabled("Local road network — embedded", "feature_group_")
    assert layer_is_enabled(
        "SUMO ground truth — evaluator only", "feature_group_"
    )
    assert layer_is_enabled(
        "Proposed model — dummy-only trajectories", "feature_group_"
    )
    assert not layer_is_enabled(
        "TransProtect adaptation — pseudolocation trajectory", "feature_group_"
    )
    assert not layer_is_enabled(
        "AnotherMe adaptation — replacement trajectory", "feature_group_"
    )
    assert not layer_is_enabled(
        "Semantic-correlation adaptation — public candidate tracks",
        "feature_group_",
    )
    assert "Proposed model" in rendered
    assert "SUMO ground truth" in rendered


def test_ground_truth_label_matches_explicit_mobility_source():
    assert _ground_truth_label("sumo") == "SUMO ground truth"
    assert _ground_truth_label("geolife") == "GeoLife ground truth"
    try:
        _ground_truth_label("synthetic")
    except ValueError as exc:
        assert "unsupported mobility source" in str(exc)
    else:
        raise AssertionError("unsupported mobility sources must fail closed")


def test_static_preview_writer_creates_an_offline_png():
    rn, record, runs = _tiny_map_fixture()
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "preview.png")
        _write_preview(path, record, runs, rn, mobility_source="sumo")
        with open(path, "rb") as handle:
            signature = handle.read(8)
        size = os.path.getsize(path)

    assert signature == b"\x89PNG\r\n\x1a\n"
    assert size > 10_000
