"""Dependency-light checks for SUMO-first SOTA demo orchestration."""

from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from experiments.run_sota_demo import (
    DEMO_SCHEMA,
    _load_geolife_records,
    _load_sumo_record,
    parse_args,
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
    assert DEMO_SCHEMA == "msc-sota-demo-v2"


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
        "experiments.run_sota_demo.run_sumo_smoke_demo",
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
    with patch("experiments.run_sota_demo.load_trajectories", return_value=[]):
        try:
            _load_geolife_records(args)
        except FileNotFoundError as exc:
            message = str(exc)
        else:
            raise AssertionError("missing GeoLife must fail without fallback")
    assert "no synthetic fallback" in message
