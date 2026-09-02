"""Dependency-light tests for the controlled SUMO demo adapter."""

from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from data.sumo_demo import (
    BEIJING_SMOKE_BBOX,
    SumoSmokeConfig,
    SumoToolchain,
    SumoUnavailableError,
    load_sumo_road_network,
    road_network_semantic_sha256,
    parse_fcd,
    parse_vehicle_routes,
    resolve_sumo_toolchain,
    run_sumo_smoke_demo,
    select_background_trajectories,
    select_longest_trace,
)


FCD_FIXTURE = """<?xml version="1.0"?>
<fcd-export>
  <timestep time="0.00">
    <vehicle id="short" x="116.3000" y="39.9700" speed="3" lane="s_0"/>
    <vehicle id="long" x="116.3100" y="39.9800" speed="4" lane="e0_0"/>
  </timestep>
  <timestep time="1.00">
    <vehicle id="long" x="116.3110" y="39.9810" speed="5" lane="e0_0"/>
  </timestep>
  <timestep time="2.00">
    <vehicle id="short" x="116.3020" y="39.9720" speed="3" lane="s_0"/>
    <vehicle id="long" x="116.3120" y="39.9820" speed="6" lane="e1_1"/>
  </timestep>
  <timestep time="3.00">
    <vehicle id="long" x="116.3130" y="39.9830" speed="7" lane="e1_1"/>
  </timestep>
  <timestep time="4.00">
    <vehicle id="long" x="116.3140" y="39.9840" speed="8" edge="e2" lane="e2_0"/>
  </timestep>
</fcd-export>
"""


ROUTE_FIXTURE = """<?xml version="1.0"?>
<routes>
  <route id="shared" edges="a b c"/>
  <vehicle id="long" depart="0"><route edges="e0 e1 e2"/></vehicle>
  <vehicle id="ref" depart="1" route="shared"/>
</routes>
"""


class _FakeSumoNode:
    def __init__(self, node_id, coordinate):
        self._node_id = node_id
        self._coordinate = coordinate

    def getID(self):
        return self._node_id

    def getCoord(self):
        return self._coordinate


class _FakeSumoEdge:
    def __init__(self, edge_id, start, end, *, passenger, shape):
        self._edge_id = edge_id
        self._start = start
        self._end = end
        self._passenger = passenger
        self._shape = shape

    def allows(self, vehicle_class):
        assert vehicle_class == "passenger"
        return self._passenger

    def getFromNode(self):
        return self._start

    def getToNode(self):
        return self._end

    def getShape(self):
        return self._shape

    def getType(self):
        return "highway.primary"

    def getLength(self):
        return 120.0

    def getSpeed(self):
        return 12.0

    def getID(self):
        return self._edge_id


class _FakeSumoNetwork:
    def __init__(self):
        start = _FakeSumoNode("n0", (0.0, 0.0))
        end = _FakeSumoNode("n1", (2.0, 2.0))
        self._edges = (
            _FakeSumoEdge(
                "passenger_edge",
                start,
                end,
                passenger=True,
                shape=((0.0, 0.0), (1.0, 1.5), (2.0, 2.0)),
            ),
            _FakeSumoEdge(
                "pedestrian_only",
                end,
                start,
                passenger=False,
                shape=((2.0, 2.0), (0.0, 0.0)),
            ),
        )

    def convertXY2LonLat(self, x, y):
        return 116.3 + x / 10_000.0, 39.9 + y / 10_000.0

    def getEdges(self, *, withInternal):
        assert withInternal is False
        return self._edges


class _FakeSumolibNetModule:
    def readNet(self, path, *, withInternal):
        assert Path(path).is_file()
        assert withInternal is False
        return _FakeSumoNetwork()


class _FakeSumolib:
    net = _FakeSumolibNetModule()


def test_parse_fcd_uses_geo_order_and_selects_longest_deterministically():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.xml"
        path.write_text(FCD_FIXTURE, encoding="utf-8")
        traces = parse_fcd(path)
        vehicle_id, samples = select_longest_trace(
            traces, interval_s=2.0, max_points=3, min_points=2
        )

    assert vehicle_id == "long"
    assert [sample.timestamp_s for sample in samples] == [0.0, 2.0, 4.0]
    assert (samples[0].lat, samples[0].lon) == (39.98, 116.31)
    assert [sample.speed_m_s for sample in samples] == [4.0, 6.0, 8.0]
    assert [sample.edge_id for sample in samples] == ["e0", "e1", "e2"]


def test_background_training_traces_exclude_held_out_vehicle_and_identifiers():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.xml"
        path.write_text(FCD_FIXTURE, encoding="utf-8")
        traces = parse_fcd(path)
        background = select_background_trajectories(
            traces,
            held_out_vehicle_id="long",
            interval_s=2.0,
            max_points=3,
            min_points=2,
        )

    assert len(background) == 1
    assert background[0].points == (
        (39.97, 116.3),
        (39.972, 116.302),
    )
    model_input = background[0].to_model_input()
    assert set(model_input) == {"points", "times"}
    assert "long" not in str(model_input)
    assert "short" not in str(model_input)
    assert "lane" not in str(model_input)


def test_parse_vehicle_routes_handles_embedded_and_referenced_routes():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "routes.xml"
        path.write_text(ROUTE_FIXTURE, encoding="utf-8")
        routes = parse_vehicle_routes(path)

    assert routes["long"] == ("e0", "e1", "e2")
    assert routes["ref"] == ("a", "b", "c")


def test_load_sumo_road_network_preserves_exact_passenger_edges_and_geometry():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "passenger.net.xml"
        path.write_text("<net/>", encoding="utf-8")
        with patch("data.sumo_demo._load_sumolib", return_value=_FakeSumolib()):
            road_network = load_sumo_road_network(path)

    graph = road_network.graph
    assert list(graph.nodes) == ["n0", "n1"]
    assert graph.number_of_edges() == 1
    edge = graph["n0"]["n1"]["passenger_edge"]
    assert edge["sumo_edge_id"] == "passenger_edge"
    assert edge["highway"] == "primary"
    assert edge["length"] == 120.0
    assert edge["speed"] == 12.0
    assert edge["travel_time"] == 10.0
    assert len(edge["geometry"].coords) == 3
    assert graph.graph["source"] == "eclipse_sumo_passenger_network"

    semantic_hash = road_network_semantic_sha256(road_network)
    graph.graph["source_path"] = "/a/volatile/generated/path.net.xml"
    assert road_network_semantic_sha256(road_network) == semantic_hash
    edge["length"] += 1.0
    assert road_network_semantic_sha256(road_network) != semantic_hash


def test_missing_sumo_fails_clearly_without_fallback():
    with patch("data.sumo_demo._module_roots", return_value=[]), patch(
        "data.sumo_demo.shutil.which", return_value=None
    ):
        try:
            resolve_sumo_toolchain()
        except SumoUnavailableError as exc:
            message = str(exc)
        else:
            raise AssertionError("missing SUMO should raise SumoUnavailableError")
    assert "No GeoLife/synthetic fallback" in message
    assert "randomTrips.py" in message


def test_pipeline_uses_real_argv_contract_and_keeps_metadata_private():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        osm = root / "Beijing.osm"
        osm.write_text("<osm version='0.6'/>", encoding="utf-8")
        random_trips = root / "sumo" / "tools" / "randomTrips.py"
        random_trips.parent.mkdir(parents=True)
        random_trips.write_text("# fixture", encoding="utf-8")
        toolchain = SumoToolchain(
            sumo=root / "sumo" / "bin" / "sumo",
            netconvert=root / "sumo" / "bin" / "netconvert",
            random_trips=random_trips,
            sumo_home=root / "sumo",
        )
        calls: list[list[str]] = []

        def fake_run(argv, *, check, capture_output, text, env):
            assert isinstance(argv, list)
            assert check and capture_output and text
            assert env["SUMO_HOME"] == str(toolchain.sumo_home)
            assert env["PATH"].split(":", 1)[0] == str(toolchain.sumo.parent)
            calls.append(argv)
            if "--version" in argv:
                return subprocess.CompletedProcess(argv, 0, "Eclipse SUMO 1.test\n", "")
            if argv[0] == str(toolchain.netconvert):
                output = Path(argv[argv.index("--output-file") + 1])
                output.write_text("<net/>", encoding="utf-8")
            elif "--route-file" in argv:
                trip_output = Path(argv[argv.index("-o") + 1])
                trip_output.write_text("<routes/>", encoding="utf-8")
                route_output = Path(argv[argv.index("--route-file") + 1])
                route_output.write_text(ROUTE_FIXTURE, encoding="utf-8")
            elif "--fcd-output" in argv:
                fcd = Path(argv[argv.index("--fcd-output") + 1])
                fcd.write_text(FCD_FIXTURE, encoding="utf-8")
                routes = Path(argv[argv.index("--vehroute-output") + 1])
                routes.write_text(ROUTE_FIXTURE, encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")

        config = SumoSmokeConfig(
            resample_interval_s=2,
            max_points=3,
            min_points=2,
        )
        with patch("data.sumo_demo.subprocess.run", side_effect=fake_run):
            record = run_sumo_smoke_demo(
                osm_path=osm,
                workdir=root / "cache",
                config=config,
                toolchain=toolchain,
            )

    stage_calls = [argv for argv in calls if "--version" not in argv]
    assert len(stage_calls) == 3
    netconvert, random_trips_call, sumo = stage_calls
    assert netconvert[netconvert.index("--keep-edges.in-geo-boundary") + 1] == ",".join(
        f"{value:g}" for value in BEIJING_SMOKE_BBOX
    )
    assert netconvert[netconvert.index("--keep-edges.by-vclass") + 1] == "passenger"
    assert "--geometry.remove" not in netconvert
    assert random_trips_call[random_trips_call.index("--vehicle-class") + 1] == "passenger"
    assert random_trips_call[random_trips_call.index("--seed") + 1] == "20260905"
    assert random_trips_call[random_trips_call.index("--end") + 1] == "300"
    assert random_trips_call[random_trips_call.index("--period") + 1] == "15"
    assert random_trips_call[random_trips_call.index("--min-distance") + 1] == "2500"
    assert sumo[sumo.index("--seed") + 1] == "20260906"
    assert sumo[sumo.index("--end") + 1] == "900"
    assert sumo[sumo.index("--fcd-output.geo") + 1] == "true"
    assert record.points[0] == (39.98, 116.31)
    assert record.evaluator_only.route_edges == ("e0", "e1", "e2")
    assert record.evaluator_only.samples[1].speed_m_s == 6.0
    assert len(record.background_trajectories) == 1
    assert record.background_trajectories[0].points[0] == (39.97, 116.3)
    public = record.to_mechanism_input()
    assert "evaluator_only" not in public
    assert "route_edges" not in str(public)
    complete = record.to_evaluator_dict()
    assert complete["evaluator_only"]["samples"][1]["edge_id"] == "e1"
    provenance = complete["provenance"]
    assert set(provenance["commands"]) == {"netconvert", "randomTrips.py", "sumo"}
    assert set(provenance["sha256"]) == {
        "fcd",
        "network",
        "osm",
        "randomTrips.py",
        "routes",
        "trips",
        "vehicle_routes",
    }
