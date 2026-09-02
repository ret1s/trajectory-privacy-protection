"""Controlled SUMO mobility source for the paper-comparator benchmark.

This module is intentionally small: it builds one passenger-only road network
from the local Beijing OSM extract, creates deterministic random demand with
SUMO's official ``randomTrips.py``, runs SUMO, and converts geographic FCD
output into one trajectory.  It is a smoke/demo scenario, not a calibrated
Beijing traffic model and not the final thesis scenario generator.

There is deliberately no GeoLife or synthetic fallback.  A caller asking for
SUMO data receives SUMO data or a clear :class:`SumoUnavailableError`.

The public mechanism input contains only ``(lat, lon, timestamp)``.  Route,
speed, lane, and edge information is kept under ``evaluator_only`` so the
demo's attacker transcript cannot accidentally consume privileged simulator
state.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Mapping, Sequence
import xml.etree.ElementTree as ET

import networkx as nx
from shapely.geometry import LineString


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKDIR = PROJECT_ROOT / "cache" / "sumo_demo"
DEFAULT_OSM_CANDIDATES = (
    PROJECT_ROOT / "data" / "raw" / "Beijing.osm.gz",
    PROJECT_ROOT / "data" / "raw" / "Beijing.osm",
)

# netconvert expects lon-min,lat-min,lon-max,lat-max for a geographic boundary.
BEIJING_SMOKE_BBOX = (116.29, 39.96, 116.36, 40.02)
DEFAULT_ROUTE_SEED = 20260905
DEFAULT_SIM_SEED = 20260906


class SumoUnavailableError(RuntimeError):
    """Raised when a complete local SUMO toolchain cannot be resolved."""


class SumoCommandError(RuntimeError):
    """Raised when a SUMO command exits unsuccessfully."""


class SumoOutputError(RuntimeError):
    """Raised when SUMO output is missing or cannot yield a usable trace."""


@dataclass(frozen=True)
class SumoToolchain:
    """Paths needed for the three real subprocess stages."""

    sumo: Path
    netconvert: Path
    random_trips: Path
    sumo_home: Path


@dataclass(frozen=True)
class SumoSmokeConfig:
    """Pinned parameters for one controlled, passenger-only smoke scenario."""

    bbox: tuple[float, float, float, float] = BEIJING_SMOKE_BBOX
    route_seed: int = DEFAULT_ROUTE_SEED
    simulation_seed: int = DEFAULT_SIM_SEED
    demand_begin_s: float = 0.0
    demand_end_s: float = 300.0
    demand_period_s: float = 15.0
    simulation_end_s: float = 900.0
    fcd_period_s: float = 1.0
    resample_interval_s: float = 30.0
    max_points: int = 12
    min_points: int = 2
    # randomTrips' --min-distance is the straight-line separation between the
    # sampled origin and destination edges, not the eventual routed length.
    min_trip_distance_m: float = 2500.0

    def __post_init__(self) -> None:
        min_lon, min_lat, max_lon, max_lat = self.bbox
        if not (-180 <= min_lon < max_lon <= 180):
            raise ValueError("bbox longitude bounds are invalid")
        if not (-90 <= min_lat < max_lat <= 90):
            raise ValueError("bbox latitude bounds are invalid")
        if self.route_seed < 0 or self.simulation_seed < 0:
            raise ValueError("SUMO seeds must be non-negative integers")
        if not self.demand_begin_s < self.demand_end_s <= self.simulation_end_s:
            raise ValueError("expected demand_begin < demand_end <= simulation_end")
        if self.demand_period_s <= 0 or self.fcd_period_s <= 0:
            raise ValueError("demand and FCD periods must be positive")
        if self.resample_interval_s < 0:
            raise ValueError("resample_interval_s cannot be negative")
        if self.max_points < 2 or self.min_points < 2:
            raise ValueError("max_points and min_points must be at least 2")
        if self.min_points > self.max_points:
            raise ValueError("min_points cannot exceed max_points")
        if self.min_trip_distance_m < 0:
            raise ValueError("min_trip_distance_m cannot be negative")


@dataclass(frozen=True)
class FCDSample:
    """One geographic SUMO FCD sample, including evaluator-only attributes."""

    timestamp_s: float
    lat: float
    lon: float
    speed_m_s: float | None
    edge_id: str | None
    lane_id: str | None


@dataclass(frozen=True)
class SumoEvaluatorMetadata:
    """Privileged simulator truth; never pass this object to a mechanism."""

    vehicle_id: str
    samples: tuple[FCDSample, ...]
    route_edges: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "vehicle_id": self.vehicle_id,
            "route_edges": list(self.route_edges),
            "samples": [
                {
                    "timestamp_s": sample.timestamp_s,
                    "speed_m_s": sample.speed_m_s,
                    "edge_id": sample.edge_id,
                    "lane_id": sample.lane_id,
                }
                for sample in self.samples
            ],
        }


@dataclass(frozen=True)
class SumoBackgroundTrajectory:
    """Simulator trajectory that may be used only as model-training context.

    Vehicle identifiers and lane/edge metadata are deliberately discarded.
    Keeping this object separate from :class:`SumoEvaluatorMetadata` lets a
    predictor learn from the other simulated vehicles without seeing the
    held-out vehicle or any evaluator-only state.
    """

    points: tuple[tuple[float, float], ...]
    times: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.points) != len(self.times) or len(self.points) < 2:
            raise ValueError("background points/times must align and contain two samples")

    def to_model_input(self) -> dict[str, object]:
        return {
            "points": [list(point) for point in self.points],
            "times": list(self.times),
        }


@dataclass(frozen=True)
class SumoRunProvenance:
    """Exact commands, versions, and artifact digests for one run."""

    scenario: str
    disclaimer: str
    versions: tuple[tuple[str, str], ...]
    commands: tuple[tuple[str, tuple[str, ...]], ...]
    sha256: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "disclaimer": self.disclaimer,
            "versions": dict(self.versions),
            "commands": {name: list(argv) for name, argv in self.commands},
            "sha256": dict(self.sha256),
        }


@dataclass(frozen=True)
class SumoDemoRecord:
    """One selected trace with an explicit public/private serialization split."""

    record_id: str
    points: tuple[tuple[float, float], ...]
    times: tuple[float, ...]
    evaluator_only: SumoEvaluatorMetadata
    provenance: SumoRunProvenance
    background_trajectories: tuple[SumoBackgroundTrajectory, ...] = ()
    network_path: str | None = None

    def to_mechanism_input(self) -> dict[str, object]:
        """Return only fields needed by a protection mechanism."""

        return {
            "record_id": self.record_id,
            "points": [list(point) for point in self.points],
            "times": list(self.times),
        }

    def to_evaluator_dict(self) -> dict[str, object]:
        """Return the complete record for local evaluation/audit artifacts."""

        result = self.to_mechanism_input()
        result["evaluator_only"] = self.evaluator_only.to_dict()
        result["provenance"] = self.provenance.to_dict()
        return result


def _existing_default_osm() -> Path:
    for candidate in DEFAULT_OSM_CANDIDATES:
        if candidate.is_file():
            return candidate
    rendered = " or ".join(str(path) for path in DEFAULT_OSM_CANDIDATES)
    raise FileNotFoundError(
        f"SUMO demo requires the local Beijing OSM extract at {rendered}"
    )


def _module_roots() -> list[Path]:
    """Discover possible SUMO roots without making the package mandatory."""

    roots: list[Path] = []
    try:
        module = importlib.import_module("sumo")
    except ImportError:
        module = None
    if module is not None:
        declared_home = getattr(module, "SUMO_HOME", None)
        if declared_home:
            roots.append(Path(declared_home).expanduser())
        module_file = getattr(module, "__file__", None)
        if module_file:
            package = Path(module_file).resolve().parent
            roots.extend((package, package.parent))

    env_home = os.environ.get("SUMO_HOME")
    if env_home:
        roots.insert(0, Path(env_home).expanduser())
    roots.extend(
        (
            Path("/opt/homebrew/share/sumo"),
            Path("/usr/local/share/sumo"),
            Path("/usr/share/sumo"),
        )
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen:
            unique.append(resolved)
            seen.add(resolved)
    return unique


def _first_file(candidates: Sequence[Path]) -> Path | None:
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def resolve_sumo_toolchain() -> SumoToolchain:
    """Resolve binaries and official ``randomTrips.py`` from local installs.

    Resolution checks ``SUMO_HOME``, an importable ``sumo`` package, standard
    installation roots, and finally ``PATH`` for binaries.  No substitute
    simulator is selected when any required component is absent.
    """

    roots = _module_roots()
    sumo_on_path = shutil.which("sumo")
    netconvert_on_path = shutil.which("netconvert")
    sumo = _first_file([root / "bin" / "sumo" for root in roots])
    netconvert = _first_file([root / "bin" / "netconvert" for root in roots])
    if sumo is None and sumo_on_path:
        sumo = Path(sumo_on_path).resolve()
    if netconvert is None and netconvert_on_path:
        netconvert = Path(netconvert_on_path).resolve()

    random_trips = _first_file(
        [root / "tools" / "randomTrips.py" for root in roots]
    )
    if random_trips is None:
        inferred_roots: list[Path] = []
        for binary in (sumo, netconvert):
            if binary is not None:
                inferred_roots.extend((binary.parent.parent, binary.parent.parent / "share" / "sumo"))
        random_trips = _first_file(
            [root / "tools" / "randomTrips.py" for root in inferred_roots]
        )

    missing = []
    if sumo is None:
        missing.append("sumo")
    if netconvert is None:
        missing.append("netconvert")
    if random_trips is None:
        missing.append("official tools/randomTrips.py")
    if missing:
        raise SumoUnavailableError(
            "SUMO demo cannot run; missing "
            + ", ".join(missing)
            + ". Install Eclipse SUMO and set SUMO_HOME (or expose its bin "
            "directory on PATH). No GeoLife/synthetic fallback is used."
        )

    assert sumo is not None and netconvert is not None and random_trips is not None
    sumo_home = random_trips.parent.parent.resolve()
    return SumoToolchain(
        sumo=sumo,
        netconvert=netconvert,
        random_trips=random_trips,
        sumo_home=sumo_home,
    )


def _load_sumolib():
    """Import SUMO's bundled Python tools without assuming a global install."""

    try:
        return importlib.import_module("sumolib")
    except ImportError:
        roots = _module_roots()
        for root in roots:
            tools_path = root / "tools"
            if (tools_path / "sumolib").is_dir():
                rendered = str(tools_path)
                if rendered not in sys.path:
                    sys.path.insert(0, rendered)
                return importlib.import_module("sumolib")
    raise SumoUnavailableError(
        "SUMO network loading requires the official bundled sumolib tools"
    )


def load_sumo_road_network(path: str | Path):
    """Convert the exact passenger network used by SUMO into ``RoadNetwork``.

    Candidate vertices and edge polylines then come from the same ``.net.xml``
    on which the FCD trajectory was simulated, removing the former mismatch
    with a separately built multimodal OSMnx graph.
    """

    from core.road_network import RoadNetwork

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SUMO network does not exist: {source}")
    sumolib = _load_sumolib()
    network = sumolib.net.readNet(str(source), withInternal=False)
    graph = nx.MultiDiGraph(
        source="eclipse_sumo_passenger_network",
        source_path=str(source),
    )

    def geographic(point: Sequence[float]) -> tuple[float, float]:
        lon, lat = network.convertXY2LonLat(float(point[0]), float(point[1]))
        return float(lon), float(lat)

    for edge in network.getEdges(withInternal=False):
        if not edge.allows("passenger"):
            continue
        start_node = edge.getFromNode()
        end_node = edge.getToNode()
        start_id, end_id = start_node.getID(), end_node.getID()
        for node, node_id in ((start_node, start_id), (end_node, end_id)):
            if node_id not in graph:
                lon, lat = geographic(node.getCoord())
                graph.add_node(node_id, x=lon, y=lat)
        coordinates = [geographic(point) for point in edge.getShape()]
        if len(coordinates) < 2:
            coordinates = [
                (graph.nodes[start_id]["x"], graph.nodes[start_id]["y"]),
                (graph.nodes[end_id]["x"], graph.nodes[end_id]["y"]),
            ]
        edge_type = str(edge.getType() or "road_other")
        highway = edge_type.removeprefix("highway.")
        length = float(edge.getLength())
        speed = float(edge.getSpeed())
        graph.add_edge(
            start_id,
            end_id,
            key=edge.getID(),
            sumo_edge_id=edge.getID(),
            length=length,
            speed=speed,
            travel_time=length / speed if speed > 0.0 else length,
            highway=highway,
            geometry=LineString(coordinates),
        )
    if graph.number_of_edges() == 0:
        raise SumoOutputError("SUMO passenger network contains no usable edges")
    return RoadNetwork(graph)


def road_network_semantic_sha256(road_network) -> str:
    """Hash canonical graph semantics, excluding volatile XML metadata.

    SUMO writes generation time and absolute paths into ``.net.xml`` comments,
    so a byte hash can change across otherwise identical seeded builds.  This
    digest covers the directed passenger graph actually consumed by benchmark
    methods: node coordinates plus edge identity, geometry, length, speed,
    travel time, and road class.  Records are sorted and floats are rounded to
    twelve decimal places to make the serialization platform-stable while
    retaining sub-millimetre geographic precision.
    """

    graph = road_network.graph

    def number(value) -> float:
        return round(float(value), 12)

    nodes = sorted(
        (
            str(node),
            number(data["x"]),
            number(data["y"]),
        )
        for node, data in graph.nodes(data=True)
    )
    edges = []
    iterator = (
        graph.edges(keys=True, data=True)
        if graph.is_multigraph()
        else ((u, v, "", data) for u, v, data in graph.edges(data=True))
    )
    for start, end, key, data in iterator:
        geometry = data.get("geometry")
        coordinates = (
            [[number(lon), number(lat)] for lon, lat in geometry.coords]
            if geometry is not None
            else [
                [number(graph.nodes[start]["x"]), number(graph.nodes[start]["y"])],
                [number(graph.nodes[end]["x"]), number(graph.nodes[end]["y"])],
            ]
        )
        highway = data.get("highway", "")
        if isinstance(highway, (list, tuple, set)):
            highway = sorted(str(value) for value in highway)
        else:
            highway = str(highway)
        edges.append(
            (
                str(start),
                str(end),
                str(key),
                str(data.get("sumo_edge_id", "")),
                number(data.get("length", 0.0)),
                number(data.get("speed", 0.0)),
                number(data.get("travel_time", 0.0)),
                highway,
                coordinates,
            )
        )
    payload = {
        "schema": "sumo-passenger-road-graph-v1",
        "directed": bool(graph.is_directed()),
        "nodes": nodes,
        "edges": sorted(edges, key=lambda value: json.dumps(value, sort_keys=True)),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _run(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute one stage with a real argv array and no shell."""

    command = [str(value) for value in argv]
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=dict(environment) if environment is not None else None,
        )
    except FileNotFoundError as exc:
        raise SumoUnavailableError(
            f"SUMO executable was not found while running {command[0]!r}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or exc.stdout or "").strip()
        detail = f": {stderr}" if stderr else ""
        raise SumoCommandError(f"SUMO stage failed ({command[0]}){detail}") from exc


def _version(binary: Path, environment: Mapping[str, str]) -> str:
    completed = _run((str(binary), "--version"), environment=environment)
    text = (completed.stdout or completed.stderr or "").strip()
    return text.splitlines()[0] if text else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


_LANE_SUFFIX = re.compile(r"_(\d+)$")


def _edge_from_lane(lane_id: str | None) -> str | None:
    if not lane_id:
        return None
    match = _LANE_SUFFIX.search(lane_id)
    return lane_id[: match.start()] if match else lane_id


def parse_fcd(path: str | Path) -> dict[str, tuple[FCDSample, ...]]:
    """Parse geographic FCD XML into vehicle traces.

    With ``--fcd-output.geo true``, SUMO encodes ``x=longitude`` and
    ``y=latitude``.  This function performs that swap explicitly.
    """

    source = Path(path)
    if not source.is_file():
        raise SumoOutputError(f"SUMO did not create FCD output: {source}")
    traces: dict[str, list[FCDSample]] = {}
    try:
        for _, element in ET.iterparse(source, events=("end",)):
            if _local_name(element.tag) != "timestep":
                continue
            timestamp_s = float(element.attrib["time"])
            for vehicle in element:
                if _local_name(vehicle.tag) != "vehicle":
                    continue
                vehicle_id = vehicle.attrib.get("id", "").strip()
                if not vehicle_id:
                    raise SumoOutputError("FCD vehicle is missing an id")
                lon = float(vehicle.attrib["x"])
                lat = float(vehicle.attrib["y"])
                if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                    raise SumoOutputError(
                        "FCD coordinates are not WGS84; ensure --fcd-output.geo true"
                    )
                speed_text = vehicle.attrib.get("speed")
                speed = float(speed_text) if speed_text is not None else None
                lane = vehicle.attrib.get("lane")
                edge = vehicle.attrib.get("edge") or _edge_from_lane(lane)
                traces.setdefault(vehicle_id, []).append(
                    FCDSample(timestamp_s, lat, lon, speed, edge, lane)
                )
            element.clear()
    except (ET.ParseError, KeyError, ValueError) as exc:
        raise SumoOutputError(f"invalid SUMO FCD XML at {source}: {exc}") from exc
    return {
        vehicle_id: tuple(sorted(samples, key=lambda sample: sample.timestamp_s))
        for vehicle_id, samples in traces.items()
    }


def _resample_trace(
    samples: Sequence[FCDSample],
    *,
    interval_s: float,
    max_points: int,
) -> tuple[FCDSample, ...]:
    if not samples:
        return ()
    ordered = sorted(samples, key=lambda sample: sample.timestamp_s)
    filtered = [ordered[0]]
    for sample in ordered[1:]:
        if sample.timestamp_s - filtered[-1].timestamp_s >= interval_s:
            filtered.append(sample)
    if len(filtered) <= max_points:
        return tuple(filtered)
    # Evenly-spaced integer indices are deterministic and retain both ends.
    indices = [
        round(index * (len(filtered) - 1) / (max_points - 1))
        for index in range(max_points)
    ]
    return tuple(filtered[index] for index in indices)


def select_longest_trace(
    traces: Mapping[str, Sequence[FCDSample]],
    *,
    interval_s: float,
    max_points: int,
    min_points: int,
) -> tuple[str, tuple[FCDSample, ...]]:
    """Choose raw-longest trace (vehicle-id tie break), then resample it."""

    eligible = [vehicle_id for vehicle_id, samples in traces.items() if samples]
    if not eligible:
        raise SumoOutputError("SUMO FCD contains no vehicle trajectory")
    vehicle_id = min(eligible, key=lambda item: (-len(traces[item]), item))
    selected = _resample_trace(
        traces[vehicle_id], interval_s=interval_s, max_points=max_points
    )
    if len(selected) < min_points:
        raise SumoOutputError(
            f"longest SUMO trace has only {len(selected)} points after resampling; "
            f"need at least {min_points}"
        )
    return vehicle_id, selected


def select_background_trajectories(
    traces: Mapping[str, Sequence[FCDSample]],
    *,
    held_out_vehicle_id: str,
    interval_s: float,
    max_points: int,
    min_points: int = 2,
) -> tuple[SumoBackgroundTrajectory, ...]:
    """Return de-identified, resampled traces excluding the evaluation vehicle."""

    selected: list[SumoBackgroundTrajectory] = []
    for vehicle_id in sorted(traces):
        if vehicle_id == held_out_vehicle_id:
            continue
        samples = _resample_trace(
            traces[vehicle_id], interval_s=interval_s, max_points=max_points
        )
        if len(samples) < min_points:
            continue
        selected.append(
            SumoBackgroundTrajectory(
                points=tuple((sample.lat, sample.lon) for sample in samples),
                times=tuple(sample.timestamp_s for sample in samples),
            )
        )
    return tuple(selected)


def parse_vehicle_routes(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read embedded or referenced vehicle routes from a SUMO route file."""

    source = Path(path)
    if not source.is_file():
        return {}
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError as exc:
        raise SumoOutputError(f"invalid SUMO route XML at {source}: {exc}") from exc

    route_defs: dict[str, tuple[str, ...]] = {}
    for element in root.iter():
        if _local_name(element.tag) != "route":
            continue
        route_id = element.attrib.get("id")
        edges = tuple(element.attrib.get("edges", "").split())
        if route_id and edges:
            route_defs[route_id] = edges

    routes: dict[str, tuple[str, ...]] = {}
    for vehicle in root.iter():
        if _local_name(vehicle.tag) != "vehicle":
            continue
        vehicle_id = vehicle.attrib.get("id")
        if not vehicle_id:
            continue
        edges: tuple[str, ...] = ()
        for child in vehicle:
            if _local_name(child.tag) == "route":
                edges = tuple(child.attrib.get("edges", "").split())
                if edges:
                    break
        if not edges:
            route_ref = vehicle.attrib.get("route")
            if route_ref:
                edges = route_defs.get(route_ref, ())
        routes[vehicle_id] = edges
    return routes


def _fmt(value: float | int) -> str:
    return f"{value:g}"


def _sumo_environment(toolchain: SumoToolchain) -> dict[str, str]:
    """Give randomTrips a coherent home and access to its router binaries."""

    environment = dict(os.environ)
    environment["SUMO_HOME"] = str(toolchain.sumo_home)
    binary_dir = str(toolchain.sumo.parent)
    old_path = environment.get("PATH", "")
    environment["PATH"] = binary_dir + (os.pathsep + old_path if old_path else "")
    return environment


def _command_arrays(
    toolchain: SumoToolchain,
    osm_path: Path,
    workdir: Path,
    config: SumoSmokeConfig,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Path]]:
    net_file = workdir / "beijing_smoke.net.xml"
    trip_file = workdir / "beijing_smoke.trips.xml"
    route_file = workdir / "beijing_smoke.rou.xml"
    fcd_file = workdir / "beijing_smoke.fcd.xml"
    vehicle_route_file = workdir / "beijing_smoke.vehroute.xml"
    bbox = ",".join(_fmt(value) for value in config.bbox)
    commands = {
        "netconvert": (
            str(toolchain.netconvert),
            "--osm-files",
            str(osm_path),
            "--keep-edges.in-geo-boundary",
            bbox,
            "--keep-edges.by-vclass",
            "passenger",
            "--keep-edges.components",
            "1",
            "--junctions.join",
            "true",
            "--output-file",
            str(net_file),
        ),
        "randomTrips.py": (
            sys.executable,
            str(toolchain.random_trips),
            "--net-file",
            str(net_file),
            "-o",
            str(trip_file),
            "--route-file",
            str(route_file),
            "--seed",
            str(config.route_seed),
            "--begin",
            _fmt(config.demand_begin_s),
            "--end",
            _fmt(config.demand_end_s),
            "--period",
            _fmt(config.demand_period_s),
            "--vehicle-class",
            "passenger",
            "--edge-permission",
            "passenger",
            "--min-distance",
            _fmt(config.min_trip_distance_m),
            "--validate",
            "--remove-loops",
            "--prefix",
            "smoke_",
        ),
        "sumo": (
            str(toolchain.sumo),
            "--net-file",
            str(net_file),
            "--route-files",
            str(route_file),
            "--seed",
            str(config.simulation_seed),
            "--begin",
            _fmt(config.demand_begin_s),
            "--end",
            _fmt(config.simulation_end_s),
            "--step-length",
            _fmt(config.fcd_period_s),
            "--device.fcd.period",
            _fmt(config.fcd_period_s),
            "--fcd-output",
            str(fcd_file),
            "--fcd-output.geo",
            "true",
            "--fcd-output.attributes",
            "id,x,y,speed,lane",
            "--vehroute-output",
            str(vehicle_route_file),
            "--vehroute-output.write-unfinished",
            "true",
            "--no-step-log",
            "true",
        ),
    }
    files = {
        "osm": osm_path,
        "network": net_file,
        "trips": trip_file,
        "routes": route_file,
        "fcd": fcd_file,
        "vehicle_routes": vehicle_route_file,
        "randomTrips.py": toolchain.random_trips,
    }
    return commands, files


def run_sumo_smoke_demo(
    *,
    osm_path: str | Path | None = None,
    workdir: str | Path = DEFAULT_WORKDIR,
    config: SumoSmokeConfig | None = None,
    toolchain: SumoToolchain | None = None,
) -> SumoDemoRecord:
    """Run the pinned SUMO smoke scenario and return its longest trace.

    All three stages execute as subprocess argv arrays.  The work directory is
    under ``cache/sumo_demo`` by default and is ignored by Git.
    """

    active_config = config or SumoSmokeConfig()
    source = Path(osm_path).expanduser().resolve() if osm_path else _existing_default_osm()
    if not source.is_file():
        raise FileNotFoundError(f"SUMO OSM input does not exist: {source}")
    destination = Path(workdir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    active_toolchain = toolchain or resolve_sumo_toolchain()
    environment = _sumo_environment(active_toolchain)
    commands, files = _command_arrays(
        active_toolchain, source, destination, active_config
    )

    versions = (
        ("sumo", _version(active_toolchain.sumo, environment)),
        ("netconvert", _version(active_toolchain.netconvert, environment)),
    )
    for stage in ("netconvert", "randomTrips.py", "sumo"):
        _run(commands[stage], environment=environment)

    traces = parse_fcd(files["fcd"])
    vehicle_id, samples = select_longest_trace(
        traces,
        interval_s=active_config.resample_interval_s,
        max_points=active_config.max_points,
        min_points=active_config.min_points,
    )
    background_trajectories = select_background_trajectories(
        traces,
        held_out_vehicle_id=vehicle_id,
        interval_s=active_config.resample_interval_s,
        max_points=active_config.max_points,
        min_points=2,
    )
    actual_routes = parse_vehicle_routes(files["vehicle_routes"])
    planned_routes = parse_vehicle_routes(files["routes"])
    route_edges = actual_routes.get(vehicle_id) or planned_routes.get(vehicle_id, ())

    missing_outputs = [str(path) for path in files.values() if not path.is_file()]
    if missing_outputs:
        raise SumoOutputError(
            "SUMO pipeline did not create expected files: " + ", ".join(missing_outputs)
        )
    digests = tuple(sorted((name, _sha256(path)) for name, path in files.items()))
    command_records = tuple((name, commands[name]) for name in commands)
    provenance = SumoRunProvenance(
        scenario="controlled_beijing_passenger_smoke_v1",
        disclaimer=(
            "DEMO ONLY: deterministic random traffic, not a calibrated Beijing "
            "mobility population or final thesis scenario generator."
        ),
        versions=versions,
        commands=command_records,
        sha256=digests,
    )
    evaluator = SumoEvaluatorMetadata(
        vehicle_id=vehicle_id,
        samples=samples,
        route_edges=tuple(route_edges),
    )
    return SumoDemoRecord(
        # The simulator's vehicle ID is evaluator-only identity information.
        # This public handle identifies the selected evaluation slot without
        # embedding that private identifier.
        record_id=f"sumo/{provenance.scenario}/evaluation_0000",
        points=tuple((sample.lat, sample.lon) for sample in samples),
        times=tuple(sample.timestamp_s for sample in samples),
        evaluator_only=evaluator,
        provenance=provenance,
        background_trajectories=background_trajectories,
        network_path=str(files["network"]),
    )


__all__ = [
    "BEIJING_SMOKE_BBOX",
    "DEFAULT_ROUTE_SEED",
    "DEFAULT_SIM_SEED",
    "DEFAULT_WORKDIR",
    "FCDSample",
    "SumoCommandError",
    "SumoBackgroundTrajectory",
    "SumoDemoRecord",
    "SumoEvaluatorMetadata",
    "SumoOutputError",
    "SumoRunProvenance",
    "SumoSmokeConfig",
    "SumoToolchain",
    "SumoUnavailableError",
    "parse_fcd",
    "load_sumo_road_network",
    "road_network_semantic_sha256",
    "parse_vehicle_routes",
    "resolve_sumo_toolchain",
    "run_sumo_smoke_demo",
    "select_background_trajectories",
    "select_longest_trace",
]
