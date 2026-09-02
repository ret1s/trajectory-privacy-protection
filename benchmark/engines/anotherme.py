"""Source-mapped clean-room implementation of AnotherMe's public VTGA.

The implementation follows the standalone ``VTGAs/gen_virtual_traj.py``
pipeline published by the AnotherMe authors at repository revision
``0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f``.  It intentionally does not
copy the repository's embedded AMap credentials or make hidden network calls.
Coordinate conversion and routing are explicit injected dependencies.

The raw generator preserves the upstream algorithm's observable constants:

* walking below 3 m/s, cycling from 3 through 10 m/s, driving above 10 m/s;
* removal of consecutive navigation points less than 6 m apart;
* linear densification at approximately 2 m;
* quadratic Bezier replacement around 70--110 degree turns (five samples on
  either side);
* replay of the rounded real speed sequence on a 3 s sampling interval with a
  1.1 m distance tolerance;
* independent integer coordinate noise in ``[-20, 20] / 800000`` degrees; and
* output timestamps spaced by 3 s.

The benchmark protocol requires one released event per input event, whereas
the official VTGA emits a variable-length 3 s trajectory. ``protect_run``
therefore performs a separately labelled normalized-time alignment.  The
unaligned, source-shaped output remains available through
``generate_virtual_trajectory`` and in ``last_trace`` for audit/validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random
from typing import Any, Protocol, Sequence, cast, runtime_checkable

import networkx as nx
import numpy as np
from pyproj import Geod

from core.demo_protocol import TrajectoryPoint, make_replacement_run


LatLon = tuple[float, float]
_WGS84_GEOD = Geod(ellps="WGS84")


class AnotherMeGenerationError(RuntimeError):
    """Raised when an upstream-required route or speed sequence is absent."""


class TransportMode(str, Enum):
    """AMap route mode selected by the public standalone Python VTGA."""

    WALKING = "walking"
    BICYCLING = "bicycling"
    DRIVING = "driving"


@runtime_checkable
class CoordinateConverter(Protocol):
    """Convert a WGS84 point into the coordinate system expected by a router."""

    name: str

    def convert(self, point: LatLon) -> LatLon:
        ...


@runtime_checkable
class RouteProvider(Protocol):
    """Return a navigation polyline including start and destination."""

    name: str

    def route(self, start: LatLon, end: LatLon, mode: TransportMode) -> Sequence[LatLon]:
        ...


@runtime_checkable
class VirtualEndpointMapper(Protocol):
    """Map a real origin/destination to a virtual-user origin/destination."""

    name: str

    def map_endpoints(self, start: LatLon, end: LatLon) -> tuple[LatLon, LatLon]:
        ...


class IdentityCoordinateConverter:
    """No-op converter used with a WGS84 local road graph.

    This is a benchmark adaptation, not a substitute for AMap's WGS84-to-GCJ02
    conversion in an official-service parity run.
    """

    name = "identity_wgs84"

    def convert(self, point: LatLon) -> LatLon:
        return _normalise_latlon(point, "coordinate")


class IdentityVirtualEndpointMapper:
    """Keep endpoints unchanged for stage-level standalone-VTGA validation."""

    name = "identity_real_endpoints"

    def map_endpoints(self, start: LatLon, end: LatLon) -> tuple[LatLon, LatLon]:
        return (
            _normalise_latlon(start, "start"),
            _normalise_latlon(end, "end"),
        )


class RoadNetworkVirtualEndpointMapper:
    """Deterministic local analogue of AnotherMe's virtual-user relocation.

    The mobile system maps POIs to another city while approximately preserving
    their distance pattern.  For a SUMO/OSM graph with no portable POI corpus,
    this adapter relocates the origin and preserves the real origin-to-
    destination displacement as closely as graph vertices allow.  It is
    explicitly an adapted component, not a reconstruction of AMap POI mapping.
    """

    name = "local_distance_pattern_virtual_endpoints"

    def __init__(
        self,
        road_network: Any,
        *,
        anchor_min_m: float = 700.0,
        anchor_max_m: float = 3_000.0,
        seed: int = 0,
    ) -> None:
        if len(road_network) < 2:
            raise ValueError("virtual endpoint mapping needs at least two vertices")
        anchor_min_m = float(anchor_min_m)
        anchor_max_m = float(anchor_max_m)
        if not math.isfinite(anchor_min_m) or anchor_min_m < 0.0:
            raise ValueError("anchor_min_m must be finite and non-negative")
        if not math.isfinite(anchor_max_m) or anchor_max_m <= 0.0:
            raise ValueError("anchor_max_m must be finite and positive")
        if anchor_max_m < anchor_min_m:
            raise ValueError("anchor_max_m must be >= anchor_min_m")
        self.road_network = road_network
        self.anchor_min_m = anchor_min_m
        self.anchor_max_m = anchor_max_m
        self.seed = int(seed)
        self.last_mapping: dict[str, float | int] = {}

    def map_endpoints(self, start: LatLon, end: LatLon) -> tuple[LatLon, LatLon]:
        start = _normalise_latlon(start, "start")
        end = _normalise_latlon(end, "end")
        start_xy = np.asarray(self.road_network.point_xy(*start), dtype=float)
        end_xy = np.asarray(self.road_network.point_xy(*end), dtype=float)
        distances = np.linalg.norm(self.road_network.xy - start_xy, axis=1)
        real_start_index = int(np.argmin(distances))
        pool = np.flatnonzero(
            (distances >= self.anchor_min_m) & (distances <= self.anchor_max_m)
        )
        pool = pool[pool != real_start_index]
        if not len(pool):
            pool = np.argsort(distances)[::-1]
            pool = pool[pool != real_start_index][: min(64, len(distances) - 1)]
        target_radius = 0.5 * (self.anchor_min_m + self.anchor_max_m)
        ordered = sorted(
            (int(index) for index in pool),
            key=lambda index: (abs(float(distances[index]) - target_radius), index),
        )
        shortlist = ordered[: min(16, len(ordered))]
        virtual_start_index = shortlist[random.Random(self.seed).randrange(len(shortlist))]

        real_displacement = end_xy - start_xy
        desired_end_xy = (
            np.asarray(self.road_network.xy[virtual_start_index]) + real_displacement
        )
        end_errors = np.linalg.norm(self.road_network.xy - desired_end_xy, axis=1)
        graph = self.road_network.graph
        start_node = _python_scalar(
            self.road_network.node_ids[virtual_start_index]
        )
        virtual_end_index: int | None = None
        for candidate in np.argsort(end_errors):
            candidate = int(candidate)
            if candidate == virtual_start_index:
                continue
            end_node = _python_scalar(self.road_network.node_ids[candidate])
            if nx.has_path(graph, start_node, end_node):
                virtual_end_index = candidate
                break
        if virtual_end_index is None:
            raise AnotherMeGenerationError("no distinct reachable virtual endpoint")

        virtual_start = self.road_network.latlon(virtual_start_index)
        virtual_end = self.road_network.latlon(virtual_end_index)
        self.last_mapping = {
            "real_start_vertex_index": real_start_index,
            "virtual_start_vertex_index": virtual_start_index,
            "virtual_end_vertex_index": virtual_end_index,
            "origin_relocation_m": float(distances[virtual_start_index]),
            "destination_vector_error_m": float(end_errors[virtual_end_index]),
        }
        return virtual_start, virtual_end


class RoadNetworkRouteProvider:
    """Deterministic shortest-path router for the benchmark ``RoadNetwork``.

    The exact SUMO passenger graph supports only the driving branch and fails
    closed for walking/cycling. Other graphs retain the requested mode as an
    explicit adapter input but are still not AMap routing parity.
    """

    name = "local_road_network_shortest_path_with_edge_geometry"
    _DRIVING_EXCLUDED = frozenset(
        {"cycleway", "footway", "path", "pedestrian", "steps", "track"}
    )
    _BICYCLING_EXCLUDED = frozenset(
        {"motorway", "motorway_link", "trunk", "trunk_link", "steps"}
    )
    _WALKING_EXCLUDED = frozenset(
        {"motorway", "motorway_link", "trunk", "trunk_link"}
    )

    def __init__(self, road_network: Any, *, weight: str = "length") -> None:
        if len(road_network) == 0:
            raise ValueError("road_network must contain at least one vertex")
        self.road_network = road_network
        self.weight = str(weight)
        self.last_route_fallback = "none"
        self.last_mode_policy = "not_run"
        self._active_mode = TransportMode.DRIVING
        self._node_to_index = {
            _python_scalar(node): index
            for index, node in enumerate(road_network.node_ids)
        }

    @staticmethod
    def _highway_values(data: dict[str, Any]) -> set[str]:
        raw = data.get("highway", "")
        values = raw if isinstance(raw, (list, tuple, set)) else (raw,)
        return {str(value).removeprefix("highway.") for value in values if value}

    def _edge_allowed(self, data: dict[str, Any], mode: TransportMode) -> bool:
        if str(data.get("access", "")).lower() == "no":
            return False
        values = self._highway_values(data)
        if mode is TransportMode.DRIVING:
            if str(data.get("motor_vehicle", "")).lower() == "no":
                return False
            return not bool(values & self._DRIVING_EXCLUDED)
        if mode is TransportMode.BICYCLING:
            if str(data.get("bicycle", "")).lower() == "no":
                return False
            return not bool(values & self._BICYCLING_EXCLUDED)
        if str(data.get("foot", "")).lower() == "no":
            return False
        return not bool(values & self._WALKING_EXCLUDED)

    def _mode_graph(self, mode: TransportMode):
        graph = self.road_network.graph
        if graph.graph.get("source") == "eclipse_sumo_passenger_network":
            if mode is not TransportMode.DRIVING:
                raise AnotherMeGenerationError(
                    "the SUMO candidate graph is passenger-only and cannot support "
                    f"AnotherMe {mode.value} routing"
                )
            self.last_mode_policy = "sumo_passenger_permissions"
            return graph
        if graph.is_multigraph():
            edges = [
                (u, v, key)
                for u, v, key, data in graph.edges(keys=True, data=True)
                if self._edge_allowed(data, mode)
            ]
        else:
            edges = [
                (u, v)
                for u, v, data in graph.edges(data=True)
                if self._edge_allowed(data, mode)
            ]
        # The chosen mode is inferred from the secret speed profile.  Report
        # the routing policy without embedding that secret-derived label in
        # attacker-visible metadata.
        self.last_mode_policy = "local_mode_filtered_routing"
        return graph.edge_subgraph(edges)

    def _edge_polyline(self, start_node: Any, end_node: Any) -> tuple[LatLon, ...]:
        """Return the selected edge geometry, oriented from start to end."""

        graph = self.road_network.graph
        raw = graph.get_edge_data(start_node, end_node)
        if raw is None:
            raise AnotherMeGenerationError(
                f"route edge {start_node!r}->{end_node!r} is absent"
            )
        candidates = raw.values() if graph.is_multigraph() else (raw,)
        if graph.graph.get("source") == "eclipse_sumo_passenger_network":
            # load_sumo_road_network already excludes every edge for which
            # sumolib reports passenger=False.  Reinterpreting those exact SUMO
            # permissions through generic OSM highway labels could incorrectly
            # reject an edge that the simulator itself used.
            candidates = tuple(candidates)
        else:
            candidates = tuple(
                data
                for data in candidates
                if self._edge_allowed(data, self._active_mode)
            )
        if not candidates:
            raise AnotherMeGenerationError(
                f"route edge {start_node!r}->{end_node!r} violates "
                f"{self._active_mode.value} permissions"
            )

        def edge_cost(data: dict[str, Any]) -> tuple[float, str]:
            value = data.get(self.weight, 1.0)
            try:
                cost = float(value)
            except (TypeError, ValueError):
                cost = 1.0
            return cost, repr(sorted(data.items(), key=lambda item: str(item[0])))

        data = min(candidates, key=edge_cost)
        geometry = data.get("geometry")
        if geometry is None:
            coordinates = [
                (
                    float(graph.nodes[start_node]["x"]),
                    float(graph.nodes[start_node]["y"]),
                ),
                (
                    float(graph.nodes[end_node]["x"]),
                    float(graph.nodes[end_node]["y"]),
                ),
            ]
        else:
            coordinates = [(float(lon), float(lat)) for lon, lat in geometry.coords]
            start_xy = np.asarray(
                [graph.nodes[start_node]["x"], graph.nodes[start_node]["y"]],
                dtype=float,
            )
            if np.linalg.norm(np.asarray(coordinates[-1]) - start_xy) < np.linalg.norm(
                np.asarray(coordinates[0]) - start_xy
            ):
                coordinates.reverse()
        return tuple((lat, lon) for lon, lat in coordinates)

    def route(self, start: LatLon, end: LatLon, mode: TransportMode) -> tuple[LatLon, ...]:
        route_graph = self._mode_graph(mode)
        self._active_mode = mode
        start = _normalise_latlon(start, "start")
        end = _normalise_latlon(end, "end")
        start_index, _ = self.road_network.nearest(*start)
        end_index, _ = self.road_network.nearest(*end)
        start_node = _python_scalar(self.road_network.node_ids[start_index])
        end_node = _python_scalar(self.road_network.node_ids[end_index])
        self.last_route_fallback = "none"
        try:
            nodes = nx.shortest_path(
                route_graph,
                source=start_node,
                target=end_node,
                weight=self.weight,
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            raise AnotherMeGenerationError(
                f"no directionally valid route between graph nodes "
                f"{start_node!r} and {end_node!r}"
            ) from exc
        if len(nodes) == 1:
            return (self.road_network.latlon(self._node_to_index[nodes[0]]),)
        route: list[LatLon] = []
        for first, second in zip(nodes, nodes[1:]):
            segment = self._edge_polyline(first, second)
            route.extend(segment if not route else segment[1:])
        return tuple(route)


@dataclass(frozen=True)
class AnotherMeGenerationTrace:
    """Evaluator-only trace of every stage in the public VTGA pipeline."""

    average_speed_mps: float
    speed_sequence_mps: tuple[float, ...]
    transport_mode: TransportMode
    mapped_start: LatLon
    mapped_end: LatLon
    converted_start: LatLon
    converted_end: LatLon
    navigation_route: tuple[LatLon, ...]
    filtered_route: tuple[LatLon, ...]
    dense_route: tuple[LatLon, ...]
    shape_obfuscated_route: tuple[LatLon, ...]
    speed_obfuscated_route: tuple[LatLon, ...]
    virtual_trajectory: tuple[TrajectoryPoint, ...]

    def stage_counts(self) -> dict[str, int]:
        return {
            "navigation": len(self.navigation_route),
            "filtered": len(self.filtered_route),
            "dense": len(self.dense_route),
            "shape_obfuscated": len(self.shape_obfuscated_route),
            "speed_obfuscated": len(self.speed_obfuscated_route),
            "virtual": len(self.virtual_trajectory),
        }


def _python_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _distance_m(first: LatLon, second: LatLon) -> float:
    """WGS84 ellipsoidal distance, equivalent to upstream geopy.geodesic."""

    _azimuth_1, _azimuth_2, distance = _WGS84_GEOD.inv(
        first[1], first[0], second[1], second[0]
    )
    return float(distance)


def _normalise_latlon(point: Any, name: str) -> LatLon:
    if hasattr(point, "lat") and hasattr(point, "lon"):
        lat, lon = point.lat, point.lon
    else:
        try:
            lat, lon = point[0], point[1]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"{name} must contain latitude and longitude") from exc
    lat, lon = float(lat), float(lon)
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError(f"{name} contains a non-finite coordinate")
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise ValueError(f"{name} is outside the WGS84 coordinate range")
    return lat, lon


def _trajectory_points(points: Sequence[Any]) -> tuple[TrajectoryPoint, ...]:
    result = tuple(points)
    if len(result) < 2:
        raise ValueError("AnotherMe VTGA needs at least two trajectory points")
    if not all(isinstance(point, TrajectoryPoint) for point in result):
        raise TypeError("trajectory values must be TrajectoryPoint instances")
    if any(
        result[i].timestamp_s >= result[i + 1].timestamp_s
        for i in range(len(result) - 1)
    ):
        raise ValueError("AnotherMe VTGA needs strictly increasing timestamps")
    return cast(tuple[TrajectoryPoint, ...], result)


def trajectory_speed_profile(
    trajectory: Sequence[TrajectoryPoint],
) -> tuple[float, tuple[float, ...]]:
    """Compute the rounded speed profile used by upstream ``init_``."""

    points = _trajectory_points(trajectory)
    total_distance_m = 0.0
    total_time_s = 0.0
    speeds: list[float] = []
    for previous, current in zip(points, points[1:]):
        distance_m = _distance_m(
            (previous.lat, previous.lon), (current.lat, current.lon)
        )
        elapsed_s = current.timestamp_s - previous.timestamp_s
        total_distance_m += distance_m
        total_time_s += elapsed_s
        # The public code skips only intervals <= 0.0001. Strict timestamp
        # validation above makes this guard documentary for benchmark inputs.
        if elapsed_s > 0.0001:
            speeds.append(round(distance_m / elapsed_s, 2))
    if total_time_s <= 0.0 or not speeds:
        raise AnotherMeGenerationError("trajectory does not provide a usable speed sequence")
    return round(total_distance_m / total_time_s, 2), tuple(speeds)


def select_transport_mode(average_speed_mps: float) -> TransportMode:
    """Apply the public standalone VTGA's 3/10 m/s mode thresholds."""

    speed = float(average_speed_mps)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("average_speed_mps must be finite and non-negative")
    if speed < 3.0:
        return TransportMode.WALKING
    if speed <= 10.0:
        return TransportMode.BICYCLING
    return TransportMode.DRIVING


def filter_navigation_route(route: Sequence[Any]) -> tuple[LatLon, ...]:
    """Reproduce the upstream consecutive-point 6 m filter.

    The source deliberately omits route element zero and compares each later
    point with its immediate predecessor in the *unfiltered* navigation route.
    """

    points = tuple(_normalise_latlon(point, f"route[{i}]") for i, point in enumerate(route))
    if len(points) < 2:
        raise AnotherMeGenerationError("navigation service returned fewer than two points")
    result = tuple(
        current
        for previous, current in zip(points, points[1:])
        if _distance_m(previous, current) >= 6.0
    )
    if not result:
        raise AnotherMeGenerationError("the 6 m route filter removed every navigation point")
    return result


def _insert_between(start: LatLon, end: LatLon, count: int) -> tuple[LatLon, ...]:
    if count <= 0:
        return ()
    lat_step = (end[0] - start[0]) / (count + 1)
    lon_step = (end[1] - start[1]) / (count + 1)
    return tuple(
        (round(start[0] + i * lat_step, 6), round(start[1] + i * lon_step, 6))
        for i in range(1, count + 1)
    )


def densify_route(route: Sequence[Any]) -> tuple[LatLon, ...]:
    """Reproduce upstream ``round(distance / 2) - 1`` densification."""

    points = tuple(_normalise_latlon(point, f"route[{i}]") for i, point in enumerate(route))
    if not points:
        raise AnotherMeGenerationError("cannot densify an empty route")
    dense: list[LatLon] = [points[0]]
    for previous, current in zip(points, points[1:]):
        insert_count = round(_distance_m(previous, current) / 2.0) - 1
        if insert_count >= 1:
            dense.extend(_insert_between(previous, current, insert_count))
        dense.append(current)
    return tuple(dense)


def _turn_angle_degrees(first: LatLon, middle: LatLon, last: LatLon) -> float:
    a = np.asarray(first, dtype=float) * 100_000.0
    b = np.asarray(middle, dtype=float) * 100_000.0
    c = np.asarray(last, dtype=float) * 100_000.0
    ab, bc = b - a, c - b
    denominator = float(np.linalg.norm(ab) * np.linalg.norm(bc))
    if denominator == 0.0:
        return math.nan
    cosine = float(np.clip(np.dot(ab, bc) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _quadratic_bezier(
    first: LatLon, control: LatLon, last: LatLon, count: int
) -> tuple[LatLon, ...]:
    p0 = np.asarray(first, dtype=float)
    p1 = np.asarray(control, dtype=float)
    p2 = np.asarray(last, dtype=float)
    result: list[LatLon] = []
    for t in np.linspace(0.0, 1.0, num=count):
        point = (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * p1 + t**2 * p2
        result.append((round(float(point[0]), 6), round(float(point[1]), 6)))
    return tuple(result)


def obfuscate_shape(route: Sequence[Any]) -> tuple[LatLon, ...]:
    """Apply the source's Bezier smoothing to 70--110 degree turns."""

    points = [_normalise_latlon(point, f"route[{i}]") for i, point in enumerate(route)]
    corners = [
        index - 1
        for index in range(2, len(points))
        if 70.0 <= _turn_angle_degrees(
            points[index - 2], points[index - 1], points[index]
        ) <= 110.0
    ]
    for index in corners:
        left = max(0, index - 5)
        right = min(index + 5, len(points) - 1)
        fitted = _quadratic_bezier(
            points[left], points[index], points[right], right - left + 1
        )
        points[left : right + 1] = fitted
    return tuple(points)


def replay_speed(
    route: Sequence[Any],
    speed_sequence_mps: Sequence[float],
    *,
    interval_s: float = 3.0,
    tolerance_m: float = 1.1,
) -> tuple[LatLon, ...]:
    """Reproduce the upstream sequential distance-match speed replay."""

    points = tuple(_normalise_latlon(point, f"route[{i}]") for i, point in enumerate(route))
    speeds = tuple(float(speed) for speed in speed_sequence_mps)
    if not points:
        raise AnotherMeGenerationError("cannot replay speed over an empty route")
    if not speeds or any(not math.isfinite(speed) or speed < 0.0 for speed in speeds):
        raise AnotherMeGenerationError("speed sequence must contain finite non-negative values")
    interval_s = float(interval_s)
    tolerance_m = float(tolerance_m)
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError("interval_s must be finite and positive")
    if not math.isfinite(tolerance_m) or tolerance_m < 0.0:
        raise ValueError("tolerance_m must be finite and non-negative")

    selected: list[LatLon] = [points[0]]
    speed_index = 0
    for point in points[1:]:
        target_m = speeds[speed_index] * interval_s
        distance_m = _distance_m(selected[-1], point)
        if abs(distance_m - target_m) < tolerance_m:
            selected.append(point)
            speed_index = (speed_index + 1) % len(speeds)
    return tuple(selected)


def add_coordinate_noise(
    route: Sequence[Any], rng: random.Random
) -> tuple[LatLon, ...]:
    """Apply the exact discrete coordinate-noise support from upstream."""

    result: list[LatLon] = []
    for index, raw_point in enumerate(route):
        lat, lon = _normalise_latlon(raw_point, f"route[{index}]")
        # Upstream draws x (longitude) first and y (latitude) second.
        lon_noise = rng.randint(-20, 20) / 800_000.0
        lat_noise = rng.randint(-20, 20) / 800_000.0
        result.append((round(lat + lat_noise, 6), round(lon + lon_noise, 6)))
    return tuple(result)


def _align_normalized_time(
    trajectory: Sequence[TrajectoryPoint], target_times: Sequence[float]
) -> tuple[TrajectoryPoint, ...]:
    """Align a variable-length official release to the benchmark event grid."""

    source = tuple(trajectory)
    times = tuple(float(value) for value in target_times)
    if not source or not times:
        raise ValueError("source trajectory and target times must be non-empty")
    if len(source) == 1:
        return tuple(TrajectoryPoint(t, source[0].lat, source[0].lon) for t in times)

    source_phase = np.linspace(0.0, 1.0, num=len(source))
    if len(times) == 1 or times[-1] == times[0]:
        target_phase = np.zeros(len(times), dtype=float)
    else:
        target_phase = (np.asarray(times) - times[0]) / (times[-1] - times[0])
    lats = np.interp(target_phase, source_phase, [point.lat for point in source])
    lons = np.interp(target_phase, source_phase, [point.lon for point in source])
    return tuple(
        TrajectoryPoint(timestamp, float(lat), float(lon))
        for timestamp, lat, lon in zip(times, lats, lons)
    )


class AnotherMeVTGAEngine:
    """Runnable public-VTGA core with explicit service dependencies."""

    name = "anotherme_vtga_engine"
    source_method = "AnotherMe (Li et al., IEEE TDSC 2024)"
    implementation_level = "paper_adaptation"

    def __init__(
        self,
        route_provider: RouteProvider,
        *,
        coordinate_converter: CoordinateConverter | None = None,
        endpoint_mapper: VirtualEndpointMapper | None = None,
        minimum_raw_samples: int = 1,
        seed: int = 0,
    ) -> None:
        if not isinstance(route_provider, RouteProvider):
            raise TypeError("route_provider must implement the RouteProvider protocol")
        converter = coordinate_converter or IdentityCoordinateConverter()
        if not isinstance(converter, CoordinateConverter):
            raise TypeError("coordinate_converter must implement CoordinateConverter")
        mapper = endpoint_mapper or IdentityVirtualEndpointMapper()
        if not isinstance(mapper, VirtualEndpointMapper):
            raise TypeError("endpoint_mapper must implement VirtualEndpointMapper")
        self.route_provider = route_provider
        self.coordinate_converter = converter
        self.endpoint_mapper = mapper
        self.minimum_raw_samples = int(minimum_raw_samples)
        if self.minimum_raw_samples < 1:
            raise ValueError("minimum_raw_samples must be positive")
        self.seed = int(seed)
        self.last_trace: AnotherMeGenerationTrace | None = None

    @classmethod
    def from_road_network(cls, road_network: Any, *, seed: int = 0) -> "AnotherMeVTGAEngine":
        return cls(
            RoadNetworkRouteProvider(road_network),
            endpoint_mapper=RoadNetworkVirtualEndpointMapper(road_network, seed=seed),
            seed=seed,
        )

    def generate_virtual_trajectory(
        self, real_trajectory: Sequence[TrajectoryPoint]
    ) -> tuple[TrajectoryPoint, ...]:
        """Run the variable-length, source-shaped standalone VTGA."""

        real = _trajectory_points(real_trajectory)
        average_speed, speeds = trajectory_speed_profile(real)
        mode = select_transport_mode(average_speed)
        mapped_start, mapped_end = self.endpoint_mapper.map_endpoints(
            (real[0].lat, real[0].lon), (real[-1].lat, real[-1].lon)
        )
        start = self.coordinate_converter.convert(mapped_start)
        end = self.coordinate_converter.convert(mapped_end)
        navigation = tuple(
            _normalise_latlon(point, f"navigation[{index}]")
            for index, point in enumerate(self.route_provider.route(start, end, mode))
        )
        filtered = filter_navigation_route(navigation)
        dense = densify_route(filtered)
        shaped = obfuscate_shape(dense)
        speed_obfuscated = replay_speed(shaped, speeds)
        noised = add_coordinate_noise(speed_obfuscated, random.Random(self.seed))
        virtual = tuple(
            TrajectoryPoint(real[0].timestamp_s + index * 3.0, lat, lon)
            for index, (lat, lon) in enumerate(noised)
        )
        self.last_trace = AnotherMeGenerationTrace(
            average_speed_mps=average_speed,
            speed_sequence_mps=speeds,
            transport_mode=mode,
            mapped_start=mapped_start,
            mapped_end=mapped_end,
            converted_start=start,
            converted_end=end,
            navigation_route=navigation,
            filtered_route=filtered,
            dense_route=dense,
            shape_obfuscated_route=shaped,
            speed_obfuscated_route=speed_obfuscated,
            virtual_trajectory=virtual,
        )
        if len(virtual) < self.minimum_raw_samples:
            raise AnotherMeGenerationError(
                f"generated trajectory has {len(virtual)} samples; upstream "
                f"experiment validity requires at least {self.minimum_raw_samples}"
            )
        return virtual

    def protect_trajectory(
        self,
        real_trajectory: Sequence[Any],
        times: Sequence[float] | None = None,
    ) -> tuple[LatLon, ...]:
        """Return benchmark-aligned coordinates while retaining the raw trace."""

        values = tuple(real_trajectory)
        if values and all(isinstance(point, TrajectoryPoint) for point in values):
            if times is not None:
                raise ValueError("times must be omitted for TrajectoryPoint input")
            real = _trajectory_points(values)
        else:
            if times is None or len(values) != len(times):
                raise ValueError(
                    "coordinate input requires one aligned timestamp per point"
                )
            real = _trajectory_points(
                tuple(
                    TrajectoryPoint(float(timestamp), *_normalise_latlon(point, f"point[{index}]"))
                    for index, (point, timestamp) in enumerate(zip(values, times))
                )
            )
        raw = self.generate_virtual_trajectory(real)
        aligned = _align_normalized_time(raw, [point.timestamp_s for point in real])
        return tuple((point.lat, point.lon) for point in aligned)

    def protect_run(self, real_trajectory: Sequence[TrajectoryPoint]):
        real = _trajectory_points(real_trajectory)
        raw = self.generate_virtual_trajectory(real)
        aligned = _align_normalized_time(raw, [point.timestamp_s for point in real])
        assert self.last_trace is not None
        return make_replacement_run(
            self.name,
            real,
            aligned,
            public_parameters={
                "implementation_level": self.implementation_level,
                "implementation_origin": "benchmark.engines.anotherme.AnotherMeVTGAEngine",
                "source_method": self.source_method,
                "source_revision": "0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f",
                "route_provider": self.route_provider.name,
                "route_fallback": str(
                    getattr(self.route_provider, "last_route_fallback", "not_reported")
                ),
                "coordinate_converter": self.coordinate_converter.name,
                "endpoint_mapper": self.endpoint_mapper.name,
                "minimum_raw_samples": self.minimum_raw_samples,
                "direction_policy": "directed_routes_only_fail_closed",
                "mode_routing_policy": str(
                    getattr(self.route_provider, "last_mode_policy", "provider_defined")
                ),
                "benchmark_alignment": "normalized_time_interpolation",
                "temporal_metric_scope": "aligned_benchmark_output_not_raw_vtga_parity",
            },
        )


__all__ = [
    "AnotherMeGenerationError",
    "AnotherMeGenerationTrace",
    "AnotherMeVTGAEngine",
    "CoordinateConverter",
    "IdentityCoordinateConverter",
    "IdentityVirtualEndpointMapper",
    "RoadNetworkRouteProvider",
    "RoadNetworkVirtualEndpointMapper",
    "RouteProvider",
    "TransportMode",
    "VirtualEndpointMapper",
    "add_coordinate_noise",
    "densify_route",
    "filter_navigation_route",
    "obfuscate_shape",
    "replay_speed",
    "select_transport_mode",
    "trajectory_speed_profile",
]
