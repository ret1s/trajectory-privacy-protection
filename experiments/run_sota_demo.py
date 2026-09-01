"""Run the paper-inspired SOTA prototypes beside the thesis prototype.

This is an integration/demo runner, not the final scientific benchmark.  The
three comparators in :mod:`core.sota_demo` intentionally implement only the
high-level output contracts of their source methods.  Results therefore MUST
NOT be presented as official reproductions or as a cross-contract leaderboard.

Usage (from the repository root)::

    python -m experiments.run_sota_demo
    python -m experiments.run_sota_demo --quick
    python -m experiments.run_sota_demo --mobility-source geolife --no-map

The default run executes a controlled Eclipse SUMO scenario over the local
Beijing OpenStreetMap extract and writes both an evaluator JSON artifact and a
standalone Folium map under ``outputs/``.  GeoLife remains an explicit
real-data validation option; neither source silently falls back to synthetic
or other mobility data.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
import time
from typing import Iterable, Mapping, Sequence

import numpy as np

from core.demo_protocol import OutputKind, ProtectedRun, TrajectoryPoint
from core.road_network import RoadNetwork
from core.sota_demo import AnotherMeLite, SemanticDummyLite, TransProtectLite
from core.thesis_demo import GeoIAnchoredDummyTrajectoriesLite
from data.geolife import load_trajectories
from data.sumo_demo import (
    DEFAULT_ROUTE_SEED,
    DEFAULT_SIM_SEED,
    DEFAULT_WORKDIR,
    SumoSmokeConfig,
    run_sumo_smoke_demo,
)
from evaluation import metrics as legacy_metrics
from experiments.provenance import (
    GRAPH_PKL,
    assert_graph_matches_manifest,
    begin_run,
    provenance,
    sha256_file,
)
from experiments.rng_util import rng_from_key


DEFAULT_OUTPUT = os.path.join("outputs", "sota_demo_results.json")
DEFAULT_MAP_OUTPUT = os.path.join("outputs", "sota_demo_map.html")
DEFAULT_PREVIEW_OUTPUT = os.path.join("outputs", "sota_demo_preview.png")
DEMO_SCHEMA = "msc-sota-demo-v2"
SUMO_GRAPH_COMPATIBILITY_NOTE = (
    "SUMO mobility is generated on a passenger-only network converted from the "
    "Beijing OSM extract. Protection candidates are generated on the separately "
    "serialized, pinned OSMnx graph whose profile is unfiltered multimodal "
    "(drive+walk+cycle). The graphs cover the same study bbox but are not assumed "
    "to have identical nodes, edges, simplification, or permissions; this demo "
    "therefore does not claim exact route compatibility across the two graphs."
)
BASE_DISCLAIMER = (
    "DEMO / PAPER-INSPIRED PROTOTYPES. These implementations are not official "
    "or faithful reproductions of the cited papers. Metrics are reported within "
    "each output contract and must not be used as a cross-track leaderboard."
)
DISCLAIMER = BASE_DISCLAIMER + " " + SUMO_GRAPH_COMPATIBILITY_NOTE


def _timestamp_s(value, fallback: float) -> float:
    if value is None:
        return float(fallback)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return float(value.timestamp())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _contract_trajectory(points: Sequence, times: Sequence) -> tuple[TrajectoryPoint, ...]:
    if len(points) != len(times):
        raise ValueError("points and times must have the same length")
    return tuple(
        TrajectoryPoint(_timestamp_s(t, i * 60.0), float(lat), float(lon))
        for i, ((lat, lon), t) in enumerate(zip(points, times))
    )


def _model_rng(seed: int, epsilon: float, mechanism: str, record_id: str):
    return rng_from_key(
        int(seed), float(epsilon), mechanism, record_id, schema=f"{DEMO_SCHEMA}-rng"
    )


def _run_models(
    rn: RoadNetwork,
    points,
    times,
    record_id: str,
    *,
    epsilon: float,
    k: int,
    seed: int,
) -> list[tuple[ProtectedRun, str, float]]:
    """Return ``(protected_run, source_method, runtime_ms)`` for all four demos."""

    real = _contract_trajectory(points, times)
    models = (
        TransProtectLite(
            rn,
            candidate_k=max(16, k * 8),
            rng=_model_rng(seed, epsilon, TransProtectLite.name, record_id),
        ),
        AnotherMeLite(
            rn,
            rng=_model_rng(seed, epsilon, AnotherMeLite.name, record_id),
        ),
        SemanticDummyLite(
            rn,
            k=k,
            rng=_model_rng(seed, epsilon, SemanticDummyLite.name, record_id),
        ),
        GeoIAnchoredDummyTrajectoriesLite(
            epsilon,
            rn,
            k=k,
            rng=_model_rng(
                seed, epsilon, GeoIAnchoredDummyTrajectoriesLite.name, record_id
            ),
        ),
    )

    completed = []
    for mechanism in models:
        started = time.perf_counter()
        # Every prototype owns its adapter.  Keeping truth separation next to
        # mechanism-specific internals avoids re-encoding a private real index
        # or REM anchor in this generic runner.
        protected = mechanism.protect_run(real)
        runtime_ms = (time.perf_counter() - started) * 1_000.0
        completed.append((protected, mechanism.source_method, runtime_ms))
    return completed


def _latlon(points: Iterable[TrajectoryPoint]) -> list[tuple[float, float]]:
    return [(point.lat, point.lon) for point in points]


def _event_candidate_points(run: ProtectedRun):
    return [
        [(candidate.lat, candidate.lon) for candidate in event.candidates]
        for event in run.transcript.events
    ]


def _public_tracks(run: ProtectedRun) -> Mapping[str, list[tuple[float, float]]]:
    tracks = defaultdict(list)
    for event in run.transcript.events:
        for candidate in event.candidates:
            tracks[candidate.candidate_id].append((candidate.lat, candidate.lon))
    n_events = len(run.transcript.events)
    return {
        candidate_id: points
        for candidate_id, points in tracks.items()
        if len(points) == n_events
    }


_MAJOR_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
}

_MAP_LABELS = {
    "transprotect_lite": "TransProtect Lite — pseudolocation trajectory",
    "anotherme_lite": "AnotherMe Lite — replacement trajectory",
    "semantic_dummy_lite": "Semantic Dummy Lite — public candidate tracks",
    "geo_i_anchored_dummy_lite": "Proposed model — dummy-only trajectories",
}


def _ground_truth_label(mobility_source: str) -> str:
    if mobility_source == "sumo":
        return "SUMO ground truth"
    if mobility_source == "geolife":
        return "GeoLife ground truth"
    raise ValueError(f"unsupported mobility source: {mobility_source}")


def _run_tracks(run: ProtectedRun) -> Mapping[str, list[tuple[float, float]]]:
    """Return public, temporally stable tracks without consulting truth."""

    if run.transcript.output_kind is OutputKind.REPLACEMENT_TRAJECTORY:
        return {
            "released_trajectory": [
                (event.candidates[0].lat, event.candidates[0].lon)
                for event in run.transcript.events
            ]
        }
    return _public_tracks(run)


def _display_points(
    record: Mapping,
    runs: Sequence[tuple[ProtectedRun, str, float]],
) -> list[tuple[float, float]]:
    """Collect the coordinates that must fit in the evaluator visualisation."""

    points = [(float(lat), float(lon)) for lat, lon in record["points"]]
    for run, _source, _runtime in runs:
        for event in run.transcript.events:
            points.extend(
                (float(candidate.lat), float(candidate.lon))
                for candidate in event.candidates
            )
    if not points:
        raise ValueError("cannot render a map with no coordinates")
    return points


def _display_bounds(
    rn: RoadNetwork,
    record: Mapping,
    runs: Sequence[tuple[ProtectedRun, str, float]],
    *,
    minimum_padding_m: float = 250.0,
    padding_fraction: float = 0.10,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bounds around truth and every public output, padded in projected metres."""

    points = _display_points(record, runs)
    xy = np.asarray([rn.point_xy(lat, lon) for lat, lon in points], dtype=float)
    minimum = np.min(xy, axis=0)
    maximum = np.max(xy, axis=0)
    padding = np.maximum(
        float(minimum_padding_m), (maximum - minimum) * float(padding_fraction)
    )
    west_x, south_y = minimum - padding
    east_x, north_y = maximum + padding
    south, west = rn.proj.to_latlon(west_x, south_y)
    north, east = rn.proj.to_latlon(east_x, north_y)
    return ((float(south), float(west)), (float(north), float(east)))


def _edge_coordinates(rn: RoadNetwork, u, v, data) -> list[tuple[float, float]]:
    """Return one graph edge as GeoJSON-order ``(lon, lat)`` coordinates."""

    geometry = data.get("geometry")
    if geometry is not None:
        return [(float(lon), float(lat)) for lon, lat in geometry.coords]
    return [
        (float(rn.graph.nodes[u]["x"]), float(rn.graph.nodes[u]["y"])),
        (float(rn.graph.nodes[v]["x"]), float(rn.graph.nodes[v]["y"])),
    ]


def _line_parts(geometry) -> list[list[list[float]]]:
    """Flatten a clipped Shapely geometry into GeoJSON line coordinates."""

    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        coordinates = [[float(x), float(y)] for x, y in geometry.coords]
        return [coordinates] if len(coordinates) >= 2 else []
    if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        parts = []
        for child in geometry.geoms:
            parts.extend(_line_parts(child))
        return parts
    return []


def _embedded_road_geojson(
    rn: RoadNetwork,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> dict[str, dict]:
    """Build two compact local MultiLineStrings clipped to display bounds.

    The graph already contains OSM road geometry, so the HTML does not need a
    network tile request to provide geographic context.  Reciprocal directed
    edges with identical geometry are de-duplicated before serialisation.
    """

    from shapely.geometry import LineString, box

    (south, west), (north, east) = bounds
    clip = box(west, south, east, north)
    lines: dict[str, list[list[list[float]]]] = {"minor": [], "major": []}
    unique_edges: dict[
        tuple[tuple[float, float], ...], tuple[list[tuple[float, float]], bool]
    ] = {}
    for u, v, data in rn.graph.edges(data=True):
        coordinates = _edge_coordinates(rn, u, v, data)
        if len(coordinates) < 2:
            continue
        forward = tuple((round(lon, 7), round(lat, 7)) for lon, lat in coordinates)
        signature = min(forward, tuple(reversed(forward)))
        highway = data.get("highway", "")
        highway_values = (
            {str(value) for value in highway}
            if isinstance(highway, (list, tuple, set))
            else {str(highway)}
        )
        is_major = bool(highway_values & _MAJOR_HIGHWAYS)
        existing = unique_edges.get(signature)
        if existing is None:
            unique_edges[signature] = (coordinates, is_major)
        elif is_major and not existing[1]:
            # Parallel/reverse graph edges may disagree on their highway tag.
            # Preserve the stronger visual classification independent of edge
            # iteration order.
            unique_edges[signature] = (existing[0], True)

    for signature in sorted(unique_edges):
        coordinates, is_major = unique_edges[signature]
        line = LineString(coordinates)
        if not line.intersects(clip):
            continue
        road_class = "major" if is_major else "minor"
        lines[road_class].extend(_line_parts(line.intersection(clip)))

    return {
        road_class: {
            "type": "Feature",
            "properties": {"road_class": road_class, "source": "local_pinned_graph"},
            "geometry": {"type": "MultiLineString", "coordinates": coordinates},
        }
        for road_class, coordinates in lines.items()
    }


def _set_diagnostics(run: ProtectedRun, rn: RoadNetwork) -> dict[str, float | None]:
    real = run.truth.real_trajectory
    event_points = _event_candidate_points(run)
    nearest, centroid_error, spread = [], [], []
    dummy_distances = []
    true_hits = []

    for event_idx, (truth, candidates) in enumerate(zip(real, event_points)):
        truth_xy = np.asarray(rn.point_xy(truth.lat, truth.lon))
        candidate_xy = np.asarray([rn.point_xy(lat, lon) for lat, lon in candidates])
        distances = np.linalg.norm(candidate_xy - truth_xy, axis=1)
        nearest.append(float(np.min(distances)))
        centroid = np.mean(candidate_xy, axis=0)
        centroid_error.append(float(np.linalg.norm(centroid - truth_xy)))
        spread.append(float(np.mean(np.linalg.norm(candidate_xy - centroid, axis=1))))

        if run.transcript.output_kind is OutputKind.REAL_PLUS_DUMMIES:
            real_id = run.truth.real_candidate_ids[event_idx]
            candidates_public = run.transcript.events[event_idx].candidates
            match = [i for i, candidate in enumerate(candidates_public) if candidate.candidate_id == real_id]
            true_hits.append(float(len(match) == 1))
            dummy_distances.extend(
                distance for i, distance in enumerate(distances.tolist()) if i not in match
            )
        else:
            dummy_distances.extend(distances.tolist())

    all_public = [point for candidates in event_points for point in candidates]
    counts = [len(candidates) for candidates in event_points]
    result: dict[str, float | None] = {
        "events": float(len(event_points)),
        "mean_candidates_per_event": float(np.mean(counts)),
        "request_multiplier": float(np.mean(counts)),
        "mean_nearest_output_distance_m": float(np.mean(nearest)),
        "mean_centroid_reconstruction_error_m": float(np.mean(centroid_error)),
        "mean_candidate_spread_m": float(np.mean(spread)),
        "mean_dummy_distance_m": float(np.mean(dummy_distances)),
        "p95_dummy_distance_m": float(np.percentile(dummy_distances, 95)),
        "on_road_rate_all_outputs": float(legacy_metrics.on_road_rate(all_public, rn)),
        "stable_public_tracks": float(len(_public_tracks(run))),
    }
    if true_hits:
        result["real_member_inclusion_rate"] = float(np.mean(true_hits))
        result["uniform_guess_success_baseline"] = float(
            np.mean([1.0 / count for count in counts])
        )
    else:
        result["real_member_inclusion_rate"] = None
        result["uniform_guess_success_baseline"] = None

    tracks = _public_tracks(run)
    if tracks:
        times = [
            datetime.fromtimestamp(event.timestamp_s, tz=timezone.utc)
            for event in run.transcript.events
        ]
        result["mean_track_speed_violation_rate"] = float(
            np.mean(
                [
                    legacy_metrics.speed_violation_rate(track, times, rn.proj)
                    for track in tracks.values()
                ]
            )
        )
        result["mean_track_dtw_m"] = float(
            np.mean(
                [legacy_metrics.dtw(_latlon(real), track, rn.proj) for track in tracks.values()]
            )
        )
    else:
        result["mean_track_speed_violation_rate"] = None
        result["mean_track_dtw_m"] = None
    return result


def _replacement_diagnostics(
    run: ProtectedRun,
    rn: RoadNetwork,
    qos_radius_m: float,
) -> dict[str, float]:
    real = _latlon(run.truth.real_trajectory)
    released = [
        (event.candidates[0].lat, event.candidates[0].lon)
        for event in run.transcript.events
    ]
    if len(real) != len(released):
        raise ValueError("replacement demo metrics require aligned trajectories")
    times = [
        datetime.fromtimestamp(event.timestamp_s, tz=timezone.utc)
        for event in run.transcript.events
    ]
    displacement = legacy_metrics.displacements(real, released, rn.proj)
    return {
        "events": float(len(released)),
        "mean_displacement_m": float(np.mean(displacement)),
        "p95_displacement_m": float(np.percentile(displacement, 95)),
        "qos_satisfaction": legacy_metrics.qos_satisfaction(
            real, released, rn.proj, qos_radius_m
        ),
        "hausdorff_m": legacy_metrics.hausdorff(real, released, rn.proj),
        "dtw_m": legacy_metrics.dtw(real, released, rn.proj),
        "on_road_rate": legacy_metrics.on_road_rate(released, rn),
        "speed_violation_rate": legacy_metrics.speed_violation_rate(
            released, times, rn.proj
        ),
        "request_multiplier": 1.0,
    }


def _diagnostics(run: ProtectedRun, rn: RoadNetwork, qos_radius_m: float):
    if run.transcript.output_kind is OutputKind.REPLACEMENT_TRAJECTORY:
        return _replacement_diagnostics(run, rn, qos_radius_m)
    return _set_diagnostics(run, rn)


def _round_value(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)
    return value


def _aggregate(rows: Sequence[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["mechanism"], row["output_kind"], row["source_method"])].append(row)

    summaries = []
    for (mechanism, output_kind, source_method), group in grouped.items():
        keys = sorted({key for row in group for key in row["metrics"]})
        aggregated = {}
        for key in keys:
            values = [row["metrics"].get(key) for row in group]
            numeric = [float(value) for value in values if value is not None]
            aggregated[key] = _round_value(np.mean(numeric)) if numeric else None
        aggregated["runtime_ms"] = _round_value(
            np.mean([row["runtime_ms"] for row in group])
        )
        summaries.append(
            {
                "mechanism": mechanism,
                "output_kind": output_kind,
                "source_method": source_method,
                "records": len(group),
                "metrics": aggregated,
            }
        )
    return summaries


def _format_number(value, decimals=1):
    if value is None:
        return "n/a"
    return f"{float(value):.{decimals}f}"


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    if not rows:
        return
    widths = [
        max(len(str(header)), *(len(str(row[i])) for row in rows))
        for i, header in enumerate(headers)
    ]
    print("  " + "  ".join(str(header).ljust(width) for header, width in zip(headers, widths)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print("  " + "  ".join(str(value).ljust(width) for value, width in zip(row, widths)))


def _print_summary(summaries: Sequence[dict], disclaimer: str = DISCLAIMER) -> None:
    print("\n" + "=" * 78)
    print(disclaimer)
    print("=" * 78)

    replacement = [s for s in summaries if s["output_kind"] == OutputKind.REPLACEMENT_TRAJECTORY.value]
    print("\nTrack A — replacement trajectory (geometry/utility diagnostics)")
    _print_table(
        ("method", "disp m", "p95 m", "QoS", "on-road", "speed viol", "ms"),
        [
            (
                s["mechanism"],
                _format_number(s["metrics"].get("mean_displacement_m")),
                _format_number(s["metrics"].get("p95_displacement_m")),
                _format_number(s["metrics"].get("qos_satisfaction"), 3),
                _format_number(s["metrics"].get("on_road_rate"), 3),
                _format_number(s["metrics"].get("speed_violation_rate"), 3),
                _format_number(s["metrics"].get("runtime_ms")),
            )
            for s in replacement
        ],
    )

    real_plus = [s for s in summaries if s["output_kind"] == OutputKind.REAL_PLUS_DUMMIES.value]
    print("\nTrack B — real + K-1 dummies (set diagnostics; no calibrated attacker yet)")
    _print_table(
        ("method", "K", "real in set", "spread m", "dummy dist m", "road", "ms"),
        [
            (
                s["mechanism"],
                _format_number(s["metrics"].get("mean_candidates_per_event"), 1),
                _format_number(s["metrics"].get("real_member_inclusion_rate"), 3),
                _format_number(s["metrics"].get("mean_candidate_spread_m")),
                _format_number(s["metrics"].get("mean_dummy_distance_m")),
                _format_number(s["metrics"].get("on_road_rate_all_outputs"), 3),
                _format_number(s["metrics"].get("runtime_ms")),
            )
            for s in real_plus
        ],
    )

    dummy_only = [s for s in summaries if s["output_kind"] == OutputKind.DUMMY_ONLY.value]
    print("\nTrack C — thesis dummy-only batch (diagnostics; truth is not a public member)")
    _print_table(
        ("method", "K", "nearest m", "centroid err m", "track DTW m", "road", "ms"),
        [
            (
                s["mechanism"],
                _format_number(s["metrics"].get("mean_candidates_per_event"), 1),
                _format_number(s["metrics"].get("mean_nearest_output_distance_m")),
                _format_number(s["metrics"].get("mean_centroid_reconstruction_error_m")),
                _format_number(s["metrics"].get("mean_track_dtw_m")),
                _format_number(s["metrics"].get("on_road_rate_all_outputs"), 3),
                _format_number(s["metrics"].get("runtime_ms")),
            )
            for s in dummy_only
        ],
    )
    print(
        "\nDo not compare the numeric columns across tracks: their public outputs and "
        "privacy questions differ.\n"
    )


def _load_sumo_record(args):
    """Run SUMO once and preserve privileged simulator state for evaluation.

    Only ``record.to_mechanism_input()`` is returned to the model loop.  Vehicle
    identity, route, speed, edge, and lane data live in the separate
    ``evaluator_only`` object and are never merged into an attacker transcript.
    """

    config = SumoSmokeConfig(
        route_seed=args.sumo_route_seed,
        simulation_seed=args.sumo_simulation_seed,
        demand_end_s=args.sumo_demand_end_s,
        simulation_end_s=args.sumo_simulation_end_s,
        resample_interval_s=args.interval_s,
        max_points=args.max_points,
        min_points=min(8, args.max_points),
        min_trip_distance_m=args.sumo_min_trip_distance_m,
    )
    record = run_sumo_smoke_demo(
        osm_path=args.sumo_osm_path,
        workdir=args.sumo_workdir,
        config=config,
    )
    mobility = {
        "source": "sumo",
        "label": "Eclipse SUMO controlled Beijing passenger smoke scenario",
        "sumo_runs": [record.provenance.to_dict()],
        "evaluator_only": [
            {
                "record_id": record.record_id,
                **record.evaluator_only.to_dict(),
            }
        ],
        "candidate_graph_compatibility": SUMO_GRAPH_COMPATIBILITY_NOTE,
    }
    return [record.to_mechanism_input()], mobility


def _load_geolife_records(args):
    """Load explicit real-data validation records, failing closed if absent."""

    loaded = load_trajectories(
        n_trajectories=args.n_trajectories,
        min_points=min(8, args.max_points),
        max_points=args.max_points,
        interval_s=args.interval_s,
        min_span_m=200,
    )
    if not loaded:
        raise FileNotFoundError(
            "--mobility-source geolife requested, but no qualifying local "
            "GeoLife trace was found. Install the dataset or choose SUMO; no "
            "synthetic fallback is available."
        )

    records = []
    for index, record in enumerate(loaded):
        copied = dict(record)
        copied["record_id"] = f"{record['user']}/{record['file']}#{index}"
        records.append(copied)
    mobility = {
        "source": "geolife",
        "label": "Microsoft GeoLife v1.3 real-data validation",
        "sumo_runs": [],
        "evaluator_only": [],
        "candidate_graph_compatibility": (
            "GeoLife coordinates are evaluated against the pinned OSMnx graph; "
            "this optional mode does not execute SUMO."
        ),
    }
    return records, mobility


def _load_records(args):
    if args.mobility_source == "sumo":
        return _load_sumo_record(args)
    if args.mobility_source == "geolife":
        return _load_geolife_records(args)
    raise ValueError(f"unsupported mobility source: {args.mobility_source!r}")


def _write_map(
    path: str,
    record: Mapping,
    runs: Sequence[tuple[ProtectedRun, str, float]],
    rn: RoadNetwork,
    *,
    mobility_source: str,
):
    import folium

    truth = record["points"]
    truth_label = _ground_truth_label(mobility_source)
    bounds = _display_bounds(rn, record, runs)
    roads = _embedded_road_geojson(rn, bounds)
    centre = [
        (bounds[0][0] + bounds[1][0]) / 2.0,
        (bounds[0][1] + bounds[1][1]) / 2.0,
    ]
    map_obj = folium.Map(
        location=centre,
        tiles=None,
        zoom_start=14,
        control_scale=True,
        prefer_canvas=True,
    )

    # This optional network layer is deliberately disabled.  The useful map
    # context comes from the embedded graph below, so opening the local HTML
    # never depends on an OSM tile response.
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Online OpenStreetMap tiles (optional)",
        overlay=True,
        show=False,
        control=True,
        opacity=0.82,
    ).add_to(map_obj)

    road_layer = folium.FeatureGroup(
        "Local road network — embedded",
        show=True,
        overlay=True,
        control=True,
    )
    folium.GeoJson(
        roads["minor"],
        name="minor roads",
        style_function=lambda _feature: {
            "color": "#d5d2cc",
            "weight": 1.1,
            "opacity": 0.84,
        },
        smooth_factor=1.0,
    ).add_to(road_layer)
    folium.GeoJson(
        roads["major"],
        name="major roads",
        style_function=lambda _feature: {
            "color": "#aba69d",
            "weight": 2.0,
            "opacity": 0.94,
        },
        smooth_factor=1.0,
    ).add_to(road_layer)
    road_layer.add_to(map_obj)

    truth_layer = folium.FeatureGroup(
        f"{truth_label} — evaluator only", show=True
    )
    folium.PolyLine(
        truth,
        color="#171717",
        weight=5,
        opacity=0.95,
        tooltip=f"{truth_label} trajectory (evaluator only)",
    ).add_to(truth_layer)
    folium.CircleMarker(
        truth[0],
        radius=7,
        color="#171717",
        fill=True,
        fill_color="#2e7d32",
        fill_opacity=1.0,
        weight=2,
        tooltip="Ground truth: start",
    ).add_to(truth_layer)
    folium.CircleMarker(
        truth[-1],
        radius=7,
        color="#171717",
        fill=True,
        fill_color="#c62828",
        fill_opacity=1.0,
        weight=2,
        tooltip="Ground truth: end",
    ).add_to(truth_layer)
    truth_layer.add_to(map_obj)

    palette = ("#1565c0", "#ef6c00", "#7b1fa2", "#00897b")
    for colour, (run, _source, _runtime) in zip(palette, runs):
        transcript = run.transcript
        is_thesis = transcript.output_kind is OutputKind.DUMMY_ONLY
        layer = folium.FeatureGroup(
            _MAP_LABELS.get(transcript.mechanism, transcript.mechanism),
            show=is_thesis,
        )
        tracks = _run_tracks(run)
        for candidate_id, track in tracks.items():
            if len(track) < 2:
                continue
            method_label = _MAP_LABELS.get(
                transcript.mechanism, transcript.mechanism
            )
            tooltip = f"{method_label}: {candidate_id}"
            folium.PolyLine(
                track,
                color=colour,
                weight=3 if transcript.output_kind is OutputKind.REPLACEMENT_TRAJECTORY else 2.4,
                opacity=0.82,
                tooltip=tooltip,
            ).add_to(layer)
            for point, phase in ((track[0], "start"), (track[-1], "end")):
                folium.CircleMarker(
                    point,
                    radius=4,
                    color=colour,
                    fill=True,
                    fill_color=colour,
                    fill_opacity=0.85,
                    weight=1,
                    tooltip=f"{candidate_id}: {phase}",
                ).add_to(layer)

        # Candidate-set methods expose opaque public IDs.  Every candidate is
        # rendered identically; evaluator-only real IDs are never consulted.
        if transcript.output_kind is OutputKind.REAL_PLUS_DUMMIES:
            for event in transcript.events:
                for candidate in event.candidates:
                    folium.CircleMarker(
                        (candidate.lat, candidate.lon),
                        radius=2.5,
                        color=colour,
                        fill=True,
                        fill_color=colour,
                        fill_opacity=0.62,
                        weight=1,
                        tooltip=f"{candidate.candidate_id}: {event.event_id}",
                    ).add_to(layer)
        layer.add_to(map_obj)

    if mobility_source == "sumo":
        mobility_note = (
            "Ground-truth movement was generated by Eclipse SUMO on a "
            "passenger-only network converted from OpenStreetMap."
        )
        graph_note = (
            "SUMO passenger graph and the protection models' pinned multimodal "
            "OSMnx candidate graph are separate graph builds and are not "
            "assumed identical."
        )
    else:
        mobility_note = (
            "Ground-truth movement comes from the optional Microsoft GeoLife "
            "validation dataset."
        )
        graph_note = "This optional validation mode does not execute SUMO."
    title = f"""
    <style>
      .leaflet-container {{ background:#f7f6f1 !important; }}
      .leaflet-control-layers {{ margin-top:8px !important; max-height:62vh;
                                 overflow-y:auto; font:12px sans-serif; }}
      @media (max-width:760px) {{
        .sota-demo-note {{ max-width:calc(100vw - 125px) !important;
                           font-size:11px !important; }}
      }}
    </style>
    <div class="sota-demo-note"
         style="position:fixed;top:10px;left:58px;z-index:9999;max-width:430px;
                background:rgba(255,255,255,.96);border:1px solid #8f261f;
                border-radius:4px;padding:7px 10px;font:12px/1.3 sans-serif;
                color:#222;box-shadow:0 1px 5px rgba(0,0,0,.22);">
      <b>DEMO — paper-inspired prototypes, not official reproductions</b><br>
      {mobility_note}<br>
      The road map is embedded from the pinned local graph; online tiles are optional.<br>
      Geometry is for inspection only; output contracts differ. {graph_note}<br>
      Road/map data © OpenStreetMap contributors (ODbL).
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(title))
    folium.LayerControl(collapsed=False).add_to(map_obj)
    map_obj.fit_bounds(bounds, padding=(8, 8))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    map_obj.save(path)
    # Folium's template emits whitespace-only line suffixes.  Normalising them
    # keeps the committed standalone artifact stable under ``git diff --check``.
    with open(path, "r", encoding="utf-8") as handle:
        rendered = handle.read()
    rendered = "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def _write_preview(
    path: str,
    record: Mapping,
    runs: Sequence[tuple[ProtectedRun, str, float]],
    rn: RoadNetwork,
    *,
    mobility_source: str,
) -> None:
    """Write an offline 2×2 PNG comparison over the same embedded roads."""

    import matplotlib

    # The demo runs from a terminal and may have no GUI session.  Selecting a
    # non-interactive backend before importing pyplot keeps PNG generation
    # deterministic on macOS, CI, and headless Linux.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    bounds = _display_bounds(rn, record, runs)
    roads = _embedded_road_geojson(rn, bounds)
    truth_label = _ground_truth_label(mobility_source)

    def projected(lines):
        projected_lines = []
        for coordinates in lines:
            values = np.asarray(coordinates, dtype=float)
            xs, ys = rn.proj.to_xy(values[:, 1], values[:, 0])
            projected_lines.append(np.column_stack([xs, ys]))
        return projected_lines

    minor_roads = projected(roads["minor"]["geometry"]["coordinates"])
    major_roads = projected(roads["major"]["geometry"]["coordinates"])
    truth_xy = np.asarray(
        [rn.point_xy(float(lat), float(lon)) for lat, lon in record["points"]]
    )
    (south, west), (north, east) = bounds
    west_x, south_y = rn.point_xy(south, west)
    east_x, north_y = rn.point_xy(north, east)

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    figure.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.91,
        bottom=0.105,
        hspace=0.16,
        wspace=0.08,
    )
    palette = ("#1565c0", "#ef6c00", "#7b1fa2", "#00897b")
    for axis, colour, (run, _source, _runtime) in zip(axes.flat, palette, runs):
        axis.set_facecolor("#f7f6f1")
        axis.add_collection(
            LineCollection(minor_roads, colors="#d5d2cc", linewidths=0.35, zorder=1)
        )
        axis.add_collection(
            LineCollection(major_roads, colors="#aaa59c", linewidths=0.8, zorder=2)
        )
        axis.plot(
            truth_xy[:, 0],
            truth_xy[:, 1],
            color="#171717",
            linewidth=2.5,
            zorder=4,
        )
        axis.scatter(
            truth_xy[0, 0], truth_xy[0, 1], s=42, c="#2e7d32",
            edgecolors="#171717", linewidths=0.8, zorder=6,
        )
        axis.scatter(
            truth_xy[-1, 0], truth_xy[-1, 1], s=42, c="#c62828",
            edgecolors="#171717", linewidths=0.8, zorder=6,
        )

        for track in _run_tracks(run).values():
            track_xy = np.asarray([rn.point_xy(lat, lon) for lat, lon in track])
            if len(track_xy) < 2:
                continue
            axis.plot(
                track_xy[:, 0],
                track_xy[:, 1],
                color=colour,
                linewidth=1.6,
                alpha=0.78,
                zorder=5,
            )
            axis.scatter(
                [track_xy[0, 0], track_xy[-1, 0]],
                [track_xy[0, 1], track_xy[-1, 1]],
                s=10,
                c=colour,
                zorder=6,
            )
        if run.transcript.output_kind is OutputKind.REAL_PLUS_DUMMIES:
            public_points = np.asarray(
                [
                    rn.point_xy(candidate.lat, candidate.lon)
                    for event in run.transcript.events
                    for candidate in event.candidates
                ]
            )
            axis.scatter(
                public_points[:, 0], public_points[:, 1], s=7,
                c=colour, alpha=0.58, zorder=5,
            )

        axis.set_title(
            _MAP_LABELS.get(run.transcript.mechanism, run.transcript.mechanism),
            fontsize=11,
            fontweight="bold",
        )
        axis.set_xlim(west_x, east_x)
        axis.set_ylim(south_y, north_y)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#d2cec7")

    figure.suptitle(
        f"{truth_label.removesuffix(' ground truth')} trajectory-protection demo "
        "over the embedded local road network",
        fontsize=15,
        fontweight="bold",
    )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#171717",
                lw=2.5,
                label=f"{truth_label} (evaluator only)",
            ),
            Line2D([0], [0], color="#666666", lw=1.6, label="Method public output"),
            Line2D([0], [0], color="#aaa59c", lw=1.0, label="Pinned local OSM road graph"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.045),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.text(
        0.5,
        0.012,
        "Paper-inspired prototypes, not official reproductions. Candidate tracks "
        "use opaque public IDs; no candidate is marked as real. "
        "Road/map data © OpenStreetMap contributors (ODbL).",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    figure.savefig(path, dpi=160, facecolor="#ffffff", bbox_inches="tight")
    plt.close(figure)


def run(args) -> dict:
    run_context = begin_run()
    print("Loading the pinned Beijing road graph...", flush=True)
    rn = RoadNetwork.from_pickle(GRAPH_PKL)
    assert_graph_matches_manifest(rn)
    print(f"  {len(rn)} road vertices (manifest verified)", flush=True)

    records, mobility = _load_records(args)
    print(
        f"Loaded {len(records)} {mobility['label']} "
        f"record(s), {sum(len(record['points']) for record in records)} points total.",
        flush=True,
    )

    result_rows = []
    first_runs = None
    for record in records:
        runs = _run_models(
            rn,
            record["points"],
            record["times"],
            record["record_id"],
            epsilon=args.epsilon,
            k=args.k,
            seed=args.seed,
        )
        if first_runs is None:
            first_runs = runs
        for protected, source_method, runtime_ms in runs:
            evaluator = protected.to_evaluator_dict()
            row = {
                "record_id": record["record_id"],
                "mechanism": protected.transcript.mechanism,
                "source_method": source_method,
                "demo_only": True,
                "output_kind": protected.transcript.output_kind.value,
                "runtime_ms": _round_value(runtime_ms),
                "metrics": {
                    key: _round_value(value)
                    for key, value in _diagnostics(
                        protected, rn, args.qos_radius_m
                    ).items()
                },
                # The split is deliberate: attack code should receive only the
                # first object.  Truth is serialized solely for offline scoring.
                "attacker_view": evaluator["public"],
                "evaluator_truth": evaluator["truth"],
            }
            result_rows.append(row)

    summaries = _aggregate(result_rows)
    if mobility["source"] == "sumo":
        artifact_disclaimer = DISCLAIMER
    else:
        artifact_disclaimer = (
            BASE_DISCLAIMER
            + " GeoLife is an explicitly selected validation source; this run "
            "does not execute SUMO."
        )
    _print_summary(summaries, artifact_disclaimer)

    if mobility["source"] == "sumo":
        dataset = {
            "name": "Eclipse SUMO controlled Beijing passenger smoke scenario",
            "raw_bytes_in_git": False,
            "note": (
                "Movement is generated at run time from the local Beijing OSM "
                "extract. Exact SUMO commands, versions, and input/output SHA-256 "
                "digests are recorded in sumo_runs. This is not a calibrated "
                "Beijing population model."
            ),
        }
    else:
        dataset = {
            "name": "GeoLife v1.3 (explicit optional validation mode)",
            "raw_bytes_in_git": False,
            "note": (
                "Raw GeoLife data are gitignored; selected record IDs are pinned "
                "below. This mode was explicitly requested and is never a fallback."
            ),
        }
    prov = provenance(
        rn,
        [args.epsilon],
        root_seeds=[args.seed],
        quick=args.quick,
        begin=run_context,
        extra={
            "artifact_schema": DEMO_SCHEMA,
            "artifact_status": "integration demo; not a scientific benchmark",
            "dataset": dataset,
            "mobility_source": mobility["source"],
            "mobility_label": mobility["label"],
            "sumo_runs": mobility["sumo_runs"],
            "candidate_graph_compatibility": mobility[
                "candidate_graph_compatibility"
            ],
            "selected_record_ids": [record["record_id"] for record in records],
            "n_points_per_record": [len(record["points"]) for record in records],
            "k": args.k,
            "qos_radius_m": args.qos_radius_m,
            "cross_contract_comparison_allowed": False,
        },
    )
    artifact = {
        "schema": DEMO_SCHEMA,
        "status": "DEMO_ONLY",
        "disclaimer": artifact_disclaimer,
        "provenance": prov,
        # Route, speed, edge, lane, and simulator vehicle ID are privileged
        # ground truth for offline evaluation. They are intentionally absent
        # from every run's attacker_view below.
        "mobility_evaluator_only": mobility["evaluator_only"],
        "summaries": summaries,
        "runs": result_rows,
    }
    visual_artifacts = None
    if not args.no_map and first_runs is not None:
        _write_map(
            args.map_output,
            records[0],
            first_runs,
            rn,
            mobility_source=mobility["source"],
        )
        print(f"Saved standalone map: {args.map_output}")
        _write_preview(
            args.preview_output,
            records[0],
            first_runs,
            rn,
            mobility_source=mobility["source"],
        )
        print(f"Saved static preview: {args.preview_output}")
        visual_artifacts = {
            "evaluator_only": True,
            "contains_ground_truth": True,
            "interactive_map": {
                "path": args.map_output,
                "sha256": sha256_file(args.map_output),
                "embedded_local_roads": True,
                "online_tiles_enabled_by_default": False,
                "byte_reproducible": False,
            },
            "static_preview": {
                "path": args.preview_output,
                "sha256": sha256_file(args.preview_output),
                "embedded_local_roads": True,
                "byte_reproducible_across_platforms": False,
            },
        }
    artifact["visual_artifacts"] = visual_artifacts
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Saved evaluator JSON: {args.output}")
    return artifact


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run three paper-inspired SOTA demos and the thesis prototype."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use at most 8 points (the mobility source remains SUMO by default)",
    )
    parser.add_argument(
        "--mobility-source",
        choices=("sumo", "geolife"),
        default="sumo",
        help=(
            "mobility ground truth: controlled SUMO simulation (default), or "
            "an explicitly requested local GeoLife validation trace"
        ),
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=1,
        help="number of records in GeoLife mode; current SUMO pilot emits exactly 1",
    )
    parser.add_argument("--max-points", type=int, default=12)
    parser.add_argument("--interval-s", type=int, default=20)
    parser.add_argument(
        "--sumo-osm-path",
        default=None,
        help="Beijing .osm or .osm.gz input (default: local pinned extract)",
    )
    parser.add_argument(
        "--sumo-workdir",
        default=str(DEFAULT_WORKDIR),
        help="directory for generated SUMO network, routes, and FCD files",
    )
    parser.add_argument(
        "--sumo-route-seed",
        type=int,
        default=DEFAULT_ROUTE_SEED,
        help="randomTrips.py demand/route seed",
    )
    parser.add_argument(
        "--sumo-simulation-seed",
        type=int,
        default=DEFAULT_SIM_SEED,
        help="SUMO simulation seed",
    )
    parser.add_argument(
        "--sumo-min-trip-distance-m",
        type=float,
        default=2500.0,
        help=(
            "minimum straight-line origin/destination separation requested "
            "from randomTrips.py"
        ),
    )
    parser.add_argument(
        "--sumo-demand-end-s",
        type=float,
        default=300.0,
        help="last time at which random demand may depart",
    )
    parser.add_argument(
        "--sumo-simulation-end-s",
        type=float,
        default=900.0,
        help="SUMO simulation end time",
    )
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--qos-radius-m", type=float, default=200.0)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--map-output", default=DEFAULT_MAP_OUTPUT)
    parser.add_argument(
        "--preview-output",
        default=DEFAULT_PREVIEW_OUTPUT,
        help="offline PNG preview written whenever map generation is enabled",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="disable both the interactive HTML map and static PNG preview",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.n_trajectories = 1
        args.max_points = min(args.max_points, 8)
    if args.n_trajectories < 1:
        parser.error("--n-trajectories must be at least 1")
    if args.mobility_source == "sumo" and args.n_trajectories != 1:
        parser.error(
            "the current SUMO pilot emits exactly one selected trajectory; "
            "use --n-trajectories 1"
        )
    if args.max_points < 2:
        parser.error("--max-points must be at least 2")
    if args.interval_s < 1:
        parser.error("--interval-s must be at least 1")
    if args.epsilon <= 0:
        parser.error("--epsilon must be positive")
    if args.k < 2:
        parser.error("--k must be at least 2")
    if args.qos_radius_m <= 0:
        parser.error("--qos-radius-m must be positive")
    if args.sumo_route_seed < 0:
        parser.error("--sumo-route-seed must be non-negative")
    if args.sumo_simulation_seed < 0:
        parser.error("--sumo-simulation-seed must be non-negative")
    if args.sumo_min_trip_distance_m < 0:
        parser.error("--sumo-min-trip-distance-m must be non-negative")
    if args.sumo_demand_end_s <= 0:
        parser.error("--sumo-demand-end-s must be positive")
    if args.sumo_simulation_end_s < args.sumo_demand_end_s:
        parser.error(
            "--sumo-simulation-end-s must be at least --sumo-demand-end-s"
        )
    return args


if __name__ == "__main__":
    run(parse_args())
