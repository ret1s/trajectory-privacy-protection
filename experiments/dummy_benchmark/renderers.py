"""Interactive and offline visual renderers for benchmark outputs."""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np

from benchmark.contracts import MethodCard
from core.demo_protocol import OutputKind, ProtectedRun
from core.road_network import RoadNetwork

from .constants import MAJOR_HIGHWAYS, MAP_LABELS
from .results import public_tracks


def ground_truth_label(mobility_source: str) -> str:
    if mobility_source == "sumo":
        return "SUMO ground truth"
    if mobility_source == "geolife":
        return "GeoLife ground truth"
    raise ValueError(f"unsupported mobility source: {mobility_source}")


def run_tracks(run: ProtectedRun) -> Mapping[str, list[tuple[float, float]]]:
    """Return public, temporally stable tracks without consulting truth."""

    if run.transcript.output_kind is OutputKind.REPLACEMENT_TRAJECTORY:
        return {
            "released_trajectory": [
                (event.candidates[0].lat, event.candidates[0].lon)
                for event in run.transcript.events
            ]
        }
    return public_tracks(run)


def display_points(
    record: Mapping,
    runs: Sequence[tuple[ProtectedRun, MethodCard | str, object]],
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


def display_bounds(
    rn: RoadNetwork,
    record: Mapping,
    runs: Sequence[tuple[ProtectedRun, MethodCard | str, object]],
    *,
    minimum_padding_m: float = 250.0,
    padding_fraction: float = 0.10,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bounds around truth and every public output, padded in projected metres."""

    points = display_points(record, runs)
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


def edge_coordinates(
    rn: RoadNetwork, u, v, data
) -> list[tuple[float, float]]:
    """Return one graph edge as GeoJSON-order ``(lon, lat)`` coordinates."""

    geometry = data.get("geometry")
    if geometry is not None:
        return [(float(lon), float(lat)) for lon, lat in geometry.coords]
    return [
        (float(rn.graph.nodes[u]["x"]), float(rn.graph.nodes[u]["y"])),
        (float(rn.graph.nodes[v]["x"]), float(rn.graph.nodes[v]["y"])),
    ]


def line_parts(geometry) -> list[list[list[float]]]:
    """Flatten a clipped Shapely geometry into GeoJSON line coordinates."""

    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        coordinates = [[float(x), float(y)] for x, y in geometry.coords]
        return [coordinates] if len(coordinates) >= 2 else []
    if geometry.geom_type in {"MultiLineString", "GeometryCollection"}:
        parts = []
        for child in geometry.geoms:
            parts.extend(line_parts(child))
        return parts
    return []


def embedded_road_geojson(
    rn: RoadNetwork,
    bounds: tuple[tuple[float, float], tuple[float, float]],
) -> dict[str, dict]:
    """Build two compact local MultiLineStrings clipped to display bounds.

    The graph already contains OSM road geometry, so geographic context does
    not require raster-tile requests. Folium's HTML runtime assets may still be
    CDN references; the separately generated PNG is the fully offline view.
    Reciprocal directed edges with identical geometry are de-duplicated before
    serialisation.
    """

    from shapely.geometry import LineString, box

    (south, west), (north, east) = bounds
    clip = box(west, south, east, north)
    lines: dict[str, list[list[list[float]]]] = {"minor": [], "major": []}
    unique_edges: dict[
        tuple[tuple[float, float], ...], tuple[list[tuple[float, float]], bool]
    ] = {}
    for u, v, data in rn.graph.edges(data=True):
        coordinates = edge_coordinates(rn, u, v, data)
        if len(coordinates) < 2:
            continue
        forward = tuple(
            (round(lon, 7), round(lat, 7)) for lon, lat in coordinates
        )
        signature = min(forward, tuple(reversed(forward)))
        highway = data.get("highway", "")
        highway_values = (
            {str(value) for value in highway}
            if isinstance(highway, (list, tuple, set))
            else {str(highway)}
        )
        is_major = bool(highway_values & MAJOR_HIGHWAYS)
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
        lines[road_class].extend(line_parts(line.intersection(clip)))

    return {
        road_class: {
            "type": "Feature",
            "properties": {
                "road_class": road_class,
                "source": "local_pinned_graph",
            },
            "geometry": {
                "type": "MultiLineString",
                "coordinates": coordinates,
            },
        }
        for road_class, coordinates in lines.items()
    }


def write_map(
    path: str,
    record: Mapping,
    runs: Sequence[tuple[ProtectedRun, MethodCard | str, object]],
    rn: RoadNetwork,
    *,
    mobility_source: str,
):
    """Write the interactive evaluator map."""

    import folium

    truth = record["points"]
    truth_label = ground_truth_label(mobility_source)
    bounds = display_bounds(rn, record, runs)
    roads = embedded_road_geojson(rn, bounds)
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

    # This optional raster-tile layer is deliberately disabled. The road
    # geometry comes from the embedded graph below. Folium/Leaflet runtime
    # assets may remain CDN references; the static PNG is the offline fallback.
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
            MAP_LABELS.get(transcript.mechanism, transcript.mechanism),
            show=is_thesis,
        )
        tracks = run_tracks(run)
        for candidate_id, track in tracks.items():
            if len(track) < 2:
                continue
            method_label = MAP_LABELS.get(
                transcript.mechanism, transcript.mechanism
            )
            tooltip = f"{method_label}: {candidate_id}"
            folium.PolyLine(
                track,
                color=colour,
                weight=(
                    3
                    if transcript.output_kind
                    is OutputKind.REPLACEMENT_TRAJECTORY
                    else 2.4
                ),
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

        # Candidate-set methods expose opaque public IDs. Every candidate is
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
            "Mobility and protection candidates use the same generated SUMO "
            "passenger network."
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
        .benchmark-note {{ max-width:calc(100vw - 125px) !important;
                           font-size:11px !important; }}
      }}
    </style>
    <div class="benchmark-note"
         style="position:fixed;top:10px;left:58px;z-index:9999;max-width:430px;
                background:rgba(255,255,255,.96);border:1px solid #8f261f;
                border-radius:4px;padding:7px 10px;font:12px/1.3 sans-serif;
                color:#222;box-shadow:0 1px 5px rgba(0,0,0,.22);">
      <b>Benchmark harness — clean-room adaptations, not reproduced SOTA</b><br>
      {mobility_note}<br>
      Road geometry is embedded; raster tiles are optional. The static PNG is the fully offline view.<br>
      Geometry is for inspection only; output contracts differ. {graph_note}<br>
      Road/map data © OpenStreetMap contributors (ODbL).
    </div>
    """
    map_obj.get_root().html.add_child(folium.Element(title))
    folium.LayerControl(collapsed=False).add_to(map_obj)
    map_obj.fit_bounds(bounds, padding=(8, 8))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    map_obj.save(path)
    # Folium's template emits whitespace-only line suffixes. Normalising them
    # keeps the committed standalone artifact stable under ``git diff --check``.
    with open(path, "r", encoding="utf-8") as handle:
        rendered = handle.read()
    rendered = "\n".join(line.rstrip() for line in rendered.splitlines()) + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(rendered)


def write_preview(
    path: str,
    record: Mapping,
    runs: Sequence[tuple[ProtectedRun, MethodCard | str, object]],
    rn: RoadNetwork,
    *,
    mobility_source: str,
) -> None:
    """Write an offline 2×2 PNG comparison over the same embedded roads."""

    import matplotlib

    # The benchmark runs from a terminal and may have no GUI session. Selecting
    # a non-interactive backend before importing pyplot keeps PNG generation
    # deterministic on macOS, CI, and headless Linux.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    bounds = display_bounds(rn, record, runs)
    roads = embedded_road_geojson(rn, bounds)
    truth_label = ground_truth_label(mobility_source)

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
    for axis, colour, (run, _source, _runtime) in zip(
        axes.flat, palette, runs
    ):
        axis.set_facecolor("#f7f6f1")
        axis.add_collection(
            LineCollection(
                minor_roads, colors="#d5d2cc", linewidths=0.35, zorder=1
            )
        )
        axis.add_collection(
            LineCollection(
                major_roads, colors="#aaa59c", linewidths=0.8, zorder=2
            )
        )
        axis.plot(
            truth_xy[:, 0],
            truth_xy[:, 1],
            color="#171717",
            linewidth=2.5,
            zorder=4,
        )
        axis.scatter(
            truth_xy[0, 0],
            truth_xy[0, 1],
            s=42,
            c="#2e7d32",
            edgecolors="#171717",
            linewidths=0.8,
            zorder=6,
        )
        axis.scatter(
            truth_xy[-1, 0],
            truth_xy[-1, 1],
            s=42,
            c="#c62828",
            edgecolors="#171717",
            linewidths=0.8,
            zorder=6,
        )

        for track in run_tracks(run).values():
            track_xy = np.asarray(
                [rn.point_xy(lat, lon) for lat, lon in track]
            )
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
                public_points[:, 0],
                public_points[:, 1],
                s=7,
                c=colour,
                alpha=0.58,
                zorder=5,
            )

        axis.set_title(
            MAP_LABELS.get(run.transcript.mechanism, run.transcript.mechanism),
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
        f"{truth_label.removesuffix(' ground truth')} trajectory-protection benchmark "
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
            Line2D(
                [0],
                [0],
                color="#666666",
                lw=1.6,
                label="Method public output",
            ),
            Line2D(
                [0],
                [0],
                color="#aaa59c",
                lw=1.0,
                label="Pinned local OSM road graph",
            ),
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
        "Source-mapped adaptations, not official/faithful reproductions. Candidate tracks "
        "use opaque public IDs; no candidate is marked as real. "
        "Road/map data © OpenStreetMap contributors (ODbL).",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    figure.savefig(path, dpi=160, facecolor="#ffffff", bbox_inches="tight")
    plt.close(figure)
