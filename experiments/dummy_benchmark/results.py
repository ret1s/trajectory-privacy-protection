"""Metric calculation, aggregation, and console reporting."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable, Mapping, Sequence

import numpy as np

from core.demo_protocol import OutputKind, ProtectedRun, TrajectoryPoint
from core.road_network import RoadNetwork
from evaluation import metrics as legacy_metrics

from .constants import DISCLAIMER


def latlon(points: Iterable[TrajectoryPoint]) -> list[tuple[float, float]]:
    return [(point.lat, point.lon) for point in points]


def event_candidate_points(run: ProtectedRun):
    return [
        [(candidate.lat, candidate.lon) for candidate in event.candidates]
        for event in run.transcript.events
    ]


def public_tracks(run: ProtectedRun) -> Mapping[str, list[tuple[float, float]]]:
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


def set_diagnostics(
    run: ProtectedRun, rn: RoadNetwork
) -> dict[str, float | None]:
    real = run.truth.real_trajectory
    event_points = event_candidate_points(run)
    nearest, centroid_error, spread = [], [], []
    dummy_distances = []
    true_hits = []
    vertex_fingerprint_hits = []

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
            match = [
                i
                for i, candidate in enumerate(candidates_public)
                if candidate.candidate_id == real_id
            ]
            true_hits.append(float(len(match) == 1))
            vertex_offsets = np.asarray(
                [rn.nearest(lat, lon)[1] for lat, lon in candidates], dtype=float
            )
            maximum = float(np.max(vertex_offsets))
            fingerprint_ties = np.flatnonzero(
                np.isclose(vertex_offsets, maximum, rtol=0.0, atol=1e-3)
            )
            vertex_fingerprint_hits.append(
                1.0 / len(fingerprint_ties)
                if len(match) == 1 and match[0] in fingerprint_ties
                else 0.0
            )
            dummy_distances.extend(
                distance
                for i, distance in enumerate(distances.tolist())
                if i not in match
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
        "on_road_rate_all_outputs": float(
            legacy_metrics.on_road_rate(all_public, rn)
        ),
        "stable_public_tracks": float(len(public_tracks(run))),
    }
    if true_hits:
        result["real_member_inclusion_rate"] = float(np.mean(true_hits))
        result["uniform_guess_success_baseline"] = float(
            np.mean([1.0 / count for count in counts])
        )
        result["vertex_offset_fingerprint_attack_success_rate"] = float(
            np.mean(vertex_fingerprint_hits)
        )
    else:
        result["real_member_inclusion_rate"] = None
        result["uniform_guess_success_baseline"] = None

    tracks = public_tracks(run)
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
                [
                    legacy_metrics.dtw(latlon(real), track, rn.proj)
                    for track in tracks.values()
                ]
            )
        )
    else:
        result["mean_track_speed_violation_rate"] = None
        result["mean_track_dtw_m"] = None
    return result


def replacement_diagnostics(
    run: ProtectedRun,
    rn: RoadNetwork,
    qos_radius_m: float,
) -> dict[str, float]:
    real = latlon(run.truth.real_trajectory)
    released = [
        (event.candidates[0].lat, event.candidates[0].lon)
        for event in run.transcript.events
    ]
    if len(real) != len(released):
        raise ValueError("replacement benchmark metrics require aligned trajectories")
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


def diagnostics(run: ProtectedRun, rn: RoadNetwork, qos_radius_m: float):
    if run.transcript.output_kind is OutputKind.REPLACEMENT_TRAJECTORY:
        return replacement_diagnostics(run, rn, qos_radius_m)
    return set_diagnostics(run, rn)


def round_value(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)
    return value


def aggregate(rows: Sequence[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["mechanism"],
                row["output_kind"],
                row["source_method"],
                row["implementation"]["implementation_level"],
            )
        ].append(row)

    summaries = []
    for (
        mechanism,
        output_kind,
        source_method,
        implementation_level,
    ), group in grouped.items():
        keys = sorted({key for row in group for key in row["metrics"]})
        aggregated = {}
        for key in keys:
            values = [row["metrics"].get(key) for row in group]
            numeric = [float(value) for value in values if value is not None]
            aggregated[key] = round_value(np.mean(numeric)) if numeric else None
        aggregated["runtime_ms"] = round_value(
            np.mean([row["runtime_ms"] for row in group])
        )
        for runtime_key in (
            "setup_runtime_ms",
            "inference_runtime_ms",
            "end_to_end_runtime_ms",
        ):
            aggregated[runtime_key] = round_value(
                np.mean(
                    [row["runtime_breakdown_ms"][runtime_key] for row in group]
                )
            )
        summaries.append(
            {
                "mechanism": mechanism,
                "output_kind": output_kind,
                "source_method": source_method,
                "implementation_level": implementation_level,
                "reportable_as_reproduced_sota": group[0]["implementation"][
                    "reportable_as_reproduced_sota"
                ],
                "records": len(group),
                "metrics": aggregated,
            }
        )
    return summaries


def format_number(value, decimals=1):
    if value is None:
        return "n/a"
    return f"{float(value):.{decimals}f}"


def print_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    if not rows:
        return
    widths = [
        max(len(str(header)), *(len(str(row[i])) for row in rows))
        for i, header in enumerate(headers)
    ]
    print(
        "  "
        + "  ".join(
            str(header).ljust(width) for header, width in zip(headers, widths)
        )
    )
    print("  " + "  ".join("-" * width for width in widths))
    for row in rows:
        print(
            "  "
            + "  ".join(
                str(value).ljust(width) for value, width in zip(row, widths)
            )
        )


def print_summary(
    summaries: Sequence[dict], disclaimer: str = DISCLAIMER
) -> None:
    print("\n" + "=" * 78)
    print(disclaimer)
    print("=" * 78)

    replacement = [
        summary
        for summary in summaries
        if summary["output_kind"] == OutputKind.REPLACEMENT_TRAJECTORY.value
    ]
    print("\nTrack A — replacement trajectory (geometry/utility diagnostics)")
    print_table(
        ("method", "disp m", "p95 m", "QoS", "on-road", "speed viol", "ms"),
        [
            (
                summary["mechanism"],
                format_number(summary["metrics"].get("mean_displacement_m")),
                format_number(summary["metrics"].get("p95_displacement_m")),
                format_number(summary["metrics"].get("qos_satisfaction"), 3),
                format_number(summary["metrics"].get("on_road_rate"), 3),
                format_number(summary["metrics"].get("speed_violation_rate"), 3),
                format_number(summary["metrics"].get("runtime_ms")),
            )
            for summary in replacement
        ],
    )

    real_plus = [
        summary
        for summary in summaries
        if summary["output_kind"] == OutputKind.REAL_PLUS_DUMMIES.value
    ]
    print("\nTrack B — real + K-1 dummies (set diagnostics; no calibrated attacker yet)")
    print_table(
        ("method", "K", "real in set", "spread m", "dummy dist m", "road", "ms"),
        [
            (
                summary["mechanism"],
                format_number(
                    summary["metrics"].get("mean_candidates_per_event"), 1
                ),
                format_number(
                    summary["metrics"].get("real_member_inclusion_rate"), 3
                ),
                format_number(summary["metrics"].get("mean_candidate_spread_m")),
                format_number(summary["metrics"].get("mean_dummy_distance_m")),
                format_number(
                    summary["metrics"].get("on_road_rate_all_outputs"), 3
                ),
                format_number(summary["metrics"].get("runtime_ms")),
            )
            for summary in real_plus
        ],
    )

    dummy_only = [
        summary
        for summary in summaries
        if summary["output_kind"] == OutputKind.DUMMY_ONLY.value
    ]
    print("\nTrack C — thesis dummy-only batch (diagnostics; truth is not a public member)")
    print_table(
        ("method", "K", "nearest m", "centroid err m", "track DTW m", "road", "ms"),
        [
            (
                summary["mechanism"],
                format_number(
                    summary["metrics"].get("mean_candidates_per_event"), 1
                ),
                format_number(
                    summary["metrics"].get("mean_nearest_output_distance_m")
                ),
                format_number(
                    summary["metrics"].get("mean_centroid_reconstruction_error_m")
                ),
                format_number(summary["metrics"].get("mean_track_dtw_m")),
                format_number(
                    summary["metrics"].get("on_road_rate_all_outputs"), 3
                ),
                format_number(summary["metrics"].get("runtime_ms")),
            )
            for summary in dummy_only
        ],
    )
    print(
        "\nDo not compare the numeric columns across tracks: their public outputs and "
        "privacy questions differ.\n"
    )
