"""Dependency-light engines for source-mapped paper adaptations.

These engines provide stable, seeded implementations of three different
dummy-generation output contracts.  They are intentionally separated from the
method cards in :mod:`benchmark.methods`: executable does not imply faithful.
The learned models, original data pipelines, and mechanism-aware attackers
listed below are still required before any reproduced-SOTA claim is allowed.

``TransProtectEngine``
    One road-vertex pseudolocation per real point.  A hand-written score favors
    candidates that are useful, reachable from the previous release, and (when
    labels are supplied) context-compatible.  It does *not* implement the GCN,
    transformer, traffic model, or VehiTrack attacker from TransProtect
    (Yadav et al., ACM SIGSPATIAL 2024).

``AnotherMeEngine``
    A whole replacement trajectory made by applying one consistent local
    translation/rotation/scale and snapping the result to the local road graph.
    It does *not* implement AnotherMe's virtual-user construction, POI mapping,
    learned mobility patterns, or mobile system (Li et al., IEEE TDSC 2024).

``SemanticDummyEngine``
    A real-plus-``K-1`` candidate set at every time step.  Dummies receive a
    simple temporal/reachability and optional semantic-category score.  It does
    *not* implement the LSTM/attention model or the exact candidate filters in
    Liu, Peng, and Zhou's semantic-correlation scheme (2026).

None of these adaptations has a Geo-Indistinguishability,
differential-privacy, or k-anonymity theorem.  Results must be labelled as
paper adaptations and must not be reported as reproductions of the source
papers.  All geographic computation is local: no map or routing web API is
called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Optional, Sequence, Tuple

import networkx as nx
import numpy as np


LatLon = Tuple[float, float]


@dataclass(frozen=True)
class CandidateSetRelease:
    """One public candidate set plus evaluator-only truth for evaluation.

    ``candidates`` is the attacker/LSP-visible value.  ``real_index`` and the
    association with ``candidate_vertex_indices`` are ground truth and MUST be
    removed before constructing an attacker-visible transcript.
    """

    candidates: Tuple[LatLon, ...]
    real_index: int
    candidate_vertex_indices: Tuple[int, ...]
    candidate_ids: Tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def public_candidates(self) -> Tuple[LatLon, ...]:
        """The only portion safe to expose to the benchmark attacker/LSP."""

        return self.candidates


def _require_positive(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive number")
    return result


def _normalise_points(points: Sequence[Any]) -> list[LatLon]:
    """Accept ``(lat, lon)`` values or objects exposing ``lat`` and ``lon``."""

    result: list[LatLon] = []
    for i, point in enumerate(points):
        if hasattr(point, "lat") and hasattr(point, "lon"):
            lat, lon = point.lat, point.lon
        else:
            try:
                lat, lon = point[0], point[1]
            except (TypeError, IndexError, KeyError) as exc:
                raise ValueError(f"point {i} must contain latitude and longitude") from exc
        lat, lon = float(lat), float(lon)
        if not math.isfinite(lat) or not math.isfinite(lon):
            raise ValueError(f"point {i} contains a non-finite coordinate")
        result.append((lat, lon))
    return result


def _aligned(values: Optional[Sequence[Any]], n: int, name: str) -> list[Any]:
    if values is None:
        return [None] * n
    result = list(values)
    if len(result) != n:
        raise ValueError(f"{name} must have the same length as points")
    return result


def _protocol_trajectory(real_trajectory: Sequence[Any]):
    """Validate and unpack ``demo_protocol.TrajectoryPoint`` values lazily."""

    from core.demo_protocol import TrajectoryPoint

    real = tuple(real_trajectory)
    if not real or not all(isinstance(point, TrajectoryPoint) for point in real):
        raise TypeError("protect_run expects a non-empty TrajectoryPoint sequence")
    points = [(point.lat, point.lon) for point in real]
    times = [point.timestamp_s for point in real]
    return real, points, times


def _elapsed_seconds(current: Any, previous: Any, default: float = 60.0) -> float:
    if current is None or previous is None:
        return default
    try:
        delta = current - previous
        if hasattr(delta, "total_seconds"):
            seconds = float(delta.total_seconds())
        elif isinstance(delta, np.timedelta64):
            seconds = float(delta / np.timedelta64(1, "s"))
        else:
            seconds = float(delta)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1.0, seconds) if math.isfinite(seconds) else default


def _nearest_indices(road_network: Any, xy: Sequence[float], k: int) -> np.ndarray:
    k = max(1, min(int(k), len(road_network)))
    _, idxs = road_network.tree.query(np.asarray(xy, dtype=float), k=k)
    idxs = np.atleast_1d(idxs).astype(int)
    # A graph can contain coincident OSM vertices; indices, rather than
    # coordinates, remain the stable candidate identity.
    return np.asarray(list(dict.fromkeys(idxs.tolist())), dtype=int)


def _candidate_pool(
    road_network: Any,
    real_xy: Sequence[float],
    candidate_k: int,
    radius_m: float,
) -> np.ndarray:
    nearby = np.asarray(
        road_network.tree.query_ball_point(np.asarray(real_xy), radius_m), dtype=int
    )
    if nearby.size:
        distances = np.linalg.norm(road_network.xy[nearby] - np.asarray(real_xy), axis=1)
        nearby = nearby[np.argsort(distances)[:candidate_k]]
    if nearby.size < min(2, len(road_network)):
        nearby = _nearest_indices(road_network, real_xy, candidate_k)
    return nearby


def _sample_index(rng: np.random.Generator, idxs: np.ndarray, logits: np.ndarray) -> int:
    """Sample a finite categorical distribution by Gumbel-max."""

    if len(idxs) == 0:
        raise ValueError("cannot sample from an empty candidate set")
    finite = np.isfinite(logits)
    if not finite.any():
        logits = np.zeros_like(logits, dtype=float)
    else:
        floor = float(np.min(logits[finite])) - 100.0
        logits = np.where(finite, logits, floor)
    return int(idxs[int(np.argmax(logits + rng.gumbel(size=len(idxs))))])


def _category_at(categories: Any, idx: int) -> Any:
    if categories is None:
        return None
    if isinstance(categories, Mapping):
        return categories.get(int(idx))
    try:
        return categories[int(idx)]
    except (IndexError, KeyError, TypeError):
        return None


def _network_distances(
    road_network: Any,
    source_idx: int,
    candidates: np.ndarray,
    cutoff: Optional[float] = None,
) -> np.ndarray:
    """Road-path distances, with projected distance as a graph-less fallback.

    Real OSM graphs have edge ``length`` attributes.  Tiny smoke-test graphs or
    disconnected extracts may not; falling back keeps the adaptation runnable while
    the returned values remain explicitly heuristic rather than evidence of
    traffic-realistic reachability.
    """

    source_xy = np.asarray(road_network.xy[int(source_idx)])
    euclidean = np.linalg.norm(road_network.xy[candidates] - source_xy, axis=1)
    graph = road_network.graph
    if graph.number_of_edges() == 0:
        return euclidean
    source_node = road_network.node_ids[int(source_idx)].item()
    try:
        lengths = nx.single_source_dijkstra_path_length(
            graph, source_node, cutoff=cutoff, weight="length"
        )
    except (nx.NetworkXError, ValueError, TypeError):
        return euclidean

    values = np.full(len(candidates), np.inf, dtype=float)
    for pos, idx in enumerate(candidates):
        node = road_network.node_ids[int(idx)].item()
        if node in lengths:
            values[pos] = float(lengths[node])
    if not np.isfinite(values).any():
        return euclidean
    # Disconnected/directionally unreachable candidates should lose to a
    # reachable one, but retain a finite score so a smoke run never crashes.
    finite_max = max(float(np.max(values[np.isfinite(values)])), 1.0)
    missing = np.maximum(euclidean, finite_max) + finite_max
    return np.where(np.isfinite(values), values, missing)


class TransProtectEngine:
    """Non-learned local TransProtect adaptation engine.

    The source method ranks synthetic road candidates using learned road/traffic
    representations and integrates the chosen set with a location mechanism.
    This local version implements filtering candidates by
    utility, road reachability, and optional public context.  It has no formal
    privacy theorem and is not a substitute for the authors' implementation.
    """

    name = "transprotect_engine"
    source_method = "TransProtect (Yadav et al., ACM SIGSPATIAL 2024)"
    implementation_level = "paper_adaptation"

    def __init__(
        self,
        road_network: Any,
        candidate_k: int = 64,
        candidate_radius_m: float = 1_500.0,
        utility_scale_m: float = 350.0,
        v_max: float = 25.0,
        slack_m: float = 100.0,
        reachability_weight: float = 3.0,
        context_weight: float = 2.0,
        exclude_nearest: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if len(road_network) == 0:
            raise ValueError("road_network must contain at least one vertex")
        if int(candidate_k) < 2 and len(road_network) > 1:
            raise ValueError("candidate_k must be at least 2")
        self.rn = road_network
        self.candidate_k = min(int(candidate_k), len(road_network))
        self.candidate_radius_m = _require_positive(candidate_radius_m, "candidate_radius_m")
        self.utility_scale_m = _require_positive(utility_scale_m, "utility_scale_m")
        self.v_max = _require_positive(v_max, "v_max")
        self.slack_m = max(0.0, float(slack_m))
        self.reachability_weight = max(0.0, float(reachability_weight))
        self.context_weight = max(0.0, float(context_weight))
        self.exclude_nearest = bool(exclude_nearest)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        self._previous_choice: Optional[int] = None
        self._previous_t: Any = None

    def protect_point(
        self,
        lat: float,
        lon: float,
        t: Any = None,
        desired_category: Any = None,
        vertex_categories: Any = None,
    ) -> LatLon:
        real_xy = np.asarray(self.rn.point_xy(lat, lon))
        candidates = _candidate_pool(
            self.rn, real_xy, self.candidate_k, self.candidate_radius_m
        )
        nearest = int(self.rn.tree.query(real_xy)[1])
        if self.exclude_nearest and len(candidates) > 1:
            candidates = candidates[candidates != nearest]
        if not len(candidates):
            candidates = np.asarray([nearest], dtype=int)

        utility_distance = np.linalg.norm(self.rn.xy[candidates] - real_xy, axis=1)
        logits = -utility_distance / self.utility_scale_m

        if self._previous_choice is not None:
            dt = _elapsed_seconds(t, self._previous_t)
            allowance = self.v_max * dt + self.slack_m
            route_distance = _network_distances(
                self.rn,
                self._previous_choice,
                candidates,
                cutoff=allowance + 2.0 * self.candidate_radius_m,
            )
            excess = np.maximum(0.0, route_distance - allowance)
            logits -= self.reachability_weight * excess / self.utility_scale_m

        if desired_category is not None and vertex_categories is not None:
            mismatch = np.asarray(
                [
                    0.0 if _category_at(vertex_categories, int(idx)) == desired_category else 1.0
                    for idx in candidates
                ]
            )
            logits -= self.context_weight * mismatch

        choice = _sample_index(self.rng, candidates, logits)
        self._previous_choice = choice
        self._previous_t = t
        return self.rn.latlon(choice)

    def perturb(self, lat: float, lon: float, t: Any = None) -> LatLon:
        """Compatibility alias for the legacy one-point benchmark contract."""

        return self.protect_point(lat, lon, t=t)

    def protect_trajectory(
        self,
        points: Sequence[Any],
        times: Optional[Sequence[Any]] = None,
        poi_categories: Optional[Sequence[Any]] = None,
        vertex_categories: Any = None,
    ) -> list[LatLon]:
        points_ll = _normalise_points(points)
        aligned_times = _aligned(times, len(points_ll), "times")
        categories = _aligned(poi_categories, len(points_ll), "poi_categories")
        self.reset()
        return [
            self.protect_point(
                lat,
                lon,
                t=t,
                desired_category=category,
                vertex_categories=vertex_categories,
            )
            for (lat, lon), t, category in zip(points_ll, aligned_times, categories)
        ]

    def protect_run(
        self,
        real_trajectory: Sequence[Any],
        poi_categories: Optional[Sequence[Any]] = None,
        vertex_categories: Any = None,
    ):
        """Adapt the raw API to :mod:`core.demo_protocol`."""

        from core.demo_protocol import TrajectoryPoint, make_replacement_run

        real, points, times = _protocol_trajectory(real_trajectory)
        output = self.protect_trajectory(
            points,
            times,
            poi_categories=poi_categories,
            vertex_categories=vertex_categories,
        )
        replacement = tuple(
            TrajectoryPoint(t, lat, lon) for t, (lat, lon) in zip(times, output)
        )
        return make_replacement_run(
            self.name,
            real,
            replacement,
            public_parameters={
                "implementation_level": self.implementation_level,
                "implementation_origin": "benchmark.engines.TransProtectEngine",
                "source_method": self.source_method,
            },
        )


class AnotherMeEngine:
    """Paper-inspired whole-trajectory replacement using only the local graph.

    One random-but-consistent translation, rotation, and small scale change is
    applied to the complete input trace.  Each transformed point is then
    snapped to a road vertex with a continuity score.  This preserves more of
    the trace's shape than independent noise while relocating it to a virtual
    area; it does not reproduce AnotherMe's virtual-user or POI workflow.
    """

    name = "anotherme_engine"
    source_method = "AnotherMe (Li et al., IEEE TDSC 2024)"
    implementation_level = "paper_adaptation"

    def __init__(
        self,
        road_network: Any,
        anchor_min_m: float = 700.0,
        anchor_max_m: float = 3_000.0,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        rotation_max_deg: float = 45.0,
        snap_k: int = 12,
        continuity_weight: float = 1.5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if len(road_network) == 0:
            raise ValueError("road_network must contain at least one vertex")
        self.rn = road_network
        self.anchor_min_m = max(0.0, float(anchor_min_m))
        self.anchor_max_m = _require_positive(anchor_max_m, "anchor_max_m")
        if self.anchor_max_m < self.anchor_min_m:
            raise ValueError("anchor_max_m must be >= anchor_min_m")
        lo, hi = map(float, scale_range)
        if not (0.0 < lo <= hi and math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError("scale_range must contain finite positive bounds")
        self.scale_range = (lo, hi)
        self.rotation_max_deg = max(0.0, float(rotation_max_deg))
        self.snap_k = max(1, min(int(snap_k), len(road_network)))
        self.continuity_weight = max(0.0, float(continuity_weight))
        self.rng = rng if rng is not None else np.random.default_rng()
        self.last_transform: Mapping[str, float] = {}

    def reset(self) -> None:
        self.last_transform = {}

    def _choose_anchor(self, origin_xy: np.ndarray) -> int:
        distances = np.linalg.norm(self.rn.xy - origin_xy, axis=1)
        pool = np.flatnonzero(
            (distances >= self.anchor_min_m) & (distances <= self.anchor_max_m)
        )
        nearest = int(np.argmin(distances))
        pool = pool[pool != nearest]
        if not len(pool):
            pool = np.argsort(distances)[::-1]
            pool = pool[pool != nearest][: min(64, max(1, len(self.rn) - 1))]
        if not len(pool):
            return nearest
        target = 0.5 * (self.anchor_min_m + self.anchor_max_m)
        logits = -np.abs(distances[pool] - target) / max(target, 1.0)
        return _sample_index(self.rng, np.asarray(pool, dtype=int), logits)

    def protect_trajectory(
        self,
        points: Sequence[Any],
        times: Optional[Sequence[Any]] = None,
    ) -> list[LatLon]:
        points_ll = _normalise_points(points)
        _aligned(times, len(points_ll), "times")  # validate; timestamps are retained by adapter
        self.reset()
        if not points_ll:
            return []

        real_xy = np.asarray([self.rn.point_xy(lat, lon) for lat, lon in points_ll])
        anchor_idx = self._choose_anchor(real_xy[0])
        anchor_xy = np.asarray(self.rn.xy[anchor_idx])
        angle = math.radians(
            self.rng.uniform(-self.rotation_max_deg, self.rotation_max_deg)
        )
        scale = float(self.rng.uniform(*self.scale_range))
        rotation = np.asarray(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        transformed = anchor_xy + (real_xy - real_xy[0]) @ rotation.T * scale

        output_indices: list[int] = []
        previous_idx: Optional[int] = None
        for i, target_xy in enumerate(transformed):
            candidates = _nearest_indices(self.rn, target_xy, self.snap_k)
            target_error = np.linalg.norm(self.rn.xy[candidates] - target_xy, axis=1)
            logits = -target_error / max(self.anchor_min_m * 0.15, 30.0)
            if previous_idx is not None:
                expected_step = float(np.linalg.norm(real_xy[i] - real_xy[i - 1]) * scale)
                route_distance = _network_distances(self.rn, previous_idx, candidates)
                consistency = np.abs(route_distance - expected_step)
                logits -= self.continuity_weight * consistency / max(expected_step, 50.0)
            choice = _sample_index(self.rng, candidates, logits)
            output_indices.append(choice)
            previous_idx = choice

        self.last_transform = {
            "anchor_vertex_index": int(anchor_idx),
            "rotation_degrees": math.degrees(angle),
            "scale": scale,
        }
        return [self.rn.latlon(idx) for idx in output_indices]

    def protect_run(self, real_trajectory: Sequence[Any]):
        """Adapt the raw API to :mod:`core.demo_protocol`."""

        from core.demo_protocol import TrajectoryPoint, make_replacement_run

        real, points, times = _protocol_trajectory(real_trajectory)
        output = self.protect_trajectory(points, times)
        replacement = tuple(
            TrajectoryPoint(t, lat, lon) for t, (lat, lon) in zip(times, output)
        )
        return make_replacement_run(
            self.name,
            real,
            replacement,
            public_parameters={
                "implementation_level": self.implementation_level,
                "implementation_origin": "benchmark.engines.AnotherMeEngine",
                "source_method": self.source_method,
            },
        )


class SemanticDummyEngine:
    """Temporal/semantic real-plus-dummies adaptation engine.

    Dummies are selected without replacement around the real point.  The score
    favors a configurable displacement band, compatibility with an optional POI
    category, separation from already selected dummies, and a movement length
    similar to the real trace since the preceding event.  Dummy identities are
    retained internally across time only to compute that temporal score.
    """

    name = "semantic_dummy_engine"
    source_method = (
        "Semantic-correlation dummy paths "
        "(Liu, Peng, and Zhou, Journal of King Saud University C&IS 2026)"
    )
    implementation_level = "paper_adaptation"

    def __init__(
        self,
        road_network: Any,
        k: int = 5,
        candidate_radius_m: float = 2_000.0,
        target_dummy_distance_m: float = 700.0,
        min_separation_m: float = 120.0,
        v_max: float = 25.0,
        slack_m: float = 100.0,
        temporal_weight: float = 2.5,
        semantic_weight: float = 2.0,
        separation_weight: float = 2.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        if int(k) < 2:
            raise ValueError("k must be at least 2 (one real plus at least one dummy)")
        if len(road_network) < int(k):
            raise ValueError("road_network needs at least k distinct vertices")
        self.rn = road_network
        self.k = int(k)
        self.candidate_radius_m = _require_positive(candidate_radius_m, "candidate_radius_m")
        self.target_dummy_distance_m = _require_positive(
            target_dummy_distance_m, "target_dummy_distance_m"
        )
        self.min_separation_m = max(0.0, float(min_separation_m))
        self.v_max = _require_positive(v_max, "v_max")
        self.slack_m = max(0.0, float(slack_m))
        self.temporal_weight = max(0.0, float(temporal_weight))
        self.semantic_weight = max(0.0, float(semantic_weight))
        self.separation_weight = max(0.0, float(separation_weight))
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        self._previous_dummy_indices: Optional[list[int]] = None
        self._previous_real_xy: Optional[np.ndarray] = None
        self._previous_t: Any = None
        # IDs are opaque and randomly assigned to roles once per run.  Their
        # values therefore do not reveal which persistent track is the real one.
        labels = [f"candidate_{i:04d}" for i in range(self.k)]
        assignment = self.rng.permutation(self.k)
        self._role_candidate_ids = [labels[int(i)] for i in assignment]

    def protect_point(
        self,
        lat: float,
        lon: float,
        t: Any = None,
        category: Any = None,
        vertex_categories: Any = None,
    ) -> CandidateSetRelease:
        real = (float(lat), float(lon))
        real_xy = np.asarray(self.rn.point_xy(*real))
        real_vertex = int(self.rn.tree.query(real_xy)[1])
        pool = np.asarray(
            self.rn.tree.query_ball_point(real_xy, self.candidate_radius_m), dtype=int
        )
        pool = pool[pool != real_vertex]
        if len(pool) < self.k - 1:
            pool = np.asarray([i for i in range(len(self.rn)) if i != real_vertex], dtype=int)

        real_step = 0.0
        dt = _elapsed_seconds(t, self._previous_t)
        if self._previous_real_xy is not None:
            real_step = float(np.linalg.norm(real_xy - self._previous_real_xy))

        selected: list[int] = []
        for dummy_slot in range(self.k - 1):
            available = pool[~np.isin(pool, np.asarray(selected, dtype=int))]
            distances = np.linalg.norm(self.rn.xy[available] - real_xy, axis=1)
            logits = -np.abs(distances - self.target_dummy_distance_m) / max(
                self.target_dummy_distance_m, 1.0
            )

            if category is not None and vertex_categories is not None:
                mismatch = np.asarray(
                    [
                        0.0 if _category_at(vertex_categories, int(idx)) == category else 1.0
                        for idx in available
                    ]
                )
                logits -= self.semantic_weight * mismatch

            if selected and self.min_separation_m > 0.0:
                chosen_xy = self.rn.xy[np.asarray(selected, dtype=int)]
                pair_dist = np.min(
                    np.linalg.norm(
                        self.rn.xy[available, None, :] - chosen_xy[None, :, :], axis=2
                    ),
                    axis=1,
                )
                shortfall = np.maximum(0.0, self.min_separation_m - pair_dist)
                logits -= self.separation_weight * shortfall / self.min_separation_m

            if self._previous_dummy_indices is not None:
                previous_idx = self._previous_dummy_indices[dummy_slot]
                route_distance = _network_distances(self.rn, previous_idx, available)
                allowance = self.v_max * dt + self.slack_m
                excess = np.maximum(0.0, route_distance - allowance)
                movement_mismatch = np.abs(route_distance - real_step)
                logits -= self.temporal_weight * (
                    movement_mismatch / max(real_step, 100.0)
                    + excess / max(allowance, 100.0)
                )

            selected.append(_sample_index(self.rng, available, logits))

        unshuffled_points: list[LatLon] = [real] + [self.rn.latlon(i) for i in selected]
        unshuffled_vertices = [real_vertex] + selected
        unshuffled_ids = self._role_candidate_ids
        permutation = self.rng.permutation(self.k)
        candidates = tuple(unshuffled_points[int(i)] for i in permutation)
        vertex_indices = tuple(int(unshuffled_vertices[int(i)]) for i in permutation)
        candidate_ids = tuple(unshuffled_ids[int(i)] for i in permutation)
        real_index = int(np.flatnonzero(permutation == 0)[0])

        matched = None
        if category is not None and vertex_categories is not None:
            matched = sum(
                _category_at(vertex_categories, idx) == category for idx in selected
            )
        self._previous_dummy_indices = selected
        self._previous_real_xy = real_xy
        self._previous_t = t
        return CandidateSetRelease(
            candidates=candidates,
            real_index=real_index,
            candidate_vertex_indices=vertex_indices,
            candidate_ids=candidate_ids,
            metadata={
                "k": self.k,
                "requested_category": None if category is None else str(category),
                "semantic_dummy_matches": matched,
                "implementation_level": self.implementation_level,
            },
        )

    def protect_trajectory(
        self,
        points: Sequence[Any],
        times: Optional[Sequence[Any]] = None,
        poi_categories: Optional[Sequence[Any]] = None,
        vertex_categories: Any = None,
    ) -> list[CandidateSetRelease]:
        points_ll = _normalise_points(points)
        aligned_times = _aligned(times, len(points_ll), "times")
        categories = _aligned(poi_categories, len(points_ll), "poi_categories")
        self.reset()
        return [
            self.protect_point(
                lat,
                lon,
                t=t,
                category=category,
                vertex_categories=vertex_categories,
            )
            for (lat, lon), t, category in zip(points_ll, aligned_times, categories)
        ]

    def protect_run(
        self,
        real_trajectory: Sequence[Any],
        poi_categories: Optional[Sequence[Any]] = None,
        vertex_categories: Any = None,
    ):
        """Build a protocol run without leaking ``real_index`` publicly."""

        from core.demo_protocol import (
            EvaluationTruth,
            OutputKind,
            ProtectedRun,
            PublicCandidate,
            PublicEvent,
            PublicTranscript,
        )

        real, points, times = _protocol_trajectory(real_trajectory)
        releases = self.protect_trajectory(
            points,
            times,
            poi_categories=poi_categories,
            vertex_categories=vertex_categories,
        )
        events = []
        real_ids = []
        for event_idx, (timestamp, release) in enumerate(zip(times, releases)):
            events.append(
                PublicEvent(
                    event_id=f"event_{event_idx:04d}",
                    timestamp_s=timestamp,
                    candidates=tuple(
                        PublicCandidate(candidate_id, lat, lon)
                        for candidate_id, (lat, lon) in zip(
                            release.candidate_ids, release.candidates
                        )
                    ),
                )
            )
            real_ids.append(release.candidate_ids[release.real_index])
        transcript = PublicTranscript(
            mechanism=self.name,
            output_kind=OutputKind.REAL_PLUS_DUMMIES,
            events=tuple(events),
            public_parameters=(
                ("implementation_level", self.implementation_level),
                (
                    "implementation_origin",
                    "benchmark.engines.SemanticDummyEngine",
                ),
                ("k", self.k),
                ("source_method", self.source_method),
            ),
        )
        return ProtectedRun(
            transcript,
            EvaluationTruth(real, tuple(real_ids)),
        )


__all__ = [
    "AnotherMeEngine",
    "CandidateSetRelease",
    "SemanticDummyEngine",
    "TransProtectEngine",
]
