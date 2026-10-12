"""Algorithm engine for the evolving dummy-only thesis architecture.

This module implements the current executable research candidate described in
the September 2026 plan:

1. release a road-network Geo-I anchor trajectory with REM; then
2. generate ``K`` plausible road trajectories using only that released anchor,
   the public road graph, timestamps, and fresh randomness.

Because the second stage never reads the true locations, its visible batch is
post-processing of the anchor transcript.  This preserves the *ideal-kernel*
privacy statement of the anchors (subject to the composition and finite-
precision caveats documented in :mod:`core.mechanisms`).  It does not yet prove
that the generated tracks are indistinguishable under the thesis' final
context-aware attacker; that is exactly what the later experiments must test.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import networkx as nx

from core.mechanisms import RoadExponential


Point = Tuple[float, float]


@dataclass(frozen=True)
class AnchoredDummyBatch:
    """Raw output before it is wrapped as a public transcript.

    ``anchors`` are retained only for evaluator diagnostics.  A caller that
    models the LSP must expose ``trajectories`` and never infer that an anchor is
    a true location: the true trajectory is not a member of the dummy batch.
    """

    anchors: Tuple[Point, ...]
    trajectories: Tuple[Tuple[Point, ...], ...]


def _delta_seconds(current, previous, default_s=60.0):
    if current is None or previous is None:
        return default_s
    delta = current - previous
    if hasattr(delta, "total_seconds"):
        delta = delta.total_seconds()
    try:
        return max(1.0, float(delta))
    except (TypeError, ValueError):
        return default_s


class GeoIAnchoredDummyEngine:
    """Current engine for the proposed Geo-I-anchor + dummy-only design.

    The generator uses persistent public-space offsets around the REM anchor
    and penalises road vertices that would require implausible speed relative
    to the same dummy track's previous public point.  It deliberately does not
    consult the true trajectory after anchors have been generated.

    This is suitable for integration experiments but not a final claim. It has no trained
    population prior, POI semantics, group model, formal trajectory-level
    budget manager, or empirical resistance claim yet.
    """

    name = "geo_i_anchored_dummy_engine"
    source_method = "Thesis candidate: REM anchor followed by dummy-only tracks"
    implementation_level = "thesis_candidate"
    output_kind = "dummy_only"

    def __init__(
        self,
        epsilon,
        road_network,
        *,
        k=3,
        offset_m=180.0,
        candidate_radius_m=220.0,
        v_max=25.0,
        reachability_slack_m=100.0,
        road_constrained=False,
        rng=None,
    ):
        if int(k) < 1:
            raise ValueError("k must be at least 1")
        if offset_m <= 0 or candidate_radius_m <= 0 or v_max <= 0:
            raise ValueError("offset_m, candidate_radius_m, and v_max must be positive")
        self.epsilon = float(epsilon)
        self.rn = road_network
        self.k = int(k)
        self.offset_m = float(offset_m)
        self.candidate_radius_m = float(candidate_radius_m)
        self.v_max = float(v_max)
        self.reachability_slack_m = float(reachability_slack_m)
        self.road_constrained = bool(road_constrained)
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.rng = rng or np.random.default_rng()
        # Independent streams: future anchor draws must not move the random
        # state used for earlier public dummy points.
        seeds = self.rng.integers(0, 2**63, size=2, dtype=np.int64)
        self.anchor_rng = np.random.default_rng(int(seeds[0]))
        self.dummy_rng = np.random.default_rng(int(seeds[1]))
        self._state = None
        self._node_to_index = {node: i for i, node in enumerate(self.rn.node_ids)}

    def _sample_public_candidate(self, target_xy, previous_xy, dt_s):
        idxs = np.asarray(
            self.rn.tree.query_ball_point(target_xy, self.candidate_radius_m),
            dtype=int,
        )
        if self.road_constrained and previous_xy is not None:
            previous_index = int(self.rn.tree.query(previous_xy)[1])
            # Public, directed shortest free-flow travel time. Staying at the
            # previous vertex is the fallback. This is a graph feasibility
            # constraint; acceleration, parking legality and turn-state are
            # not represented by this node graph.
            def travel_time(u, v, attributes):
                edges = attributes.values() if self.rn.graph.is_multigraph() else [attributes]
                return min(float(e.get("length", np.linalg.norm(
                    self.rn.xy[self._node_to_index[u]] - self.rn.xy[self._node_to_index[v]])))
                           / min(self.v_max, max(1e-9, float(e.get("speed", self.v_max)))) for e in edges)
            reachable = nx.single_source_dijkstra_path_length(
                self.rn.graph, self.rn.node_ids[previous_index], cutoff=dt_s,
                weight=travel_time)
            idxs = np.array([i for i in idxs if self.rn.node_ids[i] in reachable], dtype=int)
            if not len(idxs):
                return previous_index
        if len(idxs) == 0:
            _, nearest = self.rn.tree.query(target_xy)
            return int(nearest)

        target_dist = np.linalg.norm(self.rn.xy[idxs] - target_xy, axis=1)
        logits = -target_dist / max(25.0, self.candidate_radius_m / 3.0)
        if previous_xy is not None:
            step_dist = np.linalg.norm(self.rn.xy[idxs] - previous_xy, axis=1)
            reach = self.v_max * dt_s + self.reachability_slack_m
            excess = np.maximum(0.0, step_dist - reach)
            logits -= excess / max(25.0, self.reachability_slack_m)
        return int(idxs[int(np.argmax(logits + self.dummy_rng.gumbel(size=len(idxs))))])

    def protect_trajectory(
        self,
        points: Sequence[Point],
        times: Optional[Sequence] = None,
    ) -> AnchoredDummyBatch:
        if len(points) == 0:
            raise ValueError("points must not be empty")
        if times is not None and len(times) != len(points):
            raise ValueError("times must have the same length as points")

        times = list(times) if times is not None else [i * 60. for i in range(len(points))]
        self.reset()
        anchors, tracks = [], [[] for _ in range(self.k)]
        for (lat, lon), timestamp in zip(points, times):
            anchor, candidates = self.protect_step(lat, lon, timestamp)
            anchors.append(anchor)
            for track, candidate in zip(tracks, candidates):
                track.append(candidate)
        return AnchoredDummyBatch(tuple(anchors), tuple(tuple(t) for t in tracks))

    def reset(self):
        """Start a new session; do not rewind the random streams."""
        self._state = None
        self._anchor_mechanism = RoadExponential(self.epsilon, self.rn, rng=self.anchor_rng)
        self._anchor_mechanism.reset()

    def protect_step(self, lat, lon, timestamp_s):
        """Produce one release from the current point and retained state only.

        The anchor return is diagnostic/private to the local caller. Only the
        candidate tuple is sent to the LSP. Batch replay calls this same method.
        """
        if not hasattr(self, "_anchor_mechanism"):
            self.reset()
        self._validate_time(timestamp_s)
        anchor = self._anchor_mechanism.perturb(lat, lon, t=timestamp_s)
        return anchor, self._generate_step(anchor, timestamp_s)

    def _validate_time(self, timestamp):
        if isinstance(timestamp, (int, float, np.number)) and not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._state is not None and timestamp <= self._state["time"]:
            raise ValueError("timestamps must be strictly increasing")

    def _generate_step(self, anchor, timestamp):
        self._validate_time(timestamp)
        if self._state is None:
            angle = self.dummy_rng.uniform(0., 2. * np.pi)
            self._state = {
                "angles": angle + np.linspace(0., 2. * np.pi, self.k, endpoint=False),
                "radii": self.offset_m * self.dummy_rng.uniform(.75, 1.25, self.k),
                "previous": [None] * self.k, "time": timestamp, "step": 0,
            }
        state = self._state
        dt = _delta_seconds(timestamp, state["time"]) if state["step"] else 60.
        result = []
        for j in range(self.k):
            angle = state["angles"][j] + .12 * np.sin(state["step"] / 3. + j)
            target = np.asarray(self.rn.point_xy(*anchor)) + state["radii"][j] * np.array([np.cos(angle), np.sin(angle)])
            choice = self._sample_public_candidate(target, state["previous"][j], dt)
            result.append(self.rn.latlon(choice))
            state["previous"][j] = self.rn.xy[choice]
        state["time"], state["step"] = timestamp, state["step"] + 1
        return tuple(result)

    def generate_from_anchors(
        self,
        anchors: Sequence[Point],
        times: Optional[Sequence] = None,
    ) -> AnchoredDummyBatch:
        """Generate the visible batch from an already-public anchor trace.

        Keeping this boundary explicit makes the post-processing dependency
        auditable and lets tests verify that the second stage needs no secret
        trajectory input.
        """
        if len(anchors) == 0:
            raise ValueError("anchors must not be empty")
        if times is not None and len(times) != len(anchors):
            raise ValueError("times must have the same length as anchors")
        anchors = tuple((float(lat), float(lon)) for lat, lon in anchors)
        times = list(times) if times is not None else [i * 60. for i in range(len(anchors))]

        # All remaining computation is a function of public anchors, public
        # timestamps/graph, and fresh randomness: it has no ``points`` input.
        self._state = None
        tracks = [[] for _ in range(self.k)]
        # Time-major order: every prefix consumes exactly the same draws.
        for step, ((anchor_lat, anchor_lon), current_t) in enumerate(zip(anchors, times)):
            candidates = self._generate_step((anchor_lat, anchor_lon), current_t)
            for track, point in zip(tracks, candidates):
                track.append(point)

        return AnchoredDummyBatch(
            anchors=anchors,
            trajectories=tuple(tuple(track) for track in tracks),
        )

    def protect_run(self, real_trajectory):
        """Adapt the engine to the truth-separated benchmark protocol.

        The public transcript contains only the ``K`` generated tracks.  REM
        anchors remain evaluator-side diagnostics and are deliberately absent
        from the attacker serialization.
        """

        from core.demo_protocol import TrajectoryPoint, make_dummy_only_run

        real = tuple(real_trajectory)
        if not real or not all(isinstance(point, TrajectoryPoint) for point in real):
            raise TypeError("protect_run expects a non-empty TrajectoryPoint sequence")
        points = [(point.lat, point.lon) for point in real]
        times = [point.timestamp_s for point in real]
        batch = self.protect_trajectory(points, times)
        tracks = {
            f"candidate_{track_idx:04d}": tuple(
                TrajectoryPoint(timestamp, lat, lon)
                for timestamp, (lat, lon) in zip(times, track)
            )
            for track_idx, track in enumerate(batch.trajectories)
        }
        return make_dummy_only_run(
            self.name,
            real,
            tracks,
            public_parameters={
                "implementation_level": self.implementation_level,
                "implementation_origin": "benchmark.engines.GeoIAnchoredDummyEngine",
                "epsilon_per_release": self.epsilon,
                "k": self.k,
                "offset_m": self.offset_m,
                "candidate_radius_m": self.candidate_radius_m,
                "v_max_m_s": self.v_max,
                "reachability_slack_m": self.reachability_slack_m,
                "road_constrained": self.road_constrained,
                "online_api": "protect_step",
                "source_method": self.source_method,
            },
        )


__all__ = ["AnchoredDummyBatch", "GeoIAnchoredDummyEngine"]
