"""Executable prototype of the thesis' proposed dummy-only architecture.

This module is intentionally a *demo*, not the final thesis mechanism.  It
implements the concept described in the September 2026 research plan:

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

from core.mechanisms import RoadExponential


Point = Tuple[float, float]


@dataclass(frozen=True)
class AnchoredDummyBatch:
    """Raw output of the prototype before it is wrapped as a public transcript.

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


class GeoIAnchoredDummyTrajectoriesLite:
    """Paper-concept demo for the proposed Geo-I-anchor + dummy-only design.

    The generator uses persistent public-space offsets around the REM anchor
    and penalises road vertices that would require implausible speed relative
    to the same dummy track's previous public point.  It deliberately does not
    consult the true trajectory after anchors have been generated.

    This is suitable for a visual and contract demo only.  It has no trained
    population prior, POI semantics, group model, formal trajectory-level
    budget manager, or empirical resistance claim yet.
    """

    name = "geo_i_anchored_dummy_lite"
    source_method = "Thesis prototype: REM anchor followed by dummy-only tracks"
    demo_only = True
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
        self.rng = rng or np.random.default_rng()

    def _sample_public_candidate(self, target_xy, previous_xy, dt_s):
        idxs = np.asarray(
            self.rn.tree.query_ball_point(target_xy, self.candidate_radius_m),
            dtype=int,
        )
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
        return int(idxs[int(np.argmax(logits + self.rng.gumbel(size=len(idxs))))])

    def protect_trajectory(
        self,
        points: Sequence[Point],
        times: Optional[Sequence] = None,
    ) -> AnchoredDummyBatch:
        if len(points) == 0:
            raise ValueError("points must not be empty")
        if times is not None and len(times) != len(points):
            raise ValueError("times must have the same length as points")

        times = list(times) if times is not None else [None] * len(points)
        anchor_mechanism = RoadExponential(
            self.epsilon,
            self.rn,
            rng=self.rng,
        )
        anchor_mechanism.reset()
        anchors = tuple(
            anchor_mechanism.perturb(lat, lon, t=t)
            for (lat, lon), t in zip(points, times)
        )

        return self.generate_from_anchors(anchors, times)

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
        times = list(times) if times is not None else [None] * len(anchors)

        # All remaining computation is a function of public anchors, public
        # timestamps/graph, and fresh randomness: it has no ``points`` input.
        base_angle = self.rng.uniform(0.0, 2.0 * np.pi)
        angles = base_angle + np.linspace(0.0, 2.0 * np.pi, self.k, endpoint=False)
        radii = self.offset_m * self.rng.uniform(0.75, 1.25, size=self.k)

        tracks = []
        for track_idx in range(self.k):
            previous_xy = None
            track = []
            for step, ((anchor_lat, anchor_lon), current_t) in enumerate(zip(anchors, times)):
                anchor_xy = np.asarray(self.rn.point_xy(anchor_lat, anchor_lon))
                # Slow public drift prevents a perfectly rigid translated copy.
                angle = angles[track_idx] + 0.12 * np.sin(step / 3.0 + track_idx)
                target_xy = anchor_xy + radii[track_idx] * np.array(
                    [np.cos(angle), np.sin(angle)]
                )
                previous_t = times[step - 1] if step else None
                dt_s = _delta_seconds(current_t, previous_t)
                choice = self._sample_public_candidate(target_xy, previous_xy, dt_s)
                point = self.rn.latlon(choice)
                track.append(point)
                previous_xy = self.rn.xy[choice]
            tracks.append(tuple(track))

        return AnchoredDummyBatch(
            anchors=anchors,
            trajectories=tuple(tracks),
        )

    def protect_run(self, real_trajectory):
        """Adapt the raw prototype to the truth-separated demo protocol.

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
                "demo_only": True,
                "epsilon_per_release": self.epsilon,
                "k": self.k,
                "source_method": self.source_method,
            },
        )
