"""BR-Dummy: bounded-horizon private anchors + viable public-road postprocessing.

The noisy reuse test is inherited from PrivateReuseSMREM (predictive Geo-I
literature); the contribution being evaluated is its combination with a fixed
ledger and progress-aware dummy generation. No novelty or SOTA claim here.
"""
import math
import time

import networkx as nx
import numpy as np

from core.demo_protocol import TrajectoryPoint, make_dummy_only_run
from core.mechanisms import PrivateReuseSMREM, RoadExponential
from evaluation.scenario_metrics import RoadTravelTimes


class BudgetedReachableDummy:
    name = "budgeted_reachable_dummy"

    def __init__(self, rn, *, budget=.24, horizon=12, k=3, anchor_mode="private_reuse",
                 theta_m=200., offset_m=80., temperature_m=60., rng=None):
        if not math.isfinite(budget) or budget <= 0 or horizon < 1 or int(horizon) != horizon:
            raise ValueError("Positive finite budget and integer public horizon required")
        if k < 1 or int(k) != k or anchor_mode not in {"fresh", "private_reuse"}:
            raise ValueError("Positive integer K and known anchor mode required")
        if not all(math.isfinite(x) for x in (theta_m, offset_m, temperature_m)) or theta_m < 0 or offset_m < 0 or temperature_m <= 0:
            raise ValueError("Public distances must be finite/nonnegative; temperature positive")
        if not rn.graph.is_directed():
            raise ValueError("Directed road graph required")
        self.rn, self.budget, self.horizon, self.k = rn, float(budget), int(horizon), int(k)
        self.anchor_mode, self.theta_m, self.offset_m = anchor_mode, float(theta_m), float(offset_m)
        self.temperature_m = float(temperature_m)
        rng = rng if rng is not None else np.random.default_rng()
        seeds = rng.integers(0, 2**63, size=2, dtype=np.int64)
        self.anchor_rng, self.dummy_rng = (np.random.default_rng(int(s)) for s in seeds)
        # Context-only viability: a strongly connected component prevents
        # choosing a one-way dead-end; it does not verify lane/turn constraints.
        components = list(nx.strongly_connected_components(rn.graph))
        viable_nodes = max(components, key=lambda c: (len(c), tuple(sorted(map(str, c)))))
        self.viable = np.array([node in viable_nodes for node in rn.node_ids])
        self.travel = RoadTravelTimes(rn)
        self.reset()

    def reset(self):
        cap = self.budget / self.horizon
        self.anchor = (PrivateReuseSMREM(self.rn, epsilon_step_cap=cap, theta=self.theta_m, rng=self.anchor_rng)
                       if self.anchor_mode == "private_reuse" else RoadExponential(cap, self.rn, rng=self.anchor_rng))
        self.n, self.spent_bound, self.last_t = 0, 0., None
        self.last_anchor, self.previous = None, None
        self.offsets = None
        self.step_ms = []

    def protect_step(self, lat, lon, timestamp_s):
        started = time.perf_counter()
        if not math.isfinite(timestamp_s) or (self.last_t is not None and timestamp_s <= self.last_t):
            raise ValueError("Strictly increasing finite timestamps required")
        if self.n < self.horizon:
            self.last_anchor = self.anchor.perturb(lat, lon, t=timestamp_s)
            cost = self.budget / self.horizon
            if self.anchor_mode == "private_reuse" and self.n == 0:
                cost /= 2
            self.spent_bound += cost  # worst-case, never a hidden-branch discount
        # Beyond the declared horizon: no read of the new private coordinates.
        # Continue service using only the last protected anchor/public state.
        center = np.array(self.rn.point_xy(*self.last_anchor))
        if self.offsets is None:
            angle = self.dummy_rng.uniform(0, 2 * np.pi)
            angles = angle + np.arange(self.k) * 2 * np.pi / self.k
            self.offsets = self.offset_m * np.c_[np.cos(angles), np.sin(angles)]
        selected = []
        for j in range(self.k):
            if self.previous is None:
                ids = np.flatnonzero(self.viable)
            else:
                reachable = self.travel.reachable(self.previous[j], timestamp_s - self.last_t)
                ids = np.array([i for i, node in enumerate(self.rn.node_ids)
                                if self.viable[i] and node in reachable], dtype=int)
            # Previous state always remains feasible; initial SCC is nonempty.
            if not len(ids):
                raise RuntimeError("Empty public reachable set")
            targets = center + self.offsets[j]
            logits = -np.linalg.norm(self.rn.xy[ids] - targets, axis=1) / self.temperature_m
            # Encourage distinct candidates using PUBLIC already-selected points.
            # Duplicates remain possible and are measured, never called K-anonymity.
            for prior in selected:
                logits -= 2 * np.exp(-np.linalg.norm(self.rn.xy[ids] - self.rn.xy[prior], axis=1) / 40.)
            selected.append(int(ids[np.argmax(logits + self.dummy_rng.gumbel(size=len(ids)))]))
        self.previous, self.last_t = selected, timestamp_s
        self.n += 1
        self.step_ms.append((time.perf_counter() - started) * 1000)
        return tuple(self.rn.latlon(i) for i in selected)

    def protect_run(self, points):
        points = tuple(points)
        if not points or not all(isinstance(p, TrajectoryPoint) for p in points):
            raise ValueError("Nonempty TrajectoryPoint sequence required")
        self.reset()
        tracks = {f"candidate_{j:04d}": [] for j in range(self.k)}
        for p in points:
            released = self.protect_step(p.lat, p.lon, p.timestamp_s)
            for j, location in enumerate(released):
                tracks[f"candidate_{j:04d}"].append(TrajectoryPoint(p.timestamp_s, *location))
        return make_dummy_only_run(self.name, points, tracks, public_parameters={
            "implementation_level": "thesis_candidate", "anchor_mode": self.anchor_mode,
            "budget_per_m": self.budget, "horizon_events": self.horizon, "k": self.k,
            "theta_m": self.theta_m, "offset_m": self.offset_m, "temperature_m": self.temperature_m,
            "support": "full_V_anchors__public_largest_SCC_dummies",
            "after_horizon": "postprocessing_only_no_private_read",
        })
