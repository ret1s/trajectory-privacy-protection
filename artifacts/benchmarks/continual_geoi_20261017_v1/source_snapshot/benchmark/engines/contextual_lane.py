"""Contextual BR-lane: directed-route potential and marginal public POI cover.

Only the inherited anchor step reads GPS. Public map/POIs and protected anchors
drive postprocessing. The ideal privacy ledger is unchanged; empirical privacy
can differ, and floating-point sampling is still only an approximation.
"""
from collections import OrderedDict
from dataclasses import replace
import math
import time

import numpy as np
from scipy.sparse.csgraph import dijkstra

from benchmark.engines.lane_budgeted import LaneBudgetedDummy
from evaluation.lane_travel import matrix


class ContextualLaneDummy(LaneBudgetedDummy):
    name = 'contextual_lane_dummy'

    def __init__(self, rn, *, context=None, route_weight=0., coverage_weight=0., **kwargs):
        if not math.isfinite(route_weight) or not 0 <= route_weight <= 1:
            raise ValueError('route_weight must be in [0,1]')
        if not math.isfinite(coverage_weight) or coverage_weight < 0:
            raise ValueError('coverage_weight must be finite/nonnegative')
        if coverage_weight and (context is None or context.rn is not rn):
            raise ValueError('Matching public POI context required')
        self.context, self.route_weight, self.coverage_weight = context, route_weight, coverage_weight
        super().__init__(rn, **kwargs)
        self.reverse = matrix(rn).transpose().tocsr() if route_weight else None
        self.viable_ids = np.flatnonzero(self.viable)

    def reset(self):
        super().reset()
        self.goal_cache = OrderedDict()
        self.evaluator_anchors = []  # never serialized into PublicTranscript

    def distances_to_goal(self, target_xy):
        distances = np.linalg.norm(self.rn.xy[self.viable_ids] - target_xy, axis=1)
        goal = int(self.viable_ids[np.argmin(distances)])
        if goal not in self.goal_cache:
            self.goal_cache[goal] = dijkstra(self.reverse, directed=True, indices=goal)
            if len(self.goal_cache) > 32:
                self.goal_cache.popitem(last=False)
        self.goal_cache.move_to_end(goal)
        return self.goal_cache[goal]

    def postprocess(self, anchor, timestamp_s):
        """No private GPS argument; safe only for an already-protected anchor."""
        center = np.asarray(self.rn.point_xy(*anchor))
        if self.offsets is None:
            angle = self.dummy_rng.uniform(0, 2 * np.pi)
            angles = angle + np.arange(self.k) * 2 * np.pi / self.k
            self.offsets = self.offset_m * np.c_[np.cos(angles), np.sin(angles)]
        weights = self.context.reference_weights(anchor) if self.coverage_weight else None
        selected = []
        for j in range(self.k):
            if self.previous is None:
                ids = self.viable_ids
            else:
                reached = self.travel.reachable(self.previous[j], timestamp_s - self.last_t)
                ids = np.array(sorted(i for i in reached if self.viable[i]), dtype=int)
            if not len(ids):
                raise RuntimeError('Empty public reachable set')
            target = center + self.offsets[j]
            distance = np.linalg.norm(self.rn.xy[ids] - target, axis=1)
            if self.route_weight:
                road = self.distances_to_goal(target)[ids]
                if not np.isfinite(road).all():
                    raise RuntimeError('SCC must reach its public goal')
                distance = (1 - self.route_weight) * distance + self.route_weight * road
            logits = -distance / self.temperature_m
            for other in selected:
                logits -= 2 * np.exp(-np.linalg.norm(self.rn.xy[ids] - self.rn.xy[other], axis=1) / 40.)
            if self.coverage_weight:
                logits += self.coverage_weight * self.context.marginal_gain(ids, weights, selected)
            selected.append(int(ids[np.argmax(logits + self.dummy_rng.gumbel(size=len(ids)))]))
        self.previous, self.last_t = selected, timestamp_s
        self.evaluator_states.append(list(selected))
        return tuple(self.rn.latlon(i) for i in selected)

    def protect_step(self, lat, lon, timestamp_s):
        started = time.perf_counter()
        if not math.isfinite(timestamp_s) or (self.last_t is not None and timestamp_s <= self.last_t):
            raise ValueError('Strictly increasing finite timestamps required')
        if self.n < self.horizon:
            if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError('Finite valid private coordinate required within horizon')
            self.last_anchor = self.anchor.perturb(lat, lon, t=timestamp_s)
            cost = self.budget / self.horizon
            if self.anchor_mode == 'private_reuse' and self.n == 0:
                cost /= 2
            self.spent_bound += cost
        output = self.postprocess(self.last_anchor, timestamp_s)
        self.evaluator_anchors.append(list(self.last_anchor))
        self.n += 1
        self.step_ms.append((time.perf_counter() - started) * 1000)
        return output

    def protect_run(self, points):
        run = super().protect_run(points)
        params = {**dict(run.transcript.public_parameters),
                  'route_weight': self.route_weight, 'coverage_weight': self.coverage_weight,
                  'public_poi_context_sha256': self.context.sha256 if self.coverage_weight else None,
                  'coverage_target': 'macro_POI_recall_at_protected_anchor_not_true_location'}
        return replace(run, transcript=replace(run.transcript, public_parameters=params))
