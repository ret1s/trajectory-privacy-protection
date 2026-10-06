"""Public-time pacing of private reads, independent of query/output cadence.

Allocating privacy over time is established practice; this mixin tests its
interaction with the directed service-cover planner. No new private decision
is taken when a public request arrives between scheduled private reads.
"""
from dataclasses import replace
import math
import time
from benchmark.engines.matched_filter import MatchedFilteredProgressCoverLaneDummy
from benchmark.engines.origin_guard import OriginGuardProgressLaneDummy


class _PublicPacing:
    def __init__(self, rn, *, read_interval_s=60., **kwargs):
        if not math.isfinite(read_interval_s) or read_interval_s < 0:
            raise ValueError('Nonnegative finite public private-read interval required')
        self.read_interval_s = float(read_interval_s)
        super().__init__(rn, **kwargs)

    def reset(self):
        super().reset()
        self.last_private_read_s = None

    def protect_step(self, lat, lon, timestamp_s):
        if not math.isfinite(timestamp_s) or (self.last_t is not None and timestamp_s <= self.last_t):
            raise ValueError('Strictly increasing finite public times required')
        if self.last_private_read_s is None or timestamp_s-self.last_private_read_s >= self.read_interval_s:
            result = super().protect_step(lat, lon, timestamp_s)
            if self.privacy_read_this_step:
                self.last_private_read_s = timestamp_s
            return result
        started = time.perf_counter()
        self.privacy_read_this_step = False
        result = self.postprocess(self.last_anchor, timestamp_s)
        self.evaluator_anchors.append(list(self.last_anchor))
        self.evaluator_ledger.append({'cost_units': 0, 'spent_units': self.spent_units,
                                     'branch': 'public_clock_skip', 'private_read': False})
        self.n += 1
        self.step_ms.append((time.perf_counter()-started)*1000)
        return result

    def protect_run(self, points):
        run = super().protect_run(points)
        params = dict(run.transcript.public_parameters)
        params['minimum_private_read_interval_s'] = self.read_interval_s
        params['between_private_reads'] = 'public_belief_prediction_and_reachable_cover_progress'
        return replace(run, transcript=replace(run.transcript, public_parameters=params))


class PacedProgressLaneDummy(_PublicPacing, MatchedFilteredProgressCoverLaneDummy):
    name = 'paced_progress_lane_dummy'


class PacedOriginGuardLaneDummy(_PublicPacing, OriginGuardProgressLaneDummy):
    name = 'paced_origin_guard_lane_dummy'
