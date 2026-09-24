"""Slow private reads as the fixed privacy reservoir is consumed.

The policy reads only the already-accounted extended-transcript ledger. It
changes neither epsilon nor the cap and never needs a trip end time. Allocation
over time is prior art; this is an ablation of its service-planner interaction.
"""
from dataclasses import replace
import math
from benchmark.engines.paced_guard import PacedProgressLaneDummy
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy


class _ReservePacing:
    def __init__(self, rn, *, reserve_strength=1., **kwargs):
        if not math.isfinite(reserve_strength) or not 0 <= reserve_strength <= 1:
            raise ValueError('Public reserve strength must lie in [0,1]')
        self.reserve_strength = float(reserve_strength)
        super().__init__(rn, **kwargs)

    def protect_step(self, lat, lon, timestamp_s):
        base = self.read_interval_s
        remaining = max(1, self.max_units-self.spent_units)
        # This ledger depends only on the protected primitive/branch history,
        # not on a fresh GPS lookup, a scenario label or the unobserved suffix.
        factor = 1+self.reserve_strength*(self.max_units/remaining-1)
        self.read_interval_s = base*factor
        try:
            return super().protect_step(lat, lon, timestamp_s)
        finally:
            self.read_interval_s = base

    def protect_run(self, points):
        run = super().protect_run(points)
        params = dict(run.transcript.public_parameters)
        params.update(read_pacing='base_interval_times_1_plus_strength_times_cap_over_remaining_minus_1',
                      reserve_strength=self.reserve_strength,
                      read_schedule_inputs='public_time_and_accounted_protected_branch_history',
                      epsilon_per_read_unchanged=True, future_trip_duration_used=False)
        return replace(run, transcript=replace(run.transcript, public_parameters=params))


class ReservePacedProgressLaneDummy(_ReservePacing, PacedProgressLaneDummy):
    name = 'reserve_paced_progress_lane_dummy'


class ReservePacedSlackProgressLaneDummy(_ReservePacing, PacedSlackProgressLaneDummy):
    name = 'reserve_paced_slack_progress_lane_dummy'
