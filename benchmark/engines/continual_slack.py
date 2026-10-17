"""Opt-in fixed-unit Geo-I updater; no finite whole-session upstream cap.

Only the protected belief is intended for the bounded-Q external channel. The
legacy dummy coordinates returned here are LOCAL intermediates, not covered by
the Q-session theorem. Ideal Geo-I composes at every finite public prefix;
numerical REM and trusted full-state restore retain their existing limitations.
"""
from dataclasses import replace
import math
import time

from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.engines.slack_progress import SlackProgressCoverLaneDummy


class ContinualPacedSlackProgressLaneDummy(PacedSlackProgressLaneDummy):
    name = 'continual_paced_slack_progress_lane_dummy'

    def __init__(self, rn, *, unit_epsilon=.01, **kwargs):
        if (isinstance(unit_epsilon, bool) or not math.isfinite(unit_epsilon)
                or unit_epsilon <= 0):
            raise ValueError('Positive finite per-operation epsilon required')
        if any(key in kwargs for key in ('budget', 'horizon', 'anchor_mode')):
            raise ValueError('Set unit_epsilon explicitly; no session budget/horizon policy')
        self.fixed_unit_epsilon = float(unit_epsilon)
        # Compatibility parameters initialize the UNCHANGED primitive/planner.
        # They neither stop reads nor express a cap in this opt-in engine.
        super().__init__(rn, budget=24*self.fixed_unit_epsilon, horizon=12,
                         anchor_mode='private_reuse', **kwargs)

    def reset(self):
        super().reset()
        self.unit_epsilon = self.fixed_unit_epsilon
        self.max_units = None
        self.private_read_count = 0
        self.prefix_epsilon_bound = 0.

    def protect_step(self, lat, lon, timestamp_s):
        if (isinstance(timestamp_s, bool) or not math.isfinite(timestamp_s)
                or (self.last_t is not None and timestamp_s <= self.last_t)):
            raise ValueError('Strictly increasing finite public times required')
        started = time.perf_counter()
        due = (self.last_private_read_s is None
               or timestamp_s-self.last_private_read_s >= self.read_interval_s)
        if due and (not math.isfinite(lat) or not math.isfinite(lon)
                    or not -90 <= lat <= 90 or not -180 <= lon <= 180):
            raise ValueError('Valid private coordinates required when read')
        self.privacy_read_this_step = due
        cost, branch = 0, 'public_clock_skip'
        if due:
            first = self.last_anchor is None
            before = self.anchor.n_resample
            self.last_anchor = self.anchor.perturb(lat, lon, t=timestamp_s)
            fresh = self.anchor.n_resample > before
            if first and not fresh:
                raise RuntimeError('First anchor must be fresh')
            cost = (0 if first else 1)+int(fresh)
            self.spent_units += cost
            self.private_read_count += 1
            self.last_private_read_s = timestamp_s
            branch = 'fresh' if fresh else 'reuse'
        # Branch-dependent spending is a PRIVATE diagnostic. State a public
        # prefix theorem with the worst case, not with a realized hidden count.
        self.spent_bound = self.spent_units*self.unit_epsilon
        self.prefix_epsilon_bound = (2*self.private_read_count-1)*self.unit_epsilon
        output = self.postprocess(self.last_anchor, timestamp_s)
        self.evaluator_anchors.append(list(self.last_anchor))
        self.evaluator_ledger.append(dict(cost_units=cost, spent_units=self.spent_units,
            branch=branch, private_read=due))
        self.n += 1
        self.step_ms.append((time.perf_counter()-started)*1000)
        return output

    def protect_run(self, points):
        # Bypass fixed-filter/matched-cap wrappers; retain the existing local
        # planner's metadata. Never advertise the legacy .23/m cap for this arm.
        run = SlackProgressCoverLaneDummy.protect_run(self, points)
        params = dict(run.transcript.public_parameters)
        for key in ('budget_per_m', 'horizon_events', 'after_horizon'):
            params.pop(key, None)
        params.update(privacy_unit_epsilon=self.unit_epsilon,
            minimum_private_read_interval_s=self.read_interval_s,
            total_session_cap_per_m=None,
            ideal_prefix_bound='(2*N-1)*u for N public scheduled reads',
            external_use='protected belief to bounded-Q only; returned dummies are local',
            numerical_sampler_certified=False)
        return replace(run, transcript=replace(run.transcript, public_parameters=params))
