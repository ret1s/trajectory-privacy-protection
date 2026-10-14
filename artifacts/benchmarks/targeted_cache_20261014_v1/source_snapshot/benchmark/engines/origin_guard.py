"""Public-clock early-phase budget allocation for the progress candidate.

Early releases/tests spend one quarter of the usual epsilon. This is a
prespecified allocation experiment, not a new privacy primitive or a claim that
later observations cannot reconstruct the origin.
"""
from dataclasses import replace
import math
import time
import numpy as np
from benchmark.anchor_belief import AnchorBelief
from benchmark.engines.progress_cover import ProgressCoverLaneDummy


class _PhaseEmission:
    def __init__(self, base, owner):
        self.base, self.owner = base, owner

    def __getattr__(self, key):
        return getattr(self.base, key)

    def emission(self, anchor, previous=None):
        return self.owner.active_emission_model.emission(anchor, previous)


class _PhaseBelief(AnchorBelief):
    def __init__(self, base, owner):
        super().__init__(_PhaseEmission(base, owner))
        self.owner = owner

    def update(self, anchor, timestamp, *, observed=True):
        return super().update(anchor, timestamp,
                              observed=self.owner.privacy_read_this_step)


class OriginGuardProgressLaneDummy(ProgressCoverLaneDummy):
    name = 'origin_guard_progress_lane_dummy'

    def __init__(self, rn, *, early_belief_model, guard_seconds=60., **kwargs):
        if not math.isfinite(guard_seconds) or guard_seconds < 0:
            raise ValueError('Nonnegative finite public guard duration required')
        base = kwargs['belief_model']
        if (early_belief_model.rn is not rn or early_belief_model.context is not base.context
                or not np.array_equal(early_belief_model.state_ids, base.state_ids)
                or not np.array_equal(early_belief_model.prior, base.prior)
                or early_belief_model.theta_m != base.theta_m
                or not np.isclose(4*early_belief_model.epsilon_release, base.epsilon_release,
                                  rtol=1e-12, atol=0)
                or not np.isclose(4*early_belief_model.epsilon_test, base.epsilon_test,
                                  rtol=1e-12, atol=0)):
            raise ValueError('Early emission must share context/grid/prior and quarter epsilon')
        self.early_belief_model = early_belief_model
        self.guard_seconds = float(guard_seconds)
        super().__init__(rn, **kwargs)

    def reset(self):
        super().reset()
        self.spent_units = 0
        self.max_units = 4*(2*self.horizon-1)
        self.unit_epsilon = self.budget/(8*self.horizon)
        self.privacy_read_this_step = False
        self.filter_stopped = False
        self.start_s = None
        self.active_emission_model = self.early_belief_model
        self.evaluator_ledger = []
        self.belief = _PhaseBelief(self.belief_model, self)

    def protect_step(self, lat, lon, timestamp_s):
        started = time.perf_counter()
        if not math.isfinite(timestamp_s) or (self.last_t is not None and timestamp_s <= self.last_t):
            raise ValueError('Strictly increasing finite public times required')
        if self.start_s is None:
            self.start_s = timestamp_s
        first = self.last_anchor is None
        early = first or timestamp_s-self.start_s < self.guard_seconds
        units = 1 if early else 4
        reserve = units*(1 if first else 2)
        if self.spent_units+reserve > self.max_units:
            self.filter_stopped = True
        self.privacy_read_this_step = not self.filter_stopped
        cost, branch = 0, 'postprocess'
        self.active_emission_model = self.early_belief_model if early else self.belief_model
        if self.privacy_read_this_step:
            if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                raise ValueError('Finite valid private coordinates required when read')
            # No cached logits in the primitive; preserve previous anchor and RNG.
            self.anchor.epsilon = self.active_emission_model.epsilon_release
            self.anchor.eps_test = self.active_emission_model.epsilon_test
            self.anchor.privacy_cost_per_step_max = self.anchor.epsilon+self.anchor.eps_test
            before = self.anchor.n_resample
            self.last_anchor = self.anchor.perturb(lat, lon, t=timestamp_s)
            fresh = self.anchor.n_resample > before
            if first and not fresh:
                raise RuntimeError('First anchor must be fresh')
            cost = units*((0 if first else 1)+int(fresh))
            self.spent_units += cost
            branch = 'fresh' if fresh else 'reuse'
        assert self.spent_units <= self.max_units
        self.spent_bound = self.spent_units*self.unit_epsilon
        output = self.postprocess(self.last_anchor, timestamp_s)
        self.evaluator_anchors.append(list(self.last_anchor))
        self.evaluator_ledger.append({'cost_units': cost, 'spent_units': self.spent_units,
                                     'phase_unit_cost': units, 'early_phase': early,
                                     'branch': branch, 'private_read': self.privacy_read_this_step})
        self.n += 1
        self.step_ms.append((time.perf_counter()-started)*1000)
        return output

    def protect_run(self, points):
        run = super().protect_run(points)
        params = dict(run.transcript.public_parameters)
        params.pop('horizon_events', None)
        params.pop('after_horizon', None)
        params.update(origin_guard_seconds=self.guard_seconds,
                      first_release_always_guarded=True, early_epsilon_factor=.25,
                      early_belief_sha256=self.early_belief_model.sha256,
                      privacy_unit_epsilon=self.unit_epsilon, max_privacy_units=self.max_units,
                      tight_session_bound_per_m=self.max_units*self.unit_epsilon,
                      budget_filter='public_phase_worst_step_reservation_then_extended_branch_charge',
                      after_filter_stop='no_private_read_public_postprocessing',
                      guarantee_scope='ideal_kernel_fixed_public_clock_and_session_length')
        return replace(run, transcript=replace(run.transcript, public_parameters=params))
