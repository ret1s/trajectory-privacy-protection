"""Opt-in service alignment and public-purpose lower-tail Q postprocessing.

Legacy/nearest modes instantiate the unchanged existing paced engine. Multi
modes override only protected-history postprocessing; inherited Geo-I, private
reuse, prospective units, GPS pacing and public request cadence stay untouched.
Risk exchanges have no global/submodular/trajectory utility guarantee.
"""
from dataclasses import replace
import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment

from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.engines.fair_cover import CoverageObjective, exchange_refine
from benchmark.engines.quotient_cover import reduce_groups
from benchmark.engines.service_cover import greedy_cover
from benchmark.public_service_profiles import PublicProfileAnchorModel
from benchmark.response_aware_belief import ResponseAwareAnchorModel


MODES = ('legacy_l10', 'aligned_nearest', 'aligned_mean_multi', 'risk_multi')


class _ReplyPrefix:
    def __init__(self, context, k):
        if context.k < k:
            raise ValueError('Public reply lacks the requested legacy prefix')
        self.base, self.k = context, k
        self.signatures = context.signatures[:, :, :k].copy()
        self.sha256 = hashlib.sha256((context.sha256+'/public-prefix/'+str(k)).encode()).hexdigest()

    def query_indices(self, state):
        return self.signatures[self.access[int(state)]]

    def __getattr__(self, name):
        return getattr(self.base, name)


class PublicPurposePacedLaneDummy(PacedSlackProgressLaneDummy):
    name = 'public_purpose_paced_lane_dummy'

    def __init__(self, rn, *, belief_model, planner_mode='aligned_mean_multi',
                 risk_weight=.5, tail_mass=.25, mean_slack=.01,
                 max_risk_exchanges=3, **kwargs):
        if not isinstance(belief_model, PublicProfileAnchorModel) or planner_mode not in MODES[2:]:
            raise ValueError('Matched public-purpose model and opt-in multi mode required')
        if (isinstance(risk_weight, bool) or isinstance(tail_mass, bool) or isinstance(mean_slack, bool)
                or not np.isfinite(risk_weight) or not 0 <= risk_weight <= 1
                or not np.isfinite(tail_mass) or not 0 < tail_mass <= 1
                or not np.isfinite(mean_slack) or not 0 <= mean_slack <= 1
                or isinstance(max_risk_exchanges, bool)
                or not isinstance(max_risk_exchanges, (int, np.integer)) or max_risk_exchanges < 0):
            raise ValueError('Valid fixed public risk/floor/iteration parameters required')
        self.planner_mode, self.risk_weight, self.tail_mass = planner_mode, float(risk_weight), float(tail_mass)
        self.mean_slack, self.max_risk_exchanges = float(mean_slack), int(max_risk_exchanges)
        super().__init__(rn, belief_model=belief_model, **kwargs)

    def _groups(self, timestamp_s, previous, previous_t):
        if previous is None:
            return [self.viable_ids]*self.k
        groups = []
        for prior in previous:
            reached = self.travel.reachable(prior, timestamp_s-previous_t)
            ids = np.asarray(sorted(i for i in reached if self.viable[i]), dtype=int)
            if not len(ids):
                raise RuntimeError('Empty directed public reachable bucket')
            groups.append(ids)
        return groups

    def _risk_refine(self, profile_objective, groups, baseline):
        selected = list(baseline)
        score = profile_objective.score(selected)
        floor = score.mean-self.mean_slack
        history = [dict(states=list(selected), **score.__dict__)]
        evaluations, termination = 0, 'iteration_limit'
        if self.planner_mode != 'risk_multi' or self.risk_weight == 0:
            return selected, history, floor, evaluations, 'mean_baseline'
        for _ in range(self.max_risk_exchanges):
            best = None
            for j, group in enumerate(groups):
                ids = np.asarray(group, dtype=int)
                means, tails, values = profile_objective.scores_replacing(ids, selected[:j]+selected[j+1:])
                evaluations += len(ids)
                eligible = np.flatnonzero((means >= floor-1e-12) & (values > score.objective+1e-12))
                if len(eligible):
                    # Within a fixed track, only its greatest objective and
                    # smallest state ID can win the full-action lexical tie.
                    # This preserves exhaustive scoring without a Python loop
                    # over every improving candidate.
                    tied = eligible[values[eligible] == values[eligible].max()]
                    at = tied[np.argmin(ids[tied])]
                    proposal = selected.copy(); proposal[j] = int(ids[at])
                    key = (-float(values[at]), tuple(proposal))
                    if best is None or key < best[0]:
                        best = key, proposal
            if best is None:
                termination = 'no_improving_exchange'; break
            _, proposal = best
            checked = profile_objective.score(proposal)
            if checked.mean < floor-1e-12 or checked.objective <= score.objective+1e-12:
                termination = 'numeric_full_action_guard'; break
            if any(state not in group for state, group in zip(proposal, groups)):
                raise RuntimeError('Risk exchange left its public reachable bucket')
            selected, score = proposal, checked
            history.append(dict(states=list(selected), **score.__dict__))
        return selected, history, floor, evaluations, termination

    def postprocess(self, anchor, timestamp_s):
        previous, previous_t = self.previous, self.last_t
        belief = self.belief.update(anchor, timestamp_s, observed=self.n < self.horizon)
        profiles = self.belief_model.profiles
        try:
            profile_objective = profiles.objective(belief, tail_mass=self.tail_mass, risk_weight=self.risk_weight)
        except ValueError as exc:
            if 'undefined' not in str(exc):
                raise
            # Deterministic zero-service fallback; never invent an empty-reference score.
            profile_objective = None
        context = profiles.context
        weights = np.zeros(len(context.pois)+1) if profile_objective is None else profile_objective.poi_weights
        objective = CoverageObjective(context.signatures, context.access, weights, None, self.category_ids)
        groups = self._groups(timestamp_s, previous, previous_t)
        def ties(j, ids):
            movement = (np.zeros(len(ids)) if previous is None else np.linalg.norm(self.rn.xy[ids]-self.rn.xy[previous[j]], axis=1))
            return movement, np.linalg.norm(self.rn.xy[ids]-self.public_center, axis=1)
        reduced = reduce_groups(groups, self.service_profiles, ties)
        greedy, gains = greedy_cover(reduced, objective.marginal, ties)
        selected, history = exchange_refine(reduced, greedy, objective, ties, self.max_exchanges)
        mean_greedy = list(selected)
        goals = None
        if previous is not None:
            center = belief @ self.belief_model.xy
            def global_ties(j, ids):
                return np.linalg.norm(self.rn.xy[ids]-center, axis=1), np.zeros(len(ids))
            distinct = reduce_groups([self.viable_ids], self.service_profiles, global_ties)[0]
            global_groups = [distinct]*self.k
            goals, _ = greedy_cover(global_groups, objective.marginal, global_ties)
            goals, _ = exchange_refine(global_groups, goals, objective, global_ties, self.max_exchanges)
            distances = [self._to_goal(goal) for goal in goals]
            costs = np.asarray([[d[state] for d in distances] for state in previous])
            rr, cc = linear_sum_assignment(costs)
            assigned = dict(zip(rr, cc)); moved = []
            for j, (prior, base) in enumerate(zip(previous, selected)):
                ids = np.asarray([i for i in groups[j] if self.service_profiles[i] == self.service_profiles[base]], dtype=int)
                movement = np.linalg.norm(self.rn.xy[ids]-self.rn.xy[prior], axis=1)
                distance = distances[assigned[j]][ids]
                moved.append(int(ids[np.lexsort((ids, movement, distance))[0]]))
            if not np.array_equal(context.signatures[context.access[selected]], context.signatures[context.access[moved]]):
                raise RuntimeError('Mean progress changed the selected response signatures')
            selected = moved
            goals = [goals[assigned[j]] for j in range(self.k)]
            if self.utility_slack:
                floor = objective.value(selected)-self.utility_slack
                for j, prior in enumerate(previous):
                    ids = groups[j]
                    others = selected[:j]+selected[j+1:]
                    values = objective.value(others)+objective.marginal(ids, others)
                    eligible = ids[values >= floor-1e-12]
                    d = self._to_goal(goals[j])
                    movement = np.linalg.norm(self.rn.xy[eligible]-self.rn.xy[prior], axis=1)
                    proposal = int(eligible[np.lexsort((eligible, movement, d[eligible]))[0]])
                    candidate = selected.copy(); candidate[j] = proposal
                    if d[proposal] < d[selected[j]]-1e-12 and objective.value(candidate) >= floor-1e-12:
                        selected = candidate
        baseline = list(selected)
        if profile_objective is None:
            risk_history, mean_floor, evaluations, termination = [], None, 0, 'undefined_public_reference'
            final_score = None
        else:
            # Include fallback states even when quotient representatives differ after progress.
            risk_groups = [np.unique(np.r_[bucket, state]) for bucket, state in zip(reduced, baseline)]
            selected, risk_history, mean_floor, evaluations, termination = self._risk_refine(profile_objective, risk_groups, baseline)
            final_score = profile_objective.score(selected)
        self.evaluator_objective.append(dict(planner_mode=self.planner_mode, greedy_states=greedy,
            greedy_gains=gains, mean_exchange_history=history, mean_greedy_states=mean_greedy,
            mean_baseline_states=baseline, risk_states=list(selected), progress_goals=goals,
            reachable_counts=list(map(len, groups)), quotient_counts=list(map(len, reduced)),
            value=None if final_score is None else final_score.objective,
            mean_value=None if final_score is None else final_score.mean,
            lower_tail_cvar=None if final_score is None else final_score.lower_tail_cvar,
            empty_profile_mass=1. if profile_objective is None else profile_objective.empty_profile_mass,
            active_profile_count=0 if profile_objective is None else profile_objective.active_profile_count,
            risk_history=risk_history, risk_mean_floor=mean_floor,
            risk_candidate_evaluations=evaluations, risk_termination=termination))
        self.previous, self.last_t = list(selected), timestamp_s
        self.evaluator_states.append(list(selected))
        return tuple(self.rn.latlon(i) for i in selected)

    def protect_run(self, points):
        run = super().protect_run(points)
        params = dict(run.transcript.public_parameters)
        params.update(planner_mode=self.planner_mode, public_profiles_sha256=self.belief_model.profiles.sha256,
            planner_reply_l=self.belief_model.context.k,
            planner_reference_top_k=self.belief_model.profiles.reference_k,
            objective='public_profile_mean' if self.planner_mode == 'aligned_mean_multi' else 'public_profile_mean_and_lower_tail_CVaR',
            public_risk_weight=self.risk_weight, public_tail_mass=self.tail_mass,
            risk_mean_slack=self.mean_slack, max_risk_exchanges=self.max_risk_exchanges,
            service_equivalent_motion='mean_initialization_only; risk may change response profiles',
            optimization_claim='bounded_one_track_exchanges; no global_or_trajectory_utility_bound')
        return replace(run, transcript=replace(run.transcript, public_parameters=params))


def make_service_planner_engine(mode, rn, base, reply20, publicprofiles=None, *,
                               legacy_context=None, risk_weight=.5, tail_mass=.25,
                               mean_slack=.01, max_risk_exchanges=3, **old_engine_kwargs):
    """Runner switch preserving all existing private-mechanism configuration.

    Pass the underlying reference-top5 PublicAnchorModel as base. For exact old
    replay pass the original reply10 context as legacy_context. No allocation,
    RNG streams, cache, traffic depth or schedule is changed by this factory.
    """
    if mode not in MODES or rn is not base.rn or rn is not reply20.rn:
        raise ValueError('Known public planner mode and same map required')
    if mode in MODES[:2]:
        context = (legacy_context or _ReplyPrefix(reply20, 10)) if mode == 'legacy_l10' else reply20
        if mode == 'legacy_l10' and context.k != 10:
            raise ValueError('Legacy control requires actual L10 planner context')
        model = ResponseAwareAnchorModel(base, context)
        return PacedSlackProgressLaneDummy(rn, belief_model=model, **old_engine_kwargs)
    if publicprofiles is None or publicprofiles.context is not reply20:
        raise ValueError('Fixed public profiles matched to the actual reply context required')
    model = PublicProfileAnchorModel(base, publicprofiles)
    return PublicPurposePacedLaneDummy(rn, belief_model=model, planner_mode=mode,
        risk_weight=risk_weight, tail_mass=tail_mass, mean_slack=mean_slack,
        max_risk_exchanges=max_risk_exchanges, **old_engine_kwargs)
