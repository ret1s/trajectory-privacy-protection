"""Joint reachable query-set coverage; private anchors and ledger stay unchanged.

Greedy weighted coverage under one-state-per-track partition constraints.
The per-step 1/2 bound is not a trajectory, actual-user utility or privacy bound.
"""
from dataclasses import replace

import numpy as np

from benchmark.engines.belief_lane import BeliefLaneDummy


def greedy_cover(groups, marginal_gain, tie_cost):
    """Exact global marginal oracle. No cross-track coordinate uniqueness rule.

    groups[j] is a fixed nonempty candidate array for track j. tie_cost(j, ids)
    returns primary and secondary *public* tie costs. Exact gains take priority.
    Output is in persistent track order, not greedy selection order.
    """
    groups = [np.asarray(g, dtype=int) for g in groups]
    if not groups or any(g.ndim != 1 or not len(g) for g in groups):
        raise ValueError('Nonempty one-dimensional candidate groups required')
    pending, chosen, order, gains = set(range(len(groups))), {}, [], []
    while pending:
        best = None
        for j in sorted(pending):
            ids = groups[j]
            gain = np.asarray(marginal_gain(ids, order))
            if gain.shape != ids.shape or not np.isfinite(gain).all() or np.any(gain < 0):
                raise ValueError('Finite nonnegative marginal gains required')
            tied = np.flatnonzero(gain == gain.max())
            primary, secondary = tie_cost(j, ids[tied])
            q = tied[np.lexsort((ids[tied], secondary, primary))[0]]
            key = (-float(gain[q]), float(primary[np.where(tied == q)[0][0]]),
                   float(secondary[np.where(tied == q)[0][0]]), j, int(ids[q]))
            if best is None or key < best[0]:
                best = key, j, int(ids[q]), float(gain[q])
        _, j, state, gain = best
        pending.remove(j)
        chosen[j] = state
        order.append(state)
        gains.append(gain)
    return [chosen[j] for j in range(len(groups))], gains


class ServiceCoverLaneDummy(BeliefLaneDummy):
    name = 'service_cover_lane_dummy'

    def __init__(self, rn, *, prior_only=False, **kwargs):
        self.prior_only = bool(prior_only)
        super().__init__(rn, coverage_weight=0., **kwargs)
        self.public_center = np.average(self.belief_model.xy, axis=0,
                                        weights=self.belief_model.prior)

    def reset(self):
        super().reset()
        self.evaluator_objective = []

    def postprocess(self, anchor, timestamp_s):
        if self.prior_only:
            weights = self.belief_model.prior
        else:
            weights = self.belief.update(anchor, timestamp_s, observed=self.n < self.horizon)
        poi_weights = np.asarray(weights @ self.belief_model.poi_weights).ravel()
        base = self.belief_model.context
        groups = []
        for j in range(self.k):
            if self.previous is None:
                ids = self.viable_ids
            else:
                reached = self.travel.reachable(self.previous[j], timestamp_s - self.last_t)
                ids = np.array(sorted(i for i in reached if self.viable[i]), dtype=int)
            groups.append(ids)

        def ties(j, ids):
            distance = np.linalg.norm(self.rn.xy[ids] - self.public_center, axis=1)
            movement = (np.zeros(len(ids)) if self.previous is None else
                        np.linalg.norm(self.rn.xy[ids] - self.rn.xy[self.previous[j]], axis=1))
            return movement, distance

        selected, gains = greedy_cover(groups,
            lambda ids, chosen: base.marginal_gain(ids, poi_weights, chosen), ties)
        self.evaluator_objective.append({'greedy_gains': gains, 'value': float(sum(gains)),
                                         'reachable_counts': list(map(len, groups))})
        self.previous, self.last_t = selected, timestamp_s
        self.evaluator_states.append(list(selected))
        return tuple(self.rn.latlon(i) for i in selected)

    def protect_run(self, points):
        run = super().protect_run(points)
        params = {**dict(run.transcript.public_parameters),
                  'selector': 'global_greedy_weighted_POI_union_partition_matroid',
                  'prior_only_control': self.prior_only,
                  'public_poi_context_sha256': self.belief_model.context.sha256,
                  'coverage_target': 'fixed_training_prior' if self.prior_only else 'protected_history_belief',
                  'tie_break': 'minimum_movement_then_fixed_prior_center_then_track_state',
                  'coordinate_uniqueness_required': False}
        return replace(run, transcript=replace(run.transcript, public_parameters=params))
