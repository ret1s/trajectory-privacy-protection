"""Coverage with an optional protected-anchor directed-distance corridor.

The corridor is computed from public/protected data before greedy selection.
It is not a raw-GPS radius restriction or a guarantee about future trajectories.
"""
from dataclasses import replace
import math

import numpy as np

from benchmark.engines.service_cover import ServiceCoverLaneDummy, greedy_cover
from evaluation.lane_travel import matrix


def corridor(ids, potential, slack_m):
    ids = np.asarray(ids, dtype=int)
    potential = np.asarray(potential, dtype=float)
    if ids.ndim != 1 or not len(ids) or potential.shape != ids.shape:
        raise ValueError('Nonempty matched candidate/potential arrays required')
    if not np.isfinite(potential).all() or np.any(potential < 0):
        raise ValueError('Finite nonnegative directed distances required')
    if not math.isfinite(slack_m) or slack_m < 0:
        raise ValueError('Finite nonnegative corridor slack required')
    return ids[potential <= potential.min() + slack_m]


class RecoveryCoverLaneDummy(ServiceCoverLaneDummy):
    name = 'recovery_cover_lane_dummy'

    def __init__(self, rn, *, prior_kind, corridor_m=None, **kwargs):
        if prior_kind not in {'training_occupancy', 'uniform_public_cells'}:
            raise ValueError('Explicit public prior kind required')
        if corridor_m is not None and (not math.isfinite(corridor_m) or corridor_m < 0):
            raise ValueError('Finite nonnegative corridor slack required')
        if kwargs.get('prior_only', False):
            raise ValueError('This candidate requires protected-anchor filtering')
        self.prior_kind, self.corridor_m = prior_kind, corridor_m
        super().__init__(rn, **kwargs)
        if prior_kind == 'uniform_public_cells' and not np.allclose(
                self.belief_model.prior, 1. / len(self.belief_model.prior), rtol=1e-12, atol=1e-15):
            raise ValueError('Uniform public-cell model does not match prior kind')
        self.reverse = matrix(rn).transpose().tocsr() if corridor_m is not None else None

    def postprocess(self, anchor, timestamp_s):
        if self.corridor_m is None:
            return super().postprocess(anchor, timestamp_s)
        weights = self.belief.update(anchor, timestamp_s, observed=self.n < self.horizon)
        poi_weights = np.asarray(weights @ self.belief_model.poi_weights).ravel()
        context = self.belief_model.context
        potential = self.distances_to_goal(np.asarray(self.rn.point_xy(*anchor)))
        groups, before, lower = [], [], []
        for j in range(self.k):
            if self.previous is None:
                ids = self.viable_ids
            else:
                reached = self.travel.reachable(self.previous[j], timestamp_s - self.last_t)
                ids = np.array(sorted(i for i in reached if self.viable[i]), dtype=int)
            before.append(len(ids)); lower.append(float(potential[ids].min()))
            groups.append(corridor(ids, potential[ids], self.corridor_m))

        def ties(j, ids):
            movement = (np.zeros(len(ids)) if self.previous is None else
                        np.linalg.norm(self.rn.xy[ids] - self.rn.xy[self.previous[j]], axis=1))
            return movement, np.linalg.norm(self.rn.xy[ids] - self.public_center, axis=1)

        selected, gains = greedy_cover(groups,
            lambda ids, chosen: context.marginal_gain(ids, poi_weights, chosen), ties)
        self.evaluator_objective.append({'greedy_gains': gains, 'value': float(sum(gains)),
            'reachable_counts': before, 'corridor_counts': list(map(len, groups)),
            'minimum_goal_distance_m': lower,
            'selected_goal_distance_m': potential[selected].tolist()})
        self.previous, self.last_t = selected, timestamp_s
        self.evaluator_states.append(list(selected))
        return tuple(self.rn.latlon(i) for i in selected)

    def protect_run(self, points):
        run = super().protect_run(points)
        params = dict(run.transcript.public_parameters)
        # These inherited fields are inactive in joint coverage. Remove them only
        # for the new method; preceding frozen transcripts are left untouched.
        for field in ('offset_m', 'temperature_m', 'route_weight', 'coverage_weight', 'center_mode'):
            params.pop(field, None)
        params.update(prior_kind=self.prior_kind, corridor_slack_m=self.corridor_m,
            corridor_reference='directed_distance_to_nearest_viable_protected_anchor_state',
            coverage_target='approximate_protected_history_belief',
            motion_model='directed_lane_progress_no_lane_changes_free_flow')
        return replace(run, transcript=replace(run.transcript, public_parameters=params))
