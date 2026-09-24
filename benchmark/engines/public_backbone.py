"""Public fixed queries plus directed residual-service adaptive queries.

Total K and anchor budget are unchanged. The fixed tracks are known public
controls and can be removed by a knowledgeable attacker. This is a utility
robustness hypothesis, not a new privacy primitive or a secrecy argument.
"""
from dataclasses import replace
import numpy as np
from scipy.optimize import linear_sum_assignment
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.engines.fair_cover import CoverageObjective, exchange_refine
from benchmark.engines.quotient_cover import reduce_groups
from benchmark.engines.service_cover import greedy_cover
from evaluation.public_cover import fit_public_cover


class PublicBackboneLaneDummy(PacedSlackProgressLaneDummy):
    name = 'public_backbone_lane_dummy'

    def __init__(self, rn, *, public_queries=2, k=5, **kwargs):
        if isinstance(public_queries, bool) or int(public_queries) != public_queries or not 0 <= public_queries < k:
            raise ValueError('Integer public query count must lie in [0,K)')
        self.public_queries = int(public_queries)
        super().__init__(rn, k=k, **kwargs)
        self.fixed_plan = (fit_public_cover(rn, self.belief_model,
                           self.belief_model.context, self.public_queries) if self.public_queries else None)
        self.fixed_states = self.fixed_plan['states'] if self.fixed_plan else []

    def postprocess(self, anchor, timestamp_s):
        if not self.public_queries:
            return super().postprocess(anchor, timestamp_s)
        weights = self.belief.update(anchor, timestamp_s)
        poi_weights = np.asarray(weights @ self.belief_model.poi_weights).ravel()
        context = self.belief_model.context
        objective = CoverageObjective(context.signatures, context.access, poi_weights)
        # Condition on the public backbone; its contribution is constant over
        # all moving-query choices, so do not spend capacity covering it twice.
        residual_weights = poi_weights.copy()
        residual_weights[objective.covered(self.fixed_states)] = 0.
        residual = CoverageObjective(context.signatures, context.access, residual_weights)
        previous = self.previous
        moving_previous = previous[self.public_queries:] if previous is not None else None
        count = self.k-self.public_queries
        groups = []
        for j in range(count):
            ids = (self.viable_ids if previous is None else np.array(sorted(
                i for i in self.travel.reachable(moving_previous[j], timestamp_s-self.last_t) if self.viable[i]), dtype=int))
            groups.append(ids)
        def ties(j, ids):
            movement = (np.zeros(len(ids)) if previous is None else
                        np.linalg.norm(self.rn.xy[ids]-self.rn.xy[moving_previous[j]], axis=1))
            return movement, np.linalg.norm(self.rn.xy[ids]-self.public_center, axis=1)
        reduced = reduce_groups(groups, self.service_profiles, ties)
        greedy, gains = greedy_cover(reduced, residual.marginal, ties)
        selected, history = exchange_refine(reduced, greedy, residual, ties, self.max_exchanges)
        base = list(selected); goals = []
        if previous is not None:
            center = weights @ self.belief_model.xy
            def global_ties(j, ids):
                return np.linalg.norm(self.rn.xy[ids]-center, axis=1), np.zeros(len(ids))
            distinct = reduce_groups([self.viable_ids], self.service_profiles, global_ties)[0]
            goals, _ = greedy_cover([distinct]*count, residual.marginal, global_ties)
            goals, _ = exchange_refine([distinct]*count, goals, residual, global_ties, self.max_exchanges)
            distances = [self._to_goal(goal) for goal in goals]
            rr, cc = linear_sum_assignment(np.array([[d[s] for d in distances] for s in moving_previous]))
            assignment = dict(zip(rr, cc)); goals = [goals[assignment[j]] for j in range(count)]
            for j in range(count):
                ids = groups[j][self.service_profiles[groups[j]] == self.service_profiles[base[j]]]
                distance = distances[assignment[j]][ids]
                movement = np.linalg.norm(self.rn.xy[ids]-self.rn.xy[moving_previous[j]], axis=1)
                selected[j] = int(ids[np.lexsort((ids, movement, distance))[0]])
            assert np.array_equal(context.signatures[context.access[base]], context.signatures[context.access[selected]])
        before_slack = list(selected)
        before_value = objective.value(self.fixed_states+selected)
        floor = before_value-self.utility_slack
        if previous is not None and self.utility_slack:
            for j in range(count):
                others = selected[:j]+selected[j+1:]
                values = residual.value(others)+residual.marginal(groups[j], others)
                residual_floor = residual.value(before_slack)-self.utility_slack
                ids = groups[j][values >= residual_floor-1e-12]
                distance = self._to_goal(goals[j])
                movement = np.linalg.norm(self.rn.xy[ids]-self.rn.xy[moving_previous[j]], axis=1)
                proposal = int(ids[np.lexsort((ids, movement, distance[ids]))[0]])
                proposed = selected.copy(); proposed[j] = proposal
                if distance[proposal] < distance[selected[j]]-1e-12 and objective.value(self.fixed_states+proposed) >= floor-1e-12:
                    selected = proposed
        full = list(self.fixed_states)+selected
        after_value = objective.value(full)
        assert after_value >= floor-1e-12
        if previous is not None:
            assert all(s in g for s, g in zip(selected, groups))
        self.evaluator_objective.append({'value': after_value,
            'public_backbone_states': list(self.fixed_states), 'backbone_value': objective.value(self.fixed_states),
            'residual_greedy_states': greedy, 'residual_greedy_gains': gains,
            'residual_objective_history': history, 'moving_states_before_progress': base,
            'moving_states_before_slack': before_slack, 'moving_states_after_slack': selected,
            'moving_progress_goals': goals, 'objective_before_slack': before_value,
            'objective_after_slack': after_value, 'objective_loss': before_value-after_value,
            'reachable_counts': list(map(len, groups)), 'quotient_counts': list(map(len, reduced))})
        self.previous, self.last_t = full, timestamp_s
        self.evaluator_states.append(list(full))
        return tuple(self.rn.latlon(i) for i in full)

    def protect_run(self, points):
        run = super().protect_run(points)
        params = dict(run.transcript.public_parameters)
        params.update(public_backbone_queries=self.public_queries, adaptive_queries=self.k-self.public_queries,
                      public_backbone_states=','.join(map(str, self.fixed_states)),
                      selector='public_backbone_conditioned_residual_cover_with_directed_progress',
                      static_tracks_secret=False)
        return replace(run, transcript=replace(run.transcript, public_parameters=params))
