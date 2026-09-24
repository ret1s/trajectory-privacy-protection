"""Location-wise service saturation under the unchanged private-anchor stream."""
from dataclasses import replace
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from benchmark.capped_service_objective import CappedServiceIndex, CappedServiceObjective
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.engines.fair_cover import exchange_refine
from benchmark.engines.quotient_cover import reduce_groups
from benchmark.engines.service_cover import greedy_cover


class CappedServiceLaneDummy(PacedSlackProgressLaneDummy):
    name = 'capped_service_lane_dummy'

    def __init__(self, rn, *, location_service_cap=.9, **kwargs):
        if location_service_cap is not None and (not np.isfinite(location_service_cap) or not 0 < location_service_cap <= 1):
            raise ValueError('Cap must be None or in (0,1]')
        self.location_service_cap = location_service_cap
        super().__init__(rn, **kwargs)
        start = time.perf_counter()
        context = self.belief_model.context
        self.capped_index = (CappedServiceIndex(context.signatures, context.access, self.belief_model.poi_weights)
                             if location_service_cap is not None else None)
        self.capped_index_build_ms = (time.perf_counter()-start)*1000

    def postprocess(self, anchor, timestamp_s):
        if self.location_service_cap is None:
            return super().postprocess(anchor, timestamp_s)
        belief = self.belief.update(anchor, timestamp_s)
        context = self.belief_model.context
        objective = CappedServiceObjective(self.capped_index, belief, self.location_service_cap)
        previous, previous_t = self.previous, self.last_t
        groups = []
        for j in range(self.k):
            ids = self.viable_ids if previous is None else np.array(sorted(
                i for i in self.travel.reachable(previous[j], timestamp_s-previous_t) if self.viable[i]), dtype=int)
            groups.append(ids)
        def ties(j, ids):
            movement = np.zeros(len(ids)) if previous is None else np.linalg.norm(self.rn.xy[ids]-self.rn.xy[previous[j]], axis=1)
            return movement, np.linalg.norm(self.rn.xy[ids]-self.public_center, axis=1)
        reduced = reduce_groups(groups, self.service_profiles, ties)
        greedy, gains = greedy_cover(reduced, objective.marginal, ties)
        selected, history = exchange_refine(reduced, greedy, objective, ties, self.max_exchanges)
        base = list(selected); goals = []
        if previous is not None:
            center = belief @ self.belief_model.xy
            def global_ties(j, ids):
                return np.linalg.norm(self.rn.xy[ids]-center, axis=1), np.zeros(len(ids))
            distinct = reduce_groups([self.viable_ids], self.service_profiles, global_ties)[0]
            goals, _ = greedy_cover([distinct]*self.k, objective.marginal, global_ties)
            goals, _ = exchange_refine([distinct]*self.k, goals, objective, global_ties, self.max_exchanges)
            distances = [self._to_goal(goal) for goal in goals]
            rr, cc = linear_sum_assignment(np.array([[d[s] for d in distances] for s in previous]))
            assignment = dict(zip(rr, cc)); goals = [goals[assignment[j]] for j in range(self.k)]
            for j in range(self.k):
                ids = groups[j][self.service_profiles[groups[j]] == self.service_profiles[base[j]]]
                distance = distances[assignment[j]][ids]
                movement = np.linalg.norm(self.rn.xy[ids]-self.rn.xy[previous[j]], axis=1)
                selected[j] = int(ids[np.lexsort((ids, movement, distance))[0]])
            assert np.array_equal(context.signatures[context.access[base]], context.signatures[context.access[selected]])
        before_slack = list(selected); before_value = objective.value(selected)
        floor = before_value-self.utility_slack
        if previous is not None and self.utility_slack:
            for j in range(self.k):
                others = selected[:j]+selected[j+1:]
                values = objective.value(others)+objective.marginal(groups[j], others)
                ids = groups[j][values >= floor-1e-12]
                distance = self._to_goal(goals[j])
                movement = np.linalg.norm(self.rn.xy[ids]-self.rn.xy[previous[j]], axis=1)
                proposal = int(ids[np.lexsort((ids, movement, distance[ids]))[0]])
                proposed = selected.copy(); proposed[j] = proposal
                if distance[proposal] < distance[selected[j]]-1e-12 and objective.value(proposed) >= floor-1e-12:
                    selected = proposed
        value = objective.value(selected)
        assert value >= floor-1e-12
        assert all(s in g for s, g in zip(selected, groups))
        self.evaluator_objective.append({'value': value, 'location_service_cap': self.location_service_cap,
            'greedy_states': greedy, 'greedy_gains': gains, 'objective_history': history,
            'states_before_progress': base, 'states_before_slack': before_slack, 'states_after_slack': selected,
            'progress_goals': goals, 'objective_before_slack': before_value, 'objective_after_slack': value,
            'objective_loss': before_value-value, 'reachable_counts': list(map(len, groups)),
            'quotient_counts': list(map(len, reduced))})
        self.previous, self.last_t = selected, timestamp_s
        self.evaluator_states.append(list(selected))
        return tuple(self.rn.latlon(i) for i in selected)

    def protect_run(self, points):
        run = super().protect_run(points); params = dict(run.transcript.public_parameters)
        params.update(location_service_cap=self.location_service_cap,
                      coverage_target='expected_locationwise_capped_top5_recall_from_topL_union')
        return replace(run, transcript=replace(run.transcript, public_parameters=params))
