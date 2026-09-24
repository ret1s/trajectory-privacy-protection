"""Two-slice service-cover planning from protected belief and public roads.

Only current coordinates are released. Forecasts and finite shortest-path pools
are approximations. The current objective floor is conditional on this step's
history and does not guarantee real-user utility or future improvement.
"""
from collections import OrderedDict
from dataclasses import replace
import math
import numpy as np
from scipy.sparse.csgraph import dijkstra
from benchmark.engines.empirical_paced import EmpiricalPacedProgressLaneDummy
from benchmark.engines.fair_cover import CoverageObjective, exchange_refine
from benchmark.engines.quotient_cover import reduce_groups
from benchmark.engines.service_cover import greedy_cover


class TwoSliceObjective:
    def __init__(self, current, future, pairs):
        self.current, self.future = current, future
        self.pairs = np.asarray(pairs, dtype=int)
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError('Pairs of current/future states required')

    def value(self, selected):
        p = self.pairs[np.asarray(selected, dtype=int)]
        return .5*(self.current.value(p[:, 0])+self.future.value(p[:, 1]))

    def marginal(self, ids, selected):
        p = self.pairs[np.asarray(ids, dtype=int)]
        chosen = self.pairs[np.asarray(selected, dtype=int)]
        return .5*(self.current.marginal(p[:, 0], chosen[:, 0])+
                   self.future.marginal(p[:, 1], chosen[:, 1]))


class LookaheadCoverLaneDummy(EmpiricalPacedProgressLaneDummy):
    name = 'lookahead_cover_lane_dummy'

    def __init__(self, rn, *, lookahead_s=120., current_objective_slack=.03,
                 forecast_belief=True, **kwargs):
        if (not math.isfinite(lookahead_s) or lookahead_s < 0 or
                not math.isfinite(current_objective_slack) or not 0 <= current_objective_slack <= .1):
            raise ValueError('Nonnegative public horizon and objective slack in [0,.1] required')
        self.lookahead_s = float(lookahead_s)
        self.current_objective_slack = float(current_objective_slack)
        self.forecast_belief = bool(forecast_belief)
        self.path_cache = OrderedDict()
        super().__init__(rn, **kwargs)
        if self.category_cap is not None:
            raise ValueError('Two-slice planner currently uses mean weighted-union coverage')

    def _path_to_goal(self, goal):
        goal = int(goal)
        if goal not in self.path_cache:
            self.path_cache[goal] = dijkstra(self.progress_reverse, directed=True,
                                             indices=goal, return_predecessors=True)
            if len(self.path_cache) > 32:
                self.path_cache.popitem(last=False)
        self.path_cache.move_to_end(goal)
        return self.path_cache[goal]

    def _to_goal(self, goal):
        return self._path_to_goal(goal)[0]

    def _advance(self, start, goal, seconds):
        distances, predecessor = self._path_to_goal(goal)
        at, goal = int(start), int(goal)
        if not np.isfinite(distances[at]):
            return at
        elapsed = 0.; visited = set()
        while at != goal:
            if at in visited:
                raise RuntimeError('Cycle in public shortest-path predecessor chain')
            visited.add(at)
            nxt = int(predecessor[at])
            if nxt < 0 or not self.rn.graph.has_edge(at, nxt):
                raise RuntimeError('Invalid directed predecessor chain')
            edge = self.rn.graph[at][nxt]; cost = edge['length']/edge['speed']
            if elapsed+cost > seconds+1e-10:
                break
            elapsed += cost; at = nxt
        return at

    def postprocess(self, anchor, timestamp_s):
        previous, previous_t = self.previous, self.last_t
        original = super().postprocess(anchor, timestamp_s)
        if previous is None or self.lookahead_s == 0:
            return original
        baseline = list(self.previous); ctx = self.belief_model.context
        belief = self.belief.weights
        future_belief = (self.belief_model.predict(belief, self.lookahead_s)
                         if self.forecast_belief else belief)
        now = CoverageObjective(ctx.signatures, ctx.access,
            np.asarray(belief @ self.belief_model.poi_weights).ravel())
        future = CoverageObjective(ctx.signatures, ctx.access,
            np.asarray(future_belief @ self.belief_model.poi_weights).ravel())
        center = future_belief @ self.belief_model.xy
        def global_ties(j, ids):
            return np.linalg.norm(self.rn.xy[ids]-center, axis=1), np.zeros(len(ids))
        distinct = reduce_groups([self.viable_ids], self.service_profiles, global_ties)[0]
        future_goals, _ = greedy_cover([distinct]*self.k, future.marginal, global_ties)
        future_goals, _ = exchange_refine([distinct]*self.k, future_goals, future,
                                           global_ties, self.max_exchanges)
        goals = sorted(set(future_goals+self.evaluator_objective[-1]['progress_goals']))
        pairs, groups, baseline_actions = [], [], []
        dt = timestamp_s-previous_t
        for prior, base in zip(previous, baseline):
            local = {(int(base), int(base)), (int(prior), int(prior))}
            for goal in goals:
                current = self._advance(prior, goal, dt)
                local.add((current, self._advance(current, goal, self.lookahead_s)))
                local.add((int(base), self._advance(base, goal, self.lookahead_s)))
            local = sorted(local)
            baseline_actions.append(len(pairs)+local.index((int(base), int(base))))
            groups.append(np.arange(len(pairs), len(pairs)+len(local), dtype=int))
            pairs.extend(local)
        objective = TwoSliceObjective(now, future, pairs)
        def ties(j, ids):
            q = objective.pairs[ids, 0]
            return (np.linalg.norm(self.rn.xy[q]-self.rn.xy[previous[j]], axis=1),
                    np.linalg.norm(self.rn.xy[q]-center, axis=1))
        proposal, _ = greedy_cover(groups, objective.marginal, ties)
        proposal, history = exchange_refine(groups, proposal, objective, ties, self.max_exchanges)
        proposed_states = objective.pairs[proposal, 0].tolist()
        before = now.value(baseline); proposed_value = now.value(proposed_states)
        accepted = (proposed_value >= before-self.current_objective_slack-1e-12 and
                    objective.value(proposal) >= objective.value(baseline_actions)-1e-12)
        chosen = proposal if accepted else baseline_actions
        states = objective.pairs[chosen, 0].tolist()
        assert all(s in self.travel.reachable(p, dt) for p, s in zip(previous, states))
        assert now.value(states) >= before-self.current_objective_slack-1e-12
        self.evaluator_objective[-1].update(lookahead_accepted=bool(accepted),
            lookahead_pair_counts=list(map(len, groups)), lookahead_s=self.lookahead_s,
            current_objective_before_lookahead=before, current_objective_after_lookahead=now.value(states),
            proposed_current_objective=proposed_value, lookahead_forecast=self.forecast_belief,
            future_objective_selected=future.value(objective.pairs[chosen, 1]),
            two_slice_value=objective.value(chosen), two_slice_exchange_history=history,
            current_objective_slack=self.current_objective_slack,
            lookahead_future_states=objective.pairs[chosen, 1].tolist())
        self.previous = states; self.evaluator_states[-1] = states.copy()
        return tuple(self.rn.latlon(i) for i in states)

    def protect_run(self, points):
        run = super().protect_run(points); params = dict(run.transcript.public_parameters)
        params.update(selector='two_slice_finite_shortest_path_cover_with_current_objective_floor',
            lookahead_s=self.lookahead_s, current_objective_slack=self.current_objective_slack,
            forecast_belief=self.forecast_belief, true_future_coordinates_used=False)
        return replace(run, transcript=replace(run.transcript, public_parameters=params))
