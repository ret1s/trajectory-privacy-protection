"""Category-saturated reachable coverage and bounded single-track refinement.

All selection inputs are public or already protected. Saturation/local search
are established optimization tools, not new privacy guarantees.
"""
from dataclasses import replace

import numpy as np

from benchmark.engines.service_cover import ServiceCoverLaneDummy, greedy_cover


class CoverageObjective:
    """Weighted union, optionally truncated after category normalization.

    signatures: (state, category, slot), with -1 for absent POIs. A POI must
    occur only once per state's response and belong to a single category.
    weights includes a final zero sentinel, as PublicPoiContext does.
    """
    def __init__(self, signatures, access, weights, cap=None, category_ids=None):
        self.signatures = np.asarray(signatures, dtype=int)
        self.access = np.asarray(access, dtype=int)
        self.weights = np.asarray(weights, dtype=float).copy()
        if (self.signatures.ndim != 3 or self.access.ndim != 1
                or self.weights.ndim != 1 or not len(self.weights)
                or not np.isfinite(self.weights).all() or np.any(self.weights < 0)
                or self.weights[-1] != 0 or np.any(self.signatures < -1)
                or np.any(self.signatures >= len(self.weights) - 1)
                or np.any(self.access < 0) or np.any(self.access >= len(self.signatures))):
            raise ValueError('Valid public signatures, access and zero-sentinel weights required')
        if cap is not None and (not np.isfinite(cap) or not 0 < cap <= 1):
            raise ValueError('Category cap must be in (0,1]')
        self.cap = cap
        self.category_ids = (category_ids if category_ids is not None else
                             [np.unique(self.signatures[:, c][self.signatures[:, c] >= 0])
                              for c in range(self.signatures.shape[1])])
        # Category memberships are disjoint in the public service.
        flat = np.concatenate(self.category_ids)
        if len(flat) != len(np.unique(flat)):
            raise ValueError('A POI cannot belong to multiple categories')
        self.mass = np.array([self.weights[ids].sum() for ids in self.category_ids])
        self.active = self.mass > 0
        if cap is not None:
            for ids, mass in zip(self.category_ids, self.mass):
                if mass > 0:
                    self.weights[ids] /= mass

    def covered(self, selected):
        mask = np.zeros(len(self.weights), dtype=bool)
        if len(selected):
            ids = self.signatures[self.access[np.asarray(selected, dtype=int)]]
            mask[ids[ids >= 0]] = True
        return mask

    def category_values(self, selected):
        covered = self.covered(selected)
        return np.array([self.weights[ids[covered[ids]]].sum() for ids in self.category_ids])

    def value(self, selected):
        values = self.category_values(selected)
        if self.cap is None:
            return float(values.sum())
        return float(np.minimum(values[self.active], self.cap).mean()) if self.active.any() else 0.

    def marginal(self, states, selected):
        remaining = self.weights.copy()
        remaining[self.covered(selected)] = 0.
        ids = self.signatures[self.access[np.asarray(states, dtype=int)]]
        slot_weights = remaining[np.where(ids >= 0, ids, len(remaining) - 1)]
        if self.cap is None:
            # Preserve the old summation order: 1-ulp changes alter exact ties.
            return slot_weights.sum(axis=(1, 2))
        gains = slot_weights.sum(axis=2)
        room = np.maximum(0., self.cap - self.category_values(selected))
        return (np.minimum(gains[:, self.active], room[self.active]).mean(axis=1)
                if self.active.any() else np.zeros(len(states)))


def exchange_refine(groups, selected, objective, tie_cost, max_exchanges=3, tolerance=1e-12):
    """Globally best improving one-track replacement, at most max_exchanges.

    This never claims to reach a local or global optimum at the iteration cap.
    """
    if (int(max_exchanges) != max_exchanges or max_exchanges < 0
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError('Nonnegative integer exchange limit and tolerance required')
    selected = list(map(int, selected))
    if len(selected) != len(groups) or any(s not in g for s, g in zip(selected, groups)):
        raise ValueError('One feasible state per track required')
    history = [objective.value(selected)]
    for _ in range(max_exchanges):
        best = None
        for j, group in enumerate(groups):
            ids = np.asarray(group, dtype=int)
            others = selected[:j] + selected[j+1:]
            scores = objective.value(others) + objective.marginal(ids, others)
            eligible = np.flatnonzero(scores > history[-1] + tolerance)
            if not len(eligible):
                continue
            tied = eligible[scores[eligible] == scores[eligible].max()]
            primary, secondary = tie_cost(j, ids[tied])
            at = int(np.lexsort((ids[tied], secondary, primary))[0])
            q = tied[at]
            key = (-float(scores[q]), float(primary[at]), float(secondary[at]), j, int(ids[q]))
            if best is None or key < best[0]:
                best = key, j, int(ids[q])
        if best is None:
            break
        _, j, state = best
        proposal = selected.copy(); proposal[j] = state
        value = objective.value(proposal)
        # Check the full objective too; an approximate marginal cannot force a
        # numerically non-improving exchange.
        if value <= history[-1] + tolerance:
            break
        selected = proposal; history.append(value)
    return selected, history


class FairCoverLaneDummy(ServiceCoverLaneDummy):
    name = 'fair_cover_lane_dummy'

    def __init__(self, rn, *, category_cap=None, max_exchanges=3, **kwargs):
        if category_cap is not None and (not np.isfinite(category_cap) or not 0 < category_cap <= 1):
            raise ValueError('Category cap must be in (0,1]')
        if int(max_exchanges) != max_exchanges or max_exchanges < 0:
            raise ValueError('Nonnegative integer exchange limit required')
        if kwargs.get('prior_only', False):
            raise ValueError('This candidate uses protected-anchor filtering')
        self.category_cap, self.max_exchanges = category_cap, int(max_exchanges)
        super().__init__(rn, **kwargs)
        context = self.belief_model.context
        self.category_ids = [np.array([i for i, p in enumerate(context.pois) if p['category'] == c], dtype=int)
                             for c in context.categories]

    def postprocess(self, anchor, timestamp_s):
        belief = self.belief.update(anchor, timestamp_s, observed=self.n < self.horizon)
        weights = np.asarray(belief @ self.belief_model.poi_weights).ravel()
        context = self.belief_model.context
        objective = CoverageObjective(context.signatures, context.access, weights,
                                      self.category_cap, self.category_ids)
        groups = []
        for j in range(self.k):
            if self.previous is None:
                ids = self.viable_ids
            else:
                reached = self.travel.reachable(self.previous[j], timestamp_s - self.last_t)
                ids = np.array(sorted(i for i in reached if self.viable[i]), dtype=int)
            groups.append(ids)

        def ties(j, ids):
            movement = (np.zeros(len(ids)) if self.previous is None else
                        np.linalg.norm(self.rn.xy[ids] - self.rn.xy[self.previous[j]], axis=1))
            return movement, np.linalg.norm(self.rn.xy[ids] - self.public_center, axis=1)

        greedy, gains = greedy_cover(groups, objective.marginal, ties)
        selected, history = exchange_refine(groups, greedy, objective, ties, self.max_exchanges)
        self.evaluator_objective.append({'greedy_states': greedy, 'greedy_gains': gains,
            'objective_history': history, 'value': objective.value(selected),
            'category_values': objective.category_values(selected).tolist(),
            'category_mass': objective.mass.tolist(), 'reachable_counts': list(map(len, groups))})
        self.previous, self.last_t = selected, timestamp_s
        self.evaluator_states.append(list(selected))
        return tuple(self.rn.latlon(i) for i in selected)

    def protect_run(self, points):
        run = super().protect_run(points)
        params = dict(run.transcript.public_parameters)
        for key in ('offset_m', 'temperature_m', 'route_weight', 'coverage_weight', 'center_mode'):
            params.pop(key, None)
        params.update(selector='greedy_then_bounded_single_track_exchange',
                      category_cap=self.category_cap, max_exchanges=self.max_exchanges,
                      objective='mean_POI_union' if self.category_cap is None else 'normalized_category_capped_POI_union')
        return replace(run, transcript=replace(run.transcript, public_parameters=params))
