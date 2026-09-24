"""Evaluator-only weighted-union oracle under fixed per-track feasible sets.

This may consume true-reference weights. Never call it from a private defender.
Reports the solver's feasible lower bound and certified upper bound separately.
"""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


def solve(groups, signatures, access, weights, *, time_limit_s=10.):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0) or weights[-1] != 0:
        raise ValueError('Nonnegative weights with a zero sentinel required')
    positive = np.flatnonzero(weights[:-1] > 0)
    npoi = len(positive); index = {int(p): i for i, p in enumerate(positive)}
    compressed, masks = [], []
    for group in groups:
        ids = np.array(sorted(set(map(int, group))), dtype=int)
        if not len(ids):
            raise ValueError('Each track needs a feasible state')
        mask = np.zeros((len(ids), npoi), dtype=bool)
        for j, state in enumerate(ids):
            for p in np.unique(signatures[access[state]]):
                if int(p) in index:
                    mask[j, index[int(p)]] = True
        _, keep = np.unique(mask, axis=0, return_index=True)
        compressed.append(ids[keep]); masks.append(mask[keep])
    mask = np.concatenate(masks); nstate = len(mask)
    costs = np.r_[np.zeros(nstate), -weights[positive]]
    rows, cols, values = [], [], []
    offset = 0
    for j, group in enumerate(compressed):
        rows.extend([j]*len(group)); cols.extend(range(offset, offset+len(group))); values.extend([1.]*len(group))
        offset += len(group)
    ri, ci = np.nonzero(mask)
    rows.extend((len(groups)+ci).tolist()); cols.extend(ri.tolist()); values.extend([-1.]*len(ri))
    rows.extend(range(len(groups), len(groups)+npoi)); cols.extend(range(nstate, nstate+npoi)); values.extend([1.]*npoi)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(groups)+npoi, nstate+npoi)).tocsc()
    constraint = LinearConstraint(matrix, np.r_[np.ones(len(groups)), np.full(npoi, -np.inf)],
                                  np.r_[np.ones(len(groups)), np.zeros(npoi)])
    result = milp(costs, integrality=np.ones(len(costs)), bounds=Bounds(0, 1), constraints=constraint,
                  options={'time_limit': float(time_limit_s), 'mip_rel_gap': 0., 'presolve': True})
    union_bound = float(weights[positive][mask.any(axis=0)].sum())
    upper = min(union_bound, -float(result.mip_dual_bound)) if getattr(result, 'mip_dual_bound', None) is not None else union_bound
    selected, covered, offset = [], np.zeros(npoi, dtype=bool), 0
    for group, coverage in zip(compressed, masks):
        at = int(np.argmax(result.x[offset:offset+len(group)])) if result.x is not None else 0
        selected.append(int(group[at])); covered |= coverage[at]; offset += len(group)
    value = float(weights[positive][covered].sum())
    if value > upper+1e-7:
        raise RuntimeError('Oracle feasible result exceeds its reported upper bound')
    return {'selected_states': selected, 'feasible_value': value, 'upper_bound': max(value, upper),
            'optimal': bool(result.status == 0 and abs(value-upper) < 1e-7),
            'solver_status': int(result.status), 'solver_message': str(result.message),
            'full_feasible_counts': list(map(len, groups)), 'compressed_counts': list(map(len, compressed)),
            'positive_reference_pois': npoi, 'evaluator_only_uses_reference_weights': True}
