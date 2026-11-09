"""One predeclared family-level rule for the frozen endpoint attack bank.

Cross-fit and ordinary selection predictions are supplied by the runner. Each
validation family contributes equally, regardless of its number of repetitions.
This conservative criterion is an empirical selector, not a privacy guarantee.
"""
import math

import numpy as np


def family_statistics(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('At least two finite family summaries required')
    mean = float(values.mean())
    se = float(values.std(ddof=1)/math.sqrt(len(values)))
    return dict(mean=mean, se=se, families=len(values))


def robust_select(rows, expected_families):
    """Minimize MAE mean+SE; maximize Hit mean-SE; lexical ties only."""
    expected = sorted(expected_families)
    if len(expected) < 2 or len(set(expected)) != len(expected):
        raise ValueError('Distinct declared validation families required')
    if not rows or sorted({r['family_id'] for r in rows}) != expected:
        raise ValueError('Every declared family must be present, without extras')
    identifiers = [(r['family_id'], r['session_id'], r['seed']) for r in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Repeated family/session/rep validation prediction')
    names = set(rows[0]['errors'])
    if not names or any(set(r['errors']) != names for r in rows):
        raise ValueError('Complete identical candidate bank required for every validation prediction')
    statistics = {}
    for name in sorted(names):
        values = np.asarray([r['errors'][name] for r in rows], float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Finite nonnegative endpoint errors required')
        grouped = [np.asarray([r['errors'][name] for r in rows if r['family_id']==family], float)
                   for family in expected]
        mae = family_statistics([v.mean() for v in grouped])
        mae['objective'] = mae['mean']+mae['se']
        mae['family_values'] = dict(zip(expected, [float(v.mean()) for v in grouped]))
        statistics[name] = {'mae': mae}
        for radius in (50, 100, 200, 500):
            hit = family_statistics([np.mean(v <= radius) for v in grouped])
            hit['objective'] = hit['mean']-hit['se']
            hit['family_values'] = dict(zip(expected, [float(np.mean(v <= radius)) for v in grouped]))
            statistics[name]['hit'+str(radius)] = hit
    # Numerical comparison uses a predeclared precision; otherwise identical
    # hit distributions can break a lexical tie through floating-point order.
    selected = {'mae': min(names, key=lambda n: (round(statistics[n]['mae']['objective'], 12), n))}
    for radius in (50, 100, 200, 500):
        key = 'hit'+str(radius)
        selected[key] = min(names, key=lambda n: (-round(statistics[n][key]['objective'], 12), n))
    return selected, statistics
