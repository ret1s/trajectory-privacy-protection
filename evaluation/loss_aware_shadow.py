"""Loss-aware decisions from empirical neighbor mass, not exact likelihoods."""
import numpy as np
from scipy.spatial.distance import cdist


def decision(points, radius=100.):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Finite nonempty XY support required')
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Positive finite hit radius required')
    mean = points.mean(axis=0)
    actions = np.vstack([points, mean])
    risk = cdist(actions, points).mean(axis=1)
    mae = actions[np.argmin(risk)]
    pairs = np.array(np.triu_indices(len(points), 1)).T
    delta = points[pairs[:, 1]] - points[pairs[:, 0]]
    lengths = np.linalg.norm(delta, axis=1)
    good = (lengths > 0) & (lengths <= 2 * radius)
    pairs, delta, lengths = pairs[good], delta[good], lengths[good]
    if len(pairs):
        middle = (points[pairs[:, 0]] + points[pairs[:, 1]]) / 2
        normal = np.c_[-delta[:, 1], delta[:, 0]] / lengths[:, None]
        height = np.sqrt(np.maximum(0., radius**2 - (lengths / 2)**2))
        actions = np.vstack([actions, middle + height[:, None] * normal,
                             middle - height[:, None] * normal])
    # Tolerance only for geometric construction; true benchmark Hit uses <=100.
    coverage = (cdist(actions, points) <= radius + 1e-7).mean(axis=1)
    hit = actions[np.argmax(coverage)]
    return {'mean': mean, 'mae_action': mae, 'hit_action': hit,
            'mae_risk': float(risk.min()), 'mean_risk': float(risk[-1]),
            'hit_mass': float(coverage.max()),
            'mean_hit_mass': float(coverage[len(points)])}


def predict(model, features):
    mean, scale = np.asarray(model['mean']), np.asarray(model['scale'])
    x, y, q = np.asarray(model['x']), np.asarray(model['y']), np.asarray(features)
    if (q.ndim != 2 or x.ndim != 2 or q.shape[1] != x.shape[1]
            or mean.shape != (x.shape[1],) or scale.shape != mean.shape
            or not len(x) or y.shape != (len(x), 2)
            or not all(np.isfinite(a).all() for a in (x, y, q, mean, scale))
            or np.any(scale <= 0)):
        raise ValueError('Valid standardized shadow model/features required')
    dist = cdist((q - mean) / scale, (x - mean) / scale)
    ordering = np.argsort(dist, axis=1, kind='stable')
    predictions, audit = {}, {}
    for n in (15, 45):
        ids = ordering[:, :min(n, len(x))]
        rows = [decision(y[i]) for i in ids]
        for key in ('mae_action', 'hit_action'):
            predictions[f'shadow_loss_{key}_{n}'] = np.asarray([r[key] for r in rows])
        if n == 45:
            predictions['shadow_mean_45'] = np.asarray([r['mean'] for r in rows])
        audit[str(n)] = {'neighbors': ids.tolist(),
            **{key: [r[key] for r in rows] for key in ('mae_risk', 'mean_risk', 'hit_mass', 'mean_hit_mass')}}
    return predictions, audit
