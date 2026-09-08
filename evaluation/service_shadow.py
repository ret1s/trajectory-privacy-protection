"""Causal public-output shadow kNN; training and selection are separate roles."""
import numpy as np
from scipy.spatial.distance import cdist


def features(public, rn):
    # Stable candidate index is public in the dummy-only output contract.
    rows = np.array([[v for c in e['candidates'] for v in rn.point_xy(c['lat'], c['lon'])]
                     for e in public['events']], dtype=float) / 1000.
    means = np.cumsum(rows, axis=0) / np.arange(1, len(rows) + 1)[:, None]
    times = np.array([e['timestamp_s'] for e in public['events']])
    return np.c_[rows, means, (times - times[0]) / 60.]


def fit(x, y, provenance):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or not len(x) or y.shape != (len(x), 2):
        raise ValueError('Nonempty training feature matrix and XY targets required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite training data required')
    scale = np.std(x, axis=0)
    scale[scale < 1e-12] = 1.
    return {'x': x.tolist(), 'y': y.tolist(), 'mean': x.mean(axis=0).tolist(),
            'scale': scale.tolist(), 'provenance': provenance}


def predict(model, x, neighbours):
    if neighbours not in (1, 5, 15):
        raise ValueError('Use prespecified neighbour counts')
    mean, scale = np.asarray(model['mean']), np.asarray(model['scale'])
    distance = cdist((np.asarray(x) - mean) / scale,
                     (np.asarray(model['x']) - mean) / scale)
    # Stable training row ordering breaks equal-distance ties reproducibly.
    ids = np.argsort(distance, axis=1, kind='stable')[:, :min(neighbours, len(model['x']))]
    return np.asarray(model['y'])[ids].mean(axis=1)
