"""Public-transcript feature boundary and fixed empirical shadow regressors.

These are mechanism-aware empirical attacks, not exact likelihoods. Current
location inference may use the whole allowed observation window (offline S3).
No evaluator labels, original timestamps, route plan or hidden session state
are accepted by the feature API.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesRegressor
from evaluation.loss_aware_shadow import decision

TREE_PARAMS = dict(n_estimators=128, min_samples_leaf=5, max_depth=18,
                   max_features=1.0, bootstrap=False, random_state=20260924, n_jobs=1)


def coordinates(public, rn):
    events = public['events']
    if not events:
        raise ValueError('Nonempty public window required')
    times = np.array([e['timestamp_s'] for e in events], dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Strictly increasing finite public timestamps required')
    xy = np.array([[rn.point_xy(c['lat'], c['lon']) for c in e['candidates']]
                   for e in events], dtype=float)
    if xy.shape != (len(events), 5, 2) or not np.isfinite(xy).all():
        raise ValueError('Finite stable-index K5 public candidate coordinates required')
    return xy, times-times[0]


def current_features(public, rn):
    xy, times = coordinates(public, rn)
    z = xy.reshape(len(xy), 10)/1000.
    past = np.cumsum(z, axis=0)/np.arange(1, len(z)+1)[:, None]
    future = (np.cumsum(z[::-1], axis=0)/np.arange(1, len(z)+1)[:, None])[::-1]
    features = np.c_[z, past, future, times/60., (times[-1]-times)/60.]
    return features, xy.mean(axis=1)


def endpoint_features(public, rn, scenario):
    if scenario not in ('S9', 'S10'):
        raise ValueError('Endpoint task must be S9 or S10')
    xy, times = coordinates(public, rn)
    z = xy.reshape(len(xy), 10)/1000.
    first = z[np.minimum(np.arange(3), len(z)-1)].ravel()
    last = z[np.maximum(0, len(z)-3+np.arange(3))].ravel()
    values = np.r_[first, last, z.mean(axis=0), z.std(axis=0),
                   np.quantile(z, [.25, .5, .75], axis=0).ravel(),
                   times[-1]/60., np.log1p(len(z))]
    return values[None, :], xy[0 if scenario == 'S9' else -1].mean(axis=0, keepdims=True)


class EmpiricalShadow:
    def __init__(self, x, y, centers):
        x, y, centers = map(lambda a: np.asarray(a, dtype=float), (x, y, centers))
        if (x.ndim != 2 or len(x) < 15 or y.shape != (len(x), 2)
                or centers.shape != y.shape or not all(np.isfinite(v).all() for v in (x, y, centers))):
            raise ValueError('Finite shadow training arrays with at least 15 rows required')
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.
        self.x, self.y = (x-self.mean)/self.scale, y
        self.direct = ExtraTreesRegressor(**TREE_PARAMS).fit(self.x, y)
        self.residual = ExtraTreesRegressor(**TREE_PARAMS).fit(self.x, y-centers)

    def predict(self, x, centers, *, decisions=True):
        x, centers = np.asarray(x, dtype=float), np.asarray(centers, dtype=float)
        if (x.ndim != 2 or x.shape[1] != self.x.shape[1] or centers.shape != (len(x), 2)
                or not np.isfinite(x).all() or not np.isfinite(centers).all()):
            raise ValueError('Finite public features and public centroids required')
        standardized = (x-self.mean)/self.scale
        distances = cdist(standardized, self.x)
        order = np.argsort(distances, axis=1, kind='stable')[:, :15]
        support = self.y[order]
        result = {'knn5': support[:, :5].mean(axis=1), 'knn15': support.mean(axis=1),
                  'tree_direct': self.direct.predict(standardized),
                  'tree_residual': self.residual.predict(standardized)+centers}
        if decisions:
            chosen = [decision(p) for p in support]
            result['knn_mae15'] = np.array([d['mae_action'] for d in chosen])
            result['knn_hit100_15'] = np.array([d['hit_action'] for d in chosen])
            for radius in (50, 200, 500):
                result[f'knn_hit{radius}_15'] = np.array([decision(p, radius)['hit_action'] for p in support])
        return result


def road_variants(predictions, rn):
    return {**predictions, **{a+'_road': rn.xy[rn.tree.query(v)[1]] for a, v in predictions.items()}}
