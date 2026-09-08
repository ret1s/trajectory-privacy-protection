"""Broader empirical shadow data and data-only ExtraTrees inference.

max_features=1 is the declared integer: ONE randomly sampled feature at a split,
not the floating-point fraction 1.0. No fitting or tuning is performed at query.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from evaluation.loss_aware_shadow import predict as loss_predict
from evaluation.service_shadow import predict as knn_predict

TREE_PARAMS = dict(n_estimators=128, min_samples_leaf=5, max_depth=18,
                   max_features=1, bootstrap=False, random_state=20260908, n_jobs=1)


def centroid(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 21 or not np.isfinite(x).all():
        raise ValueError('Finite K=5 causal features required')
    return x[:, :10].reshape(-1, 5, 2).mean(axis=1) * 1000.


def fit_trees(model):
    x, y = np.asarray(model['x']), np.asarray(model['y'])
    centered = (x - model['mean']) / model['scale']
    arrays = {}
    for mode, target in (('direct', y), ('residual', y - centroid(x))):
        forest = ExtraTreesRegressor(**TREE_PARAMS).fit(centered, target)
        for i, estimator in enumerate(forest.estimators_):
            t = estimator.tree_
            prefix = f'{mode}/{i}/'
            for name, value in (('left', t.children_left), ('right', t.children_right),
                                ('feature', t.feature), ('threshold', t.threshold), ('value', t.value[:, :, 0])):
                arrays[prefix + name] = value.copy()
        # Test our data-only inference against sklearn before persisting trees.
        actual = tree_predict(arrays, centered, mode)
        assert np.allclose(actual, forest.predict(centered), rtol=0, atol=1e-10)
    return arrays


def tree_predict(arrays, x, mode):
    x = np.asarray(x, dtype=np.float32)  # sklearn tree inference uses float32.
    if x.ndim != 2 or not np.isfinite(x).all() or mode not in ('direct', 'residual'):
        raise ValueError('Finite feature matrix and known forest mode required')
    result = np.zeros((len(x), 2))
    for i in range(TREE_PARAMS['n_estimators']):
        prefix = f'{mode}/{i}/'
        left, right = arrays[prefix + 'left'], arrays[prefix + 'right']
        features, thresholds = arrays[prefix + 'feature'], arrays[prefix + 'threshold']
        nodes = np.zeros(len(x), dtype=int)
        for _ in range(TREE_PARAMS['max_depth'] + 1):
            active = np.flatnonzero(left[nodes] != -1)
            if not len(active):
                break
            n = nodes[active]
            take_left = x[active, features[n]] <= thresholds[n]
            nodes[active] = np.where(take_left, left[n], right[n])
        assert np.all(left[nodes] == -1)
        result += arrays[prefix + 'value'][nodes]
    return result / TREE_PARAMS['n_estimators']


def predict(model, arrays, x):
    x = np.asarray(x, dtype=float)
    center = centroid(x)
    if len(x) == 0:
        raise ValueError('Nonempty feature matrix required')
    result = {f'expanded_knn_{k}': knn_predict(model, x, k) for k in (1, 5, 15)}
    extra, _ = loss_predict(model, x)
    result.update({'expanded_' + key: value for key, value in extra.items()})
    z = (x - model['mean']) / model['scale']
    result['expanded_tree_direct'] = tree_predict(arrays, z, 'direct')
    result['expanded_tree_residual'] = tree_predict(arrays, z, 'residual') + center
    return result
