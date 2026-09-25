"""Exploratory S10.B raw-prefix attacks, trained on B labels only.

Prediction accepts the permitted coordinate/time stream and the public map.
No session/role ID, hidden duration, future route or destination candidates are
features. Evaluation labels enter fit only through an explicit separate y.
"""
import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors

from evaluation.live_comparison_attacks import public_arrays


def prefix_features(events, rn):
    xy, t = public_arrays(events, rn)
    if any(len(p) != 1 for p in xy):
        raise ValueError('This diagnostic expects raw, single-coordinate events')
    points = np.array([p[0] for p in xy])
    anchor = points[-1]
    # Direction from the last nonzero observed displacement, with a public
    # fixed eastward fallback for stationary windows.
    displacement = anchor - points
    nonzero = np.flatnonzero(np.linalg.norm(displacement, axis=1) > 1.)
    heading = displacement[nonzero[-1]] if len(nonzero) else np.array([1., 0.])
    heading = heading / np.linalg.norm(heading)
    basis = np.array([heading, [-heading[1], heading[0]]]).T
    sampled = np.array([np.interp(np.linspace(0, t[-1], 16), t, points[:, j]) for j in range(2)]).T
    local_path = (sampled - anchor) @ basis / 1000.
    dt = np.diff(t)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    speeds = lengths / dt if len(dt) else np.array([0.])
    scalars = np.r_[np.log1p(t[-1]), np.log1p(len(t)),
                    lengths.sum()/1000., np.quantile(speeds, [0, .25, .5, .75, 1])/10.]
    local = np.r_[local_path.ravel(), scalars]
    absolute = np.r_[anchor/1000., points[0]/1000., heading, local]
    return absolute, local, anchor, basis


class PrefixDestinationAttack:
    """Small predeclared bank: absolute endpoints and local displacements."""
    def __init__(self, events, endpoints, rn):
        self.rn = rn
        features = [prefix_features(e, rn) for e in events]
        endpoints = np.asarray(endpoints)
        self.models = {}
        self.priors = {'b_prior_mean': endpoints.mean(axis=0),
                       'b_prior_median': np.median(endpoints, axis=0)}
        for radius in (100, 200, 500):
            counts = cKDTree(endpoints).query_ball_point(endpoints, radius, return_length=True)
            self.priors[f'b_prior_hit{radius}'] = endpoints[int(np.argmax(counts))]
        for frame, index in [('absolute', 0), ('local', 1)]:
            x = np.array([f[index] for f in features])
            y = endpoints if frame == 'absolute' else np.array([(y-f[2]) @ f[3] for y, f in zip(endpoints, features)])
            mean, scale = x.mean(axis=0), x.std(axis=0)
            scale[scale < 1e-9] = 1.
            z = (x-mean)/scale
            trees = {leaf: ExtraTreesRegressor(n_estimators=128, min_samples_leaf=leaf,
                     max_depth=16, random_state=20260925, n_jobs=1).fit(z, y) for leaf in (1, 2, 4)}
            nn = NearestNeighbors(n_neighbors=min(9, len(y))).fit(z)
            self.models[frame] = mean, scale, y, nn, trees

    def predict(self, events):
        absolute, local, anchor, basis = prefix_features(events, self.rn)
        bank = {name: value.copy() for name, value in self.priors.items()}
        bank['b_last_observed'] = anchor.copy()
        for frame, x in [('absolute', absolute), ('local', local)]:
            mean, scale, y, nn, trees = self.models[frame]
            z = ((x-mean)/scale)[None, :]
            ids = nn.kneighbors(z, return_distance=False)[0]
            estimates = {f'tree_leaf{leaf}': tree.predict(z)[0] for leaf, tree in trees.items()}
            estimates.update({f'knn{k}': y[ids[:k]].mean(axis=0) for k in (1, 3, 5, 9)})
            for name, value in estimates.items():
                bank[f'b_{frame}_{name}'] = value if frame == 'absolute' else anchor + value @ basis.T
        bank.update({name+'_road': self.rn.xy[self.rn.tree.query(value)[1]] for name, value in list(bank.items())})
        return bank
