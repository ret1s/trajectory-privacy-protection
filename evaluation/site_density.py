"""Empirical same-site density fusion from public-transcript features.

KDE and conditional-independence fusion are approximations, not calibrated
posteriors or a privacy guarantee. Training labels are auxiliary-only.
"""
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


def combine_density(probabilities, prior, mode='product'):
    q, p = np.asarray(probabilities, dtype=float), np.asarray(prior, dtype=float)
    if (q.ndim != 2 or not len(q) or p.shape != (q.shape[1],)
            or not np.isfinite(q).all() or not np.isfinite(p).all()
            or np.any(q <= 0) or np.any(p <= 0)):
        raise ValueError('Positive finite probability arrays required')
    q = q/q.sum(axis=1, keepdims=True); p = p/p.sum()
    if mode == 'pool':
        return q.mean(axis=0)
    if mode != 'product':
        raise ValueError('Known combination mode required')
    score = np.log(q).sum(axis=0)-(len(q)-1)*np.log(p)
    return np.exp(score-logsumexp(score))


class GridDecisions:
    def __init__(self, xy, radii=(50, 100, 200, 500)):
        self.xy = np.asarray(xy, dtype=float)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2 or not len(self.xy) or not np.isfinite(self.xy).all():
            raise ValueError('Finite nonempty XY grid required')
        if not radii or any(not np.isfinite(r) or r <= 0 for r in radii):
            raise ValueError('Positive finite radii required')
        tree = cKDTree(self.xy)
        distances = tree.sparse_distance_matrix(tree, max(radii), output_type='coo_matrix')
        self.hit = {r: csr_matrix((np.ones(np.sum(distances.data <= r)),
                                  (distances.row[distances.data <= r], distances.col[distances.data <= r])),
                                 shape=(len(self.xy), len(self.xy))) for r in radii}

    def predict(self, probability):
        w = np.asarray(probability, dtype=float)
        if w.shape != (len(self.xy),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
            raise ValueError('Finite nonnegative grid mass required')
        w = w/w.sum(); mean = w @ self.xy; current = mean.copy()
        for _ in range(40):
            distance = np.linalg.norm(self.xy-current, axis=1)
            adjusted = w/np.maximum(distance, 1e-9)
            proposal = adjusted @ self.xy/adjusted.sum()
            if np.linalg.norm(proposal-current) < 1e-6:
                current = proposal; break
            current = proposal
        mode = self.xy[np.argmax(w)]
        actions = np.vstack([mean, mode, current])
        best = actions[np.argmin(cdist(actions, self.xy) @ w)]
        result = {'mean': mean, 'map': mode, 'mae': best}
        result.update({f'hit{r}': self.xy[np.argmax(matrix @ w)] for r, matrix in self.hit.items()})
        return result


class SiteDensity:
    def __init__(self, x, y, grid, *, bandwidth_m=250., neighbors=64, prior_mix=.05):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        self.grid = np.asarray(grid, dtype=float)
        if (x.ndim != 2 or y.shape != (len(x), 2) or len(x) < neighbors or neighbors < 1
                or int(neighbors) != neighbors or not all(np.isfinite(a).all() for a in (x, y, self.grid))
                or not np.isfinite(bandwidth_m) or bandwidth_m <= 0
                or not np.isfinite(prior_mix) or not 0 < prior_mix < 1):
            raise ValueError('Valid auxiliary features/labels and fixed density parameters required')
        self.mean, self.scale = x.mean(axis=0), x.std(axis=0)
        self.scale[self.scale < 1e-12] = 1.
        self.features = (x-self.mean)/self.scale
        self.neighbors, self.prior_mix = int(neighbors), float(prior_mix)
        self.log_kernels = -.5*cdist(y, self.grid, 'sqeuclidean')/bandwidth_m**2
        prior_score = logsumexp(self.log_kernels, axis=0)-np.log(len(x))
        prior = np.exp(prior_score-logsumexp(prior_score))
        self.prior = (1-prior_mix)*prior+prior_mix/len(prior)

    def probabilities(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError('Finite matching public features required')
        distance = cdist((x-self.mean)/self.scale, self.features)
        order = np.argsort(distance, axis=1, kind='stable')[:, :self.neighbors]
        score = logsumexp(self.log_kernels[order], axis=1)-np.log(self.neighbors)
        q = np.exp(score-logsumexp(score, axis=1, keepdims=True))
        return (1-self.prior_mix)*q+self.prior_mix*self.prior
