"""Expected location-wise capped POI union coverage.

Unlike capping the category-average score, cap each latent location's service
before averaging. This remains a belief objective, not a true-user utility
guarantee. Sparse public profiles and memoized exact marginals avoid repeated
work without dropping any posterior mass.
"""
from collections import OrderedDict
import numpy as np
from scipy.sparse import csr_matrix


class CappedServiceIndex:
    def __init__(self, signatures, access, reference_weights):
        sig = np.asarray(signatures, dtype=int)
        self.access = np.asarray(access, dtype=int)
        self.reference = csr_matrix(reference_weights, dtype=float).copy()
        if (sig.ndim != 3 or self.access.ndim != 1 or not len(self.access)
                or np.any(self.access < 0) or np.any(self.access >= len(sig))
                or self.reference.ndim != 2 or self.reference.shape[1] < 2
                or not np.isfinite(self.reference.data).all() or np.any(self.reference.data < 0)
                or np.any(np.asarray(self.reference.sum(axis=1)).ravel() > 1+1e-10)
                or self.reference[:, -1].nnz or np.any(sig < -1)
                or np.any(sig >= self.reference.shape[1]-1)):
            raise ValueError('Valid public signatures/access and normalized reference rows required')
        profiles, labels = np.unique(sig.reshape(len(sig), -1), axis=0, return_inverse=True)
        self.state_profile = labels[self.access]
        rows, cols = [], []
        for row, values in enumerate(profiles):
            ids = np.unique(values[values >= 0])
            rows.extend([row]*len(ids)); cols.extend(ids.tolist())
        self.presence = csr_matrix((np.ones(len(rows)), (rows, cols)),
                                  shape=(len(profiles), self.reference.shape[1]))
        self.reference_t = self.reference.transpose().tocsr()


class CappedServiceObjective:
    def __init__(self, index, belief, cap=.9):
        self.index = index
        self.belief = np.asarray(belief, dtype=float).copy()
        if (self.belief.shape != (index.reference.shape[0],) or not np.isfinite(self.belief).all()
                or np.any(self.belief < 0) or not np.isclose(self.belief.sum(), 1., rtol=0, atol=1e-10)
                or not np.isfinite(cap) or not 0 < cap <= 1):
            raise ValueError('Normalized nonnegative belief and cap in (0,1] required')
        self.cap = float(cap); self.cache = OrderedDict()

    def _state(self, selected):
        profile = np.unique(self.index.state_profile[np.asarray(selected, dtype=int)])
        covered = np.zeros(self.index.reference.shape[1], dtype=bool)
        if len(profile):
            covered[self.index.presence[profile].indices] = True
        key = np.packbits(covered).tobytes()
        if key not in self.cache:
            recall = np.asarray(self.index.reference @ covered).ravel()
            self.cache[key] = {'value': float(np.minimum(recall, self.cap) @ self.belief),
                'room': np.maximum(0., self.cap-recall),
                'residual_t': self.index.reference_t.multiply((~covered)[:, None]).tocsr(),
                'marginals': np.full(self.index.presence.shape[0], np.nan)}
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def value(self, selected):
        return self._state(selected)['value']

    def marginal(self, states, selected):
        ids = self.index.state_profile[np.asarray(states, dtype=int)]
        state = self._state(selected)
        missing = np.unique(ids[np.isnan(state['marginals'][ids])])
        for start in range(0, len(missing), 128):
            batch = missing[start:start+128]
            gains = (self.index.presence[batch] @ state['residual_t']).toarray()
            state['marginals'][batch] = np.minimum(gains, state['room'][None, :]) @ self.belief
        return state['marginals'][ids].copy()
