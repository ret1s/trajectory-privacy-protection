"""Opt-in joint Q-bundle kernel after Geo-I; never accepts raw GPS or intent.

The caller supplies a fixed PUBLIC library and protected belief. Cardinality is
part of the randomized action, not a separate deterministic private decision.
Ideal-arithmetic likelihood bound: T(a|b)/T(a|b') <= exp(beta), including K.
This prototype is not switched into the production/legacy engine. Library
provenance, public scheduling and session composition remain caller obligations.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class QueryBundle:
    """An ordered public tuple; no hidden anchor is added to the transmitted Qs."""
    states: tuple[int, ...]

    def __post_init__(self):
        states = tuple(self.states)
        if (not states or any(isinstance(s, (bool, np.bool_)) or
                              not isinstance(s, (int, np.integer)) or s < 0 for s in states)
                or len(set(states)) != len(states)):
            raise ValueError('Distinct nonnegative public state IDs required')
        object.__setattr__(self, 'states', tuple(map(int, states)))


class ProbabilisticQueryBundle:
    """T(a|b) proportional to w(a) exp(beta * score(b,a) / 2).

    Coverage is a PUBLIC matrix [latent state, bundle] in [0,1]. Belief may
    depend arbitrarily on the protected history; calibration is unnecessary for
    privacy, but necessary for a real-service interpretation of its objective.
    Beta <= 40 avoids numerical zero support for moderate library weights;
    floats are still not a certified implementation of pure differential privacy.
    """

    def __init__(self, bundles, coverage, *, beta=2., cost_weight=.25, base_weights=None):
        self.bundles = tuple(bundles)
        if (not self.bundles or any(not isinstance(a, QueryBundle) for a in self.bundles)
                or len(set(self.bundles)) != len(self.bundles)):
            raise ValueError('Nonempty unique public bundle library required')
        if (isinstance(beta, (bool, np.bool_)) or not np.isfinite(beta) or not 0 <= beta <= 40
                or isinstance(cost_weight, (bool, np.bool_)) or not np.isfinite(cost_weight)
                or cost_weight < 0):
            raise ValueError('Finite public beta in [0,40] and nonnegative cost weight required')
        coverage = np.array(coverage, dtype=float, copy=True)
        if (coverage.ndim != 2 or not coverage.shape[0] or coverage.shape[1] != len(self.bundles)
                or not np.isfinite(coverage).all() or np.any(coverage < 0) or np.any(coverage > 1)):
            raise ValueError('Public coverage matrix in [0,1] required')
        self.cardinalities = np.array([len(a.states) for a in self.bundles])
        if base_weights is None:
            # Equal total public base mass per K; do not favor classes merely
            # because they have more geometrically distinct library actions.
            ks, counts = np.unique(self.cardinalities, return_counts=True)
            sizes = dict(zip(ks, counts))
            weights = np.array([1. / (len(ks) * sizes[k]) for k in self.cardinalities])
        else:
            weights = np.array(base_weights, dtype=float, copy=True)
        if (weights.shape != (len(self.bundles),) or not np.isfinite(weights).all()
                or np.any(weights <= 0)):
            raise ValueError('Every public bundle must have strictly positive base mass')
        # Normalize in log space; reject an ill-conditioned library rather than
        # silently drop actions, which would invalidate common support.
        logs = np.log(weights)
        logs -= logsumexp(logs)
        if np.min(logs) < -600:
            raise ValueError('Public base weights too imbalanced for float prototype')
        self.beta, self.cost_weight = float(beta), float(cost_weight)
        self.coverage, self.log_base_weights = coverage, logs
        self.coverage.flags.writeable = False
        self.log_base_weights.flags.writeable = False
        self.cardinalities.flags.writeable = False

    def scores(self, belief):
        belief = np.asarray(belief, dtype=float)
        if (belief.shape != (self.coverage.shape[0],) or not np.isfinite(belief).all()
                or np.any(belief < 0) or not np.isclose(belief.sum(), 1., atol=1e-12, rtol=0)):
            raise ValueError('Normalized protected belief required; no silent normalization')
        # Direct summation avoids spurious floating-point flags observed in
        # NumPy/Accelerate matmul on this audit's macOS runtime.
        coverage = np.einsum('i,ij->j', belief, self.coverage, optimize=False)
        savings = 1. - self.cardinalities / self.cardinalities.max()
        return (coverage + self.cost_weight * savings) / (1. + self.cost_weight)

    def log_probabilities(self, belief):
        logits = self.log_base_weights + .5 * self.beta * self.scores(belief)
        return logits - logsumexp(logits)

    def probabilities(self, belief):
        return np.exp(self.log_probabilities(belief))

    def sample(self, belief, *, rng):
        """Fresh secret randomness required in real use; seeded RNG is for tests."""
        return self.bundles[int(rng.choice(len(self.bundles), p=self.probabilities(belief)))]


def bundle_coverage(reference, reply, latent_states, bundles):
    """Public nearest-POI Recall proxy; actual local intent stays out of Q scores.

    Compute the per-nonempty-category reference coverage of each reply union.
    Undefined reference states are rejected, not silently assigned perfect QoS.
    """
    if (reference.rn is not reply.rn or reference.pois != reply.pois
            or reference.categories != reply.categories or reply.k < reference.k):
        raise ValueError('Matching public reference and response contexts required')
    states = tuple(map(int, latent_states))
    if not states or any(s < 0 or s >= len(reference.rn) for s in states):
        raise ValueError('Valid public latent states required')
    result = np.empty((len(states), len(bundles)))
    for j, action in enumerate(bundles):
        union = set()
        for state in action.states:
            if state >= len(reply.rn):
                raise ValueError('Bundle state outside public map')
            union.update(int(p) for p in reply.query_indices(state).ravel() if p >= 0)
        for i, state in enumerate(states):
            rows = [set(map(int, row[row >= 0])) for row in reference.query_indices(state)]
            rows = [row for row in rows if row]
            if not rows:
                raise ValueError('Undefined public reference coverage')
            result[i, j] = np.mean([len(row & union) / len(row) for row in rows])
    return result
