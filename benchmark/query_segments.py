"""Opt-in public road-valid segment PQB, after the unchanged Geo-I history.

Neither raw GPS nor a private QuerySpec enters these interfaces. The real-number
kernel is proved; the NumPy sampler is a diagnostic implementation, not a
certified finite-precision DP sampler. Public tables and previous Q define support.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np
from scipy.sparse.csgraph import dijkstra

from benchmark.probabilistic_query_bundle import QueryBundle, ProbabilisticQueryBundle
from benchmark.query_bundle_bounds import utility_certificate
from evaluation.lane_travel import matrix


@dataclass(frozen=True)
class QuerySegment:
    frames: tuple[tuple[int, ...], ...]

    def __post_init__(self):
        frames = tuple(tuple(f) for f in self.frames)
        if (len(frames) not in (1, 3) or any(len(f) != 5 for f in frames)
                or any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or s < 0
                       for f in frames for s in f)):
            raise ValueError('One or three ordered K5 public frames required')
        # Coincident tracks are retained, just as duplicate GPS coordinates in
        # the public catalogue; they do not secretly change K or support.
        object.__setattr__(self, 'frames', tuple(tuple(map(int, f)) for f in frames))


def oscillation_upper(coverage, cost_weight=.25):
    """Outward bound for the exact REAL interpretation of supplied binary floats.

    This certifies the public table's geometry, NOT rounding of scores/sampling.
    Only an exact rational zero takes the zero-loss optimization branch.
    """
    g = np.asarray(coverage, dtype=float)
    if g.ndim != 2 or not g.size or not np.isfinite(g).all() or np.any(g < 0) or np.any(g > 1):
        raise ValueError('Nonempty bounded public table required')
    if isinstance(cost_weight, bool) or not math.isfinite(cost_weight) or cost_weight < 0:
        raise ValueError('Nonnegative public cost weight required')
    divisor = Fraction(1)+Fraction.from_float(float(cost_weight))
    # Each float subtraction of [0,1] values errs by at most 2^-52. Width
    # subtraction adds another such allowance. Use rationals for singleton/
    # identical row cases, avoiding an uncertified tolerance-based zero.
    exact_zero = all(np.array_equal(row-g[0], np.full(g.shape[1], row[0]-g[0, 0]))
                     for row in g)
    if exact_zero:
        first = [Fraction.from_float(v) for v in g[0]]
        exact_zero = all(len(set(Fraction.from_float(v)-first[a] for a, v in enumerate(row))) == 1 for row in g)
    if exact_zero:
        return Fraction(0)
    maximum = 0.
    for a in range(g.shape[1]):
        maximum = max(maximum, float(np.ptp(g[:, a, None]-g, axis=0).max()))
    return (Fraction.from_float(maximum)+Fraction(3, 2**52))/divisor


class PublicSegmentLibrary:
    """Deterministic support based ONLY on map, public goals and previous Q.

    Ordered public goal bundles yield shortest-time lane paths. State-based
    representation includes waits; every transition takes at most20 seconds.
    Hold is always included. No candidate depends on the protected belief.
    """
    def __init__(self, rn, goal_bundles):
        self.rn = rn
        self.goals = tuple(sorted(set(tuple(map(int, a)) for a in goal_bundles)))
        if (not self.goals or any(len(a) != 5 for a in self.goals)
                or any(s < 0 or s >= len(rn) for a in self.goals for s in a)):
            raise ValueError('Valid public ordered K5 goal bundles required')
        self.travel = matrix(rn, time=True)
        self.reverse = self.travel.transpose().tocsr()

    @lru_cache(maxsize=128)
    def to_goal(self, goal):
        return dijkstra(self.reverse, directed=True, indices=goal, return_predecessors=True)

    def advance(self, state, goal):
        distances, successor = self.to_goal(goal)
        if not np.isfinite(distances[state]):
            return state
        left = 20.
        seen = set()
        while state != goal and state not in seen:
            seen.add(state)
            nxt = int(successor[state])
            if nxt < 0:
                break
            edge = self.rn.graph.get_edge_data(state, nxt)
            if edge is None:
                raise RuntimeError('Public shortest path has no directed arc')
            cost = edge['length']/edge['speed']
            if cost > left:  # leave this sub-grid residual as a public wait
                break
            left -= cost
            state = nxt
        return state

    @lru_cache(maxsize=4096)
    def build(self, previous=None, frames=3):
        if frames not in (1, 3):
            raise ValueError('One step or three-frame segment required')
        if previous is not None and (len(previous) != 5 or any(s < 0 or s >= len(self.rn) for s in previous)):
            raise ValueError('Previous PUBLIC K5 road frame required')
        result = set()
        if previous is not None:
            result.add(QuerySegment((previous,)*frames))
        for goal in self.goals:
            # At activation no previous Q exists: initial support is public.
            current = goal if previous is None else previous
            path = []
            for _ in range(frames):
                if previous is not None or path:
                    current = tuple(self.advance(s, g) for s, g in zip(current, goal))
                path.append(current)
            result.add(QuerySegment(tuple(path)))
        return tuple(sorted(result, key=lambda a: a.frames))


class SegmentPqbPolicy:
    """Bounded, linear public score; unknown errors never imply true utility.

    coverage_provider(frames) supplies [latent,action] table for the whole
    segment. It may average public frame/purpose tables with FIXED weights.
    A single protected belief weights that table; no private future GPS input.
    """
    def __init__(self, library, coverage_provider, *, minimum_coverage=.75, cost_weight=.25):
        if not math.isfinite(minimum_coverage) or not 0 <= minimum_coverage <= 1:
            raise ValueError('Public coverage floor in [0,1] required')
        self.library, self.provider = library, coverage_provider
        self.floor, self.cost_weight = minimum_coverage, cost_weight

    def select(self, previous, epsilon, rng, *, belief, frames=3):
        actions = self.library.build(previous, frames)
        coverage = np.asarray(self.provider(actions), dtype=float)
        floor_table = np.asarray(self.provider.floor_table(actions),dtype=float) if hasattr(self.provider,'floor_table') else coverage
        if floor_table.shape != coverage.shape or not np.isfinite(floor_table).all() or np.any(floor_table < 0) or np.any(floor_table > 1):
            raise ValueError('Matched bounded public floor table required')
        # Public floor on ALL supplied latent rows. If infeasible, preserve
        # common support and report degradation instead of private refetch.
        keep = np.flatnonzero(floor_table.min(axis=0) >= self.floor)
        degraded = not len(keep)
        if not degraded:
            actions = tuple(actions[i] for i in keep)
            coverage = coverage[:, keep]
            floor_table = floor_table[:, keep]
        kappa = oscillation_upper(coverage, self.cost_weight)
        limit = Fraction(epsilon)
        exact_beta = Fraction(40) if not kappa else min(Fraction(40), 2*limit/kappa)
        beta = float(exact_beta)
        if Fraction.from_float(beta) > exact_beta:
            beta = float(np.nextafter(beta, 0.))
        while True:
            epsilon_upper_exact = Fraction.from_float(beta)*kappa/2
            upper = float(epsilon_upper_exact)
            if Fraction.from_float(upper) < epsilon_upper_exact:
                upper = float(np.nextafter(upper, math.inf))
            if Fraction.from_float(upper) <= limit and upper <= float(limit):
                break
            beta = float(np.nextafter(beta, 0.))
        # Adapter IDs are internal action indices, NEVER road states sent to LSP.
        dummy = [QueryBundle(tuple(range(5*i, 5*i+5))) for i in range(len(actions))]
        kernel = ProbabilisticQueryBundle(dummy, coverage, beta=beta, cost_weight=self.cost_weight)
        probabilities = kernel.probabilities(belief)
        index = int(rng.choice(len(actions), p=probabilities))
        diagnostics = utility_certificate(kernel, belief)
        certificate = dict(epsilon_Q_upper=upper, kappa_upper=float(kappa), beta=beta,
            exact_zero_oscillation=not kappa, public_library_size=len(actions),
            requested_public_floor=self.floor, attained_public_floor=float(floor_table.min()),
            public_floor_scope='each frame / public latent / fixed-purpose composite' if hasattr(self.provider,'floor_table') else 'segment surrogate table only',
            floor_degraded=degraded, expected_proxy_coverage=diagnostics['expected_model_coverage'],
            utility_status='public_table_only', actual_expected_utility_lower=None,
            belief_TV_bound=None, proxy_error_bound=None,
            privacy_scope='ideal-real kernel; numeric table bound, NOT certified float sampling')
        return dict(frames=[list(f) for f in actions[index].frames], certificate=certificate)


def actual_utility_bridge(proxy_mean, *, tv_bound=None, proxy_error_bound=None):
    """Explicit three-level utility scope; no real-world zero-error defaults."""
    if not math.isfinite(proxy_mean) or not 0 <= proxy_mean <= 1:
        raise ValueError('Proxy mean in [0,1] required')
    for bound in (tv_bound, proxy_error_bound):
        if bound is not None and (isinstance(bound, bool) or not math.isfinite(bound) or not 0 <= bound <= 1):
            raise ValueError('Known error bound in [0,1] or None required')
    known = tv_bound is not None and proxy_error_bound is not None
    return dict(status='conditional_actual_bound' if known else 'unverified_actual_utility',
                proxy_expected_utility=proxy_mean,
                actual_expected_utility_lower=max(0., proxy_mean-tv_bound-proxy_error_bound) if known else None,
                tv_bound=tv_bound, proxy_error_bound=proxy_error_bound)
