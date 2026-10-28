"""Public long-range cover goals, retaining the bounded Geo-I score kernel.

The old local novelty search can stop at a zero-gain neighbourhood. These goals
use the entire public reply catalogue and directed travel times, never GPS,
belief, request, destination, cohort labels or attacker outcomes.
"""
from functools import lru_cache
import math

import numpy as np
from scipy.sparse.csgraph import dijkstra

from benchmark.cache_exploration import ExploreMotionLibrary, CacheExplorePolicy
from benchmark.query_segments import QuerySegment


class TargetedExploreLibrary(ExploreMotionLibrary):
    def __init__(self, rn, goals, portfolio, *, travel_scale_s=60):
        super().__init__(rn, goals, portfolio)
        if isinstance(travel_scale_s, bool) or not isinstance(travel_scale_s, int) or travel_scale_s <= 0:
            raise ValueError('Positive public integer travel scale required')
        self.travel_scale_s = travel_scale_s

    @lru_cache(maxsize=128)
    def distances(self, state):
        values = dijkstra(self.travel, directed=True, indices=state)
        values.flags.writeable = False
        return values

    @lru_cache(maxsize=2048)
    def targeted(self, previous, known, frames=3):
        if previous is None:
            return ()
        result = {}
        # Five cyclic orders avoid an arbitrary first-track monopoly. All
        # assignment, tie breaking and horizon parameters are public.
        for offset in range(5):
            seen, targets, times = known, list(previous), []
            for track in [(offset + j) % 5 for j in range(5)]:
                distances = self.distances(previous[track])
                candidates = dict(zip(self.portfolio.representatives, self.portfolio.pools))
                candidates[previous[track]] = self.portfolio.state_pool[previous[track]]
                proposals = []
                for state, pool in candidates.items():
                    seconds = float(distances[state])
                    if not math.isfinite(seconds):
                        continue
                    gain = (pool & ~seen).bit_count()
                    value = gain / (1 + seconds / self.travel_scale_s)
                    proposals.append((-value, -gain, seconds, state, pool))
                _, _, seconds, state, pool = min(proposals)
                targets[track] = state
                times.append(seconds)
                seen |= pool
            current, path = previous, []
            for _ in range(frames):
                current = tuple(self.advance(s, g) for s, g in zip(current, targets))
                path.append(current)
            action = QuerySegment(tuple(path))
            # A PUBLIC planning potential, not POIs already received nor a
            # claim that the sampled policy will finish this long-range goal.
            steps = max(1, math.ceil(max(times) / self.travel_scale_s))
            potential = (seen & ~known).bit_count() // steps
            result[action] = max(result.get(action, 0), potential)
        return tuple(sorted(result.items(), key=lambda item: item[0].frames))

    @lru_cache(maxsize=4096)
    def build_cached(self, previous, known, frames=3):
        actions = set(super().build_cached(previous, known, frames))
        actions.update(a for a, _ in self.targeted(previous, known, frames))
        return tuple(sorted(actions, key=lambda a: a.frames))


class TargetedCachePolicy(CacheExplorePolicy):
    def __init__(self, *args, long_range_weight=4, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(long_range_weight, bool) or not isinstance(long_range_weight, int) or long_range_weight < 0:
            raise ValueError('Public nonnegative integer long-range weight required')
        self.long_range_weight = long_range_weight

    def select(self, previous, epsilon, rng=None, *, belief, known, frames=3):
        # Numerical trouble in a PRIVATE proxy must not trigger a new request,
        # stop, or private support filter. This fallback still uses the same
        # public action set and exact bounded kernel. Diagnostic stays local.
        try:
            b = np.asarray(belief, dtype=float)
            valid = (b.shape == (len(self.table.states),) and np.isfinite(b).all()
                     and np.all(b >= 0) and np.isclose(b.sum(), 1., rtol=0, atol=1e-12))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            b = np.full(len(self.table.states), 1. / len(self.table.states))
        result = super().select(previous, epsilon, rng, belief=b, known=known, frames=frames)
        result['certificate']['private_proxy_fallback_local_only'] = not valid
        return result

    @lru_cache(maxsize=4096)
    def public_tables(self, previous, known, frames=3):
        actions, scores, gains, masses = super().public_tables(previous, known, frames)
        potentials = dict(self.library.targeted(previous, known, frames))
        masses = tuple(m + self.long_range_weight * potentials.get(a, 0) for a, m in zip(actions, masses))
        return actions, scores, gains, masses
