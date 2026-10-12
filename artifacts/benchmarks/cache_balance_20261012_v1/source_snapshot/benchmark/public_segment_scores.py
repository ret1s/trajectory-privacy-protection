"""Fixed public purpose weights and POI-reference tables for segment scores.

Undefined cases get zero AVAILABILITY score in the planner, plus a validity mask;
they are still N/A (never zero Recall) in empirical utility reporting. Geometry,
destination prototypes and radius are public, not the user's private request.
"""
from functools import lru_cache

import numpy as np

from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec


def local_sorted_pois(ranking, state, candidates, query):
    """Full local answer; QuerySpec.k is only a reference evaluation parameter."""
    mask = np.asarray(candidates, dtype=bool)
    if mask.shape != (ranking.n,):
        raise ValueError('Received POI mask required')
    scores = ranking.scores(state, query)
    ids = [i for i, p in enumerate(ranking.pois) if mask[i] and p['category'] == query.category
           and np.isfinite(scores[i])]
    return sorted(ids, key=lambda i: (float(scores[i]), ranking.pois[i]['id']))


class FixedPublicPurposeTable:
    def __init__(self, reference, reply, latent_states, *, destinations, radius_m=1000.):
        if reference.rn is not reply.rn or reference.pois != reply.pois or reply.k < reference.k:
            raise ValueError('Matched public reference and response contexts required')
        self.states = tuple(map(int, latent_states))
        self.destinations = tuple(sorted(set(map(int, destinations))))
        if (not self.states or not self.destinations
                or any(s < 0 or s >= len(reference.rn) for s in self.states+self.destinations)):
            raise ValueError('Fixed public states and destination prototypes required')
        self.reference, self.reply = reference, reply
        ranking = MultiPurposeRoadRanking(reference, cache_limit=512)
        n, p, m = len(self.states), len(QueryPurpose), len(reply.pois)
        self.weights = np.zeros((n, p, m))
        self.validity = np.zeros((n, p))
        all_ids = np.ones(m, bool)
        for x, state in enumerate(self.states):
            for j, purpose in enumerate(QueryPurpose):
                prototypes = self.destinations if purpose == QueryPurpose.MIN_DETOUR else (None,)
                for destination in prototypes:
                    rows = []
                    for category in ranking.categories:
                        spec = QuerySpec(purpose, category, k=5,
                            radius_m=radius_m if purpose == QueryPurpose.WITHIN_RADIUS else None,
                            destination_state=destination)
                        ids = local_sorted_pois(ranking, state, all_ids, spec)[:5]
                        if ids:
                            rows.append(ids)
                    if rows:
                        self.validity[x, j] += 1./len(prototypes)
                        for ids in rows:
                            self.weights[x, j, ids] += 1./(len(prototypes)*len(rows)*len(ids))
        self.weights.flags.writeable = False
        self.validity.flags.writeable = False

    @lru_cache(maxsize=16384)
    def frame_table(self, frame):
        union = np.zeros(len(self.reply.pois), bool)
        for state in frame:
            ids = self.reply.query_indices(state).ravel()
            union[ids[ids >= 0]] = True
        return self.weights[:, :, union].sum(axis=2)

    def __call__(self, actions):
        # Mean over frames and ALL four public purposes. No belief-dependent
        # renormalization of N/A, destinations, categories or purpose weights.
        columns = [np.mean([self.frame_table(f) for f in a.frames], axis=0).mean(axis=1)
                   for a in actions]
        return np.clip(np.asarray(columns).T, 0., 1.)

    def floor_table(self, actions):
        # Different real positions at different frames: minimum over frames,
        # NOT a mean evaluated at one unchanged hypothetical position.
        return np.clip(np.asarray([np.min([self.frame_table(f).mean(axis=1)
                                          for f in a.frames],axis=0) for a in actions]).T,0.,1.)
