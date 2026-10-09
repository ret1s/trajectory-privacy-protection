"""Private POI purposes over a purpose-independent public retrieval stream.

Only local ranking consumes QuerySpec, true position, radius or destination.
Cover requests consume protected coordinates and a fixed public service schema.
This hides explicit purpose conditional on the same stream/schedule; correlations
with the location stream, account identifiers and activation remain observable.
"""
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
from scipy.sparse.csgraph import dijkstra

from evaluation.lane_travel import matrix
from evaluation.live_poi import EpochResponseCache


class QueryPurpose(str, Enum):
    NEAREST = 'nearest_distance'
    FASTEST = 'fastest_travel'
    WITHIN_RADIUS = 'within_radius'
    MIN_DETOUR = 'minimum_detour'


@dataclass(frozen=True)
class QuerySpec:
    purpose: QueryPurpose
    category: str
    k: int = 5
    radius_m: float | None = None
    destination_state: int | None = None

    def __post_init__(self):
        if not isinstance(self.purpose, QueryPurpose) or not self.category:
            raise ValueError('Supported purpose and nonempty category required')
        if isinstance(self.k, bool) or int(self.k) != self.k or self.k < 1:
            raise ValueError('Positive integer top-k required')
        if self.purpose == QueryPurpose.WITHIN_RADIUS:
            if self.radius_m is None or not math.isfinite(self.radius_m) or self.radius_m <= 0:
                raise ValueError('Positive finite private radius required')
        elif self.radius_m is not None:
            raise ValueError('Radius is only meaningful for within-radius queries')
        if self.purpose == QueryPurpose.MIN_DETOUR:
            if (self.destination_state is None or isinstance(self.destination_state, bool)
                    or int(self.destination_state) != self.destination_state or self.destination_state < 0):
                raise ValueError('Private integer destination state required')
        elif self.destination_state is not None:
            raise ValueError('Destination is only meaningful for detour queries')


class MultiPurposeRoadRanking:
    """Exact directed graph objectives; lexical public POI IDs break ties.

    'Available' is a candidate mask: server/evaluator may have the complete
    availability world; the device may pass only IDs received in valid replies.
    Detour destination is private local input, never a query sent to the server.
    """
    def __init__(self, ranking, cache_limit=256):
        self.rn, self.pois = ranking.rn, ranking.pois
        self.categories = ranking.categories
        self.n = len(self.pois)
        self.vertices = np.array([p['vertex'] for p in self.pois], dtype=int)
        self.distance_graph, self.time_graph = matrix(self.rn), matrix(self.rn, time=True)
        self.cache, self.cache_limit = OrderedDict(), cache_limit

    def _costs(self, state, travel_time=False, reverse=False):
        if isinstance(state, bool) or int(state) != state or not 0 <= state < len(self.rn):
            raise ValueError('Valid integer road state required')
        key = int(state), bool(travel_time), bool(reverse)
        if key not in self.cache:
            graph = self.time_graph if travel_time else self.distance_graph
            if reverse: graph = graph.transpose().tocsr()
            self.cache[key] = dijkstra(graph, directed=True, indices=int(state))
            if len(self.cache) > self.cache_limit: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def scores(self, state, query):
        if not isinstance(query, QuerySpec) or query.category not in self.categories:
            raise ValueError('Query must use a supported public category')
        cost = self._costs(state, query.purpose == QueryPurpose.FASTEST)
        scores = cost[self.vertices].copy()
        if query.purpose == QueryPurpose.WITHIN_RADIUS:
            scores[scores > query.radius_m] = math.inf
        elif query.purpose == QueryPurpose.MIN_DETOUR:
            to_destination = self._costs(query.destination_state, reverse=True)
            direct = cost[query.destination_state]
            if not math.isfinite(direct):
                scores[:] = math.inf
            else:
                scores = scores + to_destination[self.vertices] - direct
                scores[np.isfinite(scores)] = np.maximum(0., scores[np.isfinite(scores)])
        return scores

    def top(self, state, candidates, query):
        candidates = np.asarray(candidates, dtype=bool)
        if candidates.shape != (self.n,):
            raise ValueError('Complete candidate mask required')
        scores = self.scores(state, query)
        ids = [i for i, p in enumerate(self.pois) if p['category'] == query.category
               and candidates[i] and math.isfinite(scores[i])]
        ids.sort(key=lambda i: (float(scores[i]), self.pois[i]['id']))
        return ids[:query.k]


class PurposeIndependentCoverClient:
    """No private purpose/category/radius/destination argument in network API.

    Every Q requests all public categories at a fixed response depth. Q and
    request timing must themselves be produced independently of private intent.
    This class does not hide session timing or provide identity anonymity.
    """
    def __init__(self, categories, poi_count, *, k=5, response_l=10, epoch_seconds=60.):
        self.categories = tuple(sorted(categories))
        if (not self.categories or len(set(self.categories)) != len(self.categories)
                or any(not isinstance(c, str) or not c for c in self.categories)):
            raise ValueError('Distinct nonempty public categories required')
        if (isinstance(k, bool) or int(k) != k or k < 1 or isinstance(response_l, bool)
                or int(response_l) != response_l or response_l < 1):
            raise ValueError('Fixed positive integer K and L required')
        if not math.isfinite(epoch_seconds) or epoch_seconds <= 0:
            raise ValueError('Positive public epoch duration required')
        if isinstance(poi_count, bool) or int(poi_count) != poi_count or poi_count < 1:
            raise ValueError('Positive public POI count required')
        self.k, self.response_l, self.epoch_seconds = int(k), int(response_l), float(epoch_seconds)
        self.cache = EpochResponseCache(poi_count)
        self.last_t = None

    def step(self, timestamp_s, protected_coordinates, server):
        if (not math.isfinite(timestamp_s) or timestamp_s < 0 or
                (self.last_t is not None and timestamp_s <= self.last_t)):
            raise ValueError('Strictly increasing public timestamp required')
        coordinates = tuple(tuple(c) for c in protected_coordinates)
        if len(coordinates) != self.k or any(len(c) != 2 or
                not all(math.isfinite(v) for v in c) or not -90 <= c[0] <= 90 or
                not -180 <= c[1] <= 180 for c in coordinates):
            raise ValueError('Exactly K finite protected lat/lon coordinates required')
        epoch = int(timestamp_s // self.epoch_seconds)
        requests = [{'schema':'all_category_cover_v1', 'timestamp_s':timestamp_s,
                     'coordinate':list(c), 'categories':list(self.categories),
                     'response_l':self.response_l, 'epoch':epoch} for c in coordinates]
        replies = [server(request) for request in requests]
        current, known = self.cache.receive(epoch, replies)
        self.last_t = timestamp_s
        return {'requests':requests, 'replies':replies, 'current':current,
                'known':known, 'epoch':epoch}
