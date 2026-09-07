"""Bounded sparse shortest-path caches on the common public lane catalogue."""
from collections import OrderedDict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from evaluation.scenario_metrics import PoiService


def matrix(rn, time=False):
    edges=list(rn.graph.edges(data=True))
    # Do not eliminate zeros: coincident lane-end/start connections are real.
    return csr_matrix(([d['length']/d['speed'] if time else d['length'] for a,b,d in edges],
                       ([int(a) for a,b,d in edges],[int(b) for a,b,d in edges])),shape=(len(rn),len(rn)))


class SparseTravel:
    def __init__(self,rn,cache_limit=256):
        self.rn=rn
        self.matrix=matrix(rn,time=True)
        self.cache=OrderedDict()
        self.cache_limit=cache_limit

    def reachable(self,vertex,seconds):
        key=int(vertex),float(seconds)
        if key not in self.cache:
            distances=dijkstra(self.matrix,directed=True,indices=int(vertex),limit=max(0.,float(seconds)))
            self.cache[key]={int(i):float(distances[i]) for i in np.flatnonzero(np.isfinite(distances))}
            if len(self.cache)>self.cache_limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


class LanePoiService(PoiService):
    """Same directed POI query as PoiService, with a bounded faster cache.

    All methods in the lane study use this exact service. Nearest-lane-state
    access remains an approximation; do not compare numeric scores to paper-v2.
    """
    def __init__(self,rn,pois,k=5,max_access_m=250,cache_limit=2048):
        super().__init__(rn,pois,k,max_access_m)
        self.matrix=matrix(rn)
        self._distances=OrderedDict()
        self.cache_limit=cache_limit

    def distances(self,point):
        vertex,_=self.rn.nearest(*point)
        if vertex not in self._distances:
            distances=dijkstra(self.matrix,directed=True,indices=int(vertex))
            self._distances[vertex]={p['id']:float(distances[p['vertex']]) for p in self.pois if np.isfinite(distances[p['vertex']])}
            if len(self._distances)>self.cache_limit:
                self._distances.popitem(last=False)
        self._distances.move_to_end(vertex)
        return self._distances[vertex]
