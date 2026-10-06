"""Geo-I protection -> fixed-purpose-independent retrieval -> local answer.

The network operation accepts no private query. QuerySpec/GPS used to answer
are consumed after retrieval only; answering never schedules a request.
"""
import math

from benchmark.query_purpose import QuerySpec


class GeoILbsClient:
    def __init__(self, engine, retrieval, ranking):
        if (engine.rn is not ranking.rn
                or tuple(retrieval.categories) != tuple(ranking.categories)
                or retrieval.k != engine.k):
            raise ValueError('Matched Geo-I engine, fixed retrieval and local ranking required')
        self.engine, self.retrieval, self.ranking = engine, retrieval, ranking
        self.last_fetch = None

    def protect_and_fetch(self, timestamp_s, lat, lon, server):
        """Run on the public clock, independently of actual query demand."""
        coordinates = self.engine.protect_step(lat, lon, timestamp_s)
        self.last_fetch = self.retrieval.step(timestamp_s, coordinates, server)
        return self.last_fetch

    def answer(self, query, lat, lon):
        """Current GPS and private purpose only select received valid POIs."""
        if self.last_fetch is None:
            raise ValueError('Retrieve public candidates before answering')
        if not isinstance(query, QuerySpec):
            raise ValueError('Private local QuerySpec required')
        if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError('Valid local GPS required')
        state = self.ranking.rn.nearest(lat, lon)[0]
        return self.ranking.top(state, self.last_fetch['known'], query)
