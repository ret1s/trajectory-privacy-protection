"""Optional static-metadata facade around the unchanged budgeted Geo-I client.

Use answer_static only for catalogue-version-stable POIs/static road objectives.
Use answer_live for availability-dependent answers: it retains the original
current60s response guard. Neither local answer can mutate network/cache clocks.
"""
import math
import numpy as np
from benchmark.budgeted_geoi_lbs import BudgetedGeoILbsClient
from benchmark.query_purpose import QuerySpec
from benchmark.versioned_static_poi_cache import VersionedStaticPoiCache


class VersionedStaticGeoILbsClient:
    def __init__(self, budgeted_client, *, catalogue_version, public_epoch_id, start_s, end_s):
        if not isinstance(budgeted_client, BudgetedGeoILbsClient):
            raise ValueError('Existing BudgetedGeoILbsClient required')
        self.base, self.ranking = budgeted_client, budgeted_client.ranking
        self.catalogue_version, self.public_epoch_id = catalogue_version, public_epoch_id
        self.start_s, self.end_s = start_s, end_s
        self._records = tuple({k: p[k] for k in ('id', 'category', 'lat', 'lon')} for p in self.ranking.pois)
        self.cache = VersionedStaticPoiCache([p['id'] for p in self._records], catalogue_version=catalogue_version,
            epoch_id=public_epoch_id, start_s=start_s, end_s=end_s,
            status_epoch_s=self.base._schema['epoch_seconds'])
        self._mask = np.zeros(self.ranking.n, dtype=bool)
        self._admitted = False
        self._static_invalidated = False
        self._session_start = self._last_cache_tick = None

    def _scope(self, version):
        return dict(catalogue_version=self.catalogue_version if version is None else version,
                    epoch_id=self.public_epoch_id)

    def start_session(self, public_token, public_start_s):
        self._public_scope(public_start_s, self._scope(None))
        self._admitted = self.base.start_session(public_token, public_start_s)
        self._session_start = public_start_s
        return self._admitted

    def public_tick(self, public_time_s, gps_supplier, server, *, catalogue_version=None):
        scope = self._scope(catalogue_version)
        self._public_scope(public_time_s, scope)  # Only PUBLIC ticks advance cache clocks.
        fetched = self.base.public_tick(public_time_s, gps_supplier, server)
        if fetched is None:
            return None
        current = np.flatnonzero(fetched['current'])
        ids = set(self.cache.receive(public_time_s, [self._records[int(i)] for i in current], **scope))
        self._mask = np.array([p['id'] in ids for p in self._records], dtype=bool)
        self._last_cache_tick = public_time_s
        return fetched

    def _public_scope(self, public_time_s, scope):
        try:
            self.cache.static_ids(public_time_s, **scope)
        except ValueError:
            self._mask[:] = False
            self._static_invalidated = True
            raise

    def _answer_scope(self, public_time_s, catalogue_version=None):
        """Read-only validation shared by static and live local answers."""
        if not self._admitted or not self.base._open:
            raise ValueError('No active admitted public session')
        version = self.catalogue_version if catalogue_version is None else catalogue_version
        if (self._static_invalidated or version != self.catalogue_version or not math.isfinite(public_time_s)
                or not self.start_s <= public_time_s < self.end_s
                or public_time_s < self._session_start
                or self._last_cache_tick is not None and public_time_s < self._last_cache_tick):
            raise ValueError('Static public catalogue/epoch expired or answer clock rewound')

    def answer_static(self, query, public_time_s, lat, lon, *, catalogue_version=None):
        """Read-only local answer; no cache timestamp or network state changes."""
        self._answer_scope(public_time_s, catalogue_version)
        if not isinstance(query, QuerySpec):
            raise ValueError('Private local QuerySpec required')
        if not all(math.isfinite(v) for v in (lat, lon)) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError('Valid local GPS required')
        state = self.ranking.rn.nearest(lat, lon)[0]
        return self.ranking.top(state, self._mask.copy(), query)

    def answer_live(self, query, public_time_s, lat, lon):
        """Current response required; stale static metadata never implies availability."""
        self._answer_scope(public_time_s)
        return self.base.answer(query, public_time_s, lat, lon)

    def close_session(self, public_close_s):
        self.base.close_session(public_close_s)
        self._admitted = False
