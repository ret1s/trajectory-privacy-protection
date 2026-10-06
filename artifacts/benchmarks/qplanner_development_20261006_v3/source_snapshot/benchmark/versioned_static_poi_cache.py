"""Causal static POI metadata for one public catalogue version and epoch.

This local store changes no request, protected coordinate, read or budget. A
public version/scope mismatch or epoch expiry invalidates every stored record.
Live status retains the separate current-public-epoch guard of the TTL cache.
"""
import math
from benchmark.static_poi_cache import StaticPoiReplyCache


class VersionedStaticPoiCache:
    def __init__(self, poi_ids, *, catalogue_version, epoch_id, start_s, end_s, status_epoch_s=60.):
        if any(not isinstance(v, str) or not v for v in (catalogue_version, epoch_id)):
            raise ValueError('Nonempty public catalogue version and epoch ID required')
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start_s, end_s))
                or start_s < 0 or end_s <= start_s):
            raise ValueError('Finite increasing public epoch required')
        self.catalogue_version, self.epoch_id = catalogue_version, epoch_id
        self.start_s, self.end_s = float(start_s), float(end_s)
        self._poi_ids, self._status_epoch_s = tuple(poi_ids), status_epoch_s
        self.storage_limit = len(self._poi_ids)
        self._invalidated = False
        self._new_store()

    def _new_store(self):
        # An age strictly smaller than the declared epoch duration retains all
        # causal records inside that epoch. Epoch/version validation is explicit.
        self._store = StaticPoiReplyCache(self._poi_ids, ttl_s=self.end_s-self.start_s,
                                          status_epoch_s=self._status_epoch_s)

    def _scope(self, timestamp_s, catalogue_version, epoch_id):
        if (self._invalidated or catalogue_version != self.catalogue_version or epoch_id != self.epoch_id
                or isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float))
                or not math.isfinite(timestamp_s) or not self.start_s <= timestamp_s < self.end_s):
            self._new_store()
            self._invalidated = True
            raise ValueError('Public static catalogue/epoch expired or mismatched; old records invalidated')

    def receive(self, timestamp_s, records, *, catalogue_version, epoch_id):
        self._scope(timestamp_s, catalogue_version, epoch_id)
        ids = self._store.receive(timestamp_s, records)
        assert len(ids) <= self.storage_limit
        return ids

    def static_ids(self, timestamp_s, *, catalogue_version, epoch_id):
        self._scope(timestamp_s, catalogue_version, epoch_id)
        return self._store.static_ids(timestamp_s)

    def static_records(self, timestamp_s, *, catalogue_version, epoch_id):
        self._scope(timestamp_s, catalogue_version, epoch_id)
        return self._store.static_records(timestamp_s)

    def dynamic_candidates(self, timestamp_s, *, catalogue_version, epoch_id,
                           status_epoch=None, current_known_ids=None, current_available_ids=None):
        self._scope(timestamp_s, catalogue_version, epoch_id)
        return self._store.dynamic_candidates(timestamp_s, status_epoch=status_epoch,
            current_known_ids=current_known_ids, current_available_ids=current_available_ids)
