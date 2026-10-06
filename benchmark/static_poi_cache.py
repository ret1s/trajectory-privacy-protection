"""Local cache for static POI records; it never schedules a network request.

Only IDs/category/coordinates can survive across public status epochs. Live
availability must come from a current fixed-epoch public reply, or stay unknown.
TTL decisions use the public clock, never GPS, local purpose or private demand.
"""
import math


class StaticPoiReplyCache:
    def __init__(self, poi_ids, *, ttl_s=60., status_epoch_s=60.):
        ids = tuple(poi_ids)
        if not ids or len(ids) != len(set(ids)) or any(not isinstance(i, str) or not i for i in ids):
            raise ValueError('Distinct nonempty public POI IDs required')
        for value in (ttl_s, status_epoch_s):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Positive finite public cache/status duration required')
        self.poi_ids, self.ttl_s, self.status_epoch_s = frozenset(ids), float(ttl_s), float(status_epoch_s)
        self._entries = {}
        self._clock = self._last_receive = None

    def _time(self, timestamp_s):
        if (isinstance(timestamp_s, bool) or not isinstance(timestamp_s, (int, float))
                or not math.isfinite(timestamp_s) or timestamp_s < 0
                or self._clock is not None and timestamp_s < self._clock):
            raise ValueError('Nondecreasing finite public clock required')

    def _expire(self, timestamp_s):
        self._entries = {i: row for i, row in self._entries.items() if timestamp_s-row[0] < self.ttl_s}
        self._clock = float(timestamp_s)

    def receive(self, timestamp_s, records):
        """Accept only static fields; dynamic/status/private fields are rejected."""
        self._time(timestamp_s)
        if self._last_receive is not None and timestamp_s <= self._last_receive:
            raise ValueError('Strictly increasing public reply times required')
        clean = {}
        for record in records:
            if not isinstance(record, dict) or set(record) != {'id', 'category', 'lat', 'lon'}:
                raise ValueError('Only public static id/category/lat/lon fields may be cached')
            if record['id'] not in self.poi_ids or not isinstance(record['category'], str) or not record['category']:
                raise ValueError('Valid public POI ID/category required')
            lat, lon = record['lat'], record['lon']
            if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (lat, lon))
                    or not -90 <= lat <= 90 or not -180 <= lon <= 180):
                raise ValueError('Finite public POI coordinates required')
            if record['id'] in clean and clean[record['id']] != record:
                raise ValueError('Conflicting static metadata in one public reply event')
            clean[record['id']] = dict(record)
        # Validate the entire reply before mutating existing cache state.
        self._expire(timestamp_s)
        self._entries.update({i: (float(timestamp_s), r) for i, r in clean.items()})
        self._last_receive = float(timestamp_s)
        return self.static_ids(timestamp_s)

    def static_ids(self, timestamp_s):
        """IDs for static local ranking, with a half-open age window [0,TTL)."""
        self._time(timestamp_s)
        self._expire(timestamp_s)
        return tuple(sorted(self._entries))

    def static_records(self, timestamp_s):
        return tuple(dict(self._entries[i][1]) for i in self.static_ids(timestamp_s))

    def dynamic_candidates(self, timestamp_s, *, status_epoch=None,
                           current_known_ids=None, current_available_ids=None):
        """Intersect with statuses received in the CURRENT fixed public epoch.

        Missing/stale status returns unknown without fetching. Absence from a
        current cover reply is unknown, not evidence that a POI is unavailable.
        Callers must supply status sets from public replies, not evaluator truth.
        """
        ids = set(self.static_ids(timestamp_s))
        epoch = int(timestamp_s // self.status_epoch_s)
        if (status_epoch is None or isinstance(status_epoch, bool)
                or not isinstance(status_epoch, int) or status_epoch != epoch
                or current_known_ids is None or current_available_ids is None):
            return {'state': 'unknown', 'eligible': (), 'unknown': tuple(sorted(ids)), 'unavailable': ()}
        known, available = set(current_known_ids), set(current_available_ids)
        if not known <= self.poi_ids or not available <= known:
            raise ValueError('Available IDs must be a subset of current known public status IDs')
        return {'state': 'current_epoch', 'eligible': tuple(sorted(ids & available)),
                'unknown': tuple(sorted(ids-known)), 'unavailable': tuple(sorted(ids & (known-available)))}
