"""Operational facade for Geo-I, linked-session accounting and local POI use.

Private query demand never triggers a network operation. Session admission and
ticks must follow the application's public policy. A stopped session/expired
response cannot silently serve a stale result or spend a new location read.
This composes existing components; it changes no Geo-I sampler or prediction.
"""
import math

from benchmark.query_order import PrivateOrderCoverClient
from benchmark.query_purpose import QuerySpec
from core.session_budget import FixedEpochProtectedSessions


class BudgetedGeoILbsClient:
    def __init__(self, sessions, ranking, *, k=5, response_l=20,
                 response_epoch_s=60., private_order_rng=None):
        if not isinstance(sessions, FixedEpochProtectedSessions):
            raise ValueError('Persistent fixed-epoch protected sessions required')
        self.sessions, self.ranking = sessions, ranking
        self._schema = dict(categories=ranking.categories, poi_count=ranking.n,
                            k=k, response_l=response_l, epoch_seconds=response_epoch_s)
        self._order_rng = private_order_rng
        # Validate public service parameters before reserving a session slot.
        PrivateOrderCoverClient(rng=self._order_rng, **self._schema)
        self._retrieval = self._last_fetch = self._last_fetch_s = None
        self._open = False

    def start_session(self, public_token, public_start_s):
        retrieval = PrivateOrderCoverClient(rng=self._order_rng, **self._schema)
        admitted = self.sessions.start_session(public_token, public_start_s)
        if admitted:
            try:
                self._validate_service(self.sessions._engine, retrieval)
            except Exception:
                # The public slot stays reserved. A configuration error must
                # not read GPS, transmit Q, or recycle privacy credit.
                self.sessions.close_session(public_start_s)
                self._open = False
                self._retrieval = self._last_fetch = self._last_fetch_s = None
                raise
        self._open = True
        self._last_fetch = self._last_fetch_s = None
        self._retrieval = retrieval if admitted else None
        return admitted

    def _validate_service(self, engine, retrieval):
        """Check the active public POI planner before the first private read.

        The belief API exposes its active reply context directly, including
        response-aware/public-purpose views. A deeper public response is valid
        with the frozen planner; it need not change Q or the Geo-I mechanism.
        """
        context = getattr(getattr(engine, 'belief_model', None), 'context', None)
        if (engine.rn is not self.ranking.rn or engine.k != retrieval.k
                or context is None or context.rn is not self.ranking.rn):
            raise ValueError('Matched Geo-I engine, K and public POI planner required')
        signature = lambda pois: tuple((p['id'], p['category'], p['vertex']) for p in pois)
        if (tuple(sorted(context.categories)) != retrieval.categories
                or tuple(sorted(self.ranking.categories)) != retrieval.categories
                or signature(context.pois) != signature(self.ranking.pois)
                or retrieval.response_l < context.k):
            raise ValueError('Matched public POI catalogue/categories and sufficient reply depth required')

    def public_tick(self, public_time_s, gps_supplier, server):
        """Supplier is called only when the existing privacy filter permits.

        An admitted, budget-exhausted engine continues from protected state;
        a denied session sends nothing. Neither choice uses the local purpose.
        """
        if not self._open:
            raise ValueError('Start a public session first')
        self._last_fetch = self._last_fetch_s = None
        coordinates = self.sessions.protect_step(public_time_s, gps_supplier)
        if not coordinates:
            return None
        self._last_fetch = self._retrieval.step(public_time_s, coordinates, server)
        self._last_fetch_s = public_time_s
        return self._last_fetch

    def answer(self, query, public_time_s, lat, lon):
        """Use current GPS/purpose locally, without a read of the Geo-I filter.

        Current local GPS is allowed here because it never affects transmitted
        coordinates, request timing or request contents. Expiry returns an
        explicit error; it does not fetch in response to a private question.
        """
        if not self._open or self._last_fetch is None:
            raise ValueError('No live public response available')
        self.sessions.ledger.validate_time(public_time_s)
        if (public_time_s < self._last_fetch_s or
                int(public_time_s // self._retrieval.epoch_seconds) != self._last_fetch['epoch']):
            raise ValueError('Public response expired or answer clock rewound')
        if not isinstance(query, QuerySpec):
            raise ValueError('Private local QuerySpec required')
        if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError('Valid local GPS required')
        state = self.ranking.rn.nearest(lat, lon)[0]
        return self.ranking.top(state, self._last_fetch['known'], query)

    def close_session(self, public_close_s):
        self.sessions.close_session(public_close_s)
        self._open = False
        self._retrieval = self._last_fetch = self._last_fetch_s = None
