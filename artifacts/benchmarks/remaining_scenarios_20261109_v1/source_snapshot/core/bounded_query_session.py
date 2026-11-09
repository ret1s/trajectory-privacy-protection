"""Executable bounded-Q research session; protected input and local answer.

The engine owning Geo-I/GPS remains separate. Never reset that engine's privacy
ledger when reopening this Q session. Cache and Q ledger are trusted local state;
crash recovery replays the already committed frame, not a new private selection.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from benchmark.cache_exploration import mask_bits
from benchmark.public_segment_scores import local_sorted_pois
from benchmark.session_poi_cache import SessionCoverClient
from core.query_budget import PersistentQueryBudget


class BoundedQuerySession:
    def __init__(self, directory, *, policy, session_token, catalogue_id, categories,
                 poi_count, configuration_id, gamma=1., response_l=30):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.cache_path = self.directory / 'cache.json'
        self.policy = policy
        self.context = dict(session_token=session_token, catalogue_id=catalogue_id,
                            configuration_id=configuration_id, categories=list(categories),
                            poi_count=poi_count, gamma=gamma, response_l=response_l)
        # Bind the actual PUBLIC planner, not just a user-supplied label. A
        # caller cannot silently change support/weights while reopening a ledger.
        self.context['planner'] = dict(
            policy_class=type(policy).__module__+'.'+type(policy).__qualname__,
            library_class=type(policy.library).__module__+'.'+type(policy.library).__qualname__,
            goals=[list(g) for g in policy.library.goals],
            public_score_sha256=hashlib.sha256(policy.table.weights.tobytes()).hexdigest(),
            latent_states=list(policy.table.states),
            reply_sha256=policy.portfolio.reply.sha256,
            resolution=policy.resolution, novelty_weight=policy.novelty_weight,
            long_range_weight=getattr(policy,'long_range_weight',0),
            travel_scale_s=getattr(policy.library,'travel_scale_s',None),
            proximity_weight=policy.initial.proximity_weight,
            initial_exact_floor=str(policy.initial.quality_floor_exact))
        context_id = hashlib.sha256(json.dumps(self.context, sort_keys=True).encode()).hexdigest()
        self.ledger = PersistentQueryBudget(self.directory/'query.sqlite',
            session_token=session_token, gamma=gamma, context_id=context_id)
        try:
            if self.cache_path.exists():
                saved = json.loads(self.cache_path.read_text())
                if saved['context'] != self.context:
                    raise ValueError('Immutable research configuration changed')
                self.client = SessionCoverClient.restore(saved['cache'],
                    session_token=session_token, static_catalogue_id=catalogue_id)
            else:
                self.client = SessionCoverClient(categories, poi_count, k=5,
                    response_l=response_l, session_token=session_token,
                    static_catalogue_id=catalogue_id)
            cursor = self._cursor()
            tick = -1 if self.client.last_t is None else int(self.client.last_t/20)
            if cursor not in (tick, tick+1):
                raise ValueError('Cache missing/rolled back beyond a recoverable committed frame')
        except Exception:
            self.ledger.close()
            raise

    def _cursor(self):
        return self.ledger.db.execute('SELECT cursor FROM state WHERE id=1').fetchone()[0]

    def _save_cache(self):
        payload = dict(context=self.context, cache=self.client.checkpoint())
        fd, name = tempfile.mkstemp(prefix='.cache-', dir=self.directory)
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(payload, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.cache_path)
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try: os.fsync(directory_fd)
            finally: os.close(directory_fd)
        finally:
            if os.path.exists(name): os.unlink(name)

    def validate_next_tick(self, timestamp_s):
        if (isinstance(timestamp_s, bool) or not math.isfinite(timestamp_s)
                or timestamp_s < 0 or timestamp_s/20 != int(timestamp_s/20)):
            raise ValueError('Exact nonnegative public 20-second clock required')
        cursor = self._cursor()
        cached = -1 if self.client.last_t is None else int(self.client.last_t/20)
        if cursor == cached+1 and timestamp_s != cursor*20:
            raise ValueError('Recover the already committed frame before advancing the public clock')
        if int(timestamp_s/20) not in (cursor, cursor+1):
            raise ValueError('Public clock cannot skip, reset or reorder frames')

    def step(self, timestamp_s, *, protected_belief, server):
        self.validate_next_tick(timestamp_s)
        def select(previous, epsilon, simulator_rng):
            return self.policy.select(previous, epsilon, simulator_rng,
                belief=protected_belief, known=mask_bits(self.client.session_known))
        frame, _local_certificate = self.ledger.frame(timestamp_s, select)
        coordinates = [self.policy.library.rn.latlon(s) for s in frame]
        result = self.client.step(timestamp_s, coordinates, server)
        self._save_cache()
        # Only result['requests'] and the provider's replies are public. The
        # masks, belief, scores, certificates and ledger are local diagnostics.
        return result

    def local_answer(self, ranking, true_state, query):
        return local_sorted_pois(ranking, true_state, self.client.session_known, query)

    def close(self):
        self.ledger.close()


class GeoIBoundedQPipeline:
    """Bind the existing Geo-I engine to Q selection, without sending its Z.

Resume requires the SAME fully restored upstream engine. This class does not
claim to persist/reset Geo-I itself: a fresh upstream engine is rejected for an
already-started session. Sampling/cost certificates still carry their scopes.
"""
    def __init__(self, engine, query_session, projection):
        cursor = query_session._cursor()
        compatible_times = {query_session.client.last_t, cursor*20}
        if cursor >= 0 and (engine.last_t is None or engine.last_t not in compatible_times):
            raise ValueError('Restore upstream Geo-I history/ledger; never restart its budget')
        self.engine, self.session = engine, query_session
        self.projection = np.asarray(projection, dtype=int)

    def step(self, lat, lon, timestamp_s, *, server):
        self.session.validate_next_tick(timestamp_s)
        # Core's dummy coordinates are local intermediate outputs here. The
        # external LBS is queried ONLY from the bounded policy's Q frames.
        if self.engine.last_t != timestamp_s:
            self.engine.protect_step(lat, lon, timestamp_s)
        elif self.session.client.last_t != timestamp_s and self.session._cursor()*20 != timestamp_s:
            raise ValueError('Upstream already advanced outside this Q session')
        weights = self.engine.belief.weights
        belief = np.bincount(self.projection, weights=weights,
            minlength=len(self.session.policy.table.states))
        if belief.sum() > 0: belief /= belief.sum()
        return self.session.step(timestamp_s, protected_belief=belief, server=server)
