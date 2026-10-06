"""Persistent public-epoch accounting around the existing predictive Geo-I filter.

This allocator adds no location mechanism. Each admitted session runs the
existing REM/noisy-test engine with a preallocated cap. Slots and caps depend
only on public policy and session starts; unused credit is never recycled from
private branch history. The SQLite file and random key are PRIVATE local state.
Deleting/rolling back that file or using separate ledgers for one subject breaks
the operational claim. The underlying float sampler retains its existing
ideal-kernel limitation.
"""
from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3

import numpy as np


@dataclass(frozen=True)
class FixedEpochPolicy:
    epoch_id: str
    start_s: float
    end_s: float
    total_effective_epsilon_per_m: float = .23
    session_slots: int = 6
    horizon: int = 12
    read_interval_s: float = 60.

    def __post_init__(self):
        if not isinstance(self.epoch_id, str) or not self.epoch_id:
            raise ValueError('Nonempty public epoch_id required')
        for name in ('session_slots', 'horizon'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'Positive integer public {name} required')
        for name in ('start_s', 'end_s', 'total_effective_epsilon_per_m', 'read_interval_s'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Finite public {name} required')
        if self.end_s <= self.start_s or self.total_effective_epsilon_per_m <= 0 or self.read_interval_s < 0:
            raise ValueError('Increasing epoch, positive cap and nonnegative read interval required')

    @property
    def max_units_per_session(self):
        return 2*self.horizon-1

    @property
    def unit_epsilon_per_m(self):
        return self.total_effective_epsilon_per_m/(self.session_slots*self.max_units_per_session)

    @property
    def effective_session_cap_per_m(self):
        return self.total_effective_epsilon_per_m/self.session_slots

    @property
    def nominal_session_budget_per_m(self):
        # Existing engine uses u=B/(2H), with a matched filter cap of 2H-1 units.
        return 2*self.horizon*self.unit_epsilon_per_m

    def public_parameters(self):
        return dict(asdict(self), max_units_per_session=self.max_units_per_session,
                    unit_epsilon_per_m=self.unit_epsilon_per_m,
                    effective_session_cap_per_m=self.effective_session_cap_per_m,
                    nominal_session_budget_per_m=self.nominal_session_budget_per_m,
                    nominal_epoch_budget_per_m=self.session_slots*self.nominal_session_budget_per_m,
                    allocation='equal_caps_at_public_session_start_no_recycling',
                    excess_session='no_GPS_read_no_query',
                    scope='fixed_public_clock_coordinate_GeoI_only',
                    epoch_reset='explicit_public_policy_new_epoch_composes_with_previous_epochs')


@dataclass(frozen=True)
class SessionAllocation:
    slot: int
    session_token: str
    nominal_budget_per_m: float
    effective_cap_per_m: float
    unit_epsilon_per_m: float
    max_units: int
    horizon: int
    read_interval_s: float


@dataclass
class PrivateRngStreams:
    initialization: np.random.Generator
    anchor: np.random.Generator
    dummy: np.random.Generator


class PersistentEpochBudget:
    """One ledger per linked privacy subject/vehicle and PUBLIC fixed epoch.

    A transaction reserves the whole slot before any engine/GPS access. A crash
    can lose utility, never return credit. Reopening the file preserves slots;
    repeated tokens, changed policy and nonincreasing starts are rejected.
    The caller must keep this state shared for all linked apps/devices covered
    by its claim. This cannot discover identities from coordinates.
    """
    def __init__(self, policy, path, *, private_key=None):
        if not isinstance(policy, FixedEpochPolicy):
            raise ValueError('FixedEpochPolicy required')
        if private_key is not None and (not isinstance(private_key, bytes) or len(private_key) < 32):
            raise ValueError('Private random key must contain at least 32 bytes')
        self.policy, self.path = policy, Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Create restrictively; SQLite's atomic transactions prevent concurrent
        # model processes from consuming the same slot from this one ledger.
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(self.path, 0o600)
        self.db = sqlite3.connect(self.path, timeout=30., isolation_level=None)
        self.db.execute('CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK(id=1), policy TEXT NOT NULL, random_key BLOB NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS starts (token TEXT PRIMARY KEY, time REAL NOT NULL, slot INTEGER UNIQUE)')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            encoded = json.dumps(asdict(policy), sort_keys=True, separators=(',', ':'))
            row = self.db.execute('SELECT policy,random_key FROM config WHERE id=1').fetchone()
            if row is None:
                key = private_key if private_key is not None else secrets.token_bytes(32)
                self.db.execute('INSERT INTO config VALUES (1,?,?)', (encoded, key))
            else:
                if row[0] != encoded:
                    raise ValueError('Existing epoch policy cannot be changed or reset')
                key = bytes(row[1])
                if private_key is not None and not hmac.compare_digest(private_key, key):
                    raise ValueError('Existing private random key cannot be replaced')
            self._key = key
            self.db.execute('COMMIT')
        except Exception:
            self.db.execute('ROLLBACK')
            self.db.close()
            raise

    def reserve(self, session_token, public_start_s):
        if not isinstance(session_token, str) or not session_token:
            raise ValueError('Nonempty public session token required')
        self.validate_time(public_start_s)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT 1 FROM starts WHERE token=?', (session_token,)).fetchone():
                raise ValueError('Session token already consumed; cannot restart its random stream')
            last = self.db.execute('SELECT MAX(time) FROM starts').fetchone()[0]
            if last is not None and public_start_s <= last:
                raise ValueError('Strictly increasing public session starts required')
            admitted = self.db.execute('SELECT COUNT(*) FROM starts WHERE slot IS NOT NULL').fetchone()[0]
            slot = admitted if admitted < self.policy.session_slots else None
            self.db.execute('INSERT INTO starts VALUES (?,?,?)', (session_token, public_start_s, slot))
            self.db.execute('COMMIT')
        except Exception:
            self.db.execute('ROLLBACK')
            raise
        if slot is None:
            return None
        p = self.policy
        return SessionAllocation(slot, session_token, p.nominal_session_budget_per_m,
            p.effective_session_cap_per_m, p.unit_epsilon_per_m, p.max_units_per_session,
            p.horizon, p.read_interval_s)

    def validate_time(self, public_time_s):
        if isinstance(public_time_s, bool) or not isinstance(public_time_s, (int, float)) or not math.isfinite(public_time_s):
            raise ValueError('Finite public timestamp required')
        if not self.policy.start_s <= public_time_s < self.policy.end_s:
            raise ValueError('Timestamp outside fixed public epoch; no automatic reset')

    def private_rng_streams(self, allocation):
        # A stream is never derived from a public seed/session id alone. Domain
        # separation avoids correlated anchor/dummy draws and linked sessions.
        if not isinstance(allocation, SessionAllocation):
            raise ValueError('Session allocation required')
        row = self.db.execute('SELECT token FROM starts WHERE slot=?', (allocation.slot,)).fetchone()
        if row is None or row[0] != allocation.session_token:
            raise ValueError('Allocation does not belong to this epoch ledger')
        streams = {}
        for purpose in ('initialization', 'anchor', 'dummy'):
            message = json.dumps(['geo-i-session-stream-v1', self.policy.epoch_id,
                                  allocation.slot, purpose], separators=(',', ':')).encode()
            digest = hmac.new(self._key, message, hashlib.sha256).digest()
            streams[purpose] = np.random.default_rng(np.frombuffer(digest, dtype='<u4'))
        return PrivateRngStreams(**streams)

    @property
    def reserved_slots(self):
        return self.db.execute('SELECT COUNT(*) FROM starts WHERE slot IS NOT NULL').fetchone()[0]

    @property
    def reserved_cap_per_m(self):
        return self.reserved_slots*self.policy.effective_session_cap_per_m

    def close(self):
        self.db.close()


class FixedEpochProtectedSessions:
    """GPS-supplier boundary for existing paced, matched-cap Geo-I engines.

    engine_factory(allocation, private_rng_streams) returns a NEW existing
    engine with matched emission epsilons. ``protect_step(t, gps_supplier)``
    calls the supplier only when the public clock and prospective filter allow
    a read. No reset method is exposed. Public query results exclude balances,
    branch history, keys, GPS truth and linkage identifiers.
    """
    def __init__(self, ledger, engine_factory):
        self.ledger, self.engine_factory = ledger, engine_factory
        self._engine = self._allocation = None
        self._open = False
        self._last_global_time = None
        self._session_start = self._last_session_time = None
        self._completed = []

    def start_session(self, session_token, public_start_s):
        if self._open:
            raise ValueError('Close current public session before starting another')
        if self._last_global_time is not None and public_start_s <= self._last_global_time:
            raise ValueError('Public sessions cannot overlap or rewind')
        allocation = self.ledger.reserve(session_token, public_start_s)
        # Reservation survives a factory failure or process restart.
        engine = None
        if allocation is not None:
            engine = self.engine_factory(allocation, self.ledger.private_rng_streams(allocation))
            if (engine.horizon != allocation.horizon or engine.max_units != allocation.max_units
                or engine.spent_units != 0 or engine.n != 0
                or not math.isclose(engine.unit_epsilon, allocation.unit_epsilon_per_m, rel_tol=1e-12)
                or not math.isclose(engine.anchor.epsilon, allocation.unit_epsilon_per_m, rel_tol=1e-12)
                or not math.isclose(engine.anchor.eps_test, allocation.unit_epsilon_per_m, rel_tol=1e-12)
                or not math.isclose(engine.read_interval_s, allocation.read_interval_s, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError('Engine must match the reserved public filter and start fresh')
        self._engine, self._allocation = engine, allocation
        self._session_start = public_start_s
        self._last_session_time = None
        self._open = True
        return allocation is not None

    def protect_step(self, public_time_s, gps_supplier):
        if not self._open:
            raise ValueError('Start a public session first')
        self.ledger.validate_time(public_time_s)
        if public_time_s < self._session_start or (self._last_session_time is not None and public_time_s <= self._last_session_time):
            raise ValueError('Strictly increasing public session event times required')
        self._last_session_time = self._last_global_time = public_time_s
        engine = self._engine
        if engine is None:
            return ()
        t = public_time_s-self._session_start
        reserve = 1 if engine.last_anchor is None else 2
        paced = (engine.last_private_read_s is None or t-engine.last_private_read_s >= engine.read_interval_s)
        will_read = paced and engine.spent_units+reserve <= engine.max_units
        lat, lon = gps_supplier() if will_read else (math.nan, math.nan)
        result = engine.protect_step(lat, lon, t)
        if bool(engine.privacy_read_this_step) != will_read:
            raise RuntimeError('Engine read decision differs from prospective supplier boundary')
        if engine.spent_units > engine.max_units or engine.spent_bound > self._allocation.effective_cap_per_m+1e-12:
            raise RuntimeError('Existing engine exceeded preallocated session cap')
        return result

    def close_session(self, public_close_s):
        if not self._open:
            raise ValueError('No open public session')
        self.ledger.validate_time(public_close_s)
        if public_close_s < self._session_start or (self._last_session_time is not None and public_close_s < self._last_session_time):
            raise ValueError('Close cannot precede public session events')
        self._last_global_time = public_close_s
        self._completed.append(self.evaluator_current_session())
        self._engine = self._allocation = None
        self._open = False

    def evaluator_current_session(self):
        """PRIVATE evaluator/debug state; never include in attacker inputs."""
        engine, allocation = self._engine, self._allocation
        if allocation is None:
            return dict(admitted=False, spent_per_m=0., ledger=[], states=[])
        return dict(admitted=True, allocation=asdict(allocation), spent_per_m=engine.spent_bound,
                    private_reads=sum(r['private_read'] for r in engine.evaluator_ledger),
                    ledger=[dict(r) for r in engine.evaluator_ledger],
                    states=[list(s) for s in engine.evaluator_states])

    def evaluator_summary(self):
        sessions = self._completed + ([self.evaluator_current_session()] if self._open else [])
        spent = sum(s['spent_per_m'] for s in sessions)
        if spent > self.ledger.policy.total_effective_epsilon_per_m+1e-12:
            raise RuntimeError('Linked-session epoch exceeded its composition cap')
        return dict(sessions=sessions, spent_per_m=spent,
                    reserved_cap_per_m=self.ledger.reserved_cap_per_m,
                    epoch_cap_per_m=self.ledger.policy.total_effective_epsilon_per_m,
                    scope='this_process_spent_only_persistent_reservations_cover_all_processes')
