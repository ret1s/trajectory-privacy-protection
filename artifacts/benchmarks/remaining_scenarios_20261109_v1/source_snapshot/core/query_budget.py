"""Private, persistent whole-session Q accounting; no linked-session cap.

All publication happens AFTER an atomic commit. Reopening resumes cached segment
frames; a changed token/start/policy is rejected. Deleting/rolling back this local
file invalidates the claim. Metadata and session activation are outside scope.
"""
from fractions import Fraction
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3

import numpy as np


def allowance(gamma, block, phases=1):
    if (isinstance(gamma, bool) or not math.isfinite(gamma) or gamma < 0
            or isinstance(block, bool) or not isinstance(block, int) or block < 1
            or isinstance(phases, bool) or not isinstance(phases, int) or phases < 1):
        raise ValueError('Nonnegative finite Gamma, positive block/phases required')
    return Fraction(str(gamma)) / (block * (block+1) * phases)


class PersistentQueryBudget:
    """One immutable policy and private key per session file.

    Selector(previous_public_frame, exact_allowance, private_rng) returns a
    JSON-compatible dict with frames and a local certificate. It is called once
    per public block (joint mode), or once per public frame (step mode). Its
    library must be independent of the private belief conditional on previous Q.
    Diagnostics remain PRIVATE. No sampler/ledger object is sent to the server.
    """
    def __init__(self, path, *, session_token, gamma=1., start_s=0., joint=True,
                 context_id, private_key=None):
        allowance(gamma, 1)
        if (not isinstance(session_token, str) or not session_token
                or not isinstance(context_id, str) or not context_id
                or isinstance(start_s, bool) or not math.isfinite(start_s) or start_s < 0
                or not isinstance(joint, bool)
                or (private_key is not None and (not isinstance(private_key, bytes) or len(private_key) < 32))):
            raise ValueError('Immutable public session/context and private key required')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        os.chmod(self.path, 0o600)
        self.db = sqlite3.connect(self.path, timeout=30., isolation_level=None)
        self.db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), policy TEXT, key BLOB, cursor INTEGER, payload TEXT, spent TEXT)')
        self.policy = dict(token=session_token, gamma=float(gamma), start=float(start_s),
                           joint=joint, context=context_id, cadence_s=20., block_s=60.)
        encoded = json.dumps(self.policy, sort_keys=True)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            row = self.db.execute('SELECT policy,key FROM state WHERE id=1').fetchone()
            if row is None:
                key = private_key or secrets.token_bytes(32)
                self.db.execute('INSERT INTO state VALUES (1,?,?,-1,NULL,?)', (encoded, key, '0'))
            else:
                key = bytes(row[1])
                if row[0] != encoded or (private_key is not None and not hmac.compare_digest(key, private_key)):
                    raise ValueError('Existing Q session policy/key cannot be changed or reset')
            self.key = key
            self.db.execute('COMMIT')
        except Exception:
            self.db.execute('ROLLBACK')
            self.db.close()
            raise

    def frame(self, timestamp_s, selector):
        delta = (timestamp_s-self.policy['start'])/20.
        if isinstance(timestamp_s, bool) or not math.isfinite(delta) or delta < 0 or delta != int(delta):
            raise ValueError('Exact public 20-second clock required')
        tick = int(delta)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            cursor, encoded, spent = self.db.execute('SELECT cursor,payload,spent FROM state WHERE id=1').fetchone()
            payload = json.loads(encoded) if encoded else None
            if tick == cursor:  # idempotent retry: never sample again
                self.db.execute('COMMIT')
                return tuple(payload['last_frame']), payload['certificate']
            if tick != cursor+1:
                raise ValueError('Public clock cannot skip/reset/reorder frames')
            block, phase = tick//3+1, tick%3
            choose = not self.policy['joint'] or phase == 0
            if choose:
                epsilon = allowance(self.policy['gamma'], block, 1 if self.policy['joint'] else 3)
                domain = json.dumps(['q-segment-v1', self.policy['token'], block,
                                     0 if self.policy['joint'] else phase]).encode()
                seed = hmac.new(self.key, domain, hashlib.sha256).digest()
                rng = np.random.default_rng(np.frombuffer(seed, dtype='<u4'))
                previous = None if payload is None else tuple(payload['last_frame'])
                payload = selector(previous, epsilon, rng)
                frames = payload['frames']
                if len(frames) != (3 if self.policy['joint'] else 1):
                    raise ValueError('Selector must return the public number of frames')
                if any(len(f) != 5 or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in f) for f in frames):
                    raise ValueError('Five nonnegative integer road states per frame required')
                cert = payload['certificate']
                if not math.isfinite(cert['epsilon_Q_upper']) or not 0 <= cert['epsilon_Q_upper'] <= float(epsilon):
                    raise ValueError('Q certificate exceeds prospective allowance')
                spent = str(Fraction(spent)+epsilon)  # reserve allowed, not rounded measured loss
                if Fraction(spent) > Fraction(str(self.policy['gamma'])):
                    raise RuntimeError('Whole-session Q cap exceeded')
                cert.update(block=block, allocated_epsilon_Q=float(epsilon),
                            reserved_prefix_epsilon_Q=float(Fraction(spent)), gamma=self.policy['gamma'])
            frame = payload['frames'][phase if self.policy['joint'] else 0]
            payload['last_frame'] = frame
            self.db.execute('UPDATE state SET cursor=?,payload=?,spent=? WHERE id=1',
                            (tick, json.dumps(payload, allow_nan=False), spent))
            self.db.execute('COMMIT')
            return tuple(frame), payload['certificate']
        except Exception:
            self.db.execute('ROLLBACK')
            raise

    @property
    def reserved(self):
        return Fraction(self.db.execute('SELECT spent FROM state WHERE id=1').fetchone()[0])

    def close(self):
        self.db.close()
