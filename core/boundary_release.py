"""Causal boundary release layer for protect_step-compatible BR engines.

This is a delayed-publication interface, not zero-latency LBS. It makes no
claim that hiding samples prevents endpoint inference or timing side channels.
Session start and close are observable unless a separate transport hides them.
"""
from collections import deque
from dataclasses import dataclass
import math
from core.demo_protocol import PublicCandidate, PublicEvent


def nonnegative(value):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError('Expected finite nonnegative time')
    return float(value)


@dataclass(frozen=True)
class BoundaryPolicy:
    warmup_s: float = 60.0
    delay_s: float = 60.0

    def __post_init__(self):
        object.__setattr__(self, 'warmup_s', nonnegative(self.warmup_s))
        object.__setattr__(self, 'delay_s', nonnegative(self.delay_s))


class BoundaryProtectedStream:
    """One session only; wrap an existing causal engine, never reset per window.

    ingest(t, lat, lon) returns immutable PublicEvents whose timestamp is the
    actual release time. Source timestamps, suppression masks and private
    labels are deliberately excluded. delay=warmup=0 is the passthrough control.
    close() cancels queued events; it never flushes the last sensitive window.
    No timer ticks are synthesized: release happens at the next ingest, hence
    measured latency can exceed delay_s on irregular schedules.
    """
    def __init__(self, engine, policy=BoundaryPolicy(), *, session_start_s=0.0):
        if not math.isfinite(session_start_s):
            raise ValueError('Finite session start required')
        self.engine, self.policy, self.start = engine, policy, float(session_start_s)
        engine.reset()
        self.pending = deque()
        self.last_time = None
        self.closed = False
        self.generated = self.released = self.skipped = self.cancelled = 0
        self.k = None

    def ingest(self, timestamp_s, lat, lon):
        t = float(timestamp_s)
        if self.closed or not math.isfinite(t) or t < self.start or (self.last_time is not None and t <= self.last_time):
            raise ValueError('Active session and strictly increasing finite times required')
        if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError('Valid WGS84 input required')
        self.last_time = t
        if t-self.start < self.policy.warmup_s:
            self.skipped += 1
            return ()
        # Generate at observation time using only the causal BR state.
        positions = tuple(self.engine.protect_step(lat, lon, t))
        candidates = tuple(PublicCandidate(f'candidate_{i:04d}', a, b)
                           for i, (a,b) in enumerate(positions))
        if not candidates or (self.k is not None and len(candidates) != self.k):
            raise ValueError('Nonempty, fixed K required throughout a session')
        self.k = len(candidates)
        self.pending.append((t, candidates))
        self.generated += 1
        released = []
        while self.pending and self.pending[0][0]+self.policy.delay_s <= t:
            _, candidates = self.pending.popleft()
            released.append(PublicEvent(f'event_{self.released:06d}', t-self.start, candidates))
            self.released += 1
        return tuple(released)

    def close(self, timestamp_s):
        t=float(timestamp_s)
        if self.closed or not math.isfinite(t) or t < self.start or (self.last_time is not None and t < self.last_time):
            raise ValueError('Valid nondecreasing close time required')
        self.cancelled += len(self.pending)
        self.pending.clear()
        self.closed = True
        return ()

    def evaluator_summary(self):
        """Accounting only, never pass to the attacker as public parameters."""
        return dict(input_events=self.generated+self.skipped, head_suppressed=self.skipped,
                    protected_events=self.generated, released_events=self.released,
                    tail_cancelled=self.cancelled, pending_events=len(self.pending))
