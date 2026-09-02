"""Small, model-agnostic protocol for dummy-generation benchmarks.

The production benchmark currently assumes one secret point produces one
published point.  Dummy-generation mechanisms need a slightly wider contract:

* a replacement trajectory (AnotherMe/TransProtect-style public output),
* a set containing the real trajectory and ``K-1`` dummy trajectories, or
* a batch containing only dummy trajectories (the proposed architecture).

This module represents all three as a sequence of public events.  A candidate
ID that remains stable across events identifies one public trajectory.  Most
importantly, the ID of the real candidate and the original trajectory live in
``EvaluationTruth`` -- never in ``PublicTranscript``.

The dataclasses are deliberately immutable and JSON conversion is explicit:
``ProtectedRun`` has no ambiguous ``to_dict`` method.  Callers must choose
``to_attacker_dict`` or ``to_evaluator_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Optional, Sequence, Union


PublicScalar = Union[str, int, float, bool, None]


class OutputKind(str, Enum):
    """Public-output contracts supported by the benchmark."""

    REPLACEMENT_TRAJECTORY = "replacement_trajectory"
    REAL_PLUS_DUMMIES = "real_plus_dummies"
    DUMMY_ONLY = "dummy_only"


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a finite number, got {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


@dataclass(frozen=True)
class TrajectoryPoint:
    """A ground-truth or public trajectory sample in WGS84 coordinates."""

    timestamp_s: float
    lat: float
    lon: float

    def __post_init__(self) -> None:
        timestamp_s = _finite_number(self.timestamp_s, "timestamp_s")
        lat = _finite_number(self.lat, "lat")
        lon = _finite_number(self.lon, "lon")
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"lat must be in [-90, 90], got {lat}")
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"lon must be in [-180, 180], got {lon}")
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)

    def to_dict(self) -> dict[str, float]:
        return {
            "timestamp_s": self.timestamp_s,
            "lat": self.lat,
            "lon": self.lon,
        }


@dataclass(frozen=True)
class PublicCandidate:
    """One attacker-visible location in an event.

    ``candidate_id`` is an opaque public identifier.  Reusing it in later
    events links the locations into a public trajectory.  It must not encode
    whether the candidate is real.
    """

    candidate_id: str
    lat: float
    lon: float

    def __post_init__(self) -> None:
        candidate_id = str(self.candidate_id).strip()
        if not candidate_id:
            raise ValueError("candidate_id must be non-empty")
        lat = _finite_number(self.lat, "lat")
        lon = _finite_number(self.lon, "lon")
        if not -90.0 <= lat <= 90.0:
            raise ValueError(f"lat must be in [-90, 90], got {lat}")
        if not -180.0 <= lon <= 180.0:
            raise ValueError(f"lon must be in [-180, 180], got {lon}")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)

    def to_dict(self) -> dict[str, str | float]:
        return {"candidate_id": self.candidate_id, "lat": self.lat, "lon": self.lon}


@dataclass(frozen=True)
class PublicEvent:
    """One attacker-visible request/event and its published candidates."""

    event_id: str
    timestamp_s: float
    candidates: tuple[PublicCandidate, ...]

    def __post_init__(self) -> None:
        event_id = str(self.event_id).strip()
        if not event_id:
            raise ValueError("event_id must be non-empty")
        timestamp_s = _finite_number(self.timestamp_s, "timestamp_s")
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("a public event must contain at least one candidate")
        ids = [candidate.candidate_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError(f"candidate IDs must be unique within event {event_id!r}")
        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "timestamp_s", timestamp_s)
        object.__setattr__(self, "candidates", candidates)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "timestamp_s": self.timestamp_s,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


_PRIVATE_PARAMETER_KEYS = frozenset(
    {
        "ground_truth",
        "is_real",
        "private_label",
        "real_candidate_id",
        "real_candidate_ids",
        "real_event_mask",
        "real_index",
        "secret",
        "true_candidate_index",
        "true_index",
        "truth",
    }
)


def _normalise_public_parameters(
    parameters: Mapping[str, PublicScalar] | Sequence[tuple[str, PublicScalar]] | None,
) -> tuple[tuple[str, PublicScalar], ...]:
    if parameters is None:
        return ()
    items = parameters.items() if isinstance(parameters, Mapping) else parameters
    normalised: list[tuple[str, PublicScalar]] = []
    seen: set[str] = set()
    for raw_key, value in items:
        key = str(raw_key).strip()
        if not key:
            raise ValueError("public parameter keys must be non-empty")
        if key.lower() in _PRIVATE_PARAMETER_KEYS:
            raise ValueError(f"private field {key!r} cannot be a public parameter")
        if key in seen:
            raise ValueError(f"duplicate public parameter {key!r}")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"public parameter {key!r} must be a JSON scalar")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"public parameter {key!r} must be finite")
        seen.add(key)
        normalised.append((key, value))
    return tuple(sorted(normalised))


@dataclass(frozen=True)
class PublicTranscript:
    """The complete value an LSP/attacker is allowed to receive."""

    mechanism: str
    output_kind: OutputKind
    events: tuple[PublicEvent, ...]
    public_parameters: tuple[tuple[str, PublicScalar], ...] = ()

    def __post_init__(self) -> None:
        mechanism = str(self.mechanism).strip()
        if not mechanism:
            raise ValueError("mechanism must be non-empty")
        if not isinstance(self.output_kind, OutputKind):
            raise TypeError("output_kind must be an OutputKind")
        events = tuple(self.events)
        if not events:
            raise ValueError("a public transcript must contain at least one event")
        event_ids = [event.event_id for event in events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("event IDs must be unique")
        if any(events[i].timestamp_s > events[i + 1].timestamp_s for i in range(len(events) - 1)):
            raise ValueError("events must be ordered by non-decreasing timestamp")

        counts = [len(event.candidates) for event in events]
        if self.output_kind is OutputKind.REPLACEMENT_TRAJECTORY and any(
            count != 1 for count in counts
        ):
            raise ValueError("replacement output must publish exactly one candidate per event")

        parameters = _normalise_public_parameters(self.public_parameters)
        object.__setattr__(self, "mechanism", mechanism)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "public_parameters", parameters)

    def to_dict(self) -> dict[str, object]:
        """Serialize only attacker-visible information."""
        return {
            "mechanism": self.mechanism,
            "output_kind": self.output_kind.value,
            "events": [event.to_dict() for event in self.events],
            "public_parameters": dict(self.public_parameters),
        }


@dataclass(frozen=True)
class EvaluationTruth:
    """Evaluator-only labels; never pass this object to an attacker."""

    real_trajectory: tuple[TrajectoryPoint, ...]
    real_candidate_ids: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        real_trajectory = tuple(self.real_trajectory)
        if not real_trajectory:
            raise ValueError("real_trajectory must contain at least one point")
        candidate_ids = tuple(self.real_candidate_ids)
        for candidate_id in candidate_ids:
            if candidate_id is not None and not str(candidate_id).strip():
                raise ValueError("non-null real candidate IDs must be non-empty")
        object.__setattr__(self, "real_trajectory", real_trajectory)
        object.__setattr__(self, "real_candidate_ids", candidate_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "real_trajectory": [point.to_dict() for point in self.real_trajectory],
            "real_candidate_ids": list(self.real_candidate_ids),
        }


@dataclass(frozen=True)
class ProtectedRun:
    """Paired public transcript and private truth for offline evaluation."""

    transcript: PublicTranscript
    truth: EvaluationTruth

    def __post_init__(self) -> None:
        kind = self.transcript.output_kind
        truth_ids = self.truth.real_candidate_ids
        events = self.transcript.events
        real_trajectory = self.truth.real_trajectory

        if len(events) != len(real_trajectory):
            raise ValueError(
                f"{kind.value} events must align one-to-one with the real trajectory"
            )
        for event, point in zip(events, real_trajectory):
            if event.timestamp_s != point.timestamp_s:
                raise ValueError(
                    f"event {event.event_id!r} timestamp does not match the real point"
                )

        if kind is OutputKind.REAL_PLUS_DUMMIES:
            if len(truth_ids) != len(events):
                raise ValueError("one evaluator-only real candidate ID is required per event")
            for event, point, real_id in zip(
                events, real_trajectory, truth_ids
            ):
                if real_id is None:
                    raise ValueError("real-plus-dummies events require a real candidate ID")
                matches = [c for c in event.candidates if c.candidate_id == real_id]
                if len(matches) != 1:
                    raise ValueError(
                        f"truth candidate {real_id!r} is absent from event {event.event_id!r}"
                    )
                candidate = matches[0]
                if candidate.lat != point.lat or candidate.lon != point.lon:
                    raise ValueError(
                        f"truth candidate {real_id!r} does not match the real point in "
                        f"event {event.event_id!r}"
                    )
        elif truth_ids:
            raise ValueError(
                "replacement and dummy-only outputs must not designate a real public candidate"
            )

    def attacker_view(self) -> PublicTranscript:
        """Return the only object that attack code should accept."""
        return self.transcript

    def to_attacker_dict(self) -> dict[str, object]:
        return self.transcript.to_dict()

    def to_evaluator_dict(self) -> dict[str, object]:
        return {"public": self.transcript.to_dict(), "truth": self.truth.to_dict()}


def _as_points(points: Sequence[TrajectoryPoint]) -> tuple[TrajectoryPoint, ...]:
    result = tuple(points)
    if not result:
        raise ValueError("trajectory must contain at least one point")
    if not all(isinstance(point, TrajectoryPoint) for point in result):
        raise TypeError("trajectory values must be TrajectoryPoint instances")
    return result


def _events_from_tracks(
    tracks: Mapping[str, Sequence[TrajectoryPoint]],
) -> tuple[PublicEvent, ...]:
    if not tracks:
        raise ValueError("at least one public track is required")
    prepared: list[tuple[str, tuple[TrajectoryPoint, ...]]] = []
    for raw_id, raw_points in tracks.items():
        candidate_id = str(raw_id).strip()
        if not candidate_id:
            raise ValueError("public track IDs must be non-empty")
        prepared.append((candidate_id, _as_points(raw_points)))
    if len({candidate_id for candidate_id, _ in prepared}) != len(prepared):
        raise ValueError("public track IDs must be unique")

    lengths = {len(points) for _, points in prepared}
    if len(lengths) != 1:
        raise ValueError("public tracks must have the same number of samples")
    n_events = lengths.pop()
    reference_times = tuple(point.timestamp_s for point in prepared[0][1])
    for candidate_id, points in prepared[1:]:
        times = tuple(point.timestamp_s for point in points)
        if times != reference_times:
            raise ValueError(f"track {candidate_id!r} timestamps do not align")

    return tuple(
        PublicEvent(
            event_id=f"event_{i:04d}",
            timestamp_s=reference_times[i],
            candidates=tuple(
                PublicCandidate(candidate_id, points[i].lat, points[i].lon)
                for candidate_id, points in prepared
            ),
        )
        for i in range(n_events)
    )


def make_replacement_run(
    mechanism: str,
    real_trajectory: Sequence[TrajectoryPoint],
    replacement_trajectory: Sequence[TrajectoryPoint],
    *,
    public_parameters: Mapping[str, PublicScalar] | None = None,
) -> ProtectedRun:
    """Build a run where the attacker sees one replacement trajectory."""
    real = _as_points(real_trajectory)
    replacement = _as_points(replacement_trajectory)
    transcript = PublicTranscript(
        mechanism=mechanism,
        output_kind=OutputKind.REPLACEMENT_TRAJECTORY,
        events=_events_from_tracks({"candidate_0000": replacement}),
        public_parameters=_normalise_public_parameters(public_parameters),
    )
    return ProtectedRun(transcript, EvaluationTruth(real))


def make_real_plus_dummies_run(
    mechanism: str,
    real_trajectory: Sequence[TrajectoryPoint],
    candidate_tracks: Mapping[str, Sequence[TrajectoryPoint]],
    real_candidate_id: str,
    *,
    public_parameters: Mapping[str, PublicScalar] | None = None,
) -> ProtectedRun:
    """Build a candidate-set run containing the real track and ``K-1`` dummies.

    Candidate order is preserved.  A mechanism that wants randomized order
    should randomize it with its seeded RNG before calling this helper.
    """
    real = _as_points(real_trajectory)
    events = _events_from_tracks(candidate_tracks)
    real_id = str(real_candidate_id).strip()
    if not real_id:
        raise ValueError("real_candidate_id must be non-empty")
    truth = EvaluationTruth(real, tuple(real_id for _ in events))
    transcript = PublicTranscript(
        mechanism=mechanism,
        output_kind=OutputKind.REAL_PLUS_DUMMIES,
        events=events,
        public_parameters=_normalise_public_parameters(public_parameters),
    )
    return ProtectedRun(transcript, truth)


def make_dummy_only_run(
    mechanism: str,
    real_trajectory: Sequence[TrajectoryPoint],
    dummy_tracks: Mapping[str, Sequence[TrajectoryPoint]],
    *,
    public_parameters: Mapping[str, PublicScalar] | None = None,
) -> ProtectedRun:
    """Build a run whose public batch contains no designated real member."""
    real = _as_points(real_trajectory)
    transcript = PublicTranscript(
        mechanism=mechanism,
        output_kind=OutputKind.DUMMY_ONLY,
        events=_events_from_tracks(dummy_tracks),
        public_parameters=_normalise_public_parameters(public_parameters),
    )
    return ProtectedRun(transcript, EvaluationTruth(real))
