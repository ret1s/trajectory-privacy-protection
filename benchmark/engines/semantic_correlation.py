"""Clean-room primitives for Liu, Peng, and Zhou (JKSUCIS 2026).

The source paper is DOI ``10.1007/s44443-026-00899-w``.  This module
implements the parts that the publication specifies precisely and fails closed
where a reproduction-critical choice is absent.  In particular, it does not
ship trained model weights, Amap annotations, the authors' dataset split, or a
guessed implementation of the unspecified ``Gen(delta_t)`` decay function.

The implementation is dependency-light (NumPy only) so paper-supplied weights
can eventually be loaded without forcing a deep-learning runtime into every
benchmark process.  It provides inference, not an invented training recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


PAPER_DOI = "10.1007/s44443-026-00899-w"
PAPER_LAT_BOUNDS = (39.8, 40.0)
PAPER_LON_BOUNDS = (116.2, 116.5)
PAPER_GRID_SHAPE = (100, 100)
PAPER_WEEK_EMBEDDING_DIM = 3
PAPER_TIME_SLOT_COUNT = 48
PAPER_TIME_EMBEDDING_DIM = 6
PAPER_SEMANTIC_EMBEDDING_DIM = 10
PAPER_TRAIN_VALIDATION_TEST_SPLIT = (0.7, 0.1, 0.2)
PAPER_TRAINING_EPOCHS = 20
PAPER_LEARNING_RATE = 0.01
PAPER_BATCH_SIZE = 128

# Table 3 reports percentages.  Equation (17) needs a multiplicative factor,
# hence this public API exposes the normalized values in [0, 1].
PAPER_TIME_WEIGHT_SCHEDULE = (
    (5.0, 0.75),
    (10.0, 0.21),
    (25.0, 0.10),
    (50.0, 0.05),
    (95.0, 0.03),
    (180.0, 0.02),
    (335.0, 0.01),
)


class PaperSpecificationGap(RuntimeError):
    """Raised instead of silently guessing a detail absent from the paper."""


class InsufficientPaperCandidates(RuntimeError):
    """Raised when Step 4 cannot form the requested K-location set."""


def _finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _probability(value: float, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True)
class GridCell:
    """One paper-grid cell represented by its geographic centroid."""

    row: int
    column: int
    cell_id: int
    centroid_lat: float
    centroid_lon: float


class BeijingGrid:
    """The 100 x 100 central-Beijing grid reported in Section 6.1."""

    def __init__(
        self,
        lat_bounds: tuple[float, float] = PAPER_LAT_BOUNDS,
        lon_bounds: tuple[float, float] = PAPER_LON_BOUNDS,
        shape: tuple[int, int] = PAPER_GRID_SHAPE,
    ) -> None:
        south, north = map(float, lat_bounds)
        west, east = map(float, lon_bounds)
        rows, columns = map(int, shape)
        if not south < north or not west < east:
            raise ValueError("grid bounds must be strictly increasing")
        if rows <= 0 or columns <= 0:
            raise ValueError("grid dimensions must be positive")
        self.lat_bounds = (south, north)
        self.lon_bounds = (west, east)
        self.shape = (rows, columns)
        self.lat_step = (north - south) / rows
        self.lon_step = (east - west) / columns

    @property
    def size(self) -> int:
        return self.shape[0] * self.shape[1]

    def cell(self, lat: float, lon: float) -> GridCell:
        lat = _finite(lat, "lat")
        lon = _finite(lon, "lon")
        south, north = self.lat_bounds
        west, east = self.lon_bounds
        if not south <= lat <= north or not west <= lon <= east:
            raise ValueError("coordinate lies outside the paper's Beijing grid")
        rows, columns = self.shape
        row = min(int((lat - south) / self.lat_step), rows - 1)
        column = min(int((lon - west) / self.lon_step), columns - 1)
        return self.from_indices(row, column)

    def from_indices(self, row: int, column: int) -> GridCell:
        rows, columns = self.shape
        row, column = int(row), int(column)
        if not 0 <= row < rows or not 0 <= column < columns:
            raise ValueError("grid indices out of bounds")
        south, _ = self.lat_bounds
        west, _ = self.lon_bounds
        return GridCell(
            row=row,
            column=column,
            cell_id=row * columns + column,
            centroid_lat=south + (row + 0.5) * self.lat_step,
            centroid_lon=west + (column + 0.5) * self.lon_step,
        )


def historical_query_probabilities(cell_ids: Sequence[int]) -> dict[int, float]:
    """Compute Equation (1): visits to a cell divided by all grid visits."""

    values = [int(cell_id) for cell_id in cell_ids]
    if not values:
        raise ValueError("at least one historical query is required")
    counts: dict[int, int] = {}
    for cell_id in values:
        if cell_id < 0:
            raise ValueError("cell IDs must be non-negative")
        counts[cell_id] = counts.get(cell_id, 0) + 1
    total = float(len(values))
    return {cell_id: count / total for cell_id, count in sorted(counts.items())}


@dataclass(frozen=True)
class TransitionObservation:
    """One neighboring-location transition from a historical path."""

    origin: str
    destination: str
    delta_minutes: float

    def __post_init__(self) -> None:
        if not self.origin or not self.destination:
            raise ValueError("transition endpoints must be non-empty")
        delta = _finite(self.delta_minutes, "delta_minutes")
        if delta < 0:
            raise ValueError("delta_minutes must be non-negative")
        object.__setattr__(self, "delta_minutes", delta)


def estimate_transition_probabilities(
    observations: Iterable[TransitionObservation],
) -> dict[tuple[str, str], float]:
    """Estimate the row-normalized first-order matrix in Equations (4)-(5)."""

    counts: dict[tuple[str, str], int] = {}
    origin_totals: dict[str, int] = {}
    for observation in observations:
        key = (observation.origin, observation.destination)
        counts[key] = counts.get(key, 0) + 1
        origin_totals[observation.origin] = origin_totals.get(observation.origin, 0) + 1
    return {
        key: count / origin_totals[key[0]]
        for key, count in sorted(counts.items())
    }


def conditional_transition_probabilities(
    observations: Iterable[TransitionObservation],
    lower_exclusive: float | None,
    upper_inclusive: float,
) -> dict[tuple[str, str], float]:
    """Build a time-conditioned transition set as described in Steps 2-3."""

    upper = _finite(upper_inclusive, "upper_inclusive")
    lower = None if lower_exclusive is None else _finite(lower_exclusive, "lower_exclusive")
    if upper < 0 or (lower is not None and lower >= upper):
        raise ValueError("invalid time interval")
    selected = (
        item
        for item in observations
        if item.delta_minutes <= upper
        and (lower is None or item.delta_minutes > lower)
    )
    return estimate_transition_probabilities(selected)


def transition_set_l1_difference(
    total: Mapping[tuple[str, str], float],
    conditioned: Mapping[tuple[str, str], float],
) -> float:
    """Aggregate the per-key differences described before Equation (18).

    The paper does not define Equation (18)'s ``total`` normalizer clearly, so
    this function deliberately returns only the unambiguous union-L1 sum.  It
    does not manufacture a final lambda from it.
    """

    keys = set(total) | set(conditioned)
    difference = 0.0
    for key in keys:
        left = _probability(total.get(key, 0.0), f"total[{key!r}]")
        right = _probability(conditioned.get(key, 0.0), f"conditioned[{key!r}]")
        difference += abs(left - right)
    return difference


def paper_time_weight(delta_minutes: float) -> float:
    """Return Table 3's time weight as a multiplier in [0, 1]."""

    delta = _finite(delta_minutes, "delta_minutes")
    if delta < 0:
        raise ValueError("delta_minutes must be non-negative")
    for upper, weight in PAPER_TIME_WEIGHT_SCHEDULE:
        if delta <= upper:
            return weight
    raise PaperSpecificationGap(
        "Table 3 defines no time weight for delta_t greater than 335 minutes"
    )


def adjusted_transition_probability(probability: float, delta_minutes: float) -> float:
    """Compute Equation (17), Pr' = lambda_t * Pr."""

    return _probability(probability, "probability") * paper_time_weight(delta_minutes)


def dynamic_threshold(
    base_threshold: float,
    delta_minutes: float,
    decay: Callable[[float], float] | None,
) -> float:
    """Compute Equation (19) while requiring the unspecified ``Gen`` hook."""

    threshold = _probability(base_threshold, "base_threshold")
    delta = _finite(delta_minutes, "delta_minutes")
    if delta < 0:
        raise ValueError("delta_minutes must be non-negative")
    if decay is None:
        raise PaperSpecificationGap(
            "Equation (19) requires Gen(delta_t), but the paper gives no function"
        )
    factor = _probability(decay(delta), "Gen(delta_minutes)")
    return threshold * factor


@dataclass(frozen=True)
class CoordinateNormalization:
    """Explicit coordinate statistics; the paper does not publish their values."""

    mean_lat: float
    mean_lon: float
    scale_lat: float
    scale_lon: float

    def __post_init__(self) -> None:
        for value, name in ((self.mean_lat, "mean_lat"), (self.mean_lon, "mean_lon")):
            _finite(value, name)
        for value, name in ((self.scale_lat, "scale_lat"), (self.scale_lon, "scale_lon")):
            if _finite(value, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def transform(self, coordinates: Sequence[Sequence[float]]) -> np.ndarray:
        values = np.asarray(coordinates, dtype=float)
        if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
            raise ValueError("coordinates must be a finite [sequence, 2] array")
        return (values - np.asarray([self.mean_lat, self.mean_lon])) / np.asarray(
            [self.scale_lat, self.scale_lon]
        )


@dataclass(frozen=True)
class FeatureEmbeddingTables:
    """Paper-sized weekday, half-hour, and POI embedding tables."""

    weekday: np.ndarray
    time_slot: np.ndarray
    semantic: np.ndarray

    def __post_init__(self) -> None:
        weekday = np.asarray(self.weekday, dtype=float).copy()
        time_slot = np.asarray(self.time_slot, dtype=float).copy()
        semantic = np.asarray(self.semantic, dtype=float).copy()
        if weekday.shape != (7, PAPER_WEEK_EMBEDDING_DIM):
            raise ValueError("weekday embeddings must have shape [7, 3]")
        if time_slot.shape != (PAPER_TIME_SLOT_COUNT, PAPER_TIME_EMBEDDING_DIM):
            raise ValueError("time-slot embeddings must have shape [48, 6]")
        if semantic.ndim != 2 or semantic.shape[1] != PAPER_SEMANTIC_EMBEDDING_DIM:
            raise ValueError("semantic embeddings must have shape [categories, 10]")
        if not all(np.isfinite(table).all() for table in (weekday, time_slot, semantic)):
            raise ValueError("embedding tables must be finite")
        for table in (weekday, time_slot, semantic):
            table.setflags(write=False)
        object.__setattr__(self, "weekday", weekday)
        object.__setattr__(self, "time_slot", time_slot)
        object.__setattr__(self, "semantic", semantic)


def encode_paper_features(
    coordinates: Sequence[Sequence[float]],
    weekdays: Sequence[int],
    half_hour_slots: Sequence[int],
    semantic_indices: Sequence[int],
    normalization: CoordinateNormalization,
    embeddings: FeatureEmbeddingTables,
) -> np.ndarray:
    """Encode the input-module fields using the paper's stated dimensions.

    The prose and Figure 6 say the fields are concatenated, while Equation (6)
    prints addition between differently sized vectors.  Concatenation is the
    only dimensionally valid reading and is isolated here for auditability.
    """

    spatial = normalization.transform(coordinates)
    n = len(spatial)
    weekday = np.asarray(weekdays, dtype=int)
    slots = np.asarray(half_hour_slots, dtype=int)
    semantics = np.asarray(semantic_indices, dtype=int)
    if any(len(values) != n for values in (weekday, slots, semantics)):
        raise ValueError("all feature sequences must have the same length")
    if np.any((weekday < 0) | (weekday >= 7)):
        raise ValueError("weekday indices must be in [0, 6]")
    if np.any((slots < 0) | (slots >= PAPER_TIME_SLOT_COUNT)):
        raise ValueError("half-hour slots must be in [0, 47]")
    if np.any((semantics < 0) | (semantics >= len(embeddings.semantic))):
        raise ValueError("semantic index out of range")
    return np.concatenate(
        [
            spatial,
            embeddings.weekday[weekday],
            embeddings.time_slot[slots],
            embeddings.semantic[semantics],
        ],
        axis=1,
    )


@dataclass(frozen=True)
class LSTMLayerWeights:
    """Weights for Equations (7)-(12), using [h_(t-1), x_t] input order."""

    forget_kernel: np.ndarray
    input_kernel: np.ndarray
    candidate_kernel: np.ndarray
    output_kernel: np.ndarray
    forget_bias: np.ndarray
    input_bias: np.ndarray
    candidate_bias: np.ndarray
    output_bias: np.ndarray

    @property
    def hidden_size(self) -> int:
        return int(np.asarray(self.forget_bias).shape[0])

    @property
    def input_size(self) -> int:
        return int(np.asarray(self.forget_kernel).shape[1] - self.hidden_size)

    def validate(self) -> None:
        hidden = self.hidden_size
        if hidden <= 0:
            raise ValueError("LSTM hidden size must be positive")
        width = hidden + self.input_size
        if self.input_size <= 0:
            raise ValueError("LSTM input size must be positive")
        for kernel, name in (
            (self.forget_kernel, "forget_kernel"),
            (self.input_kernel, "input_kernel"),
            (self.candidate_kernel, "candidate_kernel"),
            (self.output_kernel, "output_kernel"),
        ):
            values = np.asarray(kernel, dtype=float)
            if values.shape != (hidden, width) or not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite with shape [{hidden}, {width}]")
        for bias, name in (
            (self.forget_bias, "forget_bias"),
            (self.input_bias, "input_bias"),
            (self.candidate_bias, "candidate_bias"),
            (self.output_bias, "output_bias"),
        ):
            values = np.asarray(bias, dtype=float)
            if values.shape != (hidden,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must be finite with shape [{hidden}]")


@dataclass(frozen=True)
class DenseWeights:
    kernel: np.ndarray
    bias: np.ndarray

    def validate(self, input_size: int, name: str) -> int:
        kernel = np.asarray(self.kernel, dtype=float)
        bias = np.asarray(self.bias, dtype=float)
        if kernel.ndim != 2 or kernel.shape[1] != input_size:
            raise ValueError(f"{name}.kernel has incompatible input width")
        if bias.shape != (kernel.shape[0],):
            raise ValueError(f"{name}.bias has incompatible shape")
        if not np.isfinite(kernel).all() or not np.isfinite(bias).all():
            raise ValueError(f"{name} weights must be finite")
        return int(kernel.shape[0])


@dataclass(frozen=True)
class SemanticNetworkOutput:
    prediction: np.ndarray
    attention_weights: np.ndarray
    context: np.ndarray
    hidden_sequence: np.ndarray


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _lstm_forward(inputs: np.ndarray, weights: LSTMLayerWeights) -> np.ndarray:
    weights.validate()
    if inputs.ndim != 2 or inputs.shape[1] != weights.input_size:
        raise ValueError("LSTM input has incompatible shape")
    hidden = np.zeros(weights.hidden_size, dtype=float)
    cell = np.zeros(weights.hidden_size, dtype=float)
    output = []
    for current in inputs:
        joined = np.concatenate([hidden, current])
        forget = _sigmoid(np.asarray(weights.forget_kernel) @ joined + weights.forget_bias)
        incoming = _sigmoid(np.asarray(weights.input_kernel) @ joined + weights.input_bias)
        candidate = np.tanh(
            np.asarray(weights.candidate_kernel) @ joined + weights.candidate_bias
        )
        cell = forget * cell + incoming * candidate
        outgoing = _sigmoid(np.asarray(weights.output_kernel) @ joined + weights.output_bias)
        hidden = outgoing * np.tanh(cell)
        output.append(hidden.copy())
    return np.asarray(output)


class StackedLSTMSemanticNetwork:
    """Two-layer LSTM, sigmoid attention, and two-linear-layer inference.

    Hidden widths and all weights are constructor inputs because the paper does
    not publish them.  There is intentionally no random initializer that could
    be mistaken for the authors' trained model.
    """

    def __init__(
        self,
        layers: Sequence[LSTMLayerWeights],
        attention: DenseWeights,
        fc1: DenseWeights,
        fc2: DenseWeights,
    ) -> None:
        self.layers = tuple(layers)
        if len(self.layers) != 2:
            raise ValueError("the selected paper architecture has exactly two LSTM layers")
        for layer in self.layers:
            layer.validate()
        if self.layers[1].input_size != self.layers[0].hidden_size:
            raise ValueError("second LSTM input must equal first LSTM hidden width")
        second_hidden = self.layers[1].hidden_size
        attention_size = attention.validate(second_hidden + 1, "attention")
        if attention_size != 1:
            raise ValueError("Equation (13) requires one attention score per time step")
        fc1_size = fc1.validate(second_hidden, "fc1")
        fc2.validate(fc1_size, "fc2")
        self.attention = attention
        self.fc1 = fc1
        self.fc2 = fc2

    def forward(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        query_probabilities: Sequence[float] | np.ndarray,
    ) -> SemanticNetworkOutput:
        sequence = np.asarray(features, dtype=float)
        query = np.asarray(query_probabilities, dtype=float)
        if sequence.ndim != 2 or not len(sequence) or not np.isfinite(sequence).all():
            raise ValueError("features must be a non-empty finite matrix")
        if sequence.shape[1] != self.layers[0].input_size:
            raise ValueError("feature width does not match first LSTM layer")
        if query.shape != (len(sequence),):
            raise ValueError("query probabilities must align with the sequence")
        if np.any((query < 0.0) | (query > 1.0)):
            raise ValueError("query probabilities must be in [0, 1]")

        hidden = sequence
        for layer in self.layers:
            hidden = _lstm_forward(hidden, layer)
        attention_input = np.concatenate([hidden, query[:, None]], axis=1)
        scores = attention_input @ np.asarray(self.attention.kernel).T + self.attention.bias
        # Equation (15) explicitly uses sigmoid, not softmax normalization.
        alpha = _sigmoid(scores[:, 0])
        context = np.sum(alpha[:, None] * hidden, axis=0)
        intermediate = np.asarray(self.fc1.kernel) @ context + self.fc1.bias
        prediction = np.asarray(self.fc2.kernel) @ intermediate + self.fc2.bias
        return SemanticNetworkOutput(prediction, alpha, context, hidden)


SimilarityFunction = Callable[[np.ndarray, np.ndarray], float]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """An explicit optional interpretation; the paper names no similarity metric."""

    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("similarity vectors must be one-dimensional and aligned")
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return float(np.dot(left, right) / denominator)


def rank_semantic_types(
    prediction: Sequence[float],
    semantic_embeddings: Mapping[str, Sequence[float]],
    mu: int,
    similarity: SimilarityFunction | None,
) -> tuple[str, ...]:
    """Implement Section 4.1.3's top-mu ranking with an explicit metric hook."""

    if similarity is None:
        raise PaperSpecificationGap(
            "Section 4.1.3 requires vector similarity but does not name a metric"
        )
    if int(mu) <= 0:
        raise ValueError("mu must be positive")
    vector = np.asarray(prediction, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("prediction must be a finite vector")
    if len(semantic_embeddings) < int(mu):
        raise ValueError("fewer semantic embeddings than requested mu")
    scores = []
    for label, embedding in semantic_embeddings.items():
        candidate = np.asarray(embedding, dtype=float)
        score = _finite(similarity(vector, candidate), f"similarity[{label!r}]")
        scores.append((score, str(label)))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return tuple(label for _, label in scores[: int(mu)])


@dataclass(frozen=True)
class SemanticLocation:
    location_id: str
    lat: float
    lon: float
    semantic_type: str

    def __post_init__(self) -> None:
        if not self.location_id or not self.semantic_type:
            raise ValueError("location ID and semantic type must be non-empty")
        lat = _finite(self.lat, "lat")
        lon = _finite(self.lon, "lon")
        if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
            raise ValueError("invalid WGS84 coordinate")
        object.__setattr__(self, "lat", lat)
        object.__setattr__(self, "lon", lon)


@dataclass(frozen=True)
class HistoricalVisit:
    location_id: str
    timestamp_minutes: float

    def __post_init__(self) -> None:
        if not self.location_id:
            raise ValueError("historical location ID must be non-empty")
        object.__setattr__(
            self, "timestamp_minutes", _finite(self.timestamp_minutes, "timestamp_minutes")
        )


class SemanticHierarchy:
    """Parent links for the paper's depth-three Amap semantic tree."""

    def __init__(self, parents: Mapping[str, str | None]) -> None:
        self.parents = {str(child): None if parent is None else str(parent) for child, parent in parents.items()}
        if any(not child for child in self.parents):
            raise ValueError("semantic labels must be non-empty")

    def siblings(self, semantic_type: str) -> tuple[str, ...]:
        if semantic_type not in self.parents:
            raise PaperSpecificationGap(
                f"semantic type {semantic_type!r} is absent from the supplied hierarchy"
            )
        parent = self.parents[semantic_type]
        if parent is None:
            return ()
        return tuple(
            child
            for child, candidate_parent in self.parents.items()
            if candidate_parent == parent and child != semantic_type
        )


EligibilityRule = Callable[[Sequence[bool]], bool]
ThresholdDecay = Callable[[float], float]
TieBreaker = Callable[[SemanticLocation], object]


def require_all_transitions(flags: Sequence[bool]) -> bool:
    """One explicit interpretation of Step 1; callers must select it knowingly."""

    return bool(flags) and all(flags)


def require_any_transition(flags: Sequence[bool]) -> bool:
    """Alternative Step-1 interpretation retained for sensitivity analysis."""

    return any(flags)


@dataclass(frozen=True)
class RankedSemanticCandidate:
    location: SemanticLocation
    score: float
    adjusted_transitions: tuple[float, ...]


@dataclass(frozen=True)
class DummySelection:
    selected: tuple[SemanticLocation, ...]
    ranked: tuple[RankedSemanticCandidate, ...]
    used_sibling_expansion: bool


class SemanticDummySelector:
    """Sections 4.2.1-4.2.2 with all unspecified policies made explicit."""

    def __init__(
        self,
        locations: Sequence[SemanticLocation],
        hierarchy: SemanticHierarchy,
        transitions: Mapping[tuple[str, str], float],
        thresholds_by_origin: Mapping[str, float],
        threshold_decay: ThresholdDecay | None,
        eligibility_rule: EligibilityRule | None,
        tie_breaker: TieBreaker | None = None,
    ) -> None:
        self.locations = tuple(locations)
        if len({location.location_id for location in self.locations}) != len(self.locations):
            raise ValueError("semantic location IDs must be unique")
        if not self.locations:
            raise ValueError("at least one semantic location is required")
        self.by_id = {location.location_id: location for location in self.locations}
        self.hierarchy = hierarchy
        self.transitions = {
            (str(origin), str(destination)): _probability(value, "transition probability")
            for (origin, destination), value in transitions.items()
        }
        self.thresholds = {
            str(origin): _probability(value, "transition threshold")
            for origin, value in thresholds_by_origin.items()
        }
        if threshold_decay is None:
            raise PaperSpecificationGap(
                "SemanticDummySelector requires an explicit Gen(delta_t) implementation"
            )
        if eligibility_rule is None:
            raise PaperSpecificationGap(
                "the paper does not specify how Step-1 predicates aggregate over history"
            )
        self.threshold_decay = threshold_decay
        self.eligibility_rule = eligibility_rule
        self.tie_breaker = tie_breaker

    def _rank_types(
        self,
        semantic_types: Sequence[str],
        real_location_id: str,
        query_timestamp_minutes: float,
        history: Sequence[HistoricalVisit],
    ) -> list[RankedSemanticCandidate]:
        type_set = set(semantic_types)
        ranked: list[RankedSemanticCandidate] = []
        for location in self.locations:
            if location.location_id == real_location_id or location.semantic_type not in type_set:
                continue
            flags: list[bool] = []
            adjusted: list[float] = []
            for visit in history:
                if visit.location_id not in self.thresholds:
                    raise PaperSpecificationGap(
                        f"no transition threshold supplied for {visit.location_id!r}"
                    )
                delta = abs(query_timestamp_minutes - visit.timestamp_minutes)
                probability = self.transitions.get((visit.location_id, location.location_id), 0.0)
                threshold = dynamic_threshold(
                    self.thresholds[visit.location_id], delta, self.threshold_decay
                )
                flags.append(probability > threshold)
                adjusted.append(adjusted_transition_probability(probability, delta))
            if self.eligibility_rule(flags):
                ranked.append(
                    RankedSemanticCandidate(location, float(sum(adjusted)), tuple(adjusted))
                )
        ranked.sort(
            key=lambda candidate: (
                -candidate.score,
                self.tie_breaker(candidate.location) if self.tie_breaker else 0,
            )
        )
        return ranked

    def select(
        self,
        predicted_semantic_types: Sequence[str],
        real_location_id: str,
        query_timestamp_minutes: float,
        history: Sequence[HistoricalVisit],
        k: int,
    ) -> DummySelection:
        """Filter, expand to siblings if needed, rank, and return top K-1."""

        required = int(k) - 1
        if required <= 0:
            raise ValueError("k must be at least 2")
        if not history:
            raise ValueError("historical paths are required for transition scoring")
        if real_location_id not in self.by_id:
            raise ValueError("real location is absent from the semantic catalog")
        query_time = _finite(query_timestamp_minutes, "query_timestamp_minutes")
        primary = tuple(dict.fromkeys(map(str, predicted_semantic_types)))
        if not primary:
            raise ValueError("at least one predicted semantic type is required")

        ranked = self._rank_types(primary, real_location_id, query_time, history)
        expanded = False
        if len(ranked) < required:
            siblings: list[str] = []
            for semantic_type in primary:
                siblings.extend(self.hierarchy.siblings(semantic_type))
            expanded_types = tuple(dict.fromkeys((*primary, *siblings)))
            if expanded_types != primary:
                ranked = self._rank_types(
                    expanded_types, real_location_id, query_time, history
                )
                expanded = True

        if len(ranked) < required:
            raise InsufficientPaperCandidates(
                f"Step 4 retained {len(ranked)} candidates, but K={k} requires {required}"
            )
        if self.tie_breaker is None and len(ranked) > required:
            if math.isclose(
                ranked[required - 1].score,
                ranked[required].score,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise PaperSpecificationGap(
                    "candidate scores tie at the K-1 cutoff and the paper gives no tie rule"
                )
        selected = tuple(item.location for item in ranked[:required])
        return DummySelection(selected, tuple(ranked), expanded)


def anonymity_success_rate(
    real_location_posteriors: Sequence[float], k_values: int | Sequence[int]
) -> float:
    """Compute Section 6.3 ASR as a percentage.

    The caller must supply probabilities from an attacker.  The paper does not
    define an executable attacker that produces these posterior values.
    """

    probabilities = [_probability(value, "real-location posterior") for value in real_location_posteriors]
    if not probabilities:
        raise ValueError("at least one query posterior is required")
    if isinstance(k_values, int):
        ks = [int(k_values)] * len(probabilities)
    else:
        ks = [int(value) for value in k_values]
    if len(ks) != len(probabilities) or any(k < 2 for k in ks):
        raise ValueError("one K >= 2 is required for every query")
    successes = sum(probability <= 1.0 / k for probability, k in zip(probabilities, ks))
    return 100.0 * successes / len(probabilities)


def dummy_effectiveness_rate(effective: Sequence[bool]) -> float:
    """Compute Section 6.4's k-prime/k effectiveness ratio.

    ``Sim`` itself is unspecified by the paper, so the API consumes the final
    semantic-correlation decisions instead of fabricating a similarity metric.
    """

    flags = [bool(value) for value in effective]
    if not flags:
        raise ValueError("at least one dummy effectiveness decision is required")
    return sum(flags) / len(flags)


__all__ = [
    "BeijingGrid",
    "CoordinateNormalization",
    "DenseWeights",
    "DummySelection",
    "FeatureEmbeddingTables",
    "GridCell",
    "HistoricalVisit",
    "InsufficientPaperCandidates",
    "LSTMLayerWeights",
    "PAPER_BATCH_SIZE",
    "PAPER_DOI",
    "PAPER_GRID_SHAPE",
    "PAPER_LAT_BOUNDS",
    "PAPER_LEARNING_RATE",
    "PAPER_LON_BOUNDS",
    "PAPER_SEMANTIC_EMBEDDING_DIM",
    "PAPER_TIME_EMBEDDING_DIM",
    "PAPER_TIME_SLOT_COUNT",
    "PAPER_TIME_WEIGHT_SCHEDULE",
    "PAPER_TRAINING_EPOCHS",
    "PAPER_TRAIN_VALIDATION_TEST_SPLIT",
    "PAPER_WEEK_EMBEDDING_DIM",
    "PaperSpecificationGap",
    "RankedSemanticCandidate",
    "SemanticDummySelector",
    "SemanticHierarchy",
    "SemanticLocation",
    "SemanticNetworkOutput",
    "StackedLSTMSemanticNetwork",
    "TransitionObservation",
    "adjusted_transition_probability",
    "anonymity_success_rate",
    "conditional_transition_probabilities",
    "cosine_similarity",
    "dummy_effectiveness_rate",
    "dynamic_threshold",
    "encode_paper_features",
    "estimate_transition_probabilities",
    "historical_query_probabilities",
    "paper_time_weight",
    "rank_semantic_types",
    "require_all_transitions",
    "require_any_transition",
    "transition_set_l1_difference",
]
