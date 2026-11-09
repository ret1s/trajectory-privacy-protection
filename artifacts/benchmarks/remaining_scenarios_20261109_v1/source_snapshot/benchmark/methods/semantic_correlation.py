"""Benchmark adapter for the 2026 semantic-correlation dummy scheme.

``SemanticCorrelationComparator.from_road_network`` is deliberately runnable
on the pinned SUMO/OSM graph without Amap or unreleased training assets.  That
constructor uses a bounded empirical fallback and is therefore an adaptation,
not a faithful reproduction.  The paper-exact primitives and inference shell
live in :mod:`benchmark.engines.semantic_correlation` so released author assets
can replace the fallback without changing the public benchmark contract.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Sequence

import numpy as np

from benchmark.contracts import (
    ComponentMapping,
    ComponentStatus,
    ImplementationLevel,
    MethodCard,
    MethodCardMixin,
    SourceReference,
)
from benchmark.engines.semantic_correlation import (
    HistoricalVisit,
    SemanticDummySelector,
    SemanticHierarchy,
    SemanticLocation,
    require_any_transition,
)
from core.demo_protocol import (
    EvaluationTruth,
    OutputKind,
    ProtectedRun,
    PublicCandidate,
    PublicEvent,
    PublicTranscript,
    TrajectoryPoint,
)


_SOURCE = SourceReference(
    citation=(
        "Liu, Peng, and Zhou, A Dummy-Based Location Privacy Protection "
        "Scheme with Semantic Correlation of Moving Paths, JKSUCIS 38:478, 2026"
    ),
    doi="10.1007/s44443-026-00899-w",
)


SEMANTIC_CORRELATION_CLEAN_ROOM_CARD = MethodCard(
    method_id="semantic_correlation_local_adaptation",
    display_name="Semantic-correlation 2026 - bounded local adaptation",
    output_kind=OutputKind.REAL_PLUS_DUMMIES,
    implementation_level=ImplementationLevel.PAPER_ADAPTATION,
    source=_SOURCE,
    source_mapping=(
        ComponentMapping(
            "paper grid, statistics, time weights and evaluation equations",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.semantic_correlation",
            "Implements Equations (1), (4)-(5), (17), (19), ASR/DER and the "
            "published 100x100 grid and Table-3 time schedule.",
        ),
        ComponentMapping(
            "two-layer LSTM, sigmoid attention and two-linear-layer inference",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.semantic_correlation.StackedLSTMSemanticNetwork",
            "Forward equations are executable with externally supplied weights; "
            "no unpublished width or weight initialization is guessed.",
        ),
        ComponentMapping(
            "semantic filtering, sibling expansion and top-K-minus-one selection",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.semantic_correlation.SemanticDummySelector",
            "Paper-specified steps are executable; unspecified decay, history "
            "aggregation and tie behavior are mandatory explicit dependencies.",
        ),
        ComponentMapping(
            "offline SUMO/OSM benchmark path",
            ComponentStatus.ADAPTED,
            "benchmark.methods.semantic_correlation.SemanticCorrelationComparator",
            "Uses bounded OSM road-context labels, an empirical label predictor "
            "and a local distance kernel so the common runner needs no web API.",
        ),
        ComponentMapping(
            "paper Amap semantic tree, processed GeoLife split and trained weights",
            ComponentStatus.MISSING,
            "not publicly available",
            "The article states that data are available on reasonable request and "
            "publishes neither model weights nor an official code repository.",
        ),
        ComponentMapping(
            "paper-equivalent ASR/DER reproduction",
            ComponentStatus.MISSING,
            "not validated",
            "The LSP posterior/Sim procedures, several architecture dimensions, "
            "Gen(delta_t), and exact preprocessing/split are not specified.",
        ),
    ),
    adaptation_summary=(
        "Complete deterministic real-plus-K-1 event sets using local OSM "
        "surrogates, plus source-mapped clean-room primitives. It must not be "
        "reported as the authors' trained semantic-correlation model."
    ),
)


class RoadSemanticLabeller:
    """Deterministic offline OSM road-context surrogate for unavailable Amap POIs."""

    _GROUPS = (
        ("arterial", frozenset({"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link"})),
        ("collector", frozenset({"secondary", "secondary_link", "tertiary", "tertiary_link"})),
        ("local", frozenset({"residential", "living_street", "unclassified"})),
        ("service", frozenset({"service"})),
        ("active", frozenset({"footway", "path", "pedestrian", "cycleway", "steps", "track"})),
    )

    def __init__(self, road_network: Any) -> None:
        self.rn = road_network
        self._cache: dict[int, str] = {}

    @staticmethod
    def _node_id(value: Any) -> Any:
        return value.item() if hasattr(value, "item") else value

    def label(self, vertex_index: int) -> str:
        index = int(vertex_index)
        if index in self._cache:
            return self._cache[index]
        node = self._node_id(self.rn.node_ids[index])
        graph = self.rn.graph
        highway_values: set[str] = set()
        edge_sets = []
        try:
            edge_sets.append(graph.out_edges(node, data=True))
        except (AttributeError, TypeError):
            edge_sets.append(graph.edges(node, data=True))
        try:
            edge_sets.append(graph.in_edges(node, data=True))
        except (AttributeError, TypeError):
            pass
        for edges in edge_sets:
            for _u, _v, data in edges:
                value = data.get("highway", "")
                values = value if isinstance(value, (list, tuple, set)) else (value,)
                highway_values.update(str(item) for item in values if item)
        result = "road_other"
        for group, members in self._GROUPS:
            if highway_values & members:
                result = f"road_{group}"
                break
        self._cache[index] = result
        return result


class EmpiricalSemanticTypeProvider:
    """Bounded deterministic fallback replacing the unavailable trained model."""

    def predict(
        self,
        historical_labels: Sequence[str],
        available_labels: Sequence[str],
        mu: int,
    ) -> tuple[str, ...]:
        available = tuple(dict.fromkeys(map(str, available_labels)))
        if not available:
            raise ValueError("available_labels must not be empty")
        counts = Counter(label for label in historical_labels if label in available)
        ranked = sorted(available, key=lambda label: (-counts[label], label))
        return tuple(ranked[: min(int(mu), len(ranked))])


class SemanticCorrelationComparator(MethodCardMixin):
    """Runnable, bounded paper adaptation with a real-plus-K-1 contract."""

    name = SEMANTIC_CORRELATION_CLEAN_ROOM_CARD.method_id
    method_card = SEMANTIC_CORRELATION_CLEAN_ROOM_CARD

    def __init__(
        self,
        road_network: Any,
        *,
        k: int = 4,
        mu: int = 3,
        candidate_pool_size: int = 128,
        history_limit: int = 8,
        transition_scale_m: float = 900.0,
        decay_half_life_minutes: float = 60.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if len(road_network) < int(k):
            raise ValueError("road network must contain at least K vertices")
        if int(k) < 2 or int(mu) < 1:
            raise ValueError("k must be >= 2 and mu must be >= 1")
        if int(candidate_pool_size) < int(k):
            raise ValueError("candidate_pool_size must be at least k")
        if int(history_limit) < 1:
            raise ValueError("history_limit must be positive")
        if float(transition_scale_m) <= 0 or float(decay_half_life_minutes) <= 0:
            raise ValueError("transition scale and decay half-life must be positive")
        self.rn = road_network
        self.k = int(k)
        self.mu = int(mu)
        self.candidate_pool_size = min(int(candidate_pool_size), len(road_network))
        self.history_limit = int(history_limit)
        self.transition_scale_m = float(transition_scale_m)
        self.decay_half_life_minutes = float(decay_half_life_minutes)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.labeller = RoadSemanticLabeller(road_network)
        self.semantic_provider = EmpiricalSemanticTypeProvider()

    @classmethod
    def from_road_network(
        cls, road_network: Any, **kwargs: Any
    ) -> "SemanticCorrelationComparator":
        """Construct the explicit local adaptation used by SUMO smoke runs."""

        return cls(road_network, **kwargs)

    def _nearest_indices(self, lat: float, lon: float) -> tuple[int, ...]:
        point_xy = np.asarray(self.rn.point_xy(lat, lon), dtype=float)
        _distances, indices = self.rn.tree.query(
            point_xy, k=self.candidate_pool_size
        )
        return tuple(dict.fromkeys(np.atleast_1d(indices).astype(int).tolist()))

    def _semantic_location(self, vertex_index: int) -> SemanticLocation:
        lat, lon = self.rn.latlon(vertex_index)
        return SemanticLocation(
            location_id=f"vertex_{int(vertex_index)}",
            lat=lat,
            lon=lon,
            semantic_type=self.labeller.label(vertex_index),
        )

    def _build_selector(
        self,
        indices: Sequence[int],
        source_history: Sequence[tuple[int, float, str]],
    ) -> SemanticDummySelector:
        locations = tuple(self._semantic_location(index) for index in indices)
        labels = tuple(dict.fromkeys(location.semantic_type for location in locations))
        hierarchy = SemanticHierarchy({label: "osm_road_context" for label in labels})
        transitions: dict[tuple[str, str], float] = {}
        thresholds: dict[str, float] = {}
        candidate_xy = self.rn.xy[np.asarray(indices, dtype=int)]
        for source_index, _timestamp, _label in source_history:
            origin = f"vertex_{source_index}"
            distances = np.linalg.norm(candidate_xy - self.rn.xy[source_index], axis=1)
            unnormalized = np.exp(-distances / self.transition_scale_m)
            probabilities = unnormalized / float(np.sum(unnormalized))
            for target_index, probability in zip(indices, probabilities):
                transitions[(origin, f"vertex_{target_index}")] = float(probability)
            # The paper does not publish its learned per-origin thresholds.  A
            # median can retain fewer than K-1 points for small candidate
            # pools, making a valid public K impossible.  The local adapter
            # therefore uses the K-th largest transition probability as an
            # explicit feasibility threshold; ``nextafter`` preserves the
            # paper's strict-greater-than predicate while retaining at least K
            # points before the real point is excluded.
            ranked_probabilities = np.sort(probabilities)[::-1]
            cutoff = ranked_probabilities[min(self.k - 1, len(probabilities) - 1)]
            thresholds[origin] = float(np.nextafter(cutoff, -np.inf))

        def local_decay(delta_minutes: float) -> float:
            # Explicit local fallback for the paper's unpublished Gen(delta_t).
            return math.exp(-math.log(2.0) * delta_minutes / self.decay_half_life_minutes)

        return SemanticDummySelector(
            locations,
            hierarchy,
            transitions,
            thresholds,
            threshold_decay=local_decay,
            eligibility_rule=require_any_transition,
            tie_breaker=lambda location: location.location_id,
        )

    def protect_run(self, real_trajectory: Sequence[Any]) -> ProtectedRun:
        real = tuple(real_trajectory)
        if not real or not all(isinstance(point, TrajectoryPoint) for point in real):
            raise TypeError("protect_run expects a non-empty TrajectoryPoint sequence")

        events: list[PublicEvent] = []
        real_candidate_ids: list[str] = []
        real_candidate_representations: list[TrajectoryPoint] = []
        # (road vertex index, timestamp in minutes, local semantic label)
        history: list[tuple[int, float, str]] = []
        for event_index, point in enumerate(real):
            indices = self._nearest_indices(point.lat, point.lon)
            real_index = int(self.rn.tree.query(self.rn.point_xy(point.lat, point.lon))[1])
            if real_index not in indices:
                indices = (real_index, *indices[:-1])
            current_label = self.labeller.label(real_index)
            query_minutes = point.timestamp_s / 60.0
            recent = [
                item
                for item in history[-self.history_limit :]
                if abs(query_minutes - item[1]) <= 335.0
            ]
            # A first online query still has its known current location as the
            # one-point empirical context.  This is fallback behavior, not a
            # claim about the unavailable historical GeoLife training corpus.
            source_history = recent or [(real_index, query_minutes, current_label)]
            available_labels = [self.labeller.label(index) for index in indices]
            predicted = self.semantic_provider.predict(
                [label for _idx, _time, label in source_history],
                available_labels,
                self.mu,
            )
            selector = self._build_selector(indices, source_history)
            selection = selector.select(
                predicted,
                real_location_id=f"vertex_{real_index}",
                query_timestamp_minutes=query_minutes,
                history=tuple(
                    HistoricalVisit(f"vertex_{index}", timestamp)
                    for index, timestamp, _label in source_history
                ),
                k=self.k,
            )

            # Candidate IDs are event-local: the source paper publishes sets,
            # not stable track labels.  The evaluator keeps the real ID apart.
            # Every public set member uses the same finite road-vertex catalog.
            # Publishing a raw SUMO lane coordinate beside vertex-valued
            # dummies would reveal the real member by representation alone.
            values = [(*self.rn.latlon(real_index), True)]
            for dummy in selection.selected:
                dummy_index = int(dummy.location_id.removeprefix("vertex_"))
                lat, lon = self.rn.latlon(dummy_index)
                values.append((lat, lon, False))
            permutation = self.rng.permutation(self.k)
            candidates: list[PublicCandidate] = []
            event_real_id = ""
            for public_position, source_position in enumerate(permutation):
                lat, lon, is_real = values[int(source_position)]
                candidate_id = f"event_{event_index:04d}_candidate_{public_position:04d}"
                candidates.append(PublicCandidate(candidate_id, lat, lon))
                if is_real:
                    event_real_id = candidate_id
            events.append(
                PublicEvent(
                    event_id=f"event_{event_index:04d}",
                    timestamp_s=point.timestamp_s,
                    candidates=tuple(candidates),
                )
            )
            real_candidate_ids.append(event_real_id)
            represented_lat, represented_lon = self.rn.latlon(real_index)
            real_candidate_representations.append(
                TrajectoryPoint(
                    point.timestamp_s,
                    represented_lat,
                    represented_lon,
                )
            )
            history.append((real_index, query_minutes, current_label))

        public_parameters = {
            **self.method_card.public_parameters(),
            "implementation_origin": (
                "benchmark.methods.semantic_correlation."
                "SemanticCorrelationComparator"
            ),
            "k": self.k,
            "mu": self.mu,
            "candidate_pool_size": self.candidate_pool_size,
            "semantic_source": "offline_osm_road_context_adaptation",
            "semantic_predictor": "bounded_empirical_frequency_adaptation",
            "threshold_decay": "explicit_local_exponential_adaptation",
            "threshold_policy": "k_feasible_order_statistic_adaptation",
            "history_limit": self.history_limit,
            "transition_scale_m": self.transition_scale_m,
            "decay_half_life_minutes": self.decay_half_life_minutes,
            "history_predicate": "require_any_transition",
            "tie_break_policy": "ascending_location_id",
            "candidate_linkage": "event_local_unlinked_sets",
            "candidate_representation": "shared_nearest_vertex_catalog",
            "real_location_quantization": "nearest_network_vertex",
        }
        run = ProtectedRun(
            PublicTranscript(
                mechanism=self.name,
                output_kind=OutputKind.REAL_PLUS_DUMMIES,
                events=tuple(events),
                public_parameters=tuple(public_parameters.items()),
            ),
            EvaluationTruth(
                real,
                tuple(real_candidate_ids),
                tuple(real_candidate_representations),
            ),
        )
        return self._attach_method_card(run)


__all__ = [
    "EmpiricalSemanticTypeProvider",
    "RoadSemanticLabeller",
    "SEMANTIC_CORRELATION_CLEAN_ROOM_CARD",
    "SemanticCorrelationComparator",
]
