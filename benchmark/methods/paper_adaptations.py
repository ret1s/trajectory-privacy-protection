"""Executable, source-mapped adaptations of three recent paper methods.

These classes have stable benchmark contracts and deterministic seeded
execution, but they are *not* renamed paper reproductions.  Each method card
states exactly which paper components are adapted or absent.  A caller asking
for reproduction-grade evidence must use ``require_faithful()`` and will get a
fail-closed error until the missing learned/data artifacts are implemented and
validated.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from benchmark.contracts import (
    ComponentMapping,
    ComponentStatus,
    ImplementationLevel,
    MethodCard,
    MethodCardMixin,
    SourceReference,
)
from core.demo_protocol import OutputKind
from benchmark.engines.paper_adaptations import (
    AnotherMeEngine as _AnotherMeHeuristic,
    SemanticDummyEngine as _SemanticDummyHeuristic,
    TransProtectEngine as _TransProtectHeuristic,
)


_TRANSPROTECT_SOURCE = SourceReference(
    citation=(
        "Yadav et al., Protecting Vehicle Location Privacy with "
        "Contextually-Driven Synthetic Location Generation, ACM SIGSPATIAL 2024"
    ),
    doi="10.1145/3678717.3691211",
    repository_url="https://github.com/sourabhy1797/VehiTrack",
    repository_revision="035684c6c666a9af7cbd9984d92300000eb65536",
)

TRANSPROTECT_CARD = MethodCard(
    method_id="transprotect_adaptation",
    display_name="TransProtect — local benchmark adaptation",
    output_kind=OutputKind.REPLACEMENT_TRAJECTORY,
    implementation_level=ImplementationLevel.PAPER_ADAPTATION,
    source=_TRANSPROTECT_SOURCE,
    source_mapping=(
        ComponentMapping(
            "single-pseudolocation output contract",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.paper_adaptations.TransProtectEngine.protect_run",
            "Each input event publishes exactly one opaque replacement point.",
        ),
        ComponentMapping(
            "road-constrained candidate pool",
            ComponentStatus.ADAPTED,
            "benchmark.engines.paper_adaptations._candidate_pool",
            "Uses radius/K nearest vertices from the pinned OSMnx graph instead "
            "of the paper's learned candidate pipeline.",
        ),
        ComponentMapping(
            "contextual candidate scoring",
            ComponentStatus.ADAPTED,
            "benchmark.engines.paper_adaptations.TransProtectEngine.protect_point",
            "Explicit distance, reachability and optional category penalties "
            "replace learned road/traffic scores.",
        ),
        ComponentMapping(
            "GCN/Transformer road and traffic representation",
            ComponentStatus.MISSING,
            "not implemented",
            "The checked public snapshot does not provide a drop-in Python "
            "training pipeline for the current Beijing/SUMO graph.",
        ),
        ComponentMapping(
            "VehiTrack mechanism-aware attacker",
            ComponentStatus.MISSING,
            "not implemented",
            "The current geometry/HMM diagnostics are not VehiTrack and cannot "
            "substitute for the paper's attack evaluation.",
        ),
        ComponentMapping(
            "paper-dataset reproduction",
            ComponentStatus.MISSING,
            "not implemented",
            "Rome/San Francisco graph, traffic, split and hyperparameter "
            "reproduction has not been executed.",
        ),
    ),
    adaptation_summary=(
        "Deterministic Python road-graph adaptation of the paper's output "
        "contract and contextual selection idea; not the learned TransProtect "
        "pipeline and not eligible for a reproduced-SOTA claim."
    ),
)


class TransProtectAdaptation(MethodCardMixin, _TransProtectHeuristic):
    """Runnable local adaptation; see :data:`TRANSPROTECT_CARD` for limits."""

    name = TRANSPROTECT_CARD.method_id
    method_card = TRANSPROTECT_CARD

    def protect_run(
        self,
        real_trajectory: Sequence[Any],
        poi_categories: Optional[Sequence[Any]] = None,
        vertex_categories: Any = None,
    ):
        return self._attach_method_card(
            super().protect_run(
                real_trajectory,
                poi_categories=poi_categories,
                vertex_categories=vertex_categories,
            )
        )


_ANOTHERME_SOURCE = SourceReference(
    citation=(
        "Li et al., AnotherMe: A Location Privacy Protection System Based on "
        "Online Virtual Trajectory Generation, IEEE TDSC 2024"
    ),
    doi="10.1109/TDSC.2023.3314200",
    repository_url="https://github.com/fang-zhiyou/AnotherMe",
    repository_revision="0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f",
)

ANOTHERME_CARD = MethodCard(
    method_id="anotherme_adaptation",
    display_name="AnotherMe — local benchmark adaptation",
    output_kind=OutputKind.REPLACEMENT_TRAJECTORY,
    implementation_level=ImplementationLevel.PAPER_ADAPTATION,
    source=_ANOTHERME_SOURCE,
    source_mapping=(
        ComponentMapping(
            "replacement-trajectory output contract",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.paper_adaptations.AnotherMeEngine.protect_run",
            "Publishes one time-indexed virtual trajectory while evaluator truth "
            "remains separate.",
        ),
        ComponentMapping(
            "whole-trajectory relocation",
            ComponentStatus.ADAPTED,
            "benchmark.engines.paper_adaptations.AnotherMeEngine.protect_trajectory",
            "A seeded affine relocation plus local road snapping replaces the "
            "paper/system virtual-user generation workflow.",
        ),
        ComponentMapping(
            "trajectory continuity",
            ComponentStatus.ADAPTED,
            "benchmark.engines.paper_adaptations._network_distances",
            "A road-distance consistency score is used instead of learned "
            "mobility-pattern simulation.",
        ),
        ComponentMapping(
            "virtual-user construction and history model",
            ComponentStatus.MISSING,
            "not implemented",
            "User history, stay points and virtual identity construction from "
            "the source system are absent.",
        ),
        ComponentMapping(
            "POI mapping and local route-service workflow",
            ComponentStatus.MISSING,
            "not implemented",
            "AMap/mobile calls are intentionally not copied; an equivalent "
            "local POI/routing adapter still needs validation.",
        ),
        ComponentMapping(
            "recognition and mobile-system reproduction",
            ComponentStatus.MISSING,
            "not implemented",
            "The paper classifier, latency, energy and phone experiments have "
            "not been reproduced.",
        ),
    ),
    adaptation_summary=(
        "Seeded local replacement-trajectory adaptation with road snapping; it "
        "does not implement AnotherMe's virtual user, POI mapping, learned "
        "mobility pattern or mobile system."
    ),
)


class AnotherMeAdaptation(MethodCardMixin, _AnotherMeHeuristic):
    """Runnable local adaptation; see :data:`ANOTHERME_CARD` for limits."""

    name = ANOTHERME_CARD.method_id
    method_card = ANOTHERME_CARD

    def protect_run(self, real_trajectory: Sequence[Any]):
        return self._attach_method_card(super().protect_run(real_trajectory))


_SEMANTIC_SOURCE = SourceReference(
    citation=(
        "Liu, Peng, and Zhou, A Dummy-Based Location Privacy Protection "
        "Scheme with Semantic Correlation of Moving Paths, JKSUCIS 2026"
    ),
    doi="10.1007/s44443-026-00899-w",
)

SEMANTIC_DUMMY_CARD = MethodCard(
    method_id="semantic_dummy_adaptation",
    display_name="Semantic-correlation dummy paths — local adaptation",
    output_kind=OutputKind.REAL_PLUS_DUMMIES,
    implementation_level=ImplementationLevel.PAPER_ADAPTATION,
    source=_SEMANTIC_SOURCE,
    source_mapping=(
        ComponentMapping(
            "real-plus-K-minus-one output contract",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.paper_adaptations.SemanticDummyEngine.protect_run",
            "Every event contains one evaluator-labelled real member and K-1 "
            "opaque public candidates with stable IDs.",
        ),
        ComponentMapping(
            "road/reachability candidate selection",
            ComponentStatus.ADAPTED,
            "benchmark.engines.paper_adaptations.SemanticDummyEngine.protect_point",
            "Road distance, movement mismatch and separation penalties replace "
            "the paper's exact grid filters and learned transition weights.",
        ),
        ComponentMapping(
            "semantic-category hook",
            ComponentStatus.ADAPTED,
            "benchmark.engines.paper_adaptations._category_at",
            "Accepts externally supplied vertex categories; it is not the "
            "paper's learned semantic predictor.",
        ),
        ComponentMapping(
            "LSTM/attention semantic prediction",
            ComponentStatus.MISSING,
            "not implemented",
            "No public source code/training artifact was identified; the model "
            "requires user-separated sequences and POI/time embeddings.",
        ),
        ComponentMapping(
            "paper grid, transition matrix and released parameters",
            ComponentStatus.MISSING,
            "not implemented",
            "The reported 100x100 Beijing grid and exact learned transition "
            "pipeline have not been reconstructed on the road graph.",
        ),
        ComponentMapping(
            "ASR and dummy-effectiveness reproduction",
            ComponentStatus.MISSING,
            "not implemented",
            "A calibrated semantic/path attacker and the paper dataset split "
            "are required before reproducing the reported metrics.",
        ),
    ),
    adaptation_summary=(
        "Road-graph adaptation of the paper's continuous real-plus-dummies "
        "contract with explicit heuristic temporal/semantic scores; not its "
        "LSTM/attention clean-room reimplementation."
    ),
)


class SemanticDummyAdaptation(MethodCardMixin, _SemanticDummyHeuristic):
    """Runnable local adaptation; see :data:`SEMANTIC_DUMMY_CARD` for limits."""

    name = SEMANTIC_DUMMY_CARD.method_id
    method_card = SEMANTIC_DUMMY_CARD

    def protect_run(
        self,
        real_trajectory: Sequence[Any],
        poi_categories: Optional[Sequence[Any]] = None,
        vertex_categories: Any = None,
    ):
        return self._attach_method_card(
            super().protect_run(
                real_trajectory,
                poi_categories=poi_categories,
                vertex_categories=vertex_categories,
            )
        )


PAPER_ADAPTATION_CARDS = (
    TRANSPROTECT_CARD,
    ANOTHERME_CARD,
    SEMANTIC_DUMMY_CARD,
)


__all__ = [
    "ANOTHERME_CARD",
    "AnotherMeAdaptation",
    "PAPER_ADAPTATION_CARDS",
    "SEMANTIC_DUMMY_CARD",
    "SemanticDummyAdaptation",
    "TRANSPROTECT_CARD",
    "TransProtectAdaptation",
]
