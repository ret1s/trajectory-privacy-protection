"""Benchmark adapter for the evolving thesis method."""

from __future__ import annotations

from typing import Any, Sequence

from benchmark.contracts import (
    ComponentMapping,
    ComponentStatus,
    ImplementationLevel,
    MethodCard,
    MethodCardMixin,
    SourceReference,
)
from core.demo_protocol import OutputKind
from benchmark.engines.proposed import (
    GeoIAnchoredDummyEngine as _GeoIAnchoredDummyHeuristic,
)


GEO_I_ANCHORED_DUMMY_CARD = MethodCard(
    method_id="geo_i_anchored_dummy",
    display_name="Geo-I anchored dummy trajectories — thesis candidate",
    output_kind=OutputKind.DUMMY_ONLY,
    implementation_level=ImplementationLevel.THESIS_CANDIDATE,
    source=SourceReference(
        citation="Proposed thesis method: Geo-I anchor followed by dummy-only tracks"
    ),
    source_mapping=(
        ComponentMapping(
            "REM anchor sampling",
            ComponentStatus.IMPLEMENTED,
            "core.mechanisms.RoadExponential",
            "The first stage samples a road-network exponential-mechanism "
            "anchor with evaluator-audited privacy caveats.",
        ),
        ComponentMapping(
            "post-processing boundary",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.proposed.GeoIAnchoredDummyEngine.generate_from_anchors",
            "The dummy generator accepts public anchors, public graph/time and "
            "fresh randomness without reading the secret trajectory again.",
        ),
        ComponentMapping(
            "dummy-only output contract",
            ComponentStatus.IMPLEMENTED,
            "core.demo_protocol.make_dummy_only_run",
            "Only K opaque dummy tracks are public; anchors and truth remain on "
            "the evaluator side.",
        ),
        ComponentMapping(
            "population and POI prior",
            ComponentStatus.MISSING,
            "not implemented",
            "The present generator uses geometric offsets, not held-out "
            "population or semantic distributions.",
        ),
        ComponentMapping(
            "context-aware attacker objective",
            ComponentStatus.MISSING,
            "not implemented",
            "Dummy selection is not yet optimized against the final road/time/POI "
            "filtering attacker.",
        ),
        ComponentMapping(
            "trajectory-level privacy accounting",
            ComponentStatus.MISSING,
            "not implemented",
            "Per-release anchor arguments do not by themselves establish a "
            "fixed trajectory-level budget over repeated releases.",
        ),
    ),
    adaptation_summary=(
        "Executable thesis candidate with an explicit public-anchor "
        "post-processing boundary; algorithm and trajectory-level evaluation "
        "remain under development."
    ),
)


class GeoIAnchoredDummyTrajectories(
    MethodCardMixin, _GeoIAnchoredDummyHeuristic
):
    """Current executable thesis candidate, without the obsolete ``Lite`` name."""

    name = GEO_I_ANCHORED_DUMMY_CARD.method_id
    method_card = GEO_I_ANCHORED_DUMMY_CARD

    def protect_run(self, real_trajectory: Sequence[Any]):
        return self._attach_method_card(super().protect_run(real_trajectory))


__all__ = ["GEO_I_ANCHORED_DUMMY_CARD", "GeoIAnchoredDummyTrajectories"]
