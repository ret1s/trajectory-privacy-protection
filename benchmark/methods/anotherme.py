"""Benchmark method card for the source-mapped AnotherMe VTGA.

This module replaces the earlier affine-relocation prototype with a clean-room
implementation of every locally executable stage in the authors' standalone
VTGA.  It remains a ``paper_adaptation`` because the complete system depends on
AMap/GCJ02 services, virtual-user POI construction, and paper-equivalent
recognition/system validation that are not reproducible from the repository
alone.
"""

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
from benchmark.engines.anotherme import (
    AnotherMeVTGAEngine,
    CoordinateConverter,
    RoadNetworkRouteProvider,
    RoadNetworkVirtualEndpointMapper,
    RouteProvider,
    VirtualEndpointMapper,
)
from core.demo_protocol import OutputKind, TrajectoryPoint


ANOTHERME_SOURCE = SourceReference(
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
    display_name="AnotherMe — source-mapped public-VTGA adaptation",
    output_kind=OutputKind.REPLACEMENT_TRAJECTORY,
    implementation_level=ImplementationLevel.PAPER_ADAPTATION,
    source=ANOTHERME_SOURCE,
    source_mapping=(
        ComponentMapping(
            "speed-profile and transport-mode selection",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.anotherme.trajectory_speed_profile / select_transport_mode",
            "Clean-room implementation of VTGAs/gen_virtual_traj.py init_ and its "
            "walking <3, bicycling 3..10, driving >10 m/s dispatch; WGS84 Geod "
            "provides the same ellipsoidal distance model as geopy.geodesic.",
        ),
        ComponentMapping(
            "six-metre navigation-route filtering",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.anotherme.filter_navigation_route",
            "Preserves the source's immediate-predecessor comparison and omission "
            "of navigation element zero.",
        ),
        ComponentMapping(
            "two-metre route densification",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.anotherme.densify_route",
            "Uses the source count round(distance/2)-1 and six-decimal linear "
            "interpolation with WGS84 ellipsoidal distance.",
        ),
        ComponentMapping(
            "Bezier turn-shape obfuscation",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.anotherme.obfuscate_shape",
            "Reproduces 70--110 degree corner detection, five-sample windows and "
            "quadratic Bezier replacement.",
        ),
        ComponentMapping(
            "three-second speed replay",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.anotherme.replay_speed",
            "Sequentially accepts points within the source's strict 1.1 m tolerance "
            "of speed[i] * 3 m and cycles the speed sequence.",
        ),
        ComponentMapping(
            "discrete geographic noise and timestamps",
            ComponentStatus.IMPLEMENTED,
            "benchmark.engines.anotherme.add_coordinate_noise / AnotherMeVTGAEngine.generate_virtual_trajectory",
            "Uses independent inclusive integer draws [-20,20]/800000 degrees, "
            "six-decimal rounding and 3 s timestamps from the input start time.",
        ),
        ComponentMapping(
            "virtual endpoint relocation",
            ComponentStatus.ADAPTED,
            "benchmark.engines.anotherme.RoadNetworkVirtualEndpointMapper",
            "Relocates the origin 0.7--3 km and preserves the real OD displacement "
            "on reachable graph vertices as a deterministic local analogue of "
            "cross-city virtual POI mapping.",
        ),
        ComponentMapping(
            "navigation service adapter",
            ComponentStatus.ADAPTED,
            "benchmark.engines.anotherme.RoadNetworkRouteProvider",
            "The benchmark uses a deterministic directed shortest path, preserves "
            "edge polylines, and applies the graph's mode policy instead of AMap "
            "v5 walking/bicycling/driving.",
        ),
        ComponentMapping(
            "benchmark event-grid alignment",
            ComponentStatus.ADAPTED,
            "benchmark.engines.anotherme._align_normalized_time",
            "The variable-length official 3 s output is interpolated onto the "
            "benchmark's one-event-per-input grid; the raw output is retained.",
        ),
        ComponentMapping(
            "minimum generated-trajectory validity gate",
            ComponentStatus.ADAPTED,
            "benchmark.engines.anotherme.AnotherMeVTGAEngine",
            "The local benchmark enforces the upstream experiment's minimum of "
            "20 generated samples before normalized-time alignment.",
        ),
        ComponentMapping(
            "virtual-user stay-point and POI mapping",
            ComponentStatus.MISSING,
            "not implemented",
            "The mobile source extracts 200 m/30 min stay points, queries AMap POI "
            "types/candidates, and selects virtual-city POIs; exact city anchors, "
            "service responses and a portable end-to-end dataset are not published.",
        ),
        ComponentMapping(
            "AMap coordinate conversion and route-response parity",
            ComponentStatus.MISSING,
            "not implemented",
            "The official Python component calls WGS84-to-GCJ02 conversion and AMap "
            "v5 directions. Embedded keys are not reused and no frozen responses or "
            "service-version manifest is supplied.",
        ),
        ComponentMapping(
            "paper-equivalent privacy and mobile-system validation",
            ComponentStatus.MISSING,
            "not implemented",
            "No published split manifest, trained recognition artifacts, expected "
            "golden VTGA outputs, or reproducible Android/iOS energy/latency harness "
            "is available at the pinned revision.",
        ),
    ),
    adaptation_summary=(
        "Clean-room execution of the authors' public standalone VTGA stages with "
        "local SUMO/OSM routing and explicit benchmark alignment. This is a much "
        "closer comparator than the former affine prototype, but is not yet an "
        "official or paper-equivalent reproduction of the complete AnotherMe system."
    ),
    validation_evidence=(
        "tests/test_anotherme_faithful.py: deterministic stage-level conformance to pinned public VTGA",
    ),
)


class AnotherMeAdaptation(MethodCardMixin, AnotherMeVTGAEngine):
    """Source-mapped AnotherMe comparator for the local benchmark graph."""

    name = ANOTHERME_CARD.method_id
    method_card = ANOTHERME_CARD

    def __init__(
        self,
        road_network: Any | None = None,
        *,
        route_provider: RouteProvider | None = None,
        coordinate_converter: CoordinateConverter | None = None,
        endpoint_mapper: VirtualEndpointMapper | None = None,
        anchor_min_m: float = 700.0,
        anchor_max_m: float = 3_000.0,
        minimum_raw_samples: int = 20,
        seed: int = 0,
        rng: Any | None = None,
    ) -> None:
        if rng is not None:
            if not hasattr(rng, "integers"):
                raise TypeError("rng must provide numpy Generator.integers")
            seed = int(rng.integers(0, 2**63 - 1))
        if route_provider is None:
            if road_network is None:
                raise ValueError("road_network or route_provider is required")
            route_provider = RoadNetworkRouteProvider(road_network)
            if endpoint_mapper is None:
                endpoint_mapper = RoadNetworkVirtualEndpointMapper(
                    road_network,
                    anchor_min_m=anchor_min_m,
                    anchor_max_m=anchor_max_m,
                    seed=seed,
                )
        elif road_network is not None:
            raise ValueError("provide road_network or route_provider, not both")
        super().__init__(
            route_provider,
            coordinate_converter=coordinate_converter,
            endpoint_mapper=endpoint_mapper,
            minimum_raw_samples=minimum_raw_samples,
            seed=seed,
        )

    def protect_run(self, real_trajectory: Sequence[TrajectoryPoint]):
        return self._attach_method_card(super().protect_run(real_trajectory))


__all__ = ["ANOTHERME_CARD", "ANOTHERME_SOURCE", "AnotherMeAdaptation"]
