"""Stage-level conformance tests for the source-mapped AnotherMe VTGA.

These tests validate the public repository procedure, not parity with live
AMap responses or the paper's unreleased trained classifiers.
"""

import random

import networkx as nx
from shapely.geometry import LineString

from benchmark.contracts import ComponentStatus, ImplementationLevel
from benchmark.engines.anotherme import (
    AnotherMeGenerationError,
    AnotherMeVTGAEngine,
    RoadNetworkRouteProvider,
    RoadNetworkVirtualEndpointMapper,
    TransportMode,
    add_coordinate_noise,
    densify_route,
    filter_navigation_route,
    obfuscate_shape,
    replay_speed,
    select_transport_mode,
    trajectory_speed_profile,
)
from benchmark.methods.anotherme import ANOTHERME_CARD, AnotherMeAdaptation
from core.demo_protocol import OutputKind, TrajectoryPoint
from core.road_network import RoadNetwork


class _StaticRouter:
    name = "frozen_test_route"

    def __init__(self, route):
        self._route = tuple(route)
        self.calls = []

    def route(self, start, end, mode):
        self.calls.append((start, end, mode))
        return self._route


def _real_trace():
    # Roughly 8.5 m every 3 s at Beijing's latitude: the walking branch.
    return tuple(
        TrajectoryPoint(index * 3.0, 39.9, 116.3 + index * 0.0001)
        for index in range(5)
    )


def _navigation_route():
    return tuple((39.9, 116.3 + index * 0.0001) for index in range(7))


def _line_network(count=6):
    graph = nx.DiGraph()
    for index in range(count):
        graph.add_node(index, y=39.9, x=116.3 + index * 0.0001)
        if index:
            graph.add_edge(index - 1, index, length=8.5)
            graph.add_edge(index, index - 1, length=8.5)
    return RoadNetwork(graph)


def test_transport_mode_boundaries_match_public_python_vtga():
    assert select_transport_mode(0.0) is TransportMode.WALKING
    assert select_transport_mode(2.999) is TransportMode.WALKING
    assert select_transport_mode(3.0) is TransportMode.BICYCLING
    assert select_transport_mode(10.0) is TransportMode.BICYCLING
    assert select_transport_mode(10.001) is TransportMode.DRIVING


def test_speed_profile_uses_declared_distance_backend_and_two_decimal_rounding():
    real = _real_trace()
    average, sequence = trajectory_speed_profile(real)
    assert len(sequence) == len(real) - 1
    assert all(speed == round(speed, 2) for speed in sequence)
    assert average == round(sum(sequence) / len(sequence), 2)
    assert select_transport_mode(average) is TransportMode.WALKING


def test_route_filter_preserves_source_predecessor_semantics():
    # B is <6 m from A and is removed; C is compared with B, not A, and is
    # removed too. D is >=6 m from C and retained. The initial A is omitted.
    route = (
        (39.9, 116.30000),
        (39.9, 116.30003),
        (39.9, 116.30006),
        (39.9, 116.30016),
    )
    assert filter_navigation_route(route) == (route[-1],)


def test_densification_and_shape_obfuscation_preserve_stage_contracts():
    dense = densify_route(((39.9, 116.3), (39.9, 116.3001)))
    assert dense[0] == (39.9, 116.3)
    assert dense[-1] == (39.9, 116.3001)
    assert len(dense) == round(8.54 / 2.0) + 1

    right_angle = tuple(
        [(39.9, 116.3 + index * 0.0001) for index in range(6)]
        + [(39.9 + index * 0.0001, 116.3005) for index in range(1, 7)]
    )
    shaped = obfuscate_shape(right_angle)
    assert len(shaped) == len(right_angle)
    assert shaped[0] == right_angle[0]
    assert shaped[-1] == right_angle[-1]
    assert shaped != right_angle


def test_speed_replay_and_noise_are_seed_deterministic():
    dense = densify_route(_navigation_route()[1:])
    speed = trajectory_speed_profile(_real_trace())[1]
    selected = replay_speed(dense, speed)
    assert selected
    assert selected[0] == dense[0]

    first = add_coordinate_noise(selected, random.Random(17))
    second = add_coordinate_noise(selected, random.Random(17))
    assert first == second
    for before, after in zip(selected, first):
        assert abs(before[0] - after[0]) <= 0.0000255
        assert abs(before[1] - after[1]) <= 0.0000255


def test_raw_vtga_is_reproducible_and_keeps_three_second_timestamps():
    router_a = _StaticRouter(_navigation_route())
    router_b = _StaticRouter(_navigation_route())
    first_engine = AnotherMeVTGAEngine(router_a, seed=23)
    second_engine = AnotherMeVTGAEngine(router_b, seed=23)

    first = first_engine.generate_virtual_trajectory(_real_trace())
    second = second_engine.generate_virtual_trajectory(_real_trace())
    assert first == second
    assert first == (
        TrajectoryPoint(0.0, 39.899981, 116.300098),
        TrajectoryPoint(3.0, 39.900021, 116.300176),
        TrajectoryPoint(6.0, 39.900009, 116.300299),
        TrajectoryPoint(9.0, 39.900016, 116.300405),
        TrajectoryPoint(12.0, 39.899985, 116.300502),
        TrajectoryPoint(15.0, 39.899996, 116.300590),
    )
    assert [point.timestamp_s for point in first] == [
        index * 3.0 for index in range(len(first))
    ]
    assert router_a.calls[0][2] is TransportMode.WALKING
    assert first_engine.last_trace is not None
    counts = first_engine.last_trace.stage_counts()
    assert counts["navigation"] == len(_navigation_route())
    assert counts["virtual"] == len(first)


def test_local_router_returns_a_graph_shortest_path():
    road_network = _line_network()
    provider = RoadNetworkRouteProvider(road_network)
    route = provider.route(
        road_network.latlon(0), road_network.latlon(5), TransportMode.DRIVING
    )
    assert route == tuple(road_network.latlon(index) for index in range(6))


def test_local_router_preserves_osm_edge_geometry():
    graph = nx.MultiDiGraph()
    graph.add_node(0, y=39.9, x=116.3)
    graph.add_node(1, y=39.901, x=116.301)
    graph.add_edge(
        0,
        1,
        length=160.0,
        geometry=LineString(
            [(116.3, 39.9), (116.3007, 39.9002), (116.301, 39.901)]
        ),
    )
    provider = RoadNetworkRouteProvider(RoadNetwork(graph))

    route = provider.route((39.9, 116.3), (39.901, 116.301), TransportMode.DRIVING)

    assert route == (
        (39.9, 116.3),
        (39.9002, 116.3007),
        (39.901, 116.301),
    )


def test_local_router_filters_active_edges_for_driving():
    graph = nx.MultiDiGraph()
    graph.add_node(0, y=39.9, x=116.3)
    graph.add_node(1, y=39.901, x=116.301)
    graph.add_node(2, y=39.9, x=116.301)
    graph.add_edge(0, 1, length=1.0, highway="footway")
    graph.add_edge(0, 2, length=10.0, highway="residential")
    graph.add_edge(2, 1, length=10.0, highway="residential")
    provider = RoadNetworkRouteProvider(RoadNetwork(graph))

    route = provider.route((39.9, 116.3), (39.901, 116.301), TransportMode.DRIVING)

    assert route == ((39.9, 116.3), (39.9, 116.301), (39.901, 116.301))
    assert provider.last_mode_policy == "local_mode_filtered_routing"


def test_sumo_router_trusts_exact_passenger_permissions_over_osm_label():
    graph = nx.MultiDiGraph(source="eclipse_sumo_passenger_network")
    graph.add_node(0, y=39.9, x=116.3)
    graph.add_node(1, y=39.901, x=116.301)
    # A SUMO passenger graph contains only edges for which sumolib reported
    # allows("passenger").  A generic highway-name blacklist must not
    # reinterpret that exact simulator permission.
    graph.add_edge(0, 1, length=160.0, highway="track")
    provider = RoadNetworkRouteProvider(RoadNetwork(graph))

    route = provider.route(
        (39.9, 116.3), (39.901, 116.301), TransportMode.DRIVING
    )

    assert route == ((39.9, 116.3), (39.901, 116.301))
    assert provider.last_mode_policy == "sumo_passenger_permissions"


def test_local_virtual_endpoint_mapper_relocates_and_preserves_od_pattern():
    road_network = _line_network(count=10)
    mapper_a = RoadNetworkVirtualEndpointMapper(
        road_network, anchor_min_m=20.0, anchor_max_m=60.0, seed=13
    )
    mapper_b = RoadNetworkVirtualEndpointMapper(
        road_network, anchor_min_m=20.0, anchor_max_m=60.0, seed=13
    )
    real_start, real_end = road_network.latlon(1), road_network.latlon(3)
    mapped_a = mapper_a.map_endpoints(real_start, real_end)
    mapped_b = mapper_b.map_endpoints(real_start, real_end)
    assert mapped_a == mapped_b
    assert mapped_a[0] != real_start
    assert mapped_a[0] != mapped_a[1]
    assert mapper_a.last_mapping["origin_relocation_m"] >= 20.0


def test_local_router_fails_closed_on_one_way_disconnection():
    graph = nx.DiGraph()
    for index in range(3):
        graph.add_node(index, y=39.9, x=116.3 + index * 0.0001)
    graph.add_edge(1, 0, length=8.5)
    graph.add_edge(2, 1, length=8.5)
    road_network = RoadNetwork(graph)
    provider = RoadNetworkRouteProvider(road_network)
    try:
        provider.route(
            road_network.latlon(0), road_network.latlon(2), TransportMode.DRIVING
        )
    except AnotherMeGenerationError as exc:
        assert "directionally valid route" in str(exc)
    else:
        raise AssertionError("a one-way reversal must not be used as a route")


def test_benchmark_adapter_aligns_events_and_keeps_truth_private():
    router = _StaticRouter(_navigation_route())
    mechanism = AnotherMeAdaptation(
        route_provider=router, minimum_raw_samples=1, seed=31
    )
    run = mechanism.protect_run(_real_trace())

    assert run.transcript.output_kind is OutputKind.REPLACEMENT_TRAJECTORY
    assert run.transcript.mechanism == "anotherme_adaptation"
    assert len(run.transcript.events) == len(_real_trace())
    assert run.truth.real_candidate_ids == ()
    public = str(run.to_attacker_dict())
    assert "real_trajectory" not in public
    assert "real_candidate_ids" not in public
    parameters = dict(run.transcript.public_parameters)
    assert parameters["benchmark_alignment"] == "normalized_time_interpolation"
    assert parameters["route_provider"] == "frozen_test_route"
    assert parameters["reportable_as_reproduced_sota"] is False


def test_local_mode_label_is_not_exposed_in_attacker_parameters():
    mechanism = AnotherMeAdaptation(
        route_provider=RoadNetworkRouteProvider(_line_network()),
        minimum_raw_samples=1,
        seed=31,
    )
    run = mechanism.protect_run(_real_trace())

    parameters = dict(run.transcript.public_parameters)
    assert parameters["mode_routing_policy"] == "local_mode_filtered_routing"
    rendered = str(parameters["mode_routing_policy"])
    assert all(mode.value not in rendered for mode in TransportMode)


def test_benchmark_adapter_rejects_upstream_invalid_short_generation():
    mechanism = AnotherMeAdaptation(
        route_provider=_StaticRouter(_navigation_route()), seed=31
    )
    try:
        mechanism.protect_run(_real_trace())
    except AnotherMeGenerationError as exc:
        assert "requires at least 20" in str(exc)
    else:
        raise AssertionError("short upstream-invalid output must not be stretched")


def test_method_card_is_complete_about_implemented_and_blocked_components():
    assert ANOTHERME_CARD.implementation_level is ImplementationLevel.PAPER_ADAPTATION
    assert ANOTHERME_CARD.reportable_as_reproduced_sota is False
    statuses = {mapping.status for mapping in ANOTHERME_CARD.source_mapping}
    assert statuses == {
        ComponentStatus.IMPLEMENTED,
        ComponentStatus.ADAPTED,
        ComponentStatus.MISSING,
    }
    assert set(ANOTHERME_CARD.missing_components) == {
        "virtual-user stay-point and POI mapping",
        "AMap coordinate conversion and route-response parity",
        "paper-equivalent privacy and mobile-system validation",
    }
    assert ANOTHERME_CARD.source.repository_revision == (
        "0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f"
    )
