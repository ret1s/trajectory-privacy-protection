"""Independent verifier fixtures; no dynamic dataset or scores are opened."""
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from core.road_network import RoadNetwork
from experiments.verify_dynamic_provider_status_20261006 import (
    OrderedRoadOracle, ReceivedHistory, aggregate, public_bit, score_ordered, wire,
)


def test_hash_id_and_absolute_epoch_do_not_depend_on_order_or_gps():
    first = {pid: public_bit(pid, 25) for pid in ("a", "b", "c")}
    assert first == {pid: public_bit(pid, 25) for pid in ("c", "a", "b")}
    assert isinstance(public_bit("a", 0), bool)
    with pytest.raises(AssertionError):
        public_bit("a", True)


def test_safe_partition_and_unknown_never_becomes_unavailable():
    case = {"nearest_distance": [[0, 1, 2, 3]]}
    row = score_ordered(case, {0, 1, 2}, {0, 1}, {0}, {0})["nearest_distance"]
    assert row["recall5"] == pytest.approx(1 / 3)
    assert row["reference_poi_total"] == 3
    assert row["retrieval_miss_reference_pois"] == row["current_status_unknown_reference_pois"] == row["known_available_reference_pois"] == 1
    assert row["returned_unavailable_items"] == row["returned_current_status_unknown_items"] == 0


def test_empty_reference_stays_na_and_nonempty_with_no_answer_is_zero():
    cases = {"nearest_distance": [[0, 1]], "within_radius": [[]]}
    values = score_ordered(cases, {0}, set(), set(), set())
    assert values["nearest_distance"]["recall5"] == 0
    assert values["within_radius"]["recall5"] is None
    assert values["within_radius"]["empty_reference_categories"] == 1
    assert score_ordered(cases, set(), {0}, {0}, set())["nearest_distance"]["recall5"] is None


def test_stale_invalid_control_counts_false_availability_without_safe_credit():
    case = {"nearest_distance": [[0, 1]]}
    invalid = score_ordered(case, {1}, {0, 1}, set(), {0}, {0})["nearest_distance"]
    safe = score_ordered(case, {1}, {0, 1}, set(), set())["nearest_distance"]
    assert invalid["returned_unavailable_items"] == invalid["returned_current_status_unknown_items"] == 1
    assert safe["returned_items"] == 0 and safe["current_status_unknown_reference_pois"] == 1
    with pytest.raises(AssertionError):
        score_ordered(case, {1}, {0}, {0}, {0})


def test_epoch_boundary_cache_causal_and_malformed_call_does_not_mutate():
    cache = ReceivedHistory()
    cache.observe(0., [(0, True)])
    _, inside = cache.observe(59., [(1, False)])
    assert inside == ({0, 1}, {0, 1}, {0})
    with pytest.raises(AssertionError):
        cache.observe(60., [(2, 1)])
    assert cache.last == 59. and cache.epoch == 0 and cache.bits == {0: True, 1: False}
    _, expired = cache.observe(60., [(2, True)])
    assert expired == ({0, 1, 2}, {2}, {2})
    _, later = cache.observe(1500., [(3, False)])
    assert later == ({0, 1, 2, 3}, {3}, set())
    with pytest.raises(AssertionError):
        cache.observe(20., [(0, True)])


def test_directed_oracle_distinguishes_distance_time_radius_detour():
    graph = nx.DiGraph()
    for i in range(5):
        graph.add_node(i, x=116. + i / 10000., y=39.9)
    for a, b, length, speed in [(0, 1, 100., 1.), (0, 2, 200., 20.), (1, 3, 600., 1.), (2, 3, 50., 10.), (3, 0, 300., 10.)]:
        graph.add_edge(a, b, length=length, speed=speed)
    context = SimpleNamespace(rn=RoadNetwork(graph), categories=("cafe",), pois=(
        dict(id="a", category="cafe", vertex=1), dict(id="b", category="cafe", vertex=2),
        dict(id="c", category="cafe", vertex=4)))
    cases = OrderedRoadOracle(context).ordered(0, 3)
    assert cases["nearest_distance"] == [[0, 1]]
    assert cases["fastest_travel"] == cases["minimum_detour"] == [[1, 0]]
    assert all(2 not in rows[0] for rows in cases.values())
    assert all(v["recall5"] == 1 for v in score_ordered(cases, {0, 1}, {0, 1, 2}, {0, 1, 2}, {0, 1}).values())


def test_wire_preserves_absolute_float_and_counts_only_status_overhead():
    at0 = wire([(39.9, 116.)], 0., ("cafe",), 20)
    at1500 = wire([(39.9, 116.)], 1500., ("cafe",), 20)
    assert at0[0] == at1500[0] == 1
    assert at1500[1] - at0[1] == 3
    assert at0[2] > 0 and at1500[2] > 0
    assert wire([(39.9, 116.)], 1500., ("cafe",), 30)[1] == at1500[1]


def test_whole_family_nested_draw_mean_does_not_weight_events_as_subjects():
    # Every arm has three draws/family; one family deliberately has ten events
    # per draw. Macro must still weight the two families equally.
    from experiments.verify_dynamic_provider_status_20261006 import ARMS, PURPOSES
    blocks = []
    for f, count, value in [("a", 1, 0.), ("b", 10, 1.)]:
        for draw in (1, 2, 3):
            rows = []
            for arm in ARMS:
                for event in range(count):
                    scored = score_ordered({p: [[0]] for p in PURPOSES}, {0}, {0} if value else set(), {0} if value else set(), {0} if value else set())
                    rows.append(dict(arm=arm, family_id=f, draw=draw, slot=0, t=400 + event, purposes=scored))
            blocks.append(dict(rows=rows, cost={arm: dict(requests=count) for arm in ARMS}, static_cache_max_records={"20": 1, "30": 1}))
    result = aggregate(blocks)
    row = result["summary"][ARMS[0]]["all"]["equal_purpose_macro"]
    assert row["family_mean"] == .5 and row["within_draw_family_mean"] == {"1": .5, "2": .5, "3": .5}
