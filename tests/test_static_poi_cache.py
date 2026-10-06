import json
from types import SimpleNamespace
import networkx as nx
import numpy as np
import pytest
from benchmark.static_poi_cache import StaticPoiReplyCache
from benchmark.query_purpose import (PurposeIndependentCoverClient, MultiPurposeRoadRanking,
                                     QuerySpec, QueryPurpose)
from core.road_network import RoadNetwork


def record(name):
    return {'id': name, 'category': 'cafe', 'lat': 39.9, 'lon': 116.}


def test_rolling60_crosses_epoch_boundary_without_extending_status_validity():
    cache = StaticPoiReplyCache(['a', 'b', 'c'], ttl_s=60.)
    cache.receive(40., [record('a')])
    cache.receive(60., [record('b')])
    assert cache.static_ids(60.) == ('a', 'b')
    fresh = cache.dynamic_candidates(60., status_epoch=1, current_known_ids=['b'], current_available_ids=['b'])
    assert fresh == {'state': 'current_epoch', 'eligible': ('b',), 'unknown': ('a',), 'unavailable': ()}
    stale = cache.dynamic_candidates(60., status_epoch=0, current_known_ids=['a'], current_available_ids=['a'])
    assert stale['eligible'] == () and stale['unknown'] == ('a', 'b')
    assert cache.static_ids(100.) == ('b',)  # age60 expires exactly.


def test_no_availability_private_fields_or_conflicting_metadata_enter_static_cache():
    cache = StaticPoiReplyCache(['a'], ttl_s=120.)
    cache.receive(0., [record('a')])
    for field in ('available', 'opening_state', 'true_GPS', 'purpose', 'vehicle_id'):
        with pytest.raises(ValueError):
            cache.receive(20., [{**record('a'), field: True}])
    with pytest.raises(ValueError):
        cache.receive(20., [record('a'), {**record('a'), 'lon': 116.1}])
    assert cache.static_records(0.) == (record('a'),)
    with pytest.raises(ValueError):
        cache.dynamic_candidates(20., status_epoch=0, current_known_ids=[], current_available_ids=['a'])


def test_expired_dynamic_poi_cannot_be_reused_or_trigger_refresh():
    cache = StaticPoiReplyCache(['a', 'b'], ttl_s=180.)
    cache.receive(0., [record('a'), record('b')])
    assert cache.dynamic_candidates(20., status_epoch=0, current_known_ids=['a', 'b'], current_available_ids=['a']) == {
        'state': 'current_epoch', 'eligible': ('a',), 'unknown': (), 'unavailable': ('b',)}
    assert cache.dynamic_candidates(60., status_epoch=0, current_known_ids=['a'], current_available_ids=['a'])['eligible'] == ()
    assert cache.dynamic_candidates(180., status_epoch=3, current_known_ids=['a'], current_available_ids=['a'])['eligible'] == ()
    assert not hasattr(cache, 'server') and not hasattr(cache, 'fetch')


def test_local_ttl_and_purpose_never_change_public_request_bytes_or_q_order():
    graph = nx.DiGraph()
    for i in range(4):
        graph.add_node(i, x=116.+i/10000., y=39.9)
    for a, b, length, speed in [(0, 1, 100., 1.), (0, 2, 200., 20.),
                               (1, 3, 300., 1.), (2, 3, 10., 1.), (3, 0, 300., 10.)]:
        graph.add_edge(a, b, length=length, speed=speed)
    pois = tuple({'id': name, 'category': 'cafe', 'vertex': vertex}
                 for name, vertex in [('a', 1), ('b', 2), ('c', 3)])
    ranking = MultiPurposeRoadRanking(SimpleNamespace(rn=RoadNetwork(graph), pois=pois, categories=('cafe',)))
    queries = [QuerySpec(QueryPurpose.NEAREST, 'cafe'), QuerySpec(QueryPurpose.FASTEST, 'cafe'),
               QuerySpec(QueryPurpose.WITHIN_RADIUS, 'cafe', radius_m=150.),
               QuerySpec(QueryPurpose.MIN_DETOUR, 'cafe', destination_state=3)]
    wire = []
    local_answers = []
    for ttl in (60., 120., 180.):
        for query in queries:
            client = PurposeIndependentCoverClient(['cafe'], 3, k=2, response_l=20)
            cache = StaticPoiReplyCache(['a', 'b', 'c'], ttl_s=ttl)
            events = []
            answers = []
            for t, index in [(0., 0), (20., 1), (60., 2)]:
                q = [(39.9, 116.0002), (39.9, 116.0001)]
                reply = client.step(t, q, lambda request, i=index: [[i]])
                cache.receive(t, [record(('a', 'b', 'c')[index])])
                ids = cache.static_ids(t)
                candidate_mask = np.array([p['id'] in ids for p in pois])
                answers.append(ranking.top(0, candidate_mask, query))
                events.append(reply['requests'])
            wire.append(json.dumps(events, separators=(',', ':')))
            local_answers.append(json.dumps(answers))
    assert len(set(wire)) == 1
    assert len(set(local_answers)) > 1  # Actual road objectives/TTL change local answers.
    decoded = json.loads(wire[0])
    assert [q['coordinate'] for q in decoded[0]] == [[39.9, 116.0002], [39.9, 116.0001]]


def test_public_clock_rewind_and_reply_mutation_do_not_silently_reuse_cache():
    cache = StaticPoiReplyCache(['a'])
    payload = record('a')
    cache.receive(0., [payload])
    payload['lon'] = 115.
    answer = cache.static_records(20.)
    answer[0]['lon'] = 114.
    assert cache.static_records(20.)[0]['lon'] == 116.
    with pytest.raises(ValueError):
        cache.receive(10., [record('a')])
    with pytest.raises(ValueError):
        cache.static_ids(float('nan'))
    assert cache.static_ids(60.) == ()
