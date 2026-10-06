import json
import pytest
from benchmark.versioned_static_geoi_lbs import VersionedStaticGeoILbsClient
from benchmark.query_purpose import QuerySpec, QueryPurpose
from tests.test_budgeted_geoi_lbs import client_fixture


def opt_in(base):
    return VersionedStaticGeoILbsClient(base, catalogue_version='fixed-public-map',
        public_epoch_id='day', start_s=0., end_s=86400.)


def test_real_lazy_geoi_read_and_no_extra_requests_for_static_local_answers(tmp_path):
    base, ledger, sessions, rn = client_fixture(tmp_path)
    client = opt_in(base);wire = [];reads = []
    try:
        assert client.start_session('first', 0.)
        client.public_tick(0., lambda: reads.append(1) or rn.latlon(0), lambda q: wire.append(q) or [[0]])
        spent = sessions.evaluator_current_session()['spent_per_m']
        client.public_tick(60., lambda: pytest.fail('GPS after exhausted H1'), lambda q: wire.append(q) or [[1]])
        query = QuerySpec(QueryPurpose.NEAREST, base.ranking.pois[0]['category'])
        assert 0 in client.answer_static(query, 70., *rn.latlon(0))
        assert 0 not in client.answer_live(query, 70., *rn.latlon(0))  # Only current epoch reply1 is known.
        with pytest.raises(ValueError, match='expired'):
            client.answer_live(query, 120., *rn.latlon(0))
        assert 0 in client.answer_static(query, 120., *rn.latlon(0))
        assert len(reads) == 1 and len(wire) == 6
        assert sessions.evaluator_current_session()['spent_per_m'] == spent
        client.close_session(121.)
        assert not client.start_session('denied', 200.)
        assert client.public_tick(200., lambda: pytest.fail('denied GPS'), lambda q: pytest.fail('denied wire')) is None
        with pytest.raises(ValueError, match='No active'):
            client.answer_static(query, 200., *rn.latlon(0))
        client.close_session(201.)
    finally:
        ledger.close()


def test_static_answers_purpose_gps_or_answer_times_never_change_future_public_wire(tmp_path):
    a, la, _, ra = client_fixture(tmp_path/'a')
    b, lb, _, rb = client_fixture(tmp_path/'b')
    static = opt_in(b);wires = [[], []]
    try:
        for client, rn, wire in [(a, ra, wires[0]), (static, rb, wires[1])]:
            client.start_session('same', 0.)
            client.public_tick(0., lambda r=rn: r.latlon(0), lambda q, w=wire: w.append(q) or [[0, 1]])
        # Even a future local answer time does not advance the public/cache
        # clock. The next fixed public tick must remain byte-for-byte equal.
        for t, purpose in [(10., QueryPurpose.NEAREST), (1000., QueryPurpose.FASTEST), (30., QueryPurpose.MIN_DETOUR)]:
            query = QuerySpec(purpose, 'cafe', destination_state=15) if purpose == QueryPurpose.MIN_DETOUR else QuerySpec(purpose, 'cafe')
            static.answer_static(query, t, *rb.latlon(15))
        for client, wire in [(a, wires[0]), (static, wires[1])]:
            client.public_tick(60., lambda: pytest.fail('unexpected GPS'), lambda q, w=wire: w.append(q) or [[0, 1]])
            client.close_session(61.)
        assert json.dumps(wires[0], separators=(',', ':')) == json.dumps(wires[1], separators=(',', ':'))
    finally:
        la.close();lb.close()


def test_catalogue_mismatch_invalidates_before_gps_or_request_and_local_error_never_refetches(tmp_path):
    base, ledger, _, rn = client_fixture(tmp_path)
    client = opt_in(base)
    try:
        client.start_session('first', 0.)
        client.public_tick(0., lambda: rn.latlon(0), lambda q: [[0]])
        with pytest.raises(ValueError, match='expired'):
            client.answer_static(QuerySpec(QueryPurpose.NEAREST, 'cafe'), 10., *rn.latlon(0), catalogue_version='new-map')
        with pytest.raises(ValueError, match='expired'):
            client.public_tick(60., lambda: pytest.fail('mismatch GPS'), lambda q: pytest.fail('mismatch request'), catalogue_version='new-map')
        with pytest.raises(ValueError, match='expired'):
            client.answer_static(QuerySpec(QueryPurpose.NEAREST, 'cafe'), 60., *rn.latlon(0))
        with pytest.raises(ValueError, match='expired'):
            client.answer_live(QuerySpec(QueryPurpose.NEAREST, 'cafe'), 60., *rn.latlon(0))
        client.close_session(61.)
    finally:
        ledger.close()


def test_live_answers_enforce_declared_short_epoch_without_mutating_future_wire(tmp_path):
    a, la, sa, ra = client_fixture(tmp_path/'base')
    b, lb, sb, rb = client_fixture(tmp_path/'optin')
    client = VersionedStaticGeoILbsClient(b, catalogue_version='fixed-public-map',
        public_epoch_id='short', start_s=20., end_s=40.)
    wires = [[], []]
    try:
        for target, rn, wire in [(a, ra, wires[0]), (client, rb, wires[1])]:
            target.start_session('same', 20.)
            target.public_tick(20., lambda r=rn: r.latlon(0), lambda q, w=wire: w.append(q) or [[0]])
        query = QuerySpec(QueryPurpose.NEAREST, b.ranking.pois[0]['category'])
        # The base response remains current throughout [0,60). The opt-in
        # scope is narrower and must reject both the exact end and later time.
        assert 0 in a.answer(query, 50., *ra.latlon(0))
        assert 0 in client.answer_live(query, 39., *rb.latlon(0))
        spent = sb.evaluator_current_session()['spent_per_m']
        cache_clock = client.cache._store._clock
        for t in (19., 40., 50.):
            for answer in (client.answer_static, client.answer_live):
                with pytest.raises(ValueError, match='expired or answer clock rewound'):
                    answer(query, t, *rb.latlon(0))
        assert client.cache._store._clock == cache_clock == 20.
        assert not client._static_invalidated
        assert sb.evaluator_current_session()['spent_per_m'] == spent
        for target, wire in [(a, wires[0]), (client, wires[1])]:
            target.public_tick(30., lambda: pytest.fail('GPS after exhausted H1'), lambda q, w=wire: w.append(q) or [[0]])
            target.close_session(31.)
        assert json.dumps(wires[0], separators=(',', ':')) == json.dumps(wires[1], separators=(',', ':'))
        assert sa.evaluator_summary()['spent_per_m'] == sb.evaluator_summary()['spent_per_m']
    finally:
        la.close();lb.close()
