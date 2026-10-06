from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.budgeted_geoi_lbs import BudgetedGeoILbsClient
from benchmark.engines.endpoint_noise import EndpointNoiseProgressLaneDummy
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from core.session_budget import FixedEpochPolicy, PersistentEpochBudget, FixedEpochProtectedSessions
from tests.test_belief_lane import fixture


def client_fixture(tmp_path):
    rn, context, _ = fixture()
    policy = FixedEpochPolicy('day', 0., 86400., session_slots=1, horizon=1)
    belief = PublicAnchorModel(rn, context, np.ones(len(rn)), spacing_m=2.,
        epsilon_release=policy.unit_epsilon_per_m, epsilon_test=policy.unit_epsilon_per_m)
    def factory(allocation, streams):
        engine = EndpointNoiseProgressLaneDummy(rn, belief_model=belief, privacy_scale=1.,
            budget=allocation.nominal_budget_per_m, horizon=allocation.horizon,
            k=3, rng=streams.initialization, read_interval_s=allocation.read_interval_s)
        engine.anchor_rng, engine.dummy_rng = streams.anchor, streams.dummy
        engine.reset()
        return engine
    ledger = PersistentEpochBudget(policy, tmp_path/'budget.db', private_key=b'x'*32)
    sessions = FixedEpochProtectedSessions(ledger, factory)
    ranking = MultiPurposeRoadRanking(SimpleNamespace(rn=rn,pois=context.pois,categories=context.categories))
    client = BudgetedGeoILbsClient(sessions, ranking, k=3, response_l=context.k,
        private_order_rng=np.random.default_rng(10))
    return client, ledger, sessions, rn


def test_actual_geoi_lazy_read_local_purposes_and_excess_session(tmp_path):
    client, ledger, sessions, rn = client_fixture(tmp_path)
    reads, wire = [], []
    def gps():
        reads.append(1)
        return rn.latlon(0)
    def server(request):
        wire.append(request)
        return [[0,1]]
    try:
        assert client.start_session('first', 0.)
        client.public_tick(0., gps, server)
        before = sessions.evaluator_current_session()['spent_per_m']
        for purpose in (QueryPurpose.NEAREST, QueryPurpose.FASTEST):
            client.answer(QuerySpec(purpose,'cafe'), 10., *rn.latlon(0))
        client.answer(QuerySpec(QueryPurpose.MIN_DETOUR,'cafe',destination_state=15),
                      10., *rn.latlon(0))
        assert len(reads) == 1 and len(wire) == 3
        assert sessions.evaluator_current_session()['spent_per_m'] == before
        # Exhausted H1 continues public prediction; it must not read fresh GPS.
        client.public_tick(60., lambda: pytest.fail('GPS after cap'), server)
        assert len(wire) == 6
        client.close_session(61.)
        assert not client.start_session('second', 100.)
        assert client.public_tick(100., lambda: pytest.fail('denied GPS'),
                                  lambda _: pytest.fail('denied request')) is None
        with pytest.raises(ValueError, match='No live'):
            client.answer(QuerySpec(QueryPurpose.NEAREST,'cafe'),100.,*rn.latlon(0))
        client.close_session(101.)
        assert ledger.reserved_cap_per_m == pytest.approx(.23)
        assert all(set(q)=={'schema','timestamp_s','coordinate','categories','response_l','epoch'} for q in wire)
    finally:
        ledger.close()


def test_expired_stopped_or_rewound_answers_never_refetch(tmp_path):
    client, ledger, _, rn = client_fixture(tmp_path)
    wire = []
    try:
        query = QuerySpec(QueryPurpose.NEAREST,'cafe')
        client.start_session('first',0.)
        client.public_tick(20.,lambda: rn.latlon(0),lambda request:wire.append(request) or [[0]])
        for t in (19.,60.):
            with pytest.raises(ValueError,match='expired or answer clock rewound'):
                client.answer(query,t,*rn.latlon(0))
        client.close_session(61.)
        with pytest.raises(ValueError,match='No live'):
            client.answer(query,61.,*rn.latlon(0))
        assert len(wire) == 3
    finally:
        ledger.close()


@pytest.mark.parametrize('mismatch', ['k', 'road', 'catalogue', 'categories', 'shallow_reply'])
def test_service_mismatch_fails_before_private_read_and_keeps_reserved_cap(tmp_path, mismatch):
    client, ledger, sessions, rn = client_fixture(tmp_path)
    try:
        if mismatch == 'k':
            client._schema['k'] += 1
        elif mismatch == 'road':
            client.ranking.rn = fixture()[0]
        elif mismatch == 'catalogue':
            pois = [dict(p) for p in client.ranking.pois]
            pois[0]['id'] = 'different-public-poi'
            client.ranking.pois = tuple(pois)
        elif mismatch == 'categories':
            client._schema['categories'] = ('other-category',)
        elif mismatch == 'shallow_reply':
            client._schema['response_l'] = 2
        with pytest.raises(ValueError, match='Matched'):
            client.start_session('bad-configuration', 0.)
        assert ledger.reserved_slots == 1 and ledger.reserved_cap_per_m == pytest.approx(.23)
        assert sessions.evaluator_summary()['spent_per_m'] == 0.
        with pytest.raises(ValueError, match='Start a public session'):
            client.public_tick(0., lambda: pytest.fail('configuration failure read GPS'),
                               lambda _: pytest.fail('configuration failure sent request'))
        # Failed admission does not leave the wrapper open or recover the slot.
        assert not client.start_session('next-public-session', 100.)
        assert client.public_tick(100., lambda: pytest.fail('denied GPS'),
                                  lambda _: pytest.fail('denied request')) is None
        client.close_session(101.)
    finally:
        ledger.close()


def test_deeper_public_reply_does_not_require_changing_frozen_planner(tmp_path):
    client, ledger, sessions, rn = client_fixture(tmp_path)
    client._schema['response_l'] = 20
    wire = []
    try:
        assert client.start_session('deeper-reply', 0.)
        context = sessions._engine.belief_model.context
        assert context.k == 3
        client.public_tick(0., lambda: rn.latlon(0), lambda request: wire.append(request) or [[0]])
        assert len(wire) == 3 and all(request['response_l'] == 20 for request in wire)
        assert sessions._engine.belief_model.context is context
        client.close_session(1.)
    finally:
        ledger.close()


def test_local_gps_purpose_and_answer_time_cannot_change_future_public_requests(tmp_path):
    a, ledger_a, _, rn_a = client_fixture(tmp_path/'a')
    b, ledger_b, _, rn_b = client_fixture(tmp_path/'b')
    wires = [[], []]
    try:
        for client, rn, wire in zip((a, b), (rn_a, rn_b), wires):
            assert client.start_session('same-public-start', 0.)
            client.public_tick(0., lambda rn=rn: rn.latlon(0), lambda request, wire=wire: wire.append(request) or [[0,1]])
        a.answer(QuerySpec(QueryPurpose.NEAREST,'cafe'),1.,*rn_a.latlon(0))
        for t, purpose in ((10.,QueryPurpose.FASTEST),(30.,QueryPurpose.MIN_DETOUR),(59.,QueryPurpose.NEAREST)):
            query = QuerySpec(purpose,'cafe',destination_state=15) if purpose==QueryPurpose.MIN_DETOUR else QuerySpec(purpose,'cafe')
            b.answer(query,t,*rn_b.latlon(15))
        for client, wire in zip((a, b), wires):
            client.public_tick(60., lambda: pytest.fail('fresh GPS after exhausted H1'),
                               lambda request, wire=wire: wire.append(request) or [[0,1]])
            client.close_session(61.)
        assert wires[0] == wires[1]
        assert ledger_a.reserved_cap_per_m == ledger_b.reserved_cap_per_m == pytest.approx(.23)
    finally:
        ledger_a.close()
        ledger_b.close()
