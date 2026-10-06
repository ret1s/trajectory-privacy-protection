import json

import numpy as np
import pytest

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.endpoint_noise import EndpointNoiseProgressLaneDummy
from core.session_budget import FixedEpochPolicy, PersistentEpochBudget, FixedEpochProtectedSessions
from tests.test_belief_lane import fixture


def factory(policy):
    rn, context, _ = fixture()
    belief = PublicAnchorModel(rn, context, np.linspace(1, 2, len(rn)), spacing_m=2.,
                              epsilon_release=policy.unit_epsilon_per_m,
                              epsilon_test=policy.unit_epsilon_per_m)
    def build(allocation, streams):
        model = EndpointNoiseProgressLaneDummy(rn, belief_model=belief, privacy_scale=1.,
            k=3, budget=allocation.nominal_budget_per_m, horizon=allocation.horizon,
            read_interval_s=allocation.read_interval_s, utility_slack=.03,
            rng=streams.initialization)
        model.anchor_rng, model.dummy_rng = streams.anchor, streams.dummy
        model.reset()
        return model
    return build


def make(tmp_path, *, horizon=12, slots=6, total=.23, name='ledger.db'):
    p = FixedEpochPolicy('public-day-1', 0., 100000., total, slots, horizon, 60.)
    ledger = PersistentEpochBudget(p, tmp_path/name, private_key=b'x'*32)
    return p, ledger, FixedEpochProtectedSessions(ledger, factory(p))


def test_nominal_and_effective_caps_sum_for_public_horizons():
    for h in (1, 8, 12):
        p = FixedEpochPolicy('day', 0., 86400., horizon=h)
        assert p.session_slots*p.effective_session_cap_per_m == pytest.approx(.23)
        assert p.max_units_per_session*p.unit_epsilon_per_m == pytest.approx(.23/6)
        assert p.nominal_session_budget_per_m/(2*h) == p.unit_epsilon_per_m
    # Current Endpoint20 .0575 per trip does not imply that bound across six.
    assert 6*.0575 == pytest.approx(.345)


def test_adversarial_linked_repeated_trips_cap_and_no_gps_after_exhaustion(tmp_path):
    p, ledger, protected = make(tmp_path)
    count = 0
    def gps():
        nonlocal count
        count += 1
        # Public road graph is near (0,0); this valid remote GPS forces refresh.
        return 45., 90.
    for s in range(6):
        assert protected.start_session(f'trip-{s}', s*10000.)
        for j in range(40):
            protected.protect_step(s*10000.+j*60., gps)
        diag = protected.evaluator_current_session()
        assert diag['spent_per_m'] == pytest.approx(p.effective_session_cap_per_m)
        assert diag['private_reads'] == 12
        assert sum(row['cost_units'] for row in diag['ledger']) == 23
        assert not any(row['private_read'] for row in diag['ledger'][12:])
        protected.close_session(s*10000.+2400.)
    assert count == 72
    assert protected.evaluator_summary()['spent_per_m'] == pytest.approx(.23)
    assert ledger.reserved_cap_per_m == pytest.approx(.23)
    assert not protected.start_session('trip-7', 60000.)
    def forbidden():
        raise AssertionError('A denied session must not ask for GPS')
    for t in (60000., 60060., 60120.):
        assert protected.protect_step(t, forbidden) == ()
    protected.close_session(60121.)
    assert protected.evaluator_summary()['spent_per_m'] == pytest.approx(.23)
    ledger.close()


def test_between_public_reads_uses_no_private_gps_and_factory_failure_loses_slot(tmp_path):
    p, ledger, protected = make(tmp_path, slots=2)
    protected.start_session('short', 0.)
    protected.protect_step(0., lambda: (0., .001))
    protected.protect_step(20., lambda: (_ for _ in ()).throw(AssertionError('No GPS between ticks')))
    assert protected.evaluator_current_session()['ledger'][-1]['branch'] == 'public_clock_skip'
    protected.close_session(21.)
    def failed_factory(*args):
        raise RuntimeError('crash before reading any GPS')
    other = FixedEpochProtectedSessions(ledger, failed_factory)
    with pytest.raises(RuntimeError, match='crash'):
        other.start_session('crash', 100.)
    assert ledger.reserved_slots == 2
    assert not protected.start_session('after-crash', 200.)
    ledger.close()


def test_restart_concurrent_handles_and_policy_change_cannot_refill(tmp_path):
    p, a, _ = make(tmp_path, slots=2)
    b = PersistentEpochBudget(p, tmp_path/'ledger.db')
    assert a.reserve('one', 0.).slot == 0
    assert b.reserve('two', 1.).slot == 1
    assert a.reserve('three', 2.) is None
    a.close(); b.close()
    reopened = PersistentEpochBudget(p, tmp_path/'ledger.db')
    assert reopened.reserved_slots == 2
    assert reopened.reserve('four', 3.) is None
    with pytest.raises(ValueError, match='already consumed'):
        reopened.reserve('one', 4.)
    with pytest.raises(ValueError, match='cannot be changed'):
        PersistentEpochBudget(FixedEpochPolicy('day-2', 0., 100000., session_slots=2), tmp_path/'ledger.db')
    with pytest.raises(ValueError, match='cannot be replaced'):
        PersistentEpochBudget(p, tmp_path/'ledger.db', private_key=b'y'*32)
    reopened.close()


def test_private_keyed_streams_independent_and_equal_total_control_exact(tmp_path):
    p, ledger, model = make(tmp_path, total=.345, name='global.db')
    # Equal total/.0575 per slot implies the same primitives as per-session reset.
    assert p.unit_epsilon_per_m == pytest.approx(.0025)
    baseline = factory(p)
    all_draws = []
    for slot in range(3):
        admitted = model.start_session(f'trip-{slot}', slot*1000.)
        assert admitted
        allocation = model._allocation
        streams = ledger.private_rng_streams(allocation)
        reference = baseline(allocation, streams)
        for j in range(8):
            gps = (0., .0001+j*.0001)
            assert model.protect_step(slot*1000.+j*60., lambda: gps) == reference.protect_step(*gps, j*60.)
        samples = ledger.private_rng_streams(allocation)
        all_draws.append([samples.anchor.random(8), samples.dummy.random(8)])
        model.close_session(slot*1000.+481.)
    assert not np.array_equal(all_draws[0][0], all_draws[1][0])
    assert not np.array_equal(all_draws[0][0], all_draws[0][1])
    assert not any(word in json.dumps(p.public_parameters()) for word in ('random_key', 'person_id', 'spent_units'))
    ledger.close()


def test_public_epoch_no_auto_reset_and_horizon8_long_tail(tmp_path):
    p, ledger, model = make(tmp_path, horizon=8)
    model.start_session('long', 0.)
    for i in range(24):
        model.protect_step(i*60., lambda: (45., 90.))
    diag = model.evaluator_current_session()
    assert diag['private_reads'] == 8
    assert diag['spent_per_m'] == pytest.approx(.23/6)
    assert not any(x['private_read'] for x in diag['ledger'][8:])
    with pytest.raises(ValueError, match='outside fixed public epoch'):
        model.protect_step(100000., lambda: (_ for _ in ()).throw(AssertionError('GPS')))
    ledger.close()


@pytest.mark.parametrize('field,value', [('session_slots', True), ('horizon', 1.5),
    ('total_effective_epsilon_per_m', 0), ('read_interval_s', -1), ('end_s', float('inf'))])
def test_invalid_public_policy_rejected(field, value):
    args = dict(epoch_id='day', start_s=0., end_s=86400.)
    args[field] = value
    with pytest.raises(ValueError):
        FixedEpochPolicy(**args)
