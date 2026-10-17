import numpy as np
import pytest

from benchmark.engines.continual_slack import ContinualPacedSlackProgressLaneDummy
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


def make(seed=47):
    rn, _, belief = fixture()
    return rn, ContinualPacedSlackProgressLaneDummy(rn, belief_model=belief,
        k=3, read_interval_s=60., utility_slack=.03, rng=np.random.default_rng(seed))


def test_identical_to_frozen_engine_until_cap_and_continues_afterwards():
    rn, new = make()
    old = PacedSlackProgressLaneDummy(rn, belief_model=new.belief_model, k=3,
        read_interval_s=60., utility_slack=.03, rng=np.random.default_rng(47))
    matched = 0
    for t in range(0, 1801, 20):
        can_read = old.spent_units+(1 if old.last_anchor is None else 2) <= 23
        a = old.protect_step(*rn.latlon(0), float(t))
        b = new.protect_step(*rn.latlon(0), float(t))
        if can_read:
            assert a == b
            assert old.evaluator_anchors[-1] == new.evaluator_anchors[-1]
            assert np.array_equal(old.belief.weights, new.belief.weights)
            matched += 1
        else:
            break
    assert matched > 10
    for t in range(int(new.last_t)+20, 1801, 20):
        new.protect_step(*rn.latlon(0), float(t))
    assert new.private_read_count == 31
    assert new.spent_units > 23 and old.spent_units <= 23
    assert new.prefix_epsilon_bound == pytest.approx(.61)
    assert new.spent_bound <= new.prefix_epsilon_bound


def test_clock_skip_ignores_coordinates_and_does_not_consume_upstream_budget():
    rn, engine = make()
    engine.protect_step(*rn.latlon(0), 0.)
    engine.protect_step(float('nan'), float('nan'), 20.)
    engine.protect_step(float('nan'), float('nan'), 40.)
    assert engine.spent_units == 1 and engine.private_read_count == 1
    with pytest.raises(ValueError, match='coordinates'):
        engine.protect_step(float('nan'), float('nan'), 60.)
    assert engine.last_t == 40. and engine.private_read_count == 1
    engine.protect_step(*rn.latlon(1), 60.)
    assert engine.private_read_count == 2
    with pytest.raises(ValueError, match='times'):
        engine.protect_step(*rn.latlon(1), 60.)


def test_no_legacy_cap_claim_in_local_run_metadata():
    rn, engine = make()
    run = engine.protect_run([TrajectoryPoint(float(t), *rn.latlon(0)) for t in range(0, 121, 20)])
    params = dict(run.transcript.public_parameters)
    assert params['total_session_cap_per_m'] is None
    assert 'budget_per_m' not in params and 'horizon_events' not in params
    assert params['privacy_unit_epsilon'] == .01


@pytest.mark.parametrize('unit', [0, -1, float('nan'), float('inf'), True])
def test_invalid_fixed_unit_rejected(unit):
    rn, engine = make()
    with pytest.raises(ValueError):
        ContinualPacedSlackProgressLaneDummy(rn, belief_model=engine.belief_model, unit_epsilon=unit)
