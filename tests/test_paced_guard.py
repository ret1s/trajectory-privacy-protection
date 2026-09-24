import numpy as np
import pytest
from benchmark.engines.paced_guard import PacedProgressLaneDummy, PacedOriginGuardLaneDummy
from benchmark.engines.matched_filter import MatchedFilteredProgressCoverLaneDummy
from tests.test_belief_lane import fixture
from tests.test_origin_guard import make
from core.demo_protocol import TrajectoryPoint


def test_private_clock_does_not_read_skipped_coordinates():
    rn, _, belief = fixture()
    model = PacedProgressLaneDummy(rn, belief_model=belief, k=3,
                                   read_interval_s=60., rng=np.random.default_rng(12))
    model.protect_step(0., .001, 5.)
    model.protect_step(float('nan'), float('nan'), 25.)
    model.protect_step(float('nan'), float('nan'), 45.)
    model.protect_step(0., .001, 65.)
    assert [r['private_read'] for r in model.evaluator_ledger] == [True, False, False, True]
    assert model.spent_units == sum(r['cost_units'] for r in model.evaluator_ledger)


def test_zero_pacing_reproduces_unpaced_candidate():
    rn, _, belief = fixture()
    kwargs = dict(belief_model=belief, k=3)
    points = tuple(TrajectoryPoint(i*20., 0., .001) for i in range(25))
    a = PacedProgressLaneDummy(rn, **kwargs, read_interval_s=0., rng=np.random.default_rng(31))
    b = MatchedFilteredProgressCoverLaneDummy(rn, **kwargs, rng=np.random.default_rng(31))
    assert a.protect_run(points).to_attacker_dict()['events'] == b.protect_run(points).to_attacker_dict()['events']
    assert a.evaluator_ledger == b.evaluator_ledger


def test_paced_guard_causal_matching_emission_and_bound():
    rn, _, original = make()
    opts = dict(belief_model=original.belief_model, early_belief_model=original.early_belief_model,
                k=3, read_interval_s=60., guard_seconds=0., budget=.24, horizon=12)
    def build():
        return PacedOriginGuardLaneDummy(rn, **opts, rng=np.random.default_rng(42))
    points = tuple(TrajectoryPoint(i*20., 0., .001) for i in range(75))
    model = build(); full = model.protect_run(points).to_attacker_dict()
    assert full['events'][:8] == build().protect_run(points[:8]).to_attacker_dict()['events']
    reads = [p.timestamp_s for p, r in zip(points, model.evaluator_ledger) if r['private_read']]
    assert all(b-a >= 60 for a, b in zip(reads, reads[1:]))
    assert model.spent_bound <= .23
    assert model.spent_units == sum(r['cost_units'] for r in model.evaluator_ledger)
    assert 'public_clock_skip' not in str(full)
    with pytest.raises(ValueError):
        model.protect_step(0., .001, 1.)
