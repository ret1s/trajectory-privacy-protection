import numpy as np
from benchmark.engines.paced_guard import PacedProgressLaneDummy
from benchmark.engines.reserve_paced import ReservePacedProgressLaneDummy, ReservePacedSlackProgressLaneDummy
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


def test_zero_strength_replays_fixed_policy_and_ledger():
    rn, _, belief = fixture()
    points = tuple(TrajectoryPoint(i*20., 0., .001) for i in range(40))
    args = dict(belief_model=belief, k=3, budget=.24, horizon=12)
    a = PacedProgressLaneDummy(rn, **args, rng=np.random.default_rng(49))
    b = ReservePacedProgressLaneDummy(rn, **args, reserve_strength=0., rng=np.random.default_rng(49))
    assert a.protect_run(points).to_attacker_dict()['events'] == b.protect_run(points).to_attacker_dict()['events']
    assert a.evaluator_ledger == b.evaluator_ledger


def test_reserve_skip_accepts_unread_invalid_gps_and_preserves_public_cadence():
    rn, _, belief = fixture()
    model = ReservePacedProgressLaneDummy(rn, belief_model=belief, k=3,
                                        budget=.24, horizon=12, rng=np.random.default_rng(50))
    first = model.protect_step(0., .001, 0.)
    # One unit has been used: 60 s is no longer due; the request still gets K replies.
    skipped = model.protect_step(float('nan'), float('nan'), 60.)
    assert len(first) == len(skipped) == 3
    assert not model.evaluator_ledger[-1]['private_read']
    model.protect_step(0., .001, 80.)
    assert model.evaluator_ledger[-1]['private_read']


def test_reserve_slack_is_causal_bounded_and_reachable():
    rn, _, belief = fixture()
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%14)*.0001) for i in range(100))
    def build():
        return ReservePacedSlackProgressLaneDummy(rn, belief_model=belief, k=3,
            budget=.24, horizon=12, utility_slack=.03, rng=np.random.default_rng(51))
    model = build(); full = model.protect_run(points).to_attacker_dict()
    prefix = build().protect_run(points[:18]).to_attacker_dict()
    assert full['events'][:18] == prefix['events']
    assert len(full['events']) == len(points)
    assert model.spent_bound <= .23+1e-12
    assert model.spent_units == sum(v['cost_units'] for v in model.evaluator_ledger)
    for a, b in zip(model.evaluator_states, model.evaluator_states[1:]):
        assert all(v in model.travel.reachable(u, 20.) for u, v in zip(a, b))
    for entry in model.evaluator_objective[1:]:
        assert entry['objective_after_slack'] >= entry['objective_before_slack']-.03-1e-12
