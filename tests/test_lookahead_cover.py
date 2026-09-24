from itertools import product
import numpy as np
from scipy.sparse import eye
from benchmark.empirical_mobility import EmpiricalMobilityModel
from benchmark.engines.empirical_paced import EmpiricalPacedProgressLaneDummy
from benchmark.engines.lookahead_cover import LookaheadCoverLaneDummy, TwoSliceObjective
from benchmark.engines.fair_cover import CoverageObjective
from benchmark.engines.service_cover import greedy_cover
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


def make_models():
    rn, _, base = fixture()
    q = (base.transition(20.)-eye(len(base.xy), format='csr'))/20.
    model = EmpiricalMobilityModel(base, q, {'source': 'public_fixture'})
    return rn, model


def test_two_slice_greedy_half_bound_against_small_exhaustive_optimum():
    rng = np.random.default_rng(56)
    sig = np.arange(6).reshape(6, 1, 1); access = np.arange(6)
    for _ in range(20):
        w = rng.random((2, 6)); w /= w.sum(axis=1, keepdims=True)
        objectives = [CoverageObjective(sig, access, np.r_[a, 0.]) for a in w]
        pairs = rng.integers(0, 6, (12, 2))
        objective = TwoSliceObjective(*objectives, pairs)
        groups = [np.arange(4*j, 4*j+4) for j in range(3)]
        ties = lambda j, ids: (np.zeros(len(ids)), np.zeros(len(ids)))
        chosen, _ = greedy_cover(groups, objective.marginal, ties)
        optimum = max(objective.value(list(x)) for x in product(*groups))
        assert objective.value(chosen)+1e-12 >= .5*optimum
        for a in range(len(pairs)):
            assert np.isclose(objective.marginal([a], [0])[0], objective.value([0, a])-objective.value([0]))


def test_zero_horizon_replays_parent_exactly():
    rn, model = make_models()
    points = tuple(TrajectoryPoint(i*20., 0., .001) for i in range(30))
    args = dict(belief_model=model, k=3, budget=.24, horizon=12)
    a = EmpiricalPacedProgressLaneDummy(rn, **args, rng=np.random.default_rng(57))
    b = LookaheadCoverLaneDummy(rn, **args, lookahead_s=0., rng=np.random.default_rng(57))
    assert a.protect_run(points).to_attacker_dict()['events'] == b.protect_run(points).to_attacker_dict()['events']
    assert a.evaluator_ledger == b.evaluator_ledger


def test_directed_advance_floor_causality_and_paired_anchor_invariants():
    rn, model = make_models()
    args = dict(belief_model=model, k=3, budget=.24, horizon=12)
    def build():
        return LookaheadCoverLaneDummy(rn, **args, lookahead_s=120.,
            current_objective_slack=.03, rng=np.random.default_rng(58))
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%14)*.0001) for i in range(40))
    planner = build(); full = planner.protect_run(points).to_attacker_dict()
    assert build().protect_run(points[:14]).to_attacker_dict()['events'] == full['events'][:14]
    control = EmpiricalPacedProgressLaneDummy(rn, **args, rng=np.random.default_rng(58))
    control.protect_run(points)
    assert planner.evaluator_anchors == control.evaluator_anchors
    assert planner.evaluator_ledger == control.evaluator_ledger
    assert planner.spent_bound <= .23+1e-12
    for previous, selected, record in zip(planner.evaluator_states, planner.evaluator_states[1:], planner.evaluator_objective[1:]):
        assert all(q in planner.travel.reachable(p, 20.) for p, q in zip(previous, selected))
        assert record['current_objective_after_lookahead'] >= record['current_objective_before_lookahead']-.03-1e-12
        assert all(q in planner.travel.reachable(p, 120.) for p, q in zip(selected, record['lookahead_future_states']))
    for goal in planner.viable_ids[::max(1, len(planner.viable_ids)//5)]:
        for start in planner.viable_ids[::max(1, len(planner.viable_ids)//5)]:
            for duration in (0., .1, 1., 5., 20.):
                assert planner._advance(start, goal, duration) in planner.travel.reachable(start, duration)
