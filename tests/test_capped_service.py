from itertools import product
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from benchmark.capped_service_objective import CappedServiceIndex, CappedServiceObjective
from benchmark.engines.capped_service import CappedServiceLaneDummy
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.engines.service_cover import greedy_cover
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


def test_exact_union_marginals_submodularity_and_greedy_small_instances():
    rng = np.random.default_rng(2525)
    for _ in range(25):
        signatures = rng.integers(-1, 7, size=(8, 2, 3))
        weights = rng.random((5, 7)); weights /= weights.sum(axis=1, keepdims=True)
        weights = np.c_[weights, np.zeros(5)]
        belief = rng.random(5); belief /= belief.sum()
        index = CappedServiceIndex(signatures, np.arange(8), csr_matrix(weights))
        objective = CappedServiceObjective(index, belief, .9)
        def brute(selected):
            mask = np.zeros(8)
            if selected:
                ids = np.unique(signatures[list(selected)]); mask[ids[ids >= 0]] = 1
            return np.minimum(weights @ mask, .9) @ belief
        for selected in ([], [0], [0, 0], [0, 1], [3, 7, 2]):
            assert objective.value(selected) == pytest.approx(brute(selected), abs=1e-14)
            expected = [brute([*selected, i])-brute(selected) for i in range(8)]
            assert np.allclose(objective.marginal(np.arange(8), selected), expected, rtol=0, atol=2e-15)
        assert np.all(objective.marginal(np.arange(8), [0]) >= objective.marginal(np.arange(8), [0, 1])-1e-14)
        groups = [np.array([0, 1, 2, 3]), np.array([3, 4, 5]), np.array([6, 7])]
        ties = lambda j, ids: (np.zeros(len(ids)), np.zeros(len(ids)))
        selected, _ = greedy_cover(groups, objective.marginal, ties)
        assert objective.value(selected) >= .5*max(brute(s) for s in product(*groups))-1e-12
        # Cap=1 reduces to the original expected-union utility, not a new metric.
        linear = CappedServiceObjective(index, belief, 1.)
        mask = np.zeros(8); ids = np.unique(signatures[[0, 3]]); mask[ids[ids >= 0]] = 1
        assert linear.value([0, 3]) == pytest.approx(belief @ weights @ mask)


@pytest.mark.parametrize('slack', [0., .03])
def test_disabled_cap_exact_parent_boundary(slack):
    rn, _, belief = fixture()
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%14)*.0001) for i in range(20))
    opts = dict(belief_model=belief, k=5, budget=.24, horizon=12, utility_slack=slack)
    old = PacedSlackProgressLaneDummy(rn, **opts, rng=np.random.default_rng(91))
    new = CappedServiceLaneDummy(rn, **opts, location_service_cap=None, rng=np.random.default_rng(91))
    assert old.protect_run(points).to_attacker_dict()['events'] == new.protect_run(points).to_attacker_dict()['events']
    assert old.evaluator_ledger == new.evaluator_ledger and old.evaluator_anchors == new.evaluator_anchors


def test_capped_planner_causality_matching_ledger_and_physical_paths():
    rn, _, belief = fixture()
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%14)*.0001) for i in range(40))
    opts = dict(belief_model=belief, k=5, budget=.24, horizon=12, utility_slack=.03)
    def make(cls=CappedServiceLaneDummy):
        return cls(rn, **opts, rng=np.random.default_rng(92))
    model = make(); full = model.protect_run(points).to_attacker_dict()
    assert make().protect_run(points[:8]).to_attacker_dict()['events'] == full['events'][:8]
    control = make(PacedSlackProgressLaneDummy); control.protect_run(points)
    assert control.evaluator_anchors == model.evaluator_anchors
    assert control.evaluator_ledger == model.evaluator_ledger and model.spent_bound <= .23+1e-12
    for d in model.evaluator_objective:
        assert d['objective_after_slack'] >= d['objective_before_slack']-.03-1e-12
    for a, b in zip(model.evaluator_states, model.evaluator_states[1:]):
        assert all(v in model.travel.reachable(u, 20) for u, v in zip(a, b))
