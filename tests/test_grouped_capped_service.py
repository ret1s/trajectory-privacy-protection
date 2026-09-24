import numpy as np
import pytest
from scipy.sparse import csr_matrix
from benchmark.capped_service_objective import CappedServiceIndex, CappedServiceObjective
from benchmark.grouped_capped_service import GroupedCappedServiceIndex, GroupedCappedServiceObjective, GroupedCappedServiceLaneDummy
from benchmark.engines.capped_service import CappedServiceLaneDummy
from tests.test_belief_lane import fixture
from core.demo_protocol import TrajectoryPoint


def test_grouped_objective_preserves_all_mass_and_exact_real_formula():
    rng = np.random.default_rng(2526)
    signatures = rng.integers(-1, 15, size=(30, 3, 5))
    base = rng.random((6, 15)); base /= base.sum(axis=1, keepdims=True)
    weights = np.c_[base[rng.integers(0, 6, 80)], np.zeros(80)]
    ordinary = CappedServiceIndex(signatures, np.arange(30), csr_matrix(weights))
    grouped = GroupedCappedServiceIndex(signatures, np.arange(30), csr_matrix(weights))
    assert grouped.reference.shape[0] == 6 and grouped.original_latent_count == 80
    for _ in range(30):
        belief = rng.random(80); belief[0] = 1e-20; belief /= belief.sum()
        a = CappedServiceObjective(ordinary, belief, .9)
        b = GroupedCappedServiceObjective(grouped, belief, .9)
        assert b.belief.sum() == pytest.approx(belief.sum())
        selected = rng.integers(0, 30, 4).tolist()
        assert a.value(selected) == pytest.approx(b.value(selected), abs=1e-14)
        assert np.allclose(a.marginal(np.arange(30), selected), b.marginal(np.arange(30), selected), rtol=0, atol=2e-15)
    with pytest.raises(ValueError): GroupedCappedServiceObjective(grouped, np.ones(6)/6)


def test_grouping_keeps_anchor_ledger_and_prefix_boundary():
    rn, _, belief = fixture()
    opts = dict(belief_model=belief, k=5, budget=.24, horizon=12, utility_slack=.03)
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%14)*.0001) for i in range(20))
    slow = CappedServiceLaneDummy(rn, **opts, rng=np.random.default_rng(5))
    fast = GroupedCappedServiceLaneDummy(rn, **opts, rng=np.random.default_rng(5))
    slow.protect_run(points); public = fast.protect_run(points).to_attacker_dict()
    assert slow.evaluator_anchors == fast.evaluator_anchors and slow.evaluator_ledger == fast.evaluator_ledger
    prefix = GroupedCappedServiceLaneDummy(rn, **opts, rng=np.random.default_rng(5)).protect_run(points[:8]).to_attacker_dict()
    assert public['events'][:8] == prefix['events']
    assert fast.spent_bound <= .23+1e-12
