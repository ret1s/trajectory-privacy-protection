import numpy as np
import pytest

from benchmark.anchor_belief import AnchorBelief, PublicAnchorModel
from benchmark.prior_factors import FactorizedAnchorModel, PriorFactorCover, cell_balanced_prior
from benchmark.engines.service_cover import ServiceCoverLaneDummy
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


def test_cell_balancing_is_mean_not_sum():
    rn, c, l = fixture()
    q = np.linspace(1., 3., len(rn))
    p = cell_balanced_prior(rn, q, 20.)
    _, inv = np.unique(np.floor(rn.xy/20.).astype(int), axis=0, return_inverse=True)
    expected = np.array([q[inv == i].mean() for i in range(inv.max()+1)])
    assert np.allclose(np.bincount(inv, weights=p), expected/expected.sum())
    assert np.isclose(p.sum(), 1)
    for bad in (np.zeros(len(rn)), -q, q[:-1], q*np.nan):
        with pytest.raises(ValueError): cell_balanced_prior(rn, bad, 20.)


def test_independent_initial_and_transition_factors():
    rn, c, l = fixture()
    q = cell_balanced_prior(rn, np.ones(len(rn)), l.spacing_m)
    u = PublicAnchorModel(rn, c, q, spacing_m=l.spacing_m)
    m = FactorizedAnchorModel(l, u)
    assert np.array_equal(m.prior, l.prior)
    assert np.array_equal(m.emission(rn.latlon(0)), l.emission(rn.latlon(0)))
    assert np.allclose(m.predict(l.prior, 20), u.predict(l.prior, 20))
    assert np.allclose(m.predict(l.prior, 10000), u.prior)
    assert not np.allclose(l.prior, u.prior)
    a, b = AnchorBelief(l), AnchorBelief(FactorizedAnchorModel(l, l))
    for t in (0, 20, 40, 2000):
        assert np.array_equal(a.update(rn.latlon(15), t), b.update(rn.latlon(15), t))
    with pytest.raises(ValueError): FactorizedAnchorModel(l, fixture()[2])


def test_full_run_prefix_pairing_and_private_boundary():
    rn, _, l = fixture()
    model = FactorizedAnchorModel(l, l)
    points = tuple(TrajectoryPoint(5.*i, 0., .0001+i*.0001) for i in range(15))
    def make():
        return PriorFactorCover(rn, belief_model=model, rng=np.random.default_rng(71))
    a = make(); full = a.protect_run(points).to_attacker_dict()
    old = ServiceCoverLaneDummy(rn, belief_model=l, rng=np.random.default_rng(71))
    assert old.protect_run(points).to_attacker_dict()['events'] == full['events']
    assert old.evaluator_anchors == a.evaluator_anchors
    assert a.spent_bound == pytest.approx(.23)
    assert make().protect_run(points[:5]).to_attacker_dict()['events'] == full['events'][:5]
    b, c = make(), make()
    for m in (b, c): m.anchor.perturb = lambda *args, **kwargs: rn.latlon(15)
    for i in range(15):
        assert b.protect_step(0., 0., 5.*i) == c.protect_step(20., 30., 5.*i)
    assert b.protect_step(np.nan, np.nan, 75) == c.protect_step(80, 90, 75)
