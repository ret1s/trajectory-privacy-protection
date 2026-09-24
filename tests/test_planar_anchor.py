import json
import numpy as np
import pytest
from scipy.stats import laplace
from benchmark.planar_anchor import PlanarAnchorModel, PredictivePlanarLaplace
from benchmark.engines.planar_paced import PlanarPacedLaneDummy
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


def test_radial_sampler_contract_reuse_reset_and_unrestricted_output():
    rn, _, base = fixture()
    class Draws:
        def gamma(self, *, shape, scale):
            assert shape == 2 and scale == 100
            return 2000.
        def uniform(self, low, high):
            assert low == 0 and high == 2*np.pi
            return np.pi/2
        def laplace(self, *, scale):
            assert scale == 100
            return -10000.
    primitive = PredictivePlanarLaplace(rn, epsilon_release=.01, epsilon_test=.01, rng=Draws())
    point = rn.latlon(0); first = primitive.perturb(*point)
    assert np.allclose(rn.point_xy(*first), rn.xy[0]+[0, 2000.], rtol=0, atol=1e-7)
    assert first not in {rn.latlon(i) for i in range(len(rn))}
    assert primitive.perturb(*point) == first and primitive.n_resample == 1
    primitive.reset(); assert primitive.previous is None and primitive.n_resample == 0


def test_continuous_density_and_reuse_atom_are_distinct():
    rn, _, base = fixture(); model = PlanarAnchorModel(base)
    first = rn.proj.to_latlon(*(rn.xy[0]+[200., 30.]))
    second = rn.proj.to_latlon(*(rn.xy[0]+[10., 130.]))
    fresh = .01**2/(2*np.pi)*np.exp(-.01*np.linalg.norm(model.xy-rn.point_xy(*first), axis=1))
    assert np.allclose(model.emission(first), fresh, rtol=1e-12, atol=0)
    d = np.linalg.norm(model.xy-rn.point_xy(*first), axis=1)
    q = laplace.cdf(200-d, scale=100.)
    assert np.array_equal(model.emission(first, first), q)
    assert np.allclose(model.emission(second, first), (1-q)*model.emission(second))
    # Ideal density/atom likelihood ratios respect their declared metric bounds
    # on this finite diagnostic grid; this numerical check is not a DP proof.
    distances = np.linalg.norm(model.xy[:, None]-model.xy[None, :], axis=2)
    for values, eps in ((fresh, .01), (q, .01), (model.emission(second, first), .02)):
        logs = np.log(values)
        assert np.all(np.abs(logs[:, None]-logs[None, :]) <= eps*distances+1e-10)


def test_causal_planner_budget_stop_and_physical_paths():
    rn, _, base = fixture(); model = PlanarAnchorModel(base)
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%14)*.0001) for i in range(30))
    def make():
        return PlanarPacedLaneDummy(rn, belief_model=model, k=5, budget=.24, horizon=12,
                                   utility_slack=.03, rng=np.random.default_rng(2626))
    full_model = make(); full = full_model.protect_run(points).to_attacker_dict()
    assert make().protect_run(points[:8]).to_attacker_dict()['events'] == full['events'][:8]
    assert full_model.spent_bound <= .23+1e-12
    assert not any(s in json.dumps(full) for s in ('evaluator_', 'spent_units', 'n_resample'))
    assert full_model.anchor.name == 'predictive_planar_laplace'
    assert 'full_plane' in full['public_parameters']['support']
    for a, b in zip(full_model.evaluator_states, full_model.evaluator_states[1:]):
        assert all(v in full_model.travel.reachable(u, 20) for u, v in zip(a, b))
    # Force the public budget filter into its exhausted state, keeping both
    # already-protected histories identical. Future true GPS cannot affect it.
    a, b = make(), make()
    for engine in (a, b):
        engine.protect_step(*rn.latlon(0), 0.)
        engine.spent_units = engine.max_units
    for t in (20., 60., 120.):
        assert a.protect_step(float('nan'), float('nan'), t) == b.protect_step(30., 80., t)
        assert not a.evaluator_ledger[-1]['private_read']
    with pytest.raises(ValueError, match='continuous'):
        PlanarPacedLaneDummy(rn, belief_model=base)


@pytest.mark.parametrize('epsilon', [0., -1., float('nan')])
def test_invalid_budget_rejected(epsilon):
    rn, _, _ = fixture()
    with pytest.raises(ValueError):
        PredictivePlanarLaplace(rn, epsilon_release=epsilon, epsilon_test=.01)
