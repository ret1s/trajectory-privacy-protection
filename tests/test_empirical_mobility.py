from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye
from benchmark.empirical_mobility import EmpiricalMobilityModel, smoothed_generator
from benchmark.engines.empirical_paced import EmpiricalPacedSlackProgressLaneDummy
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


def test_generator_smoothing_preserves_inputs_and_unvisited_prior_rates():
    counts = csr_matrix([[0., 3.], [0., 0.]])
    prior = csr_matrix([[.8, .2], [.3, .7]])
    before_counts, before_prior = counts.toarray(), prior.toarray()
    q = smoothed_generator(counts, np.array([10., 0.]), prior, 20.)
    assert np.array_equal(counts.toarray(), before_counts)
    assert np.array_equal(prior.toarray(), before_prior)
    assert np.allclose(q.sum(axis=1), 0.)
    assert q[0, 1] == pytest.approx((3.+.2)/30.)
    assert q[1, 0] == pytest.approx(.3/20.)
    assert q[1, 1] == pytest.approx(-.3/20.)


def test_two_state_continuous_prediction_matches_analytic_chain_and_semigroup():
    base = SimpleNamespace(xy=np.zeros((2, 2)), sha256='toy')
    q = csr_matrix([[-.2, .2], [.1, -.1]])
    model = EmpiricalMobilityModel(base, q, {'scope': 'public_test'})
    w = np.array([1., 0.])
    p = model.predict(w, 5.)
    expected_zero = 1/3+2/3*np.exp(-.3*5.)
    assert p[0] == pytest.approx(expected_zero, abs=1e-12)
    assert np.allclose(model.predict(model.predict(w, 2.), 3.), p)
    with pytest.raises(ValueError):
        EmpiricalMobilityModel(base, csr_matrix([[-1., 2.], [0., 0.]]), {})
    with pytest.raises(ValueError):
        model.predict(np.array([.2, .2]), 2.)


def test_model_preserves_emission_and_engine_causality_budget_and_motion():
    rn, context, base = fixture()
    p = base.transition(20.)
    q = (p-eye(len(base.xy), format='csr'))/20.
    model = EmpiricalMobilityModel(base, q, {'source': 'public_fixture'})
    anchor = rn.latlon(0)
    assert np.array_equal(model.emission(anchor), base.emission(anchor))
    assert model.prior is base.prior and model.poi_weights is base.poi_weights
    def build():
        return EmpiricalPacedSlackProgressLaneDummy(rn, belief_model=model, utility_slack=.03,
            k=3, budget=.24, horizon=12, rng=np.random.default_rng(55))
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%14)*.0001) for i in range(40))
    engine = build(); full = engine.protect_run(points).to_attacker_dict()
    assert build().protect_run(points[:12]).to_attacker_dict()['events'] == full['events'][:12]
    assert engine.spent_bound <= .23+1e-12
    for a, b in zip(engine.evaluator_states, engine.evaluator_states[1:]):
        assert all(v in engine.travel.reachable(u, 20.) for u, v in zip(a, b))
