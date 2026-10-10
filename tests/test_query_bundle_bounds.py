"""Independent finite oracles for privacy oscillation and actual utility."""
import itertools

import numpy as np
import pytest

from benchmark.probabilistic_query_bundle import QueryBundle, ProbabilisticQueryBundle
from benchmark.query_bundle_bounds import (public_privacy_certificate, calibrated_beta,
                                          public_floor_indices, utility_certificate)


def kernel(coverage=None, beta=2., cost=.25):
    table = np.array([[.8, .9, .95], [.7, .95, 1.], [.6, .85, .9]]) if coverage is None else coverage
    return ProbabilisticQueryBundle([QueryBundle((0,)), QueryBundle((1, 2)), QueryBundle((3, 4, 5))],
                                    table, beta=beta, cost_weight=cost)


def simplex(n=6):
    return [np.array([a, b, n-a-b])/n for a in range(n+1) for b in range(n+1-a)]


def test_oscillation_matches_four_index_oracle_and_bounds_likelihood():
    model = kernel(beta=8.)
    c = model.coverage
    oracle = max(c[x, a]-c[y, a]-c[x, b]+c[y, b]
                 for x, y, a, b in itertools.product(range(3), repeat=4))/(1.+model.cost_weight)
    certificate = public_privacy_certificate(model)
    assert certificate['score_oscillation'] == pytest.approx(oracle)
    assert certificate['epsilon_Q'] < certificate['conservative_epsilon_Q']
    logs = np.array([model.log_probabilities(b) for b in simplex()])
    assert np.max(np.ptp(logs, axis=0)) <= certificate['epsilon_Q']+1e-12


def test_action_independent_location_offsets_have_zero_privacy_loss():
    # Score changes with belief, but only by a common offset. A naive
    # pointwise sensitivity bound misses this exact noninterference.
    model = kernel([[.1, .2, .3], [.4, .5, .6], [.6, .7, .8]], beta=8.)
    cert = public_privacy_certificate(model)
    assert cert['epsilon_Q'] == pytest.approx(0., abs=1e-14)
    assert model.probabilities([1., 0., 0.]) == pytest.approx(model.probabilities([0., 0., 1.]))


def test_public_calibration_and_floor_are_belief_independent():
    model = kernel()
    beta = calibrated_beta(model, epsilon_target=.3)
    new = kernel(beta=beta)
    assert public_privacy_certificate(new)['epsilon_Q'] <= .3+1e-12
    assert public_floor_indices(model, minimum_coverage=.8).tolist() == [1, 2]
    with pytest.raises(ValueError, match='No public bundle'):
        public_floor_indices(model, minimum_coverage=.95)
    assert calibrated_beta(kernel(np.ones((3, 3))), epsilon_target=1.) == 40


def test_utility_bridge_under_belief_error_and_wrong_purpose():
    model = kernel(beta=4.)
    # Enumerate true posteriors and beliefs independently. True utility differs
    # from proxy by a known per-action perturbation (models purpose/grid error).
    true_table = model.coverage-np.array([.03, .01, .02])
    for b, mu in itertools.product(simplex(3), repeat=2):
        tv = .5*np.abs(b-mu).sum()
        cert = utility_certificate(model, b, tv_bound=tv, proxy_error_bound=.03)
        p = model.probabilities(b)
        actual = mu @ true_table
        expected_actual = float(np.sum(p*actual))
        assert expected_actual >= cert['lower_expected_actual_utility']-1e-12
        assert actual.max()-expected_actual <= cert['additive_shortfall_to_best_actual_library_utility']+1e-12
        assert cert['exact_surrogate_score_regret'] <= cert['variational_score_regret_bound']+1e-12
        assert np.all(actual >= cert['lower_per_action_actual_utility']-1e-12)


def test_variational_bound_improves_old_exact_maximizer_mass_bound():
    model = kernel(beta=4.)
    b = np.array([.2, .3, .5])
    cert = utility_certificate(model, b)
    score = model.scores(b)
    star_mass = np.exp(model.log_base_weights[score == score.max()]).sum()
    old = min(1., (2./model.beta)*(np.log(1./star_mass)+1.))
    assert cert['variational_score_regret_bound'] < old


def test_shared_latent_two_round_privacy_is_composed_not_iid_given_x():
    model = kernel(beta=2.)
    r = np.array([[.8, .15, .05], [.1, .2, .7]])
    tq = np.array([model.probabilities(b) for b in np.eye(3)])
    pairs = np.array([r @ (tq[:, a]*tq[:, b]) for a,b in itertools.product(range(3), repeat=2)]).T
    epsilon = public_privacy_certificate(model)['epsilon_Q']
    assert np.max(np.log(pairs.max(axis=0)/pairs.min(axis=0))) <= 2*epsilon+1e-12


@pytest.mark.parametrize('name,value', [('tv_bound', -1.), ('proxy_error_bound', np.nan),
                                      ('tv_bound', True), ('proxy_error_bound', 1.01)])
def test_invalid_assumptions_are_rejected(name, value):
    with pytest.raises(ValueError):
        utility_certificate(kernel(), [1., 0., 0.], **{name: value})
