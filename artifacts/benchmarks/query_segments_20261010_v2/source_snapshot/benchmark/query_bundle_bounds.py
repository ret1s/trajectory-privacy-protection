"""Ideal-real certificates for the opt-in Geo-I bundle kernel.

Public score-oscillation/floor certificates use only the fixed coverage table.
Utility certificates depend on protected belief and MUST stay local. Numerical
values are float diagnostics, not certified finite-precision DP sampling. Belief
error bounds are caller assumptions, never silently inferred from model scores.
"""
import numpy as np

from benchmark.probabilistic_query_bundle import ProbabilisticQueryBundle


def _probability_bound(value, name):
    if (isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f'{name} must be a finite bound in [0,1]')
    return float(value)


def public_privacy_certificate(kernel):
    """epsilon_Q = beta/2 * kappa for arbitrary pairs of protected beliefs.

    kappa = max_{a,a'} range_x(g(x,a)-g(x,a'))/(1+lambda).
    Action-independent offsets cancel from the softmax. This is tighter than
    beta/(1+lambda), and does not assume true GPS lies on the belief grid.
    """
    if not isinstance(kernel, ProbabilisticQueryBundle):
        raise ValueError('A fixed public bundle kernel required')
    coverage = kernel.coverage
    kappa = 0.
    for a in range(coverage.shape[1]):
        differences = coverage[:, a, None]-coverage
        kappa = max(kappa, float(np.ptp(differences, axis=0).max()))
    kappa /= 1.+kernel.cost_weight
    return dict(score_oscillation=kappa, epsilon_Q=.5*kernel.beta*kappa,
                conservative_epsilon_Q=kernel.beta/(1.+kernel.cost_weight),
                public_min_coverage=float(coverage.min()),
                scope='ideal-real kernel on this public score table; not a float sampler certificate')


def calibrated_beta(kernel, *, epsilon_target):
    """PUBLIC calibration from score geometry, not from attacker test scores.

    The underlying prototype caps beta at40 for float support. If kappa=0,
    beta40 improves the public objective without spending privacy: scores differ
    only by action-independent belief offsets. Tiny nonzero kappa is NOT zero.
    """
    if (isinstance(epsilon_target, (bool, np.bool_)) or not np.isfinite(epsilon_target)
            or epsilon_target < 0):
        raise ValueError('Finite nonnegative public privacy target required')
    kappa = public_privacy_certificate(kernel)['score_oscillation']
    return 40. if kappa == 0 else min(40., 2.*float(epsilon_target)/kappa)


def public_floor_indices(kernel, *, minimum_coverage):
    """Safe offline pruning on ALL public proxy states, never on current Z/b.

    An empty result is explicit infeasibility. No secret-dependent fallback and
    no tolerance that silently includes below-floor actions. Bounds concern the
    public proxy table; grid/sensor/purpose errors require their own allowance.
    """
    floor = _probability_bound(minimum_coverage, 'minimum_coverage')
    indices = np.flatnonzero(kernel.coverage.min(axis=0) >= floor)
    if not len(indices):
        raise ValueError('No public bundle meets this coverage floor')
    return indices


def utility_certificate(kernel, belief, *, tv_bound=0., proxy_error_bound=0.):
    """Conditional utility bridge; optional errors must be justified separately.

    tv_bound bounds TV(true conditional grid distribution, belief).
    proxy_error_bound uniformly bounds actual-purpose/grid/local-ranking error
    relative to the public proxy. Defaults0 describes a MATCHED ideal model;
    it does not certify calibration of the current mobility filter.
    """
    tv = _probability_bound(tv_bound, 'tv_bound')
    proxy = _probability_bound(proxy_error_bound, 'proxy_error_bound')
    error = min(1., tv+proxy)
    scores = kernel.scores(belief)
    probabilities = kernel.probabilities(belief)
    best = float(scores.max())
    expected_score = float(np.sum(probabilities*scores))
    model_cover = np.einsum('i,ij->j', np.asarray(belief), kernel.coverage, optimize=False)
    expected_cover = float(np.sum(probabilities*model_cover))
    expected_cost = float(np.sum(probabilities*kernel.cardinalities/kernel.cardinalities.max()))
    gaps = best-scores
    eta = .5*kernel.beta
    regret_bound = float(gaps.max())
    if eta > 0:
        # Gibbs variational bound: compare with base measure conditioned on
        # a near-optimal set. This removes the old '+1' tail-integration term.
        for delta in np.unique(gaps):
            log_mass = np.logaddexp.reduce(kernel.log_base_weights[gaps <= delta])
            regret_bound = min(regret_bound, float(delta-log_mass/eta))
    exact_regret = max(0., best-expected_score)
    # Cost term remains explicit: a score combines coverage with query savings.
    shortfall = (1.+kernel.cost_weight)*regret_bound + 2.*error + kernel.cost_weight*(1.-expected_cost)
    pointwise_floor = float(kernel.coverage.min())
    return dict(expected_model_coverage=expected_cover, expected_normalized_cost=expected_cost,
                exact_surrogate_score_regret=exact_regret,
                variational_score_regret_bound=max(0., regret_bound),
                assumed_tv_bound=tv, assumed_proxy_error_bound=proxy,
                lower_expected_actual_utility=max(0., expected_cover-error),
                additive_shortfall_to_best_actual_library_utility=min(1., shortfall),
                lower_per_action_actual_utility=max(0., pointwise_floor-proxy),
                scope='conditional ideal utility; local certificate, not a measured calibration claim')
