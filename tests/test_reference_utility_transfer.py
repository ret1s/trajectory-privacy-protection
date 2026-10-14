"""Finite semantic counterexamples/oracles, not benchmark-score assertions."""
from fractions import Fraction as F
import itertools

import pytest

from benchmark.finite_query_kernel import FiniteCenteredKernel
from benchmark.reference_utility_transfer import reference_transfer


def test_same_marginal_service_distribution_does_not_transfer_a_public_floor():
    # Two user cases, two cached answers. Both caches have public mean Recall .5.
    # Q selection slightly favours the cache missing the current user's answer.
    # The GPS/request marginal remains exactly the uniform public reference law.
    k = FiniteCenteredKernel((1, 1), F(1, 2), resolution=1)
    conditional = [k.probabilities((1, 0)), k.probabilities((0, 1))]
    joint = {(x, a): F(1, 2)*conditional[x][a] for x in range(2) for a in range(2)}
    actual = sum(p for (x, a), p in joint.items() if x != a)
    assert actual == F(9, 20) < F(1, 2)
    assert all(sum(joint[x, a] for a in range(2)) == F(1, 2) for x in range(2))
    # Conditioning on the cache, rather than its unconditional marginal,
    # reveals TV=.05. The transfer bound is then sharp.
    cert = reference_transfer([F(1, 2)], conditional_tv_bounds=[F(1, 20)],
                              proxy_error_bounds=[0])
    assert F(cert['session_expected_recall_lower']) == actual


def test_oracle_over_all_small_correlated_history_case_tables_and_utility_vertices():
    reference = (F(1, 3), F(2, 3))
    for values in itertools.product(range(1, 4), repeat=4):
        joint = tuple(F(v, sum(values)) for v in values)
        for rewards in itertools.product((0, F(1, 2), 1), repeat=4):
            floor, tv = F(1), F(0)
            for h in range(2):
                mass = joint[2*h]+joint[2*h+1]
                posterior = (joint[2*h]/mass, joint[2*h+1]/mass)
                floor = min(floor, sum(reference[x]*rewards[2*h+x] for x in range(2)))
                tv = max(tv, sum(abs(p-r) for p, r in zip(posterior, reference))/2)
            cert = reference_transfer([floor], conditional_tv_bounds=[tv], proxy_error_bounds=[0])
            actual = sum(p*u for p, u in zip(joint, rewards))
            assert actual >= F(cert['session_expected_recall_lower'])


def test_session_expectation_and_failure_bound_without_independent_steps():
    # The same hidden Bernoulli state determines every tick, so ticks are
    # maximally dependent. Enumerate its full joint law instead of IID samples.
    for good in (F(1, 4), F(1, 2), F(3, 4)):
        cert = reference_transfer([good]*3, conditional_tv_bounds=[0]*3,
                                  proxy_error_bounds=[0]*3, recall_threshold=F(1, 2))
        assert F(cert['session_expected_recall_lower']) == good
        assert 1-good <= F(cert['session_below_threshold_probability_upper'])


def test_unknown_bounds_weighting_and_proxy_discrepancy_are_explicit():
    assert reference_transfer([F(99, 100)])['session_expected_recall_lower'] is None
    cert = reference_transfer([F(99, 100)], conditional_tv_bounds=[F(1, 10)],
                              proxy_error_bounds=[F(1, 20)])
    assert F(cert['session_expected_recall_lower']) == F(21, 25)
    assert cert['assumptions_verified_by_this_function'] is False
    weighted = reference_transfer([F(99, 100), 1], conditional_tv_bounds=[0, None],
                                  proxy_error_bounds=[0, None], time_weights=[1, 0])
    assert F(weighted['session_expected_recall_lower']) == F(99, 100)
    for kwargs in ({'conditional_tv_bounds': [True]}, {'proxy_error_bounds': [-1]},
                   {'time_weights': [F(1, 2)]}, {'recall_threshold': 1}):
        with pytest.raises(ValueError):
            reference_transfer([F(99, 100)], **kwargs)
