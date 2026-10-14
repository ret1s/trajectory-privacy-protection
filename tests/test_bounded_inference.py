from fractions import Fraction
import itertools
import pytest

from benchmark.bounded_inference import inference_certificate
from benchmark.finite_query_kernel import FiniteCenteredKernel


def test_exact_joint_TV_controls_optimal_attacker_for_unequal_prior():
    kernel=FiniteCenteredKernel((1,3,2),Fraction(1,2),resolution=2)
    scores=list(itertools.product(range(3),repeat=3))
    for prior in (Fraction(1,2),Fraction(3,4),Fraction(1,10)):
        for left in scores:
            p=kernel.probabilities(left)
            for right in scores:
                q=kernel.probabilities(right)
                success=sum(max(prior*a,(1-prior)*b) for a,b in zip(p,q))
                cert=inference_certificate(1.,1,infinite=False,
                    prior_optimal_success_upper=max(prior,1-prior))
                assert success<=Fraction(cert['average_success_upper'])


def test_unknown_prior_never_becomes_uniform_and_average_is_not_pointwise():
    c=inference_certificate()
    assert c['average_success_upper'] is None and not c['prior_bound_assumed']
    assert Fraction(c['exact_average_inference_gain_upper'])<=Fraction(553,10000)
    assert Fraction(c['exact_information_bits_upper'])<=Fraction(9,500)
    balanced=inference_certificate(prior_optimal_success_upper=Fraction(1,2))
    assert Fraction(balanced['average_success_upper'])<=Fraction(5553,10000)
    twenty=inference_certificate(prior_optimal_success_upper=Fraction(1,20))
    assert Fraction(twenty['average_success_upper'])<=Fraction(1053,10000)
    with pytest.raises(ValueError):inference_certificate(prior_optimal_success_upper=True)
