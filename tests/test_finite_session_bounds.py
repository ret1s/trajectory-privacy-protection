from fractions import Fraction
import math
import pytest
from benchmark.finite_session_bounds import prefix_ratio,all_duration_ratio,posterior_mass_bound

@pytest.mark.parametrize('gamma',[Fraction(0),Fraction(1,2),Fraction(1),Fraction(2)])
def test_all_duration_bound_includes_every_prefix(gamma):
    bound=all_duration_ratio(gamma);ratio=Fraction(1)
    for j in range(1,1001):
        ratio*=1+gamma/Fraction(j*(j+1))
        assert ratio<=bound
    if gamma:assert float(bound)<math.exp(float(gamma))


def test_closed_rational_values_and_prior_limit():
    assert prefix_ratio(1,11)==Fraction(10763282027953,4817387520000)
    r=all_duration_ratio(1)
    assert r==Fraction(10763282027953,4415938560000)
    assert posterior_mass_bound(0,r)==0 and posterior_mass_bound(1,r)==1
    assert float(posterior_mass_bound(Fraction(1,2),r))==pytest.approx(.7090800193321709)
    assert posterior_mass_bound(Fraction(9,10),r)>Fraction(9,10)

@pytest.mark.parametrize('g,j',[(-1,11),(True,11),(1,-1),(1,True),(1,1.5),(12,11)])
def test_invalid_all_duration_parameters(g,j):
    with pytest.raises(ValueError):all_duration_ratio(g,j)
