from fractions import Fraction
import math
import pytest
from benchmark.late_prefix_bounds import late_change_ratio,late_change_allowance
from benchmark.finite_session_bounds import prefix_ratio,posterior_mass_bound


def test_causal_late_prefix_composition_exact():
    for gamma in (Fraction(0),Fraction(1,2),Fraction(1),Fraction(2)):
        for j in range(1,13):
            r=late_change_ratio(gamma,j,11)
            s=late_change_allowance(gamma,j,11)
            assert r<=prefix_ratio(gamma,11) and float(r)<=math.exp(float(s))+1e-14
    assert late_change_ratio(1,11,11)==Fraction(133,132)
    assert late_change_allowance(1,5,11)==Fraction(7,60)
    assert float(posterior_mass_bound(Fraction(1,2),late_change_ratio(1,5,11)))<.53
    assert late_change_ratio(1,12,11)==1


@pytest.mark.parametrize('first,last',[(0,11),(True,11),(1,-1),(1,False),(1.5,11)])
def test_late_bounds_reject_ambiguous_public_indices(first,last):
    with pytest.raises(ValueError):late_change_ratio(1,first,last)
