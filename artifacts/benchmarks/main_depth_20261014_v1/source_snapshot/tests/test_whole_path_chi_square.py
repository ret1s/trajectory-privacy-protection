import itertools
from fractions import Fraction

import pytest

from benchmark.finite_query_kernel import FiniteCenteredKernel
from benchmark.whole_path_chi_square import whole_path_bound
from core.query_budget import allowance


def test_adaptive_path_chi_square_and_all_bounded_utility_vertices():
    for private_first in itertools.product(range(3),repeat=2):
        p,w={():Fraction(1)},{():Fraction(1)}
        for j in range(1,4):
            new_p,new_w={},{}
            for h in p:
                masses=(1,2) if not h else (1+h[-1],3,2)
                scores=private_first if not h else tuple((a+sum(h))%3 for a in range(3))
                kernel=FiniteCenteredKernel(masses,allowance(1.,j),resolution=2)
                for a,prob in enumerate(kernel.probabilities(scores)):
                    new_p[h+(a,)]=p[h]*prob
                    new_w[h+(a,)]=w[h]*Fraction(masses[a],sum(masses))
            p,w=new_p,new_w
        chi=sum((p[h]-w[h])**2/w[h] for h in p)
        tv=sum(abs(p[h]-w[h]) for h in p)/2
        bound=whole_path_bound(1.,3)
        assert chi<=Fraction(bound['exact_chi_square_upper'])
        assert tv*tv<=Fraction(bound['exact_utility_shift_squared_upper'])
        best={h:int(p[h]>w[h]) for h in p}
        assert sum((p[h]-w[h])*best[h] for h in p)==tv


def test_infinite_bound_and_long_sessions_do_not_require_unbounded_integer_strings():
    short=whole_path_bound(1.,11)
    long=whole_path_bound(1.,100000)
    infinite=whole_path_bound(1.,64,infinite=True)
    assert Fraction(infinite['exact_rational_utility_shift_upper'])**2 >= Fraction(infinite['exact_utility_shift_squared_upper'])
    assert Fraction(infinite['exact_rational_utility_shift_upper']) <= Fraction(553,10000)
    assert short['utility_shift_display_upper']<.056
    assert Fraction(short['exact_chi_square_upper'])<=Fraction(long['exact_chi_square_upper'])
    assert long['exact_chi_square_upper']==infinite['exact_chi_square_upper']
    assert infinite['utility_shift_display_upper']<.056
    assert whole_path_bound(0.,100000)['exact_chi_square_upper']=='0'
    with pytest.raises(ValueError):whole_path_bound(1.,0,infinite=True)
    with pytest.raises(ValueError):whole_path_bound(3.,10)
