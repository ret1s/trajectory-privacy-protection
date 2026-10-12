from fractions import Fraction
import itertools
import pytest
from benchmark.finite_query_kernel import FiniteCenteredKernel,quantize_scores
from benchmark.conditional_gain import conditional_actual_gain


def test_conditional_true_gain_bound_exhaustive_actual_utilities():
    v=(.1,.4,.9);f=quantize_scores(v)
    for eps in (Fraction(1,2),Fraction(1),Fraction(2)):
        k=FiniteCenteredKernel([1,3,8],eps);w=[Fraction(m,k.M) for m in k.masses];p=k.probabilities(f)
        for h in itertools.product((Fraction(0),Fraction(1,4),Fraction(1,2),Fraction(3,4),Fraction(1)),repeat=3):
            delta=max(abs(Fraction(x)-Fraction(y)) for x,y in zip(h,v))
            cert=conditional_actual_gain(k,f,v,uniform_error_bound=delta)
            actual=sum((a-b)*c for a,b,c in zip(p,w,h))
            assert actual>=Fraction(cert['actual_gain_lower'])


def test_unknown_error_is_not_assumed_zero_and_wrong_proxy_can_lose():
    k=FiniteCenteredKernel([1,41],Fraction(1,2),resolution=1)
    assert conditional_actual_gain(k,[1,0],[1.,0.])['actual_gain_lower'] is None
    cert=conditional_actual_gain(k,[1,0],[1.,0.],uniform_error_bound=1)
    assert Fraction(cert['exact_planning_gain'])>0 and Fraction(cert['actual_gain_lower'])<0


def test_invalid_calibration_bound_rejected():
    k=FiniteCenteredKernel([1,1],1)
    for delta in (True,-1):
        with pytest.raises(ValueError):conditional_actual_gain(k,[0,1],[0.,1.],uniform_error_bound=delta)


def test_universal_same_prefix_shift_for_every_bounded_utility_vertex():
    from benchmark.conditional_gain import conditional_utility_shift
    for masses in ((1,1,1),(1,3,8)):
        for eps in (Fraction(1,2),Fraction(1),Fraction(2)):
            k=FiniteCenteredKernel(masses,eps,resolution=2)
            w=[Fraction(m,k.M) for m in masses]
            for scores in itertools.product(range(3),repeat=3):
                p=k.probabilities(scores);cert=conditional_utility_shift(k,scores)
                tv=Fraction(cert['exact_same_prefix_total_variation'])
                assert tv<=Fraction(cert['universal_same_prefix_utility_shift_upper'])
                # Linear utility extremes on [0,1]^3 occur at these vertices.
                for h in itertools.product((0,1),repeat=3):
                    assert abs(sum((a-b)*c for a,b,c in zip(p,w,h)))<=tv
    first=FiniteCenteredKernel([1,1],Fraction(1,2),resolution=1)
    assert Fraction(conditional_utility_shift(first,[0,1])['universal_same_prefix_utility_shift_upper'])==Fraction(1,20)
