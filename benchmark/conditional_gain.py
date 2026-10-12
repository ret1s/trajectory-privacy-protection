"""Conditional actual-utility bound, never defaults unknown calibration to zero."""
from fractions import Fraction


def conditional_actual_gain(kernel,integer_scores,planning_scores,*,uniform_error_bound=None):
    f=tuple(integer_scores);kernel.integer_weights(f)
    if len(planning_scores)!=len(f):raise ValueError('Matched public-support score vectors required')
    v=tuple(Fraction(float(x)) for x in planning_scores)
    w=tuple(Fraction(m,kernel.M) for m in kernel.masses)
    z=tuple(Fraction(int(x),kernel.resolution) for x in f)
    mu=sum(a*b for a,b in zip(w,z))
    covariance=sum(a*b*(c-mu) for a,b,c in zip(w,v,z))
    spread=sum(a*abs(b-mu) for a,b in zip(w,z))
    result=dict(exact_planning_gain=str(kernel.alpha*covariance),exact_centered_absolute_spread=str(spread),
                actual_gain_lower=None,status='unverified calibration; no actual-gain lower bound')
    if uniform_error_bound is not None:
        if isinstance(uniform_error_bound,bool):raise ValueError('Nonnegative rational error bound required')
        delta=Fraction(uniform_error_bound)
        if delta<0:raise ValueError('Nonnegative error bound required')
        result.update(actual_gain_lower=str(kernel.alpha*(covariance-delta*spread)),
                      status='conditional on supplied uniform actual-vs-planning error bound',exact_uniform_error_bound=str(delta))
    return result


def conditional_utility_shift(kernel,integer_scores):
    """Exact same-prefix TV: protects any [0,1] utility, even a wrong proxy.

    This is NOT the difference between two whole policies with different
    public histories/cache/support. No calibration or IID assumption is used.
    """
    p=kernel.probabilities(integer_scores)
    w=tuple(Fraction(m,kernel.M) for m in kernel.masses)
    tv=sum(abs(a-b) for a,b in zip(p,w))/2
    assert tv<=kernel.alpha/4
    return dict(exact_same_prefix_total_variation=str(tv),
        universal_same_prefix_utility_shift_upper=str(kernel.alpha/4),
        scope='conditional on identical history/support/public base; any bounded [0,1] utility')
