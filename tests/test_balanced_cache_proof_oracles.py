"""Exact oracles for adaptive public bases, projection and limited utility claims."""
from collections import defaultdict
from fractions import Fraction
import itertools
from benchmark.finite_query_kernel import FiniteCenteredKernel
from benchmark.finite_session_bounds import posterior_mass_bound


def test_adaptive_public_bases_and_hidden_prefix_projection():
    eps1,eps2=Fraction(1,2),Fraction(1,6);bound=(1+eps1)*(1+eps2)
    first=FiniteCenteredKernel([1,3],eps1,resolution=1)
    distribution=[]
    for y in (0,1):
        pmf=defaultdict(Fraction)
        for h in (0,1):
            # m depends on randomized PUBLIC h, not on target y.
            second=FiniteCenteredKernel([9,1] if h==0 else [1,9],eps2,resolution=1)
            for a in (0,1):
                p=first.probabilities([y,1-y])[h]*second.probabilities([1-y,y])[a]
                # Both first states share a wire coordinate; second supports
                # overlap only partly. Projection sums compatible paths.
                pmf[h+a]+=p
        assert sum(pmf.values())==1;distribution.append(pmf)
    for event in distribution[0]:
        p,q=distribution[0][event],distribution[1][event]
        assert p/q<=bound and q/p<=bound
        prior=Fraction(1,4);posterior=prior*p/(prior*p+(1-prior)*q)
        assert posterior<=posterior_mass_bound(prior,bound)


def test_increased_private_planning_score_is_not_public_coverage_gain():
    k=FiniteCenteredKernel([1,41],Fraction(1,2),resolution=1)
    z=[1,0];d=[0,10]
    public_expected=sum(Fraction(w,k.M)*v for w,v in zip(k.masses,d))
    actual_expected=sum(p*v for p,v in zip(k.probabilities(z),d))
    assert actual_expected<public_expected
    assert k.utility_certificate(z)['guaranteed_quantized_score_gain']>0


def test_session_union_handles_losing_current_pool_and_adaptive_requests():
    initial={0,1};frames=[{2},{3},{4}];cache=set(initial)
    for frame in frames:
        cache|=frame
        assert not initial.issubset(frame)  # current-frame invariant unnecessary
        for size in range(1,6):
            for reference in itertools.combinations(range(5),size):
                r=set(reference)
                assert Fraction(len(cache&r),size)>=Fraction(len(initial&r),size)
