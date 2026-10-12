import itertools
from fractions import Fraction
import numpy as np
import pytest

from benchmark.finite_query_kernel import FiniteCenteredKernel,quantize_scores,simulator_randbelow


def test_exhaustive_integer_normalization_support_and_likelihood():
    for epsilon in (Fraction(0),Fraction(1,12),Fraction(1,2),Fraction(2)):
        kernel=FiniteCenteredKernel([1,2,3],epsilon,resolution=3)
        distributions=[kernel.probabilities(s) for s in itertools.product(range(4),repeat=3)]
        assert all(sum(p)==1 and min(p)>0 for p in distributions)
        for p,q in itertools.product(distributions,repeat=2):
            assert all(a<=(1+epsilon)*b for a,b in zip(p,q))
        assert {sum(kernel.integer_weights(s)) for s in itertools.product(range(4),repeat=3)}=={kernel.public_total}


def test_exact_sampler_partitions_all_public_integer_draws():
    k=FiniteCenteredKernel([1,1],Fraction(1),resolution=2)
    weights=k.integer_weights([0,2]);counts=[0,0]
    for value in range(k.public_total):
        counts[k.sample([0,2],randbelow=lambda n,v=value:v)]+=1
    assert tuple(counts)==weights


def test_exact_expected_utility_gain_variance_identity():
    k=FiniteCenteredKernel([1,3,2],Fraction(1,2),resolution=8)
    for scores in itertools.product(range(9),repeat=3):
        z=[Fraction(s,8) for s in scores];w=[Fraction(m,6) for m in k.masses]
        mean=sum(a*b for a,b in zip(w,z));var=sum(a*(b-mean)**2 for a,b in zip(w,z))
        expected=sum(a*b for a,b in zip(k.probabilities(scores),z))
        assert expected==mean+k.alpha*var and expected>=mean


def test_zero_allowance_ignores_all_private_scores():
    k=FiniteCenteredKernel([1,3],Fraction(0),resolution=100)
    assert k.probabilities([0,100])==k.probabilities([100,0])==(Fraction(1,4),Fraction(3,4))
    assert k.sample([0,100],randbelow=lambda n:0)==k.sample([100,0],randbelow=lambda n:0)


def test_quantizer_improves_raw_planning_score_not_only_integer_proxy():
    random=np.random.default_rng(1)
    for _ in range(100):
        v=random.normal(size=5)
        f=quantize_scores(v)
        cert=FiniteCenteredKernel([1,2,3,4,5],Fraction(1,12)).utility_certificate(f,planning_scores=v)
        assert cert['monotone_planning_gain_certified']
        assert Fraction(cert['exact_planning_score_gain'])>=0


def test_quantization_total_bounded_and_monotone():
    assert quantize_scores([.1,.4,.8],100)==(0,42,100)
    assert quantize_scores([1.,1.,1.])==(0,0,0)
    for values in ([np.nan,np.inf,-np.inf],[1e308,-1e308,0.]):
        assert all(0<=f<=65536 for f in quantize_scores(values))


def test_simulator_byte_rejection_handles_arbitrary_precision_public_bound():
    draw=simulator_randbelow(np.random.default_rng(1))
    for n in (1,3,2**130+17):
        assert all(0<=draw(n)<n for _ in range(100))


@pytest.mark.parametrize('scores',[[0],[0,4],[True,0],[-1,0]])
def test_invalid_private_integer_scores_rejected(scores):
    with pytest.raises(ValueError):FiniteCenteredKernel([1,2],Fraction(1),resolution=3).integer_weights(scores)
