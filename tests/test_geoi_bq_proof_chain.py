"""Exact hidden-state, aliasing, posterior and public-stopping proof oracles."""
from fractions import Fraction as F
import itertools

from benchmark.finite_query_kernel import FiniteCenteredKernel
from benchmark.finite_session_bounds import prefix_ratio, posterior_mass_bound


def test_marginalize_hidden_anchors_then_alias_wire_before_bayesian_inference():
    # Deliberately unrelated upstream laws and scores: the Q disclosure cap
    # does not require a calibrated posterior or successful inversion of Z.
    upstream = ((F(1, 5), F(4, 5)), (F(3, 4), F(1, 4)), (F(1, 2), F(1, 2)))
    kernel = FiniteCenteredKernel((1, 2, 3), F(1, 2), resolution=2)
    laws = []
    for x in range(3):
        law = [F(0), F(0)]
        for z in range(2):
            scores = tuple((x+2*z+a) % 3 for a in range(3))
            for a, p in enumerate(kernel.probabilities(scores)):
                law[a % 2] += upstream[x][z]*p  # several sampled lanes alias
        assert sum(law) == 1
        laws.append(law)
    ratio = 1+kernel.epsilon
    assert all(p <= ratio*q for left in laws for right in laws for p, q in zip(left, right))
    for prior in ((F(1, 3),)*3, (F(1, 10), F(1, 5), F(7, 10))):
        for event in ((0,), (1, 2), (0, 2)):
            mass = sum(prior[x] for x in event)
            lower = mass/(mass+ratio*(1-mass))
            upper = posterior_mass_bound(mass, ratio)
            for o in range(2):
                posterior = sum(prior[x]*laws[x][o] for x in event)/sum(prior[x]*laws[x][o] for x in range(3))
                assert lower <= posterior <= upper


def test_public_optional_stop_preserves_global_privacy_with_adaptive_support():
    def paths(x):
        law = {(): F(1)}
        for j in range(1, 4):
            expanded = {}
            for h, prob in law.items():
                masses = (1, 2) if not h else (2, 1, 1+sum(h))
                scores = tuple((x+a+sum(h)) % 3 for a in range(len(masses)))
                k = FiniteCenteredKernel(masses, F(1, j*(j+1)), resolution=2)
                for a, p in enumerate(k.probabilities(scores)):
                    expanded[h+(a,)] = prob*p
            law = expanded
        result = {}
        for h, p in law.items():
            stopped = h[:1] if h[0] == 0 else h
            result[stopped] = result.get(stopped, F(0))+p
        return result
    laws = [paths(x) for x in range(3)]
    ratio = prefix_ratio(1, 3)
    for left, right in itertools.product(laws, repeat=2):
        assert left.keys() == right.keys()
        assert all(left[h] <= ratio*right[h] for h in left)
