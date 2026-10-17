import itertools
from fractions import Fraction

import pytest

from benchmark.finite_query_kernel import FiniteCenteredKernel
from benchmark.session_utility_bounds import session_utility_bound
from core.query_budget import allowance


def test_whole_path_bound_with_adaptive_support_and_arbitrary_private_scores():
    # Public support/masses change with the prior sampled output. Private score
    # can depend on the entire past. Exhaustive exact path PMFs, no IID shortcut.
    for first in itertools.product(range(3), repeat=2):
        p, w = {(): Fraction(1)}, {(): Fraction(1)}
        for j in range(1, 4):
            next_p, next_w = {}, {}
            for history in p:
                masses = (1, 3) if not history else (1 + history[-1], 2, 1)
                scores = first if not history else tuple((a + sum(history)) % 3 for a in range(3))
                kernel = FiniteCenteredKernel(masses, allowance(1., j), resolution=2)
                for a, prob in enumerate(kernel.probabilities(scores)):
                    next_p[history + (a,)] = p[history] * prob
                    next_w[history + (a,)] = w[history] * Fraction(masses[a], sum(masses))
            p, w = next_p, next_w
        tv = sum(abs(p[h] - w[h]) for h in p) / 2
        certificate = session_utility_bound(1., 3)
        assert tv <= Fraction(certificate['exact_coupling_upper'])
        # Sup over every bounded function is TV, so this tests ALL [0,1]
        # whole-path utilities, including private-GPS-dependent Recall.
        maximizing = {h: int(p[h] > w[h]) for h in p}
        assert sum((p[h]-w[h])*maximizing[h] for h in p) == tv


def test_no_personalization_has_no_distortion_and_infinite_tail_is_conservative():
    assert session_utility_bound(0., 100)['exact_KL_upper'] == '0'
    finite = session_utility_bound(1., 1000)
    infinite = session_utility_bound(1., 11, infinite=True)
    assert Fraction(finite['exact_KL_upper']) <= Fraction(infinite['exact_KL_upper'])
    assert Fraction(finite['exact_coupling_upper']) <= Fraction(infinite['exact_coupling_upper'])
    assert infinite['utility_shift_display_upper'] < .081
    with pytest.raises(ValueError): session_utility_bound(1., 0, infinite=True)
    with pytest.raises(ValueError): session_utility_bound(1., True)
