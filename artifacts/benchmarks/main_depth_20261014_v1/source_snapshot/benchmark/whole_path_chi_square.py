"""Exact adaptive path-space chi-square bound vs the matched public base.

This is supplemental mathematics, not a change to the frozen experiment. A
rational squared-utility bound is authoritative; square-root displays are not
floating-point certificates. Research Gamma range is the declared [0,2].
"""
from fractions import Fraction
import math

from core.query_budget import allowance


def whole_path_bound(gamma, selections, *, infinite=False):
    allowance(gamma, 1)
    g = Fraction(str(gamma))
    if g > 2:
        raise ValueError('This research certificate supports public Gamma in [0,2]')
    if isinstance(selections, bool) or not isinstance(selections, int) or selections < 0:
        raise ValueError('Nonnegative public selection count required')
    if infinite and selections == 0:
        raise ValueError('Positive computed prefix required for infinite bound')
    prefix = min(selections, 64)
    product = Fraction(1)
    for j in range(1, prefix+1):
        epsilon = allowance(gamma, j)
        alpha = epsilon/(2+epsilon)
        product *= 1+alpha*alpha/4
    if infinite or prefix < selections:
        # Sum_{j>n} alpha_j^2/4 <= Gamma^2/(48*n^3).
        # Tail product <= 1/(1-s) by elementary symmetric sums <= s^k.
        tail = g*g/(48*prefix**3)
        if tail >= 1:
            raise ValueError('Increase the public cutoff for a nonvacuous tail')
        product /= 1-tail
    chi = product-1
    squared = min(Fraction(1), chi/4)
    resolution = 1_000_000
    scaled_numerator = squared.numerator*resolution*resolution
    ceiling = math.isqrt(scaled_numerator//squared.denominator)
    if ceiling*ceiling*squared.denominator < scaled_numerator:
        ceiling += 1
    rational_upper = Fraction(ceiling,resolution)
    display = math.nextafter(math.sqrt(float(squared)),math.inf) if squared else 0.
    return dict(exact_chi_square_upper=str(chi), exact_utility_shift_squared_upper=str(squared),
                exact_rational_utility_shift_upper=str(rational_upper),
                utility_shift_display_upper=display, floating_display_certified=False,
                computed_exact_prefix=prefix, horizon='infinite' if infinite else selections,
                scope='entire adaptive path vs IDENTICAL public-base rule; bounded [0,1] utility')
