"""Whole-session utility distortion with adaptive public support/base.

Exact rational coupling and chi-square/KL bounds. No belief calibration, IID
samples, fixed support across time, or similarity of realized caches required.
The two policies MUST share the same public rule at a matched history.
"""
from fractions import Fraction
import math

from core.query_budget import allowance


def session_utility_bound(gamma, selections, *, infinite=False):
    if isinstance(selections, bool) or not isinstance(selections, int) or selections < 0:
        raise ValueError('Nonnegative selection count required')
    allowance(gamma, 1)
    if infinite and selections == 0:
        raise ValueError('Infinite bound needs a positive computed prefix')
    # A fixed-size exact prefix plus a conservative tail avoids enormous
    # numerator/denominator strings for arbitrarily long sessions.
    prefix = min(selections, 64)
    common, kl_upper = Fraction(1), Fraction(0)
    for j in range(1, prefix + 1):
        epsilon = allowance(gamma, j)
        alpha = epsilon / (2 + epsilon)
        common *= 1 - alpha / 4
        # KL(p||w) <= chi-square(p||w) = alpha^2 Var_w(z).
        kl_upper += alpha * alpha / 4
    coupling = 1 - common
    if infinite or prefix < selections:
        # alpha_j <= Gamma/(2*j*(j+1)); union bound for uncomputed
        # coupling tail, and integral upper bound for square tail.
        g = Fraction(str(gamma))
        tail = g / (8 * (prefix + 1))
        if not infinite:
            tail -= g / (8 * (selections + 1))
        coupling = min(Fraction(1), coupling + tail)
        # For j > n: 1/[j^2(j+1)^2] <= 1/j^4; sum <= integral_n^inf.
        kl_upper += g * g / (48 * prefix**3)
    # Keep the exact KL bound as authoritative. The displayed square root is
    # binary64, rounded upwards; it is not an exact real-number certificate.
    pinsker = math.nextafter(math.sqrt(float(kl_upper / 2)), math.inf) if kl_upper else 0.
    return dict(exact_coupling_upper=str(coupling), exact_KL_upper=str(kl_upper),
                pinsker_display_upper=pinsker,
                utility_shift_display_upper=min(float(coupling), pinsker, 1.),
                scope='whole adaptive session vs same public-base policy; any utility in [0,1]',
                horizon='infinite' if infinite else selections, computed_exact_prefix=prefix,
                floating_display_certified=False)
