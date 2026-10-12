"""Public-clock rational bounds for the finite Q kernel; no private state."""
from fractions import Fraction


def _parameters(gamma, rounds):
    if isinstance(gamma,bool) or isinstance(rounds,bool) or not isinstance(rounds,int) or rounds<0:
        raise ValueError('Nonnegative rational Gamma and public integer rounds required')
    g=Fraction(gamma)
    if g<0:raise ValueError('Nonnegative Gamma required')
    return g


def prefix_ratio(gamma, rounds):
    g=_parameters(gamma,rounds);ratio=Fraction(1)
    for j in range(1,rounds+1):ratio*=1+g/Fraction(j*(j+1))
    return ratio


def all_duration_ratio(gamma, cutoff=11):
    g=_parameters(gamma,cutoff);tail=g/Fraction(cutoff+1)
    if tail>=1:raise ValueError('Public cutoff requires Gamma/(cutoff+1)<1')
    # Every elementary symmetric tail sum e_k <= tail^k. This bound
    # holds for all prefixes, including BEFORE cutoff, by monotonicity.
    return prefix_ratio(g,cutoff)/(1-tail)


def posterior_mass_bound(prior, ratio):
    if isinstance(prior,bool) or isinstance(ratio,bool):raise ValueError('Rational inputs required')
    prior,ratio=Fraction(prior),Fraction(ratio)
    if not 0<=prior<=1 or ratio<1:raise ValueError('Prior in [0,1] and ratio>=1 required')
    return ratio*prior/(ratio*prior+1-prior)
