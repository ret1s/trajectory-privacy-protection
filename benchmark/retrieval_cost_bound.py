"""Necessary record budget under balanced DISJOINT service references.

Not a universal lower bound for overlapping real POI references. All records
already available offline must count in C; a zero-query bulk control is allowed.
"""
from fractions import Fraction
from benchmark.whole_path_chi_square import whole_path_bound


def necessary_records(regions, reference_k, target_recall, *, gamma=1.):
    for value in (regions,reference_k):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError('Positive public region/reference count required')
    if isinstance(target_recall,bool):raise ValueError('Rational target Recall required')
    recall=Fraction(target_recall)
    if not 0<=recall<=1:raise ValueError('Recall target in [0,1] required')
    delta=Fraction(whole_path_bound(gamma,64,infinite=True)['exact_rational_utility_shift_upper'])
    lower=max(Fraction(0),regions*reference_k*(recall-delta))
    ceiling=(lower.numerator+lower.denominator-1)//lower.denominator
    return dict(exact_record_lower=str(lower),integer_record_lower=ceiling,
        scope='uniform exogenous region; pairwise-disjoint k-item reference sets; entire available cache including offline records; NECESSARY, not sufficient')
