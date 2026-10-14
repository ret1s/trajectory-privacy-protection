"""Ideal extended Geo-I history: event-sensitive prospective-cap bound.

Distances are for SCHEDULED private GPS inputs only. First fresh release costs
one unit; later test+refresh costs at most two; total integer cap is mandatory.
This is a pairwise mathematical profile, not a bound published from secret GPS.
"""
from fractions import Fraction


def event_log_bound(distances_m, *, unit_epsilon='0.01', cap_units=23):
    if isinstance(cap_units,bool) or not isinstance(cap_units,int) or cap_units<1:
        raise ValueError('Positive public integer unit cap required')
    if isinstance(unit_epsilon,bool):raise ValueError('Nonnegative rational unit epsilon required')
    unit=Fraction(unit_epsilon)
    if unit<0:raise ValueError('Nonnegative rational unit epsilon required')
    distances=[]
    for value in distances_m:
        if isinstance(value,bool):raise ValueError('Nonnegative rational metre distance required')
        value=Fraction(value)
        if value<0:raise ValueError('Nonnegative rational metre distance required')
        distances.append(value)
    if not distances:return Fraction(0)
    # Mandatory first one-unit release leaves at most cap-1 later units.
    candidates=sorted((d for d in distances[1:] for _ in range(2)),reverse=True)
    return unit*(distances[0]+sum(candidates[:cap_units-1],Fraction(0)))
