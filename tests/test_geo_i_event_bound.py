from fractions import Fraction
import itertools
import pytest

from benchmark.geo_i_event_bound import event_log_bound


def test_event_bound_dominates_every_adaptive_prospective_cost_path():
    distances=[Fraction(3),Fraction(1),Fraction(7),Fraction(2)]
    for cap in range(1,8):
        bound=event_log_bound(distances,unit_epsilon=Fraction(1,100),cap_units=cap)
        for later in itertools.product(range(3),repeat=3):
            costs=(1,)+later
            if sum(costs)<=cap:
                value=sum(Fraction(c,100)*d for c,d in zip(costs,distances))
                assert value<=bound


def test_single_fix_is_not_charged_the_whole_trajectory_cap():
    assert event_log_bound([10]+[0]*20)==Fraction(1,10)
    assert event_log_bound([0,10]+[0]*20)==Fraction(1,5)
    assert event_log_bound([1]*100)==Fraction(23,100)
    assert event_log_bound([])==0
    with pytest.raises(ValueError):event_log_bound([True])
    with pytest.raises(ValueError):event_log_bound([-1])
