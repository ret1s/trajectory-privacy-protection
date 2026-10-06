"""Mathematical diagnostic controls; no study artifact or fresh data access."""
import pytest

from experiments.qplanner_proxy_diagnostic_20261006 import correlation,family_nested,movement_m


def test_nested_family_average_gives_unequal_draw_counts_equal_family_weight():
    rows=[dict(family_id='a',draw=d,x=.3) for d in (1,2,3)]+[dict(family_id='b',draw=1,x=.9)]
    result=family_nested(rows,'x')
    assert result['mean']==pytest.approx(.6)
    assert result['defined_families']==result['families']==2


def test_descriptive_correlation_rejects_nonfinite_and_keeps_constant_na():
    assert correlation([.4,.4],[.2,.8]) is None
    assert correlation([None,.5],[.1,.2]) is None
    assert correlation([.1,.3,.8],[.2,.5,.9],rank=True)==pytest.approx(1.)
    with pytest.raises(ValueError):correlation([float('nan'),.5],[.1,.2])


def test_public_movement_is_same_track_straight_line_and_initially_undefined():
    old=[(0.,0.)]*5;new=[(0.,.001)]*5
    assert movement_m(None,old) is None
    assert movement_m(old,old)==0.
    assert movement_m(old,new)==pytest.approx(111.1950802,abs=1e-6)
