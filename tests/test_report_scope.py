import pytest
from evaluation.report_scope import PRIORITY_CASES, ALL_CASES, scenario_macro
from experiments.reaggregate_active_scope import cost_summary, paired_interval


def test_two_case_scenario_retains_equal_scenario_weight():
    values={c:float(c.startswith('S10.')) for c in PRIORITY_CASES}
    values['S10.B']=999
    assert len(ALL_CASES)==29 and len(PRIORITY_CASES)==14
    assert scenario_macro(values)==pytest.approx(.2)


def test_no_observed_metric_remains_missing():
    assert scenario_macro({c:None for c in PRIORITY_CASES}) is None


def test_cost_filter_keeps_full_clock_and_equal_family_weight():
    rows=[dict(family_id=f,session_id=s,request_bytes=b,service_events=10)
          for f,s,b in [('f1','a',100),('f1','b',300),('f2','c',400),('f2','excluded',100000)]]
    result=cost_summary(rows,{'a','b','c'})
    assert result['request_bytes_per_service_event']==30
    assert result['cost_completed_sessions']==3


def test_paired_bootstrap_preserves_constant_difference_with_missing_family():
    result=paired_interval({'S10.A':{'f1':-.1,'f2':-.1},'S10.C':{'f1':-.1,'f2':-.1,'f3':-.1}})
    assert result['delta']==pytest.approx(-.1)
    assert result['ci95']==pytest.approx([-.1,-.1])
    assert result['valid_draws']<3000  # Missing A in all-f3 draws stays undefined.
