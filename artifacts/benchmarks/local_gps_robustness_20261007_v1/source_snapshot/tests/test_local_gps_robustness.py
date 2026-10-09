import copy
import math

import pytest

from experiments.local_gps_robustness_20261007 import (
    ARMS, PURPOSES, VARIANTS, _baseline, estimate_xy, score_rankings,
    standardized_noise, summarize,
)


def fix(t, xy, normal=(0., 0.)):
    return {'t':float(t),'xy':list(xy),'normal':list(normal)}


def orders(categories):
    return {purpose:copy.deepcopy(categories) for purpose in PURPOSES}


def test_no_current_or_future_gps_can_repair_a_stale_fix():
    fixes = [fix(0,(0,0)),fix(60,(120,0)),fix(120,(240,0))]
    before = estimate_xy(fixes,80,'velocity_twofix60',0)
    poisoned = copy.deepcopy(fixes)
    poisoned[-1]['xy'] = ['poisoned future GPS',None]
    poisoned[-1]['normal'] = None
    poisoned.append(fix(180,(999999,-999999)))
    assert estimate_xy(poisoned,80,'velocity_twofix60',0) == before
    assert before['estimated_xy'] == [160.,0.]
    assert before['last_fix_t'] == 60
    assert before['stored_fix_count'] == 2
    assert estimate_xy(poisoned,80,'hold_last60',0)['estimated_xy'] == [120.,0.]


def test_first_fix_fallback_and_speed_clip_use_only_two_noisy_fixes():
    fixes = [fix(0,(0,0)),fix(60,(1200,0)),fix(120,(2400,0))]
    first = estimate_xy(fixes,20,'velocity_twofix60',0)
    assert first['first_fix_fallback'] and first['estimated_xy'] == [0.,0.]
    clipped = estimate_xy(fixes,80,'velocity_twofix60',0)
    assert clipped['raw_velocity_m_s'] == 20.
    assert clipped['velocity_clipped'] and clipped['estimated_xy'] == [1360.,0.]
    assert clipped['local_fixes_observed'] == 2


def test_noisy_velocity_and_hold_share_the_same_clock_offset():
    z = standardized_noise('f',1,2,60)
    assert z == standardized_noise('f',1,2,60)
    assert z != standardized_noise('f',2,2,60)
    assert all(math.isfinite(v) for v in z)
    fixes = [fix(0,(0,0)),fix(60,(120,0),z)]
    hold = estimate_xy(fixes,60,'hold_last60',5)
    velocity = estimate_xy(fixes,60,'velocity_twofix60',5)
    assert hold['estimated_xy'] == velocity['estimated_xy']
    high = estimate_xy(fixes,60,'hold_last60',15)['estimated_xy']
    for a,b,truth in zip(hold['estimated_xy'],high,[120.,0.]):
        assert b-truth == pytest.approx(3*(a-truth))


@pytest.mark.parametrize('mode,sigma', [('future_route',0),('hold_last60',-1),('hold_last60',math.nan)])
def test_invalid_estimator_parameters_fail_closed(mode,sigma):
    with pytest.raises(ValueError): estimate_xy([fix(0,(0,0))],20,mode,sigma)


def test_true_reference_does_not_move_to_estimated_ranking():
    actual = orders([[0,1,2,3,4,5]])
    estimated = orders([[5,4,3,2,1,0]])
    score = score_rankings(actual,estimated,set(range(6)))
    for result in score.values():
        assert result['recall5'] == .8
        assert result['completion'] == 1.
        assert result['reference_poi_total'] == 5
        assert result['overlap_total'] == 4


def test_defined_reference_with_unreachable_estimate_stays_zero_not_na():
    score = score_rankings(orders([[0,1],[]]),orders([[],[2]]),{0,1,2})
    for result in score.values():
        assert result['recall5'] == result['completion'] == 0
        assert result['reference_category_count'] == 1
        assert result['all_category_count'] == 2
        assert result['zero_answer_reference_categories'] == 1
        assert result['empty_reference_categories'] == 1
        assert result['answered_empty_reference_categories'] == 1
        assert result['returned_outside_true_domain'] == 1


def test_outside_true_radius_returns_are_not_counted_as_completion():
    actual = orders([[0,1]])
    estimate = orders([[2,3,0,1]])
    score = score_rankings(actual,estimate,{2,3})
    for result in score.values():
        assert result['recall5'] == result['completion'] == 0
        assert result['returned_items'] == 2
        assert result['returned_outside_true_domain'] == 2


def test_invalid_snap_keeps_reference_denominator_and_zero_answer():
    result = score_rankings(orders([[1],[]]),{},set(),invalid_estimate=True)
    for row in result.values():
        assert row['recall5'] == 0
        assert row['reference_poi_total'] == 1
        assert row['invalid_estimate_reference_categories'] == 1
    empty = score_rankings(orders([[]]),orders([[]]),set())
    assert all(row['recall5'] is None for row in empty.values())


def test_exact_control_recovery_detects_old_metric_or_denominator_tampering():
    score = score_rankings(orders([[0,1,2]]),orders([[0,1,2]]),{0,1})
    legacy = {p:{k:score[p][k] for k in ('recall5','completion','reference_category_count',
        'all_category_count','overlap_total','reference_poi_total')} for p in PURPOSES}
    _baseline(score,legacy)
    legacy[PURPOSES[0]]['reference_poi_total'] = 5
    with pytest.raises(AssertionError): _baseline(score,legacy)


def test_equal_family_nested_draw_aggregation_and_unchanged_cost():
    blocks = []
    for family in ('a','b'):
        for draw in (1,2,3):
            rows = []
            for variant in VARIANTS:
                for depth in (20,30):
                    score = score_rankings(orders([[0]]),orders([[0]]),{0} if family == 'a' else set())
                    for t in ([0] if family == 'a' else [0,20,40]):
                        rows.append(dict(arm=f'{variant}--L{depth}',slot=0,t=t,family_id=family,draw=draw,purposes=score))
            blocks.append(dict(rows=rows,estimates=[],wire=[dict(method=f'service_l{d}',requests=5,
                request_bytes=10,reply_bytes=d) for d in (20,30)]))
    result = summarize(blocks)
    for arm in ARMS:
        assert result['summary'][arm]['all']['equal_purpose_macro']['family_mean'] == .5
        assert result['summary'][arm]['all']['three_purpose_macro']['family_mean'] == .5
        assert result['cost'][arm]['requests'] == 30
        assert result['summary'][arm]['all'][PURPOSES[0]]['explicit_denominators']['total_events'] == 12
        assert result['summary'][arm]['temporal_tail_400_600']['equal_purpose_macro']['family_mean'] is None
        assert result['summary'][arm]['temporal_tail_400_600']['three_purpose_macro']['within_draw_family_mean'] == {'1':None,'2':None,'3':None}
