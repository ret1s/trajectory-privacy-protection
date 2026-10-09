import copy
import numpy as np
import pytest

from experiments.s8_companion_inference_20261007 import (
    causal_prefix, features, position_from_public, aggregate)


def public(times=(0, 60, 120), lon=116.32):
    return {'events':[dict(event_id=f'e{i}',timestamp_s=t,
        candidates=[dict(candidate_id='c0',lat=40.,lon=lon+i*.001)])
        for i,t in enumerate(times)]}


def test_actual_common_clock_offset_prevents_future_partner_lookup():
    target = causal_prefix(public(), 0., 60.)
    partner = causal_prefix(public(), 3., 60.)
    assert [e['timestamp_s'] for e in target['events']] == [0.,60.]
    assert [e['timestamp_s'] for e in partner['events']] == [3.]
    assert features(target,partner,60.,joint=True)[-2] == pytest.approx(57/60)


def test_future_coordinates_and_private_future_fields_cannot_change_features():
    tape = public()
    before = features(causal_prefix(tape,3.,60.),None,60.,joint=False)
    changed = copy.deepcopy(tape)
    changed['events'][1]['candidates'][0]['lon'] = 0.
    changed['events'][1]['secret_future_label'] = 'not read'
    assert np.array_equal(before,features(causal_prefix(changed,3.,60.),None,60.,joint=False))


@pytest.mark.parametrize('where', ['root','event','candidate'])
def test_private_fields_in_visible_prefix_are_rejected(where):
    tape = public((0,))
    obj = tape if where=='root' else tape['events'][0] if where=='event' else tape['events'][0]['candidates'][0]
    obj['declared_companions'] = True
    with pytest.raises(ValueError): causal_prefix(tape,0.,0.)


def test_missing_partner_is_explicit_and_not_imputed_as_colocation():
    target = causal_prefix(public(),0.,0.)
    assert causal_prefix(public(),3.,0.) is None
    x = features(target,None,0.,joint=True)
    assert len(x)==67 and np.all(x[32:]==0)


def test_joint_rejects_future_view_and_candidate_order_does_not_matter():
    target = public((0,))
    with pytest.raises(ValueError):features(target,public((10,)),0.,joint=True)
    twin = copy.deepcopy(target)
    target['events'][0]['candidates'].append(dict(candidate_id='c1',lat=40.001,lon=116.32))
    twin=copy.deepcopy(target);twin['events'][0]['candidates'].reverse()
    assert np.array_equal(features(target,None,0.,joint=False),features(twin,None,0.,joint=False))


def test_family_not_tick_weight_and_both_cases_equal_when_defined():
    rows = [dict(family_id='f1',case_id='S8.A',truth_xy=np.array([0.,0.]),partner_visible=True)]*100
    rows += [dict(family_id='f1',case_id='S8.B',truth_xy=np.array([100.,0.]),partner_visible=True),
             dict(family_id='f2',case_id='S8.A',truth_xy=np.array([300.,0.]),partner_visible=False)]
    result=aggregate(rows,np.zeros((102,2)))
    assert result['family_macro_mae_m']==175.
    assert result['event_pooled_mae_m']==pytest.approx(400/102)
    assert result['family_macro_hit100']==.5


def test_public_velocity_extrapolates_only_saved_prefix():
    tape=public((3,63))
    latest=position_from_public(tape,120.)
    velocity=position_from_public(tape,120.,velocity=True)
    assert velocity[0] > latest[0]
    assert velocity[1] == latest[1]


@pytest.mark.parametrize('cut', [0.,60.,120.])
def test_independent_clock_features_match_with_missing_or_lagged_partner(cut):
    from experiments.verify_s8_companion_inference_20261007 import clip,inputs
    a=public();b=public(lon=116.33)
    production=features(causal_prefix(a,0.,cut),causal_prefix(b,3.,cut),cut,joint=True)
    independent=inputs(clip(a,0.,cut),clip(b,3.,cut),cut,True)
    assert np.array_equal(production,independent)


def test_independent_family_case_arithmetic_and_changed_prediction_rejected():
    from experiments.verify_s8_companion_inference_20261007 import grouped,same
    rows=[dict(family_id='f1',case_id='S8.A',truth_xy=np.array([0.,0.]),partner_visible=True),
          dict(family_id='f2',case_id='S8.B',truth_xy=np.array([200.,0.]),partner_visible=False)]
    independent=[dict(family_id=r['family_id'],case_id=r['case_id'],truth=r['truth_xy'],
                      partner=[(0.,np.array([[0.,0.]]))] if r['partner_visible'] else []) for r in rows]
    y=np.zeros((2,2))
    same(aggregate(rows,y),grouped(independent,y))
    with pytest.raises(AssertionError):same(aggregate(rows,y),grouped(independent,y+1.))
