import numpy as np
import pytest

from evaluation.loss_aware_shadow import decision, predict
from evaluation.service_shadow import fit


def test_mean_is_not_hit_or_mae_optimum():
    p = np.array([[0., 0.], [0., 0.], [1000., 0.]])
    a = decision(p)
    assert np.array_equal(a['mae_action'], [0, 0])
    assert a['mae_risk'] < a['mean_risk']
    assert a['hit_mass'] == pytest.approx(2/3)
    assert a['mean_hit_mass'] == 0


def test_circle_centers_capture_support_missed_by_support_actions():
    p = np.array([[0., 0.], [150., 0.], [1000., 0.]])
    a = decision(p)
    assert a['hit_mass'] == pytest.approx(2/3)
    assert np.count_nonzero(np.linalg.norm(p-a['hit_action'], axis=1) <= 100+1e-7) == 2
    for bad in ([], [[np.nan, 0]], [[1, 2, 3]]):
        with pytest.raises(ValueError): decision(bad)
    with pytest.raises(ValueError): decision(p, 0)
    b = decision([[3., 7.]])
    assert b['hit_mass'] == 1 and b['mae_risk'] == 0


def test_prediction_is_deterministic_and_row_causal():
    x = np.zeros((50, 2)); y = np.c_[np.arange(50)*20., np.zeros(50)]
    m = fit(x, y, {'training_only': True})
    a, aa = predict(m, np.zeros((3, 2)))
    b, bb = predict(m, np.zeros((1, 2)))
    assert len(a) == 5
    for k in a: assert np.array_equal(a[k][:1], b[k])
    for n in (15, 45):
        assert aa[str(n)]['neighbors'][0] == list(range(n))
        assert aa[str(n)]['mae_risk'][0] <= aa[str(n)]['mean_risk'][0]
        assert aa[str(n)]['hit_mass'][0] >= aa[str(n)]['mean_hit_mass'][0]
    with pytest.raises(ValueError): predict(m, [[np.nan, 0]])
