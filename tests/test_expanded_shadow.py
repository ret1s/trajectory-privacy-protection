import numpy as np
import pytest

from evaluation.service_shadow import fit
from evaluation.expanded_shadow import centroid, fit_trees, predict, TREE_PARAMS


def test_forests_are_data_only_deterministic_and_row_causal():
    rng = np.random.default_rng(194)
    x = rng.normal(size=(55, 21))
    y = centroid(x) + rng.normal(size=(55, 2)) * 10
    model = fit(x, y, {'role': 'test_train'})
    a, b = fit_trees(model), fit_trees(model)
    assert TREE_PARAMS['max_features'] == 1 and type(TREE_PARAMS['max_features']) is int
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert all(v.dtype != object for v in a.values())
    full = predict(model, a, x[:3])
    prefix = predict(model, a, x[:1])
    assert len(full) == 10
    for name in full:
        assert np.array_equal(full[name][:1], prefix[name])
        assert full[name].shape == (3, 2) and np.isfinite(full[name]).all()
    with pytest.raises(ValueError):
        predict(model, a, np.full((1, 21), np.nan))
    with pytest.raises(ValueError):
        centroid(np.ones((2, 20)))


def test_centroid_uses_only_current_five_candidates():
    x = np.zeros((1, 21))
    x[:, :10] = np.tile([1., 2.], 5)
    x[:, 10:] = 99999
    assert np.array_equal(centroid(x), [[1000., 2000.]])


def test_attacker_selection_uses_shared_bank_and_separate_losses(monkeypatch):
    import experiments.run_expanded_shadow as run
    monkeypatch.setattr(run, 'METHODS', ('fixture',))
    ss = [
        {'method': 'fixture', 'case_id': 'S1.A',
         'mae_by_attack': {'a': 4., 'b': 3.}, 'hit_by_attack': {'a': .8, 'b': .2}},
        {'method': 'fixture', 'case_id': 'S3.A',
         'mae_by_attack': {'a': 4., 'b': 3., 'offline': 0.},
         'hit_by_attack': {'a': .8, 'b': .2, 'offline': 1.}},
    ]
    result = run.choose(ss)
    assert result['global_attackers'] == {'fixture': {'mae': 'b', 'hit': 'a'}}
    assert result['attackers']['fixture/S3.A'] == {'mae': 'offline', 'hit': 'offline'}
    # A tie must not depend on dictionary insertion order.
    ss[0]['mae_by_attack'] = {'b': 4., 'a': 4.}
    assert run.choose(ss)['attackers']['fixture/S1.A']['mae'] == 'a'
