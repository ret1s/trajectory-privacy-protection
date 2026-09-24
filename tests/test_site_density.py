import numpy as np
import pytest
from evaluation.site_density import combine_density, GridDecisions, SiteDensity


def test_prior_correction_single_view_and_uninformative_views():
    p = np.array([.8, .2]); q = np.array([.4, .6])
    assert np.allclose(combine_density([q], p), q)
    assert np.allclose(combine_density([p, p], p), p)
    expected = q*q/p; expected /= expected.sum()
    assert np.allclose(combine_density([q, q], p), expected)
    assert np.allclose(combine_density([q, p], p), q)
    with pytest.raises(ValueError): combine_density([[0., 1.]], p)


def test_grid_decisions_match_brute_grid_hit_mass_and_do_not_worsen_mean_risk():
    xy = np.array([[0., 0.], [40., 0.], [100., 0.], [250., 0.]])
    model = GridDecisions(xy, radii=(50, 100)); w = np.array([.1, .15, .6, .15])
    pred = model.predict(w)
    for radius in (50, 100):
        brute = ((np.linalg.norm(xy[:, None]-xy[None, :], axis=2) <= radius)*w).sum(axis=1)
        assert np.array_equal(pred[f'hit{radius}'], xy[np.argmax(brute)])
    risk = lambda z: np.linalg.norm(xy-z, axis=1) @ w
    assert risk(pred['mae']) <= risk(pred['mean'])+1e-12


def test_density_normalized_positive_and_query_batch_invariant():
    rng = np.random.default_rng(123)
    x, y, grid = rng.normal(size=(80, 5)), rng.normal(size=(80, 2))*100, rng.normal(size=(20, 2))*200
    model = SiteDensity(x, y, grid)
    a = model.probabilities(x[:2]); b = model.probabilities(x[:1])
    assert np.allclose(a.sum(axis=1), 1.) and np.all(a > 0)
    assert np.array_equal(a[:1], b)
    assert model.prior.sum() == pytest.approx(1.)
    with pytest.raises(ValueError): model.probabilities(np.zeros((1, 3)))
