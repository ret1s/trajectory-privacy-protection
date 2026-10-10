"""Independent privacy/utility contracts for the opt-in joint bundle kernel."""
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.probabilistic_query_bundle import ProbabilisticQueryBundle, QueryBundle, bundle_coverage


def fixture(beta=2.):
    actions = [QueryBundle((0,)), QueryBundle((1, 2)), QueryBundle((3, 4, 5))]
    coverage = np.array([[1., .1, .6], [0., .9, 1.], [.2, .4, .7]])
    return ProbabilisticQueryBundle(actions, coverage, beta=beta, cost_weight=.25)


def test_common_support_and_joint_likelihood_bound_include_cardinality():
    # Independent exhaustive simplex grid, not a mirrored normalization oracle.
    beliefs = [np.array([a, b, 8-a-b])/8 for a in range(9) for b in range(9-a)]
    kernel = fixture()
    probabilities = np.array([kernel.probabilities(b) for b in beliefs])
    assert np.all(probabilities > 0)
    assert np.allclose(probabilities.sum(axis=1), 1.)
    assert np.max(np.log(probabilities.max(axis=0)/probabilities.min(axis=0))) <= kernel.beta + 1e-12
    assert not np.allclose(probabilities[0], probabilities[-1])
    # Observed K is randomized jointly with the coordinates, not hidden/ignored.
    assert set(kernel.cardinalities) == {1, 2, 3}


def test_base_mass_is_balanced_per_k_and_beta_zero_ignores_belief():
    actions = [QueryBundle((0,)), QueryBundle((1,)), QueryBundle((2, 3))]
    kernel = ProbabilisticQueryBundle(actions, [[1., 0., .5], [0., 1., .8]], beta=0)
    assert kernel.probabilities([1., 0.]) == pytest.approx([.25, .25, .5])
    assert kernel.probabilities([0., 1.]) == pytest.approx([.25, .25, .5])


def test_whole_channel_posterior_vulnerability_and_two_round_composition():
    kernel = fixture()
    # Independent finite X -> Z matrix; arbitrary full-support privacy channel.
    rem = np.array([[.8, .15, .05], [.1, .2, .7]])
    prior = np.array([.4, .6])
    pz = prior @ rem
    posterior = (prior[:, None]*rem/pz).T
    coverage = kernel.coverage[:2]
    second = ProbabilisticQueryBundle(kernel.bundles, coverage, beta=2., cost_weight=.25)
    tq = np.array([second.probabilities(b) for b in posterior])
    observation = rem @ tq
    joint = prior[:, None]*observation
    pxq = joint / joint.sum(axis=0)
    pxz = (prior[:, None]*rem/pz)
    pzq = pz[:, None]*tq / joint.sum(axis=0)
    assert pxq == pytest.approx(pxz @ pzq)
    assert joint.max(axis=0).sum() <= (prior[:, None]*rem).max(axis=0).sum() + 1e-12
    assert np.all(observation.max(axis=0)/observation.min(axis=0) <= np.exp(2.) + 1e-12)
    # Repeat fresh randomness conditional on the SAME latent Z: integrate the
    # product inside the Z sum. Outputs are not independent given X.
    pairs = np.array([rem @ (tq[:, a]*tq[:, b])
                      for a, b in itertools.product(range(3), repeat=2)]).T
    assert np.all(pairs.max(axis=0)/pairs.min(axis=0) <= np.exp(4.) + 1e-12)
    for x in range(2):
        cap = np.exp(2.)*prior[x]/(np.exp(2.)*prior[x]+1-prior[x])
        assert pxq[x].max() <= cap + 1e-12


def test_more_independent_qs_can_make_anchor_reconstruction_easier():
    # X=Z is binary. Q_i copies Z with p=.75. More points/outputs do not imply
    # more privacy: the optimum majority decoder improves with three samples.
    one = .75
    three = .75**3 + 3*.75**2*.25
    assert three > one


@pytest.mark.parametrize('belief', [[.5, .5], [1., -1., 1.], [.4, .4, .4], [np.nan, 0., 0.]])
def test_reject_invalid_protected_belief(belief):
    with pytest.raises(ValueError):
        fixture().probabilities(belief)


def test_reject_zero_support_invalid_library_and_extreme_parameters():
    action = QueryBundle((0,))
    for kwargs in ({'base_weights': [0.]}, {'beta': -1.}, {'beta': 1000.}, {'cost_weight': -1.}):
        with pytest.raises(ValueError):
            ProbabilisticQueryBundle([action], [[1.]], **kwargs)
    with pytest.raises(ValueError):
        ProbabilisticQueryBundle([action, action], [[0., 1.]])
    for states in ((0, 0), (), (-1,), (True,)):
        with pytest.raises(ValueError):
            QueryBundle(states)


def test_float_extremes_keep_support_and_inputs_are_copied():
    coverage = np.array([[0., 1.], [1., 0.]])
    kernel = ProbabilisticQueryBundle([QueryBundle((0,)), QueryBundle((1,))],
                                     coverage, beta=40., base_weights=[1e-250, 1.])
    coverage[:] = 0.
    assert kernel.coverage[0, 1] == 1.
    assert np.all(kernel.probabilities([0., 1.]) > 0)
    assert kernel.sample([0., 1.], rng=np.random.default_rng(1)) in kernel.bundles


def test_public_reply_union_uses_unique_pois_and_nonempty_category_mean():
    class Road:
        def __len__(self):
            return 2

    rn = Road()
    reference = SimpleNamespace(rn=rn, pois=('p0', 'p1', 'p2'), categories=('a', 'b'), k=2,
                                query_indices=lambda state: np.array([[0, 1], [-1, -1]]))
    reply = SimpleNamespace(rn=rn, pois=reference.pois, categories=reference.categories, k=2,
                            query_indices=lambda state: np.array([[state, state], [2, -1]]))
    # Duplicate POIs and irrelevant category b never inflate recall.
    actions = [QueryBundle((0,)), QueryBundle((0, 1))]
    assert bundle_coverage(reference, reply, [0], actions)[0] == pytest.approx([.5, 1.])
    reference.query_indices = lambda state: np.full((2, 2), -1)
    with pytest.raises(ValueError, match='Undefined'):
        bundle_coverage(reference, reply, [0], actions)
