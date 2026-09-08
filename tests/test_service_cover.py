"""Exact tiny oracles and information-boundary regressions."""
from itertools import product
import json

import numpy as np
import pytest

from benchmark.engines.service_cover import greedy_cover, ServiceCoverLaneDummy
from benchmark.engines.contextual_lane import ContextualLaneDummy
from core.demo_protocol import TrajectoryPoint
from evaluation.service_shadow import features, fit, predict
from tests.test_belief_lane import fixture


def test_global_not_fixed_track_order_and_duplicate_slots():
    sets = [{0}, {1, 2}, {0, 1, 2}]
    def gain(ids, chosen):
        covered = set().union(*(sets[i] for i in chosen))
        return np.array([len(sets[i] - covered) for i in ids])
    tie = lambda j, ids: (ids * 0., ids * 0.)
    selected, gains = greedy_cover([[0, 1], [2]], gain, tie)
    assert selected == [0, 2] and gains == [3., 0.]
    assert greedy_cover([[2], [2]], gain, tie)[0] == [2, 2]
    with pytest.raises(ValueError): greedy_cover([[]], gain, tie)


def test_exhaustive_small_partition_optimum_half_bound():
    rng = np.random.default_rng(287)
    for _ in range(100):
        cover = rng.random((9, 7)) < .35
        weights = rng.random(7)
        groups = [np.arange(3), np.arange(3, 6), np.arange(6, 9)]
        def objective(states):
            return float(weights[np.any(cover[list(states)], axis=0)].sum()) if states else 0.
        def gain(ids, selected):
            return np.array([max(0., objective(selected + [i]) - objective(selected)) for i in ids])
        result, gains = greedy_cover(groups, gain, lambda j, ids: (ids * 0., ids * 0.))
        optimum = max(objective(s) for s in product(*groups))
        assert objective(result) + 1e-12 >= .5 * optimum
        assert sum(gains) == pytest.approx(objective(result))


@pytest.mark.parametrize('prior_only', [False, True])
def test_prefix_pairing_directed_motion_and_no_extra_private_read(prior_only):
    rn, context, belief = fixture()
    points = tuple(TrajectoryPoint(i * 5., 0., .0001 + i * .0001) for i in range(15))
    def make():
        return ServiceCoverLaneDummy(rn, belief_model=belief, prior_only=prior_only,
                                      rng=np.random.default_rng(19))
    model = make(); public = model.protect_run(points).to_attacker_dict()
    assert make().protect_run(points[:5]).to_attacker_dict()['events'] == public['events'][:5]
    control = ContextualLaneDummy(rn, rng=np.random.default_rng(19))
    control.protect_run(points)
    assert model.evaluator_anchors == control.evaluator_anchors
    assert model.spent_bound == pytest.approx(.23)
    assert not any(k in json.dumps(public) for k in ('evaluator', 'greedy_gains', 'mean_xy'))
    for a, b in zip(model.evaluator_states, model.evaluator_states[1:]):
        for x, y in zip(a, b):
            assert y in model.travel.reachable(x, 5.)
    a, b = make(), make()
    for x in (a, b): x.anchor.perturb = lambda *args, **kwargs: rn.latlon(15)
    for i in range(15):
        assert a.protect_step(0., 0., i * 5.) == b.protect_step(20., 30., i * 5.)
    assert a.protect_step(float('nan'), float('nan'), 75) == b.protect_step(80, 90, 75)


def test_prior_control_independent_of_anchor_and_private_position():
    rn, _, belief = fixture()
    a = ServiceCoverLaneDummy(rn, belief_model=belief, prior_only=True, rng=np.random.default_rng(2))
    b = ServiceCoverLaneDummy(rn, belief_model=belief, prior_only=True, rng=np.random.default_rng(99))
    for i in range(12):
        assert a.protect_step(0., 0., i * 20.) == b.protect_step(40., 116., i * 20.)


def test_shadow_features_are_causal_and_fit_uses_training_statistics():
    rn, _, belief = fixture()
    points = tuple(TrajectoryPoint(i * 5., 0., .0001) for i in range(5))
    public = ServiceCoverLaneDummy(rn, belief_model=belief).protect_run(points).to_attacker_dict()
    x = features(public, rn)
    prefix = {**public, 'events': public['events'][:3]}
    assert np.array_equal(features(prefix, rn), x[:3])
    y = np.arange(10).reshape(5, 2)
    model = fit(x, y, {'split': 'development_train'})
    assert np.allclose(model['mean'], x.mean(axis=0))
    assert predict(model, x, 5).shape == (5, 2)
    assert np.allclose(predict(model, x, 15), y.mean(axis=0))
