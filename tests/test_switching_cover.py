"""Dense forward oracle, reset/prefix and private-input boundary regressions."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from benchmark.engines.contextual_lane import ContextualLaneDummy
from benchmark.engines.switching_cover import SwitchingCoverLaneDummy
from benchmark.switching_belief import SwitchingAnchorBelief
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


def test_dense_joint_forward_and_long_gap_mass():
    diffusion = np.array([[.8, .2], [.3, .7]])
    model = SimpleNamespace(prior=np.array([.4, .6]),
        emission=lambda a, p: np.array([.2, .7]),
        transition=lambda dt: csr_matrix(.6*np.eye(2)+.4*diffusion),
        predict=lambda w, dt: .6**(dt/20)*w+(1-.6**(dt/20))*np.array([.4, .6]))
    belief = SwitchingAnchorBelief(model)
    q = np.array([[.2, .3], [.2, .3]])
    previous = None
    for time in (0., 5., 25., 326.):
        if previous is not None:
            dt = time-previous; stay = (1+np.exp(-dt/100))/2
            modes = np.array([[stay, 1-stay], [1-stay, stay]])
            kernels = [a*np.eye(2)+(1-a)*diffusion for a in (.95, .1)]
            if dt > 120:
                kernels = [(.6**(dt/20)*np.eye(2)+(1-.6**(dt/20))*np.tile(model.prior, (2, 1)))]*2
            # Explicit summation, no production helper or normalized-row shortcut.
            pred = np.zeros_like(q)
            for old_mode in range(2):
                for mode in range(2):
                    for old_state in range(2):
                        for state in range(2):
                            pred[mode, state] += q[old_mode, old_state]*modes[old_mode, mode]*kernels[mode][old_state, state]
            q = pred
        q *= [.2, .7]; q /= q.sum()
        assert np.allclose(belief.update((0, 0), time), q.sum(axis=0), atol=1e-14)
        assert np.allclose(belief.joint, q, atol=1e-14)
        assert belief.mode_probabilities.sum() == pytest.approx(1.)
        previous = time
    belief.update(None, 400., observed=False)
    assert belief.joint.sum() == pytest.approx(1.)
    assert belief.previous == (0, 0)
    with pytest.raises(ValueError): belief.update(None, 400., observed=False)
    with pytest.raises(ValueError): belief.update(None, float('nan'), observed=False)


def test_protected_history_only_reset_and_directed_prefix():
    rn, _, belief = fixture()
    def make():
        return SwitchingCoverLaneDummy(rn, belief_model=belief, rng=np.random.default_rng(29))
    points = tuple(TrajectoryPoint(i*5., 0., .0001+i*.0001) for i in range(15))
    model = make(); public = model.protect_run(points).to_attacker_dict()
    assert make().protect_run(points[:5]).to_attacker_dict()['events'] == public['events'][:5]
    control = ContextualLaneDummy(rn, rng=np.random.default_rng(29)); control.protect_run(points)
    assert model.evaluator_anchors == control.evaluator_anchors
    assert model.spent_bound == pytest.approx(control.spent_bound)
    assert not any(k in json.dumps(public) for k in ('joint', 'mode_probabilities', 'evaluator'))
    for a, b in zip(model.evaluator_states, model.evaluator_states[1:]):
        for x, y in zip(a, b): assert y in model.travel.reachable(x, 5.)
    a, b = make(), make()
    # Postprocess receives only the same released anchors, not protect_step GPS.
    for i in range(15):
        assert a.postprocess(rn.latlon(15), i*5.) == b.postprocess(rn.latlon(15), i*5.)
    model.reset()
    assert model.belief.timestamp is None
    assert np.array_equal(model.belief.weights, belief.prior)
