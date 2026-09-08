import json

import numpy as np
import pytest

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.recovery_cover import RecoveryCoverLaneDummy, corridor
from benchmark.engines.service_cover import ServiceCoverLaneDummy
from core.demo_protocol import TrajectoryPoint
from experiments.diagnose_service_cover import uniform_cell_prior
from tests.test_belief_lane import fixture


def test_corridor_nonempty_threshold_and_validation():
    assert corridor([1, 2, 3], [500, 700, 701], 200).tolist() == [1, 2]
    assert corridor([1, 2, 3], [500, 500, 700], 0).tolist() == [1, 2]
    for ids, distances, slack in [([], [], 1), ([1], [float('inf')], 1),
                                  ([1], [-1], 1), ([1], [1, 2], 1), ([1], [1], -1)]:
        with pytest.raises(ValueError): corridor(ids, distances, slack)


def test_uniform_cell_not_lane_state_multiplicity():
    rn, c, learned = fixture()
    prior = uniform_cell_prior(rn, learned.spacing_m)
    m = PublicAnchorModel(rn, c, prior, spacing_m=learned.spacing_m)
    assert np.allclose(m.prior, 1. / len(m.prior))
    assert np.array_equal(m.state_ids, learned.state_ids)
    assert np.array_equal(m.log_normalizers, learned.log_normalizers)


@pytest.mark.parametrize('slack', [None, 0., 200.])
def test_causal_paired_private_boundary_and_corridor(slack):
    rn, c, belief = fixture()
    points = tuple(TrajectoryPoint(i * 5., 0., .0001 + i * .0001) for i in range(15))
    def make():
        return RecoveryCoverLaneDummy(rn, belief_model=belief, prior_kind='training_occupancy',
                                      corridor_m=slack, rng=np.random.default_rng(19))
    model = make(); public = model.protect_run(points).to_attacker_dict()
    assert make().protect_run(points[:5]).to_attacker_dict()['events'] == public['events'][:5]
    old = ServiceCoverLaneDummy(rn, belief_model=belief, rng=np.random.default_rng(19))
    old_public = old.protect_run(points).to_attacker_dict()
    assert model.evaluator_anchors == old.evaluator_anchors
    assert model.spent_bound == pytest.approx(.23)
    if slack is None: assert public['events'] == old_public['events']
    for a, b in zip(model.evaluator_states, model.evaluator_states[1:]):
        assert all(y in model.travel.reachable(x, 5.) for x, y in zip(a, b))
    if slack is not None:
        for row in model.evaluator_objective:
            assert all(0 < a <= b for a, b in zip(row['corridor_counts'], row['reachable_counts']))
            assert np.all(np.asarray(row['selected_goal_distance_m']) <= np.asarray(row['minimum_goal_distance_m']) + slack)
    assert not any(s in json.dumps(public) for s in ('evaluator', 'greedy_gains', 'minimum_goal'))
    for key in ('offset_m', 'temperature_m', 'route_weight', 'coverage_weight', 'center_mode'):
        assert key not in public['public_parameters']
    a, b = make(), make()
    for m in (a, b): m.anchor.perturb = lambda *args, **kwargs: rn.latlon(15)
    for i in range(15):
        assert a.protect_step(0., 0., i * 5.) == b.protect_step(20., 30., i * 5.)
    assert a.protect_step(float('nan'), float('nan'), 75) == b.protect_step(80, 90, 75)


def test_false_prior_metadata_rejected():
    rn, _, model = fixture()
    with pytest.raises(ValueError, match='Uniform'):
        RecoveryCoverLaneDummy(rn, belief_model=model, prior_kind='uniform_public_cells')
