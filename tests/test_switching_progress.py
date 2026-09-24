import numpy as np
import pytest
from benchmark.engines.switching_cover import SwitchingCoverLaneDummy
from benchmark.engines.switching_progress import (
    SwitchingQuotientCoverLaneDummy, MatchedSwitchingProgressCoverLaneDummy)
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


def test_quotient_reproduces_existing_switching_transcript():
    rn, _, belief = fixture()
    points = tuple(TrajectoryPoint(i*5., 0., .001 if i < 8 else .0001*i)
                   for i in range(30))
    opts = dict(belief_model=belief, k=3, budget=.24, horizon=12)
    old = SwitchingCoverLaneDummy(rn, **opts, rng=np.random.default_rng(81))
    new = SwitchingQuotientCoverLaneDummy(rn, **opts, rng=np.random.default_rng(81))
    assert old.protect_run(points).to_attacker_dict()['events'] == new.protect_run(points).to_attacker_dict()['events']
    assert old.evaluator_anchors == new.evaluator_anchors


def test_switching_progress_is_causal_and_keeps_budget_and_step_service():
    rn, context, belief = fixture()
    opts = dict(belief_model=belief, k=3, budget=.24, horizon=12)
    points = tuple(TrajectoryPoint(i*5., 0., .001) for i in range(60))
    model = MatchedSwitchingProgressCoverLaneDummy(rn, **opts, rng=np.random.default_rng(81))
    full = model.protect_run(points).to_attacker_dict()
    prefix = MatchedSwitchingProgressCoverLaneDummy(rn, **opts, rng=np.random.default_rng(81)).protect_run(points[:15]).to_attacker_dict()
    assert full['events'][:15] == prefix['events']
    assert model.spent_bound <= .23 + 1e-12
    assert sum(x['cost_units'] for x in model.evaluator_ledger) == model.spent_units
    for record in model.evaluator_objective[1:]:
        a, b = record['pre_progress_states'], record['progress_states']
        assert np.array_equal(context.signatures[context.access[a]], context.signatures[context.access[b]])
    assert sum(model.belief.mode_probabilities) == pytest.approx(1.)
    assert 'evaluator_ledger' not in str(full)
