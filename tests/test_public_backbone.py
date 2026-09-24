import numpy as np
import pytest
from benchmark.engines.public_backbone import PublicBackboneLaneDummy
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.engines.fair_cover import CoverageObjective
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


@pytest.mark.parametrize('slack', [0., .03])
def test_zero_backbone_exact_parent_boundary(slack):
    rn, _, belief = fixture()
    opts = dict(belief_model=belief, k=5, budget=.24, horizon=12, utility_slack=slack)
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%14)*.0001) for i in range(40))
    old = PacedSlackProgressLaneDummy(rn, **opts, rng=np.random.default_rng(77))
    new = PublicBackboneLaneDummy(rn, **opts, public_queries=0, rng=np.random.default_rng(77))
    assert old.protect_run(points).to_attacker_dict()['events'] == new.protect_run(points).to_attacker_dict()['events']
    assert old.evaluator_anchors == new.evaluator_anchors and old.evaluator_ledger == new.evaluator_ledger


def test_backbone_prefix_ledger_feasibility_and_conditional_coverage():
    rn, context, belief = fixture()
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%14)*.0001) for i in range(40))
    opts = dict(belief_model=belief, k=5, budget=.24, horizon=12, utility_slack=.03)
    def make(cls=PublicBackboneLaneDummy):
        return cls(rn, **opts, rng=np.random.default_rng(77))
    model = make(); full = model.protect_run(points).to_attacker_dict()
    assert make().protect_run(points[:8]).to_attacker_dict()['events'] == full['events'][:8]
    control = make(PacedSlackProgressLaneDummy); control.protect_run(points)
    assert control.evaluator_ledger == model.evaluator_ledger
    assert control.evaluator_anchors == model.evaluator_anchors
    assert model.spent_bound <= .23+1e-12
    for record in model.evaluator_objective:
        assert record['objective_after_slack'] >= record['objective_before_slack']-.03-1e-12
    for states in model.evaluator_states:
        assert states[:2] == model.fixed_states and len(states) == 5
    for a, b in zip(model.evaluator_states, model.evaluator_states[1:]):
        assert all(v in model.travel.reachable(u, 20.) for u, v in zip(a, b))
    weights = np.asarray(belief.prior @ belief.poi_weights).ravel()
    objective = CoverageObjective(context.signatures, context.access, weights)
    residual = weights.copy(); residual[objective.covered(model.fixed_states)] = 0
    conditional = CoverageObjective(context.signatures, context.access, residual)
    for _ in range(20):
        states = np.random.default_rng(_).integers(len(rn), size=3).tolist()
        assert objective.value(model.fixed_states+states) == pytest.approx(objective.value(model.fixed_states)+conditional.value(states))
    public = str(full)
    assert 'evaluator_ledger' not in public and 'evaluator_anchors' not in public
