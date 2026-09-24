import numpy as np
from benchmark.engines.paced_guard import PacedProgressLaneDummy
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


def test_paced_zero_slack_reproduces_parent_and_positive_slack_has_declared_bound():
    rn, context, belief = fixture()
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%14)*.0001) for i in range(40))
    def make(cls, **extra):
        return cls(rn, belief_model=belief, k=3, horizon=12, budget=.24, rng=np.random.default_rng(47), **extra)
    plain = make(PacedProgressLaneDummy).protect_run(points).to_attacker_dict()
    zero = make(PacedSlackProgressLaneDummy, utility_slack=0.).protect_run(points).to_attacker_dict()
    assert plain['events'] == zero['events']
    model = make(PacedSlackProgressLaneDummy, utility_slack=.03)
    full = model.protect_run(points).to_attacker_dict()
    prefix = make(PacedSlackProgressLaneDummy, utility_slack=.03).protect_run(points[:8]).to_attacker_dict()
    assert prefix['events'] == full['events'][:8]
    assert model.spent_bound <= .23+1e-12
    for record in model.evaluator_objective[1:]:
        assert record['objective_after_slack'] >= record['objective_before_slack']-.03-1e-12
    for a, b in zip(model.evaluator_states, model.evaluator_states[1:]):
        assert all(v in model.travel.reachable(u, 20.) for u, v in zip(a, b))
