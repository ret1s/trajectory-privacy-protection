import numpy as np
import pytest
from benchmark.engines.slack_progress import FilteredSlackProgressCoverLaneDummy
from benchmark.engines.progress_cover import FilteredProgressCoverLaneDummy
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture

@pytest.mark.parametrize('slack',[0.,.01,.03])
def test_causal_bounded_belief_objective_loss_and_reachable_outputs(slack):
    rn,_,belief=fixture();kwargs=dict(belief_model=belief,k=3,budget=.24,horizon=12)
    points=tuple(TrajectoryPoint(i*5.,0.,.0001+i*.0001) for i in range(24))
    make=lambda:FilteredSlackProgressCoverLaneDummy(rn,**kwargs,utility_slack=slack,rng=np.random.default_rng(16))
    m=make();result=m.protect_run(points).to_attacker_dict()
    assert make().protect_run(points[:10]).to_attacker_dict()['events']==result['events'][:10]
    for step in m.evaluator_objective:
        assert step.get('objective_loss',0)<=slack+1e-12
    for a,b in zip(m.evaluator_states,m.evaluator_states[1:]):
        assert all(y in m.travel.reachable(x,5.) for x,y in zip(a,b))
    assert m.spent_bound<=.24
    if slack==0:
        base=FilteredProgressCoverLaneDummy(rn,**kwargs,rng=np.random.default_rng(16))
        assert result['events']==base.protect_run(points).to_attacker_dict()['events']
