import numpy as np
import pytest
from benchmark.engines.progress_cover import ProgressCoverLaneDummy,FilteredProgressCoverLaneDummy
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture

@pytest.mark.parametrize('cls',[ProgressCoverLaneDummy,FilteredProgressCoverLaneDummy])
def test_progress_keeps_service_signature_and_directed_reachability(cls):
    rn,context,belief=fixture();kwargs=dict(belief_model=belief,k=3,budget=.24,horizon=12)
    points=tuple(TrajectoryPoint(i*5.,0.,.0001+i*.0001) for i in range(20))
    m=cls(rn,**kwargs,rng=np.random.default_rng(16));full=m.protect_run(points).to_attacker_dict()
    for record in m.evaluator_objective[1:]:
        a,b=record['pre_progress_states'],record['progress_states']
        assert np.array_equal(context.signatures[context.access[a]],context.signatures[context.access[b]])
    for a,b in zip(m.evaluator_states,m.evaluator_states[1:]):
        assert all(y in m.travel.reachable(x,5.) for x,y in zip(a,b))
    assert m.spent_bound<=.24
    prefix=cls(rn,**kwargs,rng=np.random.default_rng(16)).protect_run(points[:8]).to_attacker_dict()
    assert full['events'][:8]==prefix['events']
    assert 'progress_goals' not in str(full)
