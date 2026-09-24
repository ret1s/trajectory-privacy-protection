from itertools import product
import numpy as np
import pytest
from benchmark.engines.quotient_cover import reduce_groups,QuotientCoverLaneDummy
from benchmark.engines.fair_cover import CoverageObjective,FairCoverLaneDummy,exchange_refine
from benchmark.engines.service_cover import greedy_cover
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture

@pytest.mark.parametrize('cap',[None,.9])
def test_quotient_preserves_exact_selector_over_random_partition_instances(cap):
    rng=np.random.default_rng(911)
    for _ in range(100):
        profiles=rng.integers(0,5,size=25)
        signatures=np.array([[[i,(i+1)%5],[-1,-1]] for i in range(5)])[profiles]
        weights=np.r_[rng.random(5),0.]
        objective=CoverageObjective(signatures,np.arange(25),weights,cap)
        groups=[np.sort(rng.choice(25,12,replace=False)) for _ in range(3)]
        p=rng.integers(0,5,size=(3,25));s=rng.integers(0,3,size=(3,25))
        ties=lambda j,ids:(p[j,ids],s[j,ids])
        reduced=reduce_groups(groups,profiles,ties)
        def solve(gs):
            greedy,gains=greedy_cover(gs,objective.marginal,ties)
            selected,h=exchange_refine(gs,greedy,objective,ties,3)
            return greedy,gains,selected,h
        assert solve(groups)==solve(reduced)
        assert max(objective.value(x) for x in product(*groups))==pytest.approx(
            max(objective.value(x) for x in product(*reduced)))

@pytest.mark.parametrize('cap',[None,.9])
def test_identical_transcripts_anchors_ledger_and_feasibility(cap):
    rn,_,belief=fixture()
    points=tuple(TrajectoryPoint(i*5.,0.,.0001+i*.0001) for i in range(16))
    opts=dict(belief_model=belief,k=5,budget=.24,horizon=12,category_cap=cap)
    old=FairCoverLaneDummy(rn,**opts,rng=np.random.default_rng(812))
    new=QuotientCoverLaneDummy(rn,**opts,rng=np.random.default_rng(812))
    assert old.protect_run(points).to_attacker_dict()['events']==new.protect_run(points).to_attacker_dict()['events']
    assert old.evaluator_anchors==new.evaluator_anchors
    assert old.spent_bound==new.spent_bound
    assert all(all(b<=a for a,b in zip(r['reachable_counts'],r['quotient_counts'])) for r in new.evaluator_objective)
