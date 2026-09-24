import numpy as np
import pytest
from benchmark.engines.matched_filter import MatchedFilteredCoverLaneDummy,MatchedFilteredProgressCoverLaneDummy,MatchedFilteredSlackProgressCoverLaneDummy
from tests.test_belief_lane import fixture

@pytest.mark.parametrize('cls',[MatchedFilteredCoverLaneDummy,MatchedFilteredProgressCoverLaneDummy,MatchedFilteredSlackProgressCoverLaneDummy])
def test_tight_bound_matches_fixed_H_without_changing_primitive_epsilon(cls):
    rn,_,belief=fixture();model=cls(rn,belief_model=belief,k=3,horizon=12,budget=.24,rng=np.random.default_rng(44))
    for i in range(60):model.protect_step(0.,.001,5.*i)
    assert model.unit_epsilon==pytest.approx(.01)
    assert model.max_units==23
    assert model.spent_bound<=.23+1e-12
    assert sum(x['private_read'] for x in model.evaluator_ledger)>=12
    assert sum(x['cost_units'] for x in model.evaluator_ledger)==model.spent_units
