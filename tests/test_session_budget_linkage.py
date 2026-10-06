import numpy as np
import pytest

from experiments.session_budget_linkage import nested_family_mean, factory
from benchmark.anchor_belief import PublicAnchorModel
from core.session_budget import FixedEpochPolicy, PersistentEpochBudget, FixedEpochProtectedSessions
from tests.test_belief_lane import fixture


def test_family_repetition_session_weighting_prevents_large_group_domination():
    rows=[dict(family_id='a',rep=0,value=1.) for _ in range(50)]
    rows += [dict(family_id='a',rep=1,value=0.),dict(family_id='b',rep=0,value=0.)]
    assert nested_family_mean(rows,lambda r:r['value'])==pytest.approx(.25)
    assert nested_family_mean(rows,lambda r:None) is None


def test_experiment_factory_preserves_existing_emissions_and_private_streams(tmp_path):
    rn, context, _=fixture()
    policy=FixedEpochPolicy('day',0.,86400.,horizon=8)
    model=PublicAnchorModel(rn,context,np.ones(len(rn)),spacing_m=2.,
        epsilon_release=policy.unit_epsilon_per_m,epsilon_test=policy.unit_epsilon_per_m)
    ledger=PersistentEpochBudget(policy,tmp_path/'epoch.sqlite',private_key=b'z'*32)
    stream=FixedEpochProtectedSessions(ledger,factory(rn,model))
    assert stream.start_session('trip',0.)
    engine=stream._engine
    assert engine.anchor.epsilon==pytest.approx(policy.unit_epsilon_per_m)
    assert engine.anchor.eps_test==pytest.approx(policy.unit_epsilon_per_m)
    assert engine.theta_m==200.
    assert engine.max_units==15
    stream.protect_step(0.,lambda:(0.,.001))
    stream.close_session(1.)
    assert ledger.reserved_cap_per_m==pytest.approx(.23/6)
    ledger.close()
