from fractions import Fraction
import numpy as np

from benchmark.defined_service_scores import DefinedPublicPurposeTable
from benchmark.public_pool_portfolio import PublicPoolPortfolio,FiniteSegmentPolicy
from benchmark.query_segments import PublicSegmentLibrary
from core.query_budget import PersistentQueryBudget
from tests.test_belief_lane import fixture


def objects():
    rn,c,_=fixture()
    table=DefinedPublicPurposeTable(c,c,[0,12,25],destinations=[12,25])
    pool=PublicPoolPortfolio(c,public_poi_weights=table.public_poi_weights)
    goals=pool.goals(width=4)
    library=PublicSegmentLibrary(rn,goals)
    return rn,c,table,pool,library


def test_defined_public_reference_mass_is_not_diluted_by_NA():
    _,_,table,_,_=objects()
    assert np.allclose(table.weights.sum(axis=2)[table.defined],1.)
    assert np.all(table.weights.sum(axis=2)[~table.defined]==0)
    assert table.public_poi_weights.sum()==np.float64(1.) or abs(table.public_poi_weights.sum()-1)<1e-15


def test_exact_reference_ceiling_missing_bridge_all_masks():
    _,c,_,pool,_=objects()
    frame=(0,5,10,15,20)
    for ids in ([],[0],list(range(len(c.pois)))):
        cert=pool.reference_certificate(ids,frame)
        if ids:
            assert cert['recall']>=cert['lower_recall']
            assert cert['recall']==cert['achievable_ceiling']-cert['missing_attainable']/len(ids)
        else:assert cert['status']=='NA' and cert['recall'] is None


def test_public_floor_invariant_survives_movement_and_restart(tmp_path):
    rn,c,table,pool,lib=objects()
    policy=FiniteSegmentPolicy(lib,table,pool,use_simulator=True)
    path=tmp_path/'q.sqlite'
    args=dict(session_token='s',context_id='fixed',gamma=1.,private_key=b'z'*32)
    budget=PersistentQueryBudget(path,**args)
    for t in range(0,600,20):
        def selector(prev,eps,rng):return policy.select(prev,eps,rng,belief=[.2,.3,.5])
        frame,cert=budget.frame(float(t),selector)
        assert cert['selected_public_reference_mass']>=policy.quality_floor
        assert cert['selected_pool_attainable_fraction']>=0
        assert Fraction(cert['exact_likelihood_ratio_upper'])<=1+Fraction.from_float(cert['epsilon_Q_upper'])
        assert cert['guaranteed_quantized_score_gain']>=0
        if t==100:
            budget.close();budget=PersistentQueryBudget(path,**args)
    assert budget.reserved==Fraction(10,11)
    budget.close()


def test_production_default_uses_OS_source_not_persisted_simulator_stream(monkeypatch):
    _,_,table,pool,lib=objects()
    policy=FiniteSegmentPolicy(lib,table,pool)
    calls=[]
    monkeypatch.setattr('benchmark.finite_query_kernel.secrets.randbelow',lambda n:calls.append(n) or 0)
    choice=policy.select(None,Fraction(1,2),None,belief=[.2,.3,.5])
    assert len(calls)==1 and calls[0]==int(choice['certificate']['public_draw_total'])
    assert choice['certificate']['random_source'].startswith('fresh secrets')
