import math
import numpy as np
import pytest
from scipy.stats import laplace
from benchmark.engines.filtered_cover import FilteredCoverLaneDummy
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture

def test_integer_filter_extends_private_reads_but_never_overspends_or_reads_after_stop():
    rn,_,belief=fixture();m=FilteredCoverLaneDummy(rn,belief_model=belief,horizon=12,budget=.24,k=3,rng=np.random.default_rng(44))
    for i in range(60):m.protect_step(0.,.001,5.*i)
    ledger=m.evaluator_ledger
    assert sum(x['cost_units'] for x in ledger)==m.spent_units<=24
    assert sum(x['private_read'] for x in ledger)>=12
    assert any(x['branch']=='reuse' for x in ledger)
    assert m.spent_bound<=.24
    assert not m.privacy_read_this_step
    m.protect_step(float('nan'),float('nan'),300.)
    assert not m.privacy_read_this_step

def test_causal_prefix_and_no_private_ledger_publication():
    rn,_,belief=fixture()
    def make():return FilteredCoverLaneDummy(rn,belief_model=belief,k=3,rng=np.random.default_rng(33))
    points=tuple(TrajectoryPoint(i*5.,0.,.001+i*.00001) for i in range(35))
    full=make().protect_run(points).to_attacker_dict()
    assert make().protect_run(points[:18]).to_attacker_dict()['events']==full['events'][:18]
    assert 'spent_units' not in str(full) and 'evaluator_ledger' not in str(full)
    assert 'horizon_events' not in full['public_parameters']

def test_exhaustive_extended_paths_respect_fixed_cap_likelihood_ratio():
    # Enumerate a two-location ideal mechanism, including hidden branches,
    # filter stopping and postprocessing. This checks the argument's path logic;
    # it does not replace a proof for all continuous locations.
    u=.1;cap=6;theta=.5;N=8
    def distribution(secret):
        paths={((),None,0):1.}
        for x in secret:
            new={}
            for (history,last,spent),mass in paths.items():
                if spent+(1 if last is None else 2)>cap:
                    new[history+(('stop',last),),last,spent]=mass;continue
                rem=np.exp(-.5*u*np.abs(np.array([0.,1.])-x));rem/=rem.sum()
                q=0. if last is None else float(laplace.cdf(theta-abs(x-last),scale=1/u))
                if last is not None:new[history+(('reuse',last),),last,spent+1]=mass*q
                cost=1 if last is None else 2
                for z in (0,1):new[history+(('fresh',z),),z,spent+cost]=mass*(1-q)*rem[z]
            paths=new
        return {key[0]:value for key,value in paths.items()}
    for a,b in (([0.]*N,[1.]*N),([0.,1.]*4,[1.,0.]*4)):
        p,q=distribution(a),distribution(b)
        assert p.keys()==q.keys()
        assert sum(p.values())==pytest.approx(1.)
        assert max(abs(math.log(p[k]/q[k])) for k in p)<=cap*u+1e-12

def test_first_twelve_events_match_fixed_ledger_control():
    from benchmark.engines.quotient_cover import QuotientCoverLaneDummy
    rn,_,belief=fixture()
    points=tuple(TrajectoryPoint(i*5.,0.,.001+i*.00001) for i in range(12))
    kwargs=dict(belief_model=belief,k=5,horizon=12,budget=.24)
    a=FilteredCoverLaneDummy(rn,**kwargs,rng=np.random.default_rng(99))
    b=QuotientCoverLaneDummy(rn,**kwargs,rng=np.random.default_rng(99))
    assert a.protect_run(points).to_attacker_dict()['events']==b.protect_run(points).to_attacker_dict()['events']
    assert a.evaluator_anchors==b.evaluator_anchors
    assert a.spent_bound<=b.spent_bound
