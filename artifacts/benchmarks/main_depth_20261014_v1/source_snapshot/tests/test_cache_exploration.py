import copy,itertools,json
from fractions import Fraction
import numpy as np
import pytest
from benchmark.session_poi_cache import SessionCoverClient
from benchmark.cache_exploration import CacheExplorePolicy,ExploreMotionLibrary,mask_bits
from benchmark.finite_query_kernel import FiniteCenteredKernel
from core.query_budget import PersistentQueryBudget
from evaluation.lane_travel import SparseTravel
from tests.test_public_pool_portfolio import objects


def test_session_cache_crosses_epoch_and_restarts_without_extra_requests():
    client=SessionCoverClient(['c'],4,k=5,response_l=30,session_token='s',static_catalogue_id='map')
    q=((0.,0.),)*5;calls=[]
    def server(r):calls.append(r);return [[int(r['timestamp_s']//60)%4]]
    client.step(0.,q,server);client.step(60.,q,server)
    saved=json.loads(json.dumps(client.checkpoint()))
    resumed=SessionCoverClient.restore(saved,session_token='s',static_catalogue_id='map')
    retry=resumed.step(60.,q,server)
    assert len(calls)==10 and retry['session'].tolist()==[True,True,False,False]
    changed=copy.deepcopy(retry);changed['session'][:]=False
    assert resumed.session_known.any()
    wire=resumed.step(120.,q,server)
    assert wire['current'].tolist()==[False,False,True,False]
    assert wire['session'].tolist()==[True,True,True,False]
    with pytest.raises(ValueError):resumed.step(120.,((1.,1.),)*5,server)
    with pytest.raises(ValueError):SessionCoverClient.restore(saved,session_token='different',static_catalogue_id='map')


def test_cache_support_and_masses_use_only_public_history():
    rn,c,t,p,old=objects();lib=ExploreMotionLibrary(rn,old.goals,p)
    policy=CacheExplorePolicy(lib,t,p,use_simulator=True)
    initial=old.goals[0];known=p.pool(initial)
    actions,g,d,m=policy.public_tables(initial,known)
    travel=SparseTravel(rn)
    for a in actions:
        previous=initial
        for frame in a.frames:
            assert all(y in travel.reachable(x,20.) for x,y in zip(previous,frame))
            previous=frame
    # All private score vectors share the same positive PUBLIC masses/support.
    k=FiniteCenteredKernel(m,Fraction(1,6),resolution=1)
    lo=k.probabilities([0]*len(m));hi=k.probabilities([i%2 for i in range(len(m))])
    assert all(x/y<=1+k.epsilon and y/x<=1+k.epsilon for x,y in zip(lo,hi))
    average=sum(Fraction(x) for x in d)/len(d)
    weighted=sum(Fraction(x*w,sum(m)) for x,w in zip(d,m))
    variance=sum((Fraction(x)-average)**2 for x in d)/len(d)
    assert weighted==average+4*variance/(1+4*average)


def test_private_scores_and_requests_do_not_enter_public_library():
    rn,c,t,p,old=objects();policy=CacheExplorePolicy(ExploreMotionLibrary(rn,old.goals,p),t,p,use_simulator=True)
    known=p.pool(old.goals[0]);support=policy.public_tables(old.goals[0],known)
    a=policy.select(old.goals[0],Fraction(1,6),np.random.default_rng(7),belief=[.2,.3,.5],known=known)
    b=policy.select(old.goals[0],Fraction(1,6),np.random.default_rng(7),belief=[.2,.3,.5],known=known)
    assert a==b and policy.public_tables(old.goals[0],known) is support
    assert a['certificate']['actual_expected_utility_lower'] is None


def test_multi_step_cache_invariant_and_budget_restart(tmp_path):
    rn,c,t,p,old=objects();policy=CacheExplorePolicy(ExploreMotionLibrary(rn,old.goals,p),t,p,use_simulator=True)
    args=dict(session_token='s',context_id='frozen',gamma=1.,private_key=b'z'*32)
    ledger=PersistentQueryBudget(tmp_path/'q.sqlite',**args)
    client=SessionCoverClient(c.categories,len(c.pois),k=5,response_l=30,session_token='s',static_catalogue_id='map')
    def server(r):return [[int(v) for v in row if v>=0] for row in c.query_indices(rn.nearest(*r['coordinate'])[0])]
    initial=None;previous=np.zeros(len(c.pois),bool)
    for stamp in range(0,601,20):
        def selector(q,e,rng):return policy.select(q,e,rng,belief=[.2,.3,.5],known=mask_bits(client.session_known))
        frame,cert=ledger.frame(float(stamp),selector)
        wire=client.step(float(stamp),[rn.latlon(s) for s in frame],server)
        if initial is None:initial=wire['current'].copy()
        assert np.all(~previous|wire['session']) and np.all(~initial|wire['session'])
        for reference in itertools.combinations(range(len(c.pois)),min(2,len(c.pois))):
            assert sum(wire['session'][i] for i in reference)>=sum(initial[i] for i in reference)
        previous=wire['session'].copy()
        if stamp==100:
            checkpoint=client.checkpoint();ledger.close();ledger=PersistentQueryBudget(tmp_path/'q.sqlite',**args)
            client=SessionCoverClient.restore(checkpoint,session_token='s',static_catalogue_id='map')
    assert ledger.reserved==Fraction(11,12);ledger.close()
