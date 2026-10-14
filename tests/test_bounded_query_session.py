from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from benchmark.targeted_cache_exploration import TargetedExploreLibrary, TargetedCachePolicy
from core.bounded_query_session import BoundedQuerySession, GeoIBoundedQPipeline
from tests.test_public_pool_portfolio import objects


def setup(directory, **extra):
    rn, reply, table, portfolio, library = objects()
    policy = TargetedCachePolicy(TargetedExploreLibrary(rn, library.goals, portfolio), table, portfolio)
    session = BoundedQuerySession(directory, policy=policy, session_token='s',
        catalogue_id='public-static', categories=reply.categories, poi_count=len(reply.pois),
        configuration_id='bounded-main-v1', **extra)
    calls=[]
    def server(request):
        calls.append(request)
        return [list(map(int,row[row>=0])) for row in reply.query_indices(rn.nearest(*request['coordinate'])[0])]
    return rn, reply, session, server, calls


def test_restart_retry_and_local_requests_do_not_spend_or_send_again(tmp_path, monkeypatch):
    monkeypatch.setattr('benchmark.finite_query_kernel.secrets.randbelow', lambda n: n//2)
    rn, reply, session, server, calls = setup(tmp_path)
    for t in range(0, 121, 20): session.step(float(t), protected_belief=[.2,.3,.5], server=server)
    spent=session.ledger.reserved
    ranking=MultiPurposeRoadRanking(reply)
    for state in (0,12,25):
        for purpose in (QueryPurpose.NEAREST, QueryPurpose.FASTEST):
            session.local_answer(ranking,state,QuerySpec(purpose,reply.categories[0],k=5))
    assert len(calls)==35
    session.close()
    _, _, resumed, replay_server, replay_calls=setup(tmp_path)
    wire=resumed.step(120.,protected_belief=[.5,.4,.1],server=replay_server)
    assert not replay_calls and resumed.ledger.reserved==spent==Fraction(3,4)
    assert wire['session'].any()
    with pytest.raises(ValueError): GeoIBoundedQPipeline(SimpleNamespace(last_t=None),resumed,[0,1,2])
    resumed.close()


def test_crash_after_q_commit_recovers_same_frame_before_next_tick(tmp_path, monkeypatch):
    monkeypatch.setattr('benchmark.finite_query_kernel.secrets.randbelow', lambda n: 0)
    _, _, session, server, _ = setup(tmp_path)
    def unavailable(request): raise OSError('public service failure')
    with pytest.raises(OSError): session.step(0., protected_belief=[.2,.3,.5],server=unavailable)
    spent=session.ledger.reserved;session.close()
    _, _, resumed, server, calls=setup(tmp_path)
    with pytest.raises(ValueError): GeoIBoundedQPipeline(SimpleNamespace(last_t=None),resumed,[0,1,2])
    with pytest.raises(ValueError): resumed.step(20.,protected_belief=[.2,.3,.5],server=server)
    resumed.step(0.,protected_belief=[.5,.4,.1],server=server)
    assert len(calls)==5 and resumed.ledger.reserved==spent==Fraction(1,2)
    resumed.step(20.,protected_belief=[.5,.4,.1],server=server)
    assert resumed.ledger.reserved==spent
    resumed.close()


def test_invalid_or_pending_clock_rejected_before_upstream_GPS_access(tmp_path):
    _, _, session, server, _=setup(tmp_path)
    class Engine:
        last_t=None
        belief=SimpleNamespace(weights=np.array([.2,.3,.5]))
        reads=0
        def protect_step(self,lat,lon,t):self.last_t=t;self.reads+=1
    engine=Engine();pipeline=GeoIBoundedQPipeline(engine,session,[0,1,2])
    with pytest.raises(ValueError):pipeline.step(0.,0.,35.,server=server)
    assert engine.reads==0
    def unavailable(request):raise OSError('public failure')
    with pytest.raises(OSError):pipeline.step(0.,0.,0.,server=unavailable)
    with pytest.raises(ValueError):pipeline.step(0.,0.,20.,server=server)
    assert engine.reads==1 and engine.last_t==0.
    pipeline.step(0.,0.,0.,server=server)
    assert engine.reads==1
    session.close()


def test_same_label_cannot_hide_a_changed_public_planner(tmp_path):
    _, _, session, _, _=setup(tmp_path);session.close()
    rn,reply,table,portfolio,library=objects()
    changed=TargetedCachePolicy(TargetedExploreLibrary(rn,library.goals,portfolio),table,portfolio,novelty_weight=5)
    with pytest.raises(ValueError):
        BoundedQuerySession(tmp_path,policy=changed,session_token='s',catalogue_id='public-static',
            categories=reply.categories,poi_count=len(reply.pois),configuration_id='bounded-main-v1')


def test_pipeline_matches_component_loop_with_real_unchanged_GeoI(tmp_path, monkeypatch):
    from scipy.spatial import cKDTree
    from benchmark.anchor_belief import PublicAnchorModel
    from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
    monkeypatch.setattr('benchmark.finite_query_kernel.secrets.randbelow',lambda n:n//2)
    rn,reply,session,server,_=setup(tmp_path/'pipeline')
    model=PublicAnchorModel(rn,reply,np.linspace(1,2,len(rn)),spacing_m=2.,epsilon_release=.01,epsilon_test=.01)
    projection=cKDTree(rn.xy[list(session.policy.table.states)]).query(model.xy)[1]
    def make():
        return PacedSlackProgressLaneDummy(rn,belief_model=model,k=5,horizon=12,budget=.24,
            read_interval_s=60.,utility_slack=.03,rng=np.random.default_rng(47))
    engine,control=make(),make()
    pipeline=GeoIBoundedQPipeline(engine,session,projection)
    _,_,direct,direct_server,_=setup(tmp_path/'component-control')
    for t in range(0,301,20):
        gps=rn.latlon((t//20)%len(rn))
        wire=pipeline.step(*gps,float(t),server=server)
        control.protect_step(*gps,float(t))
        belief=np.bincount(projection,weights=control.belief.weights,minlength=3);belief/=belief.sum()
        expected=direct.step(float(t),protected_belief=belief,server=direct_server)
        assert wire['requests']==expected['requests']
        assert np.array_equal(wire['session'],expected['session'])
    assert engine.evaluator_anchors==control.evaluator_anchors
    assert engine.evaluator_ledger==control.evaluator_ledger
    assert engine.spent_bound<=.23+1e-12
    # Retry uses the committed result and does not consume another GPS read.
    count=len(engine.evaluator_ledger)
    pipeline.step(float('nan'),float('nan'),300.,server=server)
    assert len(engine.evaluator_ledger)==count
    session.close();direct.close()


def test_same_protected_input_and_bits_keep_wire_independent_of_local_request(tmp_path, monkeypatch):
    monkeypatch.setattr('benchmark.finite_query_kernel.secrets.randbelow', lambda n: n//3)
    wires=[]
    for name,local_state in [('a',0),('b',25)]:
        _,reply,session,server,calls=setup(tmp_path/name)
        for t in range(0,181,20):
            session.step(float(t),protected_belief=[.2,.3,.5],server=server)
            session.local_answer(MultiPurposeRoadRanking(reply),local_state,QuerySpec(QueryPurpose.NEAREST,reply.categories[0],k=5))
        wires.append(calls);session.close()
    assert wires[0]==wires[1]
