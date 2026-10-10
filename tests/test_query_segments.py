import itertools
from fractions import Fraction

import numpy as np
import pytest

from benchmark.probabilistic_query_bundle import ProbabilisticQueryBundle, QueryBundle
from benchmark.query_bundle_bounds import calibrated_beta
from benchmark.query_segments import (QuerySegment, PublicSegmentLibrary, SegmentPqbPolicy,
                                      oscillation_upper, actual_utility_bridge)
from benchmark.public_segment_scores import FixedPublicPurposeTable, local_sorted_pois
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from core.query_budget import PersistentQueryBudget, allowance
from benchmark.segment_cover_client import SegmentCoverClient
from tests.test_belief_lane import fixture
from evaluation.lane_travel import SparseTravel


def policy():
    rn, _, _ = fixture()
    lib = PublicSegmentLibrary(rn, [(0, 2, 4, 6, 8), (12, 14, 16, 18, 20)])
    def table(actions):
        return np.array([[np.mean(a.frames)/30. for a in actions],
                         [1.-np.mean(a.frames)/30. for a in actions]])
    return rn, SegmentPqbPolicy(lib, table, minimum_coverage=.95)


def test_all_public_segments_directed_and_common_support():
    rn, p = policy()
    travel = SparseTravel(rn)
    for previous in (None, (1, 3, 5, 7, 9), (29,)*5):
        actions = p.library.build(previous)
        if previous:
            assert QuerySegment((previous,)*3) in actions
        for action in actions:
            sequence = (previous,)+action.frames if previous else action.frames
            for a, b in zip(sequence, sequence[1:]):
                assert all(y in travel.reachable(x, 20.) for x, y in zip(a, b))
        assert p.library.build(previous) is actions


def test_public_outward_oscillation_matches_rational_oracle():
    g = np.array([[.1, .8, .4], [.7, .2, .3], [.2, .5, .9]])
    oracle = max(Fraction.from_float(g[x,a])-Fraction.from_float(g[x,b])
                 -Fraction.from_float(g[y,a])+Fraction.from_float(g[y,b])
                 for x,y,a,b in itertools.product(range(3), repeat=4))/Fraction(5,4)
    assert oscillation_upper(g) >= oracle
    assert float(oscillation_upper(g)-oracle) < 1e-14
    assert oscillation_upper([[.125,.5],[.375,.75]]) == 0
    assert oscillation_upper([[.125,.5],[.375,np.nextafter(.75,1.)]]) > 0


def test_zero_oscillation_optimizes_utility_without_privacy():
    aa = [QueryBundle((0,)), QueryBundle((1,)), QueryBundle((2,))]
    table = [[.1,.5,.9],[.1,.5,.9]]
    before = ProbabilisticQueryBundle(aa, table, beta=0., cost_weight=0.)
    beta = calibrated_beta(before, epsilon_target=0.)
    after = ProbabilisticQueryBundle(aa, table, beta=beta, cost_weight=0.)
    assert beta == 40. and oscillation_upper(table, 0.) == 0
    assert after.probabilities([1.,0.]) == pytest.approx(after.probabilities([0.,1.]))
    assert after.probabilities([1.,0.]) @ np.array(table)[0] > .89
    assert before.probabilities([1.,0.]) @ np.array(table)[0] == pytest.approx(.5)


@pytest.mark.parametrize('joint', [True, False])
def test_resume_retry_no_refill_public_clock_and_long_session(tmp_path, joint):
    _, p = policy()
    args = dict(session_token='trip', context_id='public-map-config', joint=joint,
                private_key=b'a'*32, gamma=1.)
    path = tmp_path/'q.sqlite'
    calls = []
    def select(prev, eps, rng):
        calls.append(eps)
        return p.select(prev, eps, rng, belief=[.3,.7], frames=3 if joint else 1)
    budget = PersistentQueryBudget(path, **args)
    q0, c0 = budget.frame(0., select)
    assert c0['floor_degraded'] and c0['actual_expected_utility_lower'] is None
    assert budget.frame(0., lambda *a: pytest.fail('retry resampled')) == (q0,c0)
    budget.close()
    budget = PersistentQueryBudget(path, **args)
    if joint:
        budget.frame(20., lambda *a: pytest.fail('cached segment resampled'))
    else:
        budget.frame(20., select)
    for tick in range(2, 300):
        budget.frame(tick*20., select)
    assert budget.reserved == Fraction(100,101)
    assert len(calls) == (100 if joint else 300)
    before = budget.reserved
    with pytest.raises(ValueError, match='skip/reset'):
        budget.frame(0., select)
    assert budget.reserved == before
    budget.close()
    for changed in (dict(gamma=2.), dict(session_token='new-trip'), dict(start_s=20.), dict(context_id='different')):
        with pytest.raises(ValueError, match='cannot be changed'):
            PersistentQueryBudget(path, **(args|changed))


def test_correlated_adaptive_two_round_likelihood_not_iid():
    _, p = policy()
    # Enumerate public actions: the second support depends on first PUBLIC Q.
    # Beliefs under the two private histories can be arbitrary and correlated.
    def channel(previous, belief, eps):
        actions=p.library.build(previous, frames=1)
        g=p.provider(actions); k=oscillation_upper(g)
        beta=float(min(Fraction(40),2*eps/k)) if k else 40.
        aa=[QueryBundle(tuple(range(5*i,5*i+5))) for i in range(len(actions))]
        m=ProbabilisticQueryBundle(aa,g,beta=beta)
        return actions,m.probabilities(belief)
    acts, pa=channel(None,[1.,0.],allowance(1.,1))
    _, pb=channel(None,[0.,1.],allowance(1.,1))
    ratios=[]
    for i,a in enumerate(acts):
        nxt, qa=channel(a.frames[-1],[.2,.8],allowance(1.,2))
        same,qb=channel(a.frames[-1],[.9,.1],allowance(1.,2))
        assert nxt==same and np.all(qa>0) and qa.sum()==pytest.approx(1.)
        ratios.extend(np.log(pa[i]*qa/(pb[i]*qb)))
    assert max(np.abs(ratios)) <= float(allowance(1.,1)+allowance(1.,2))+1e-12


def test_actual_utility_requires_explicit_error_assumptions():
    assert actual_utility_bridge(.8)['actual_expected_utility_lower'] is None
    assert actual_utility_bridge(.8,tv_bound=.1)['actual_expected_utility_lower'] is None
    assert actual_utility_bridge(.8,tv_bound=.1,proxy_error_bound=.05)['actual_expected_utility_lower']==pytest.approx(.65)
    with pytest.raises(ValueError): actual_utility_bridge(.8,tv_bound=True)


def test_fixed_purpose_table_and_full_local_answer():
    rn,c,_=fixture()
    table=FixedPublicPurposeTable(c,c,[0,12,25],destinations=[12,25])
    acts=[QuerySegment(((0,5,10,15,20),)*3),QuerySegment(((5,10,15,20,25),)*3)]
    g=table(acts)
    assert g.shape==(3,2) and np.all((g>=0)&(g<=1))
    assert np.all(table.weights.sum(axis=2)<=1.+1e-12)
    ranking=MultiPurposeRoadRanking(c)
    spec=QuerySpec(QueryPurpose.NEAREST,c.categories[0],k=1)
    full=local_sorted_pois(ranking,0,np.ones(ranking.n,bool),spec)
    assert len(full)>1
    assert ranking.top(0,np.ones(ranking.n,bool),spec)==full[:1]


def test_invalid_selector_never_publishes_or_charges(tmp_path):
    ledger=PersistentQueryBudget(tmp_path/'q.sqlite',session_token='s',context_id='c')
    def bad(*a): return dict(frames=[[0]*5]*3,certificate=dict(epsilon_Q_upper=2.))
    with pytest.raises(ValueError,match='exceeds'):
        ledger.frame(0.,bad)
    assert ledger.reserved==0
    ledger.close()


def test_end_to_end_local_request_and_gps_do_not_control_wire(tmp_path):
    rn,c,_=fixture()
    lib=PublicSegmentLibrary(rn,[(0,2,4,6,8),(12,14,16,18,20)])
    table=FixedPublicPurposeTable(c,c,[0,12,25],destinations=[12,25])
    p=SegmentPqbPolicy(lib,table,minimum_coverage=0.)
    def server(request):
        state=rn.nearest(*request['coordinate'])[0]
        return [list(map(int,row[row>=0])) for row in c.query_indices(state)]
    clients=[]
    for i in range(2):
        ledger=PersistentQueryBudget(tmp_path/f'{i}.sqlite',session_token='s',context_id='same',private_key=b'z'*32)
        clients.append(SegmentCoverClient(rn,p,ledger,c.categories,len(c.pois)))
    rank=MultiPurposeRoadRanking(c)
    requests=[QuerySpec(QueryPurpose.NEAREST,c.categories[0]),
              QuerySpec(QueryPurpose.WITHIN_RADIUS,c.categories[0],radius_m=200.)]
    for tick in range(9):
        outputs=[client.step(tick*20.,[.2,.3,.5],server) for client in clients]
        assert outputs[0][0]['requests']==outputs[1][0]['requests']
        assert outputs[0][0]['replies']==outputs[1][0]['replies']
        assert set(outputs[0][0]['requests'][0])=={'schema','timestamp_s','coordinate','categories','response_l','epoch'}
        for i,client in enumerate(clients):
            client.local_answer(rank,1+i*15,outputs[i][0]['known'],requests[i])
    for client in clients:client.ledger.close()
