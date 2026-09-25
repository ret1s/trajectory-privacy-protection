import numpy as np
from benchmark.engines.dls import DLSGraph, entropy
from benchmark.paper_comparators import PreparedDLS, rdg_choice, FakeQuerySequence, path_similarity


class Catalogue:
    def __init__(self,n):self.n=n
    def __len__(self):return self.n


def test_prepared_dls_preserves_original_pools_and_seeded_choices():
    for q in ([1]*31,list(range(1,32)),[1,1,2,2,2,3,5,5,5,5,5,7,8]):
        for k in (2,5):
            a=DLSGraph(Catalogue(len(q)),q,k=k,rng=np.random.default_rng(104))
            b=PreparedDLS(Catalogue(len(q)),q,k=k,rng=np.random.default_rng(104))
            for true in range(len(q)):
                np.testing.assert_array_equal(a.candidate_pool(true),b.candidate_pool(true))
                x,h=a.select(true);y,g=b.select(true)
                np.testing.assert_array_equal(x,y);assert h==g


def test_rdg_uses_max_product_for_choice_and_forward_for_posterior():
    matrix=np.array([[9,1,2,8],[1,9,8,2]],float)
    transition=lambda a,b:matrix[np.ix_(a,b)]
    previous=[0,1];posterior=np.array([.7,.3])
    selected,p=rdg_choice(previous,posterior,0,[1,2,3],transition,2)
    scores=[]
    for d in (1,2,3):
        t=matrix[:,[0,d]];t=t/t.sum(axis=1,keepdims=True)
        scores.append(entropy(np.max(posterior[:,None]*t,axis=0)))
    assert selected.tolist()==[0,1+int(np.argmax(scores))]
    t=matrix[:,selected];t=t/t.sum(axis=1,keepdims=True)
    np.testing.assert_allclose(p,posterior@t)
    assert not np.allclose(p, np.max(posterior[:,None]*t,axis=0)/np.max(posterior[:,None]*t,axis=0).sum())


def test_path_similarity_has_direction_and_stationary_handling():
    assert path_similarity([10,0],[10,0])==0
    assert path_similarity([10,0],[-10,0])==1
    assert path_similarity([0,0],[0,0])==0
    assert path_similarity([10,0],[20,0])==1


def test_fake_timer_does_not_read_next_true_location_and_has_no_public_fake_flag():
    # Spy: fake continuation can only depend on its prior state, not current GPS.
    class DLS:
        k=2
        def select(self,real):return np.array([real,real+1]),0
    def run(second):
        f=FakeQuerySequence(None,None,DLS(),np.random.default_rng(8))
        f.step(0,3);f.remaining=2;f.next_time=5
        seen=[]
        def continuation(dt,fixed_reference=None):
            seen.append(fixed_reference)
            return np.array([f.anchor+2,f.anchor+3]) if fixed_reference is None else None
        f.continuation=continuation
        result=f.step(30,second)
        return [(t,ids.tolist(),real) for t,ids,real in result],seen
    a,seen=run(50);b,_=run(99)
    assert a[:-1]==b[:-1] and len(a)>1
    assert seen[:-1]==[None]*(len(seen)-1)
    assert a[-1][1][0]==50 and b[-1][1][0]==99


def test_attack_view_keeps_inserted_queries_but_strips_evaluator_labels():
    from evaluation.live_comparison_attacks import public_view
    ex={'clock_indices':[0,20,40], 'service_event_positions':{'0':0,'1':2,'2':4},
        'events':[{'timestamp_s':t,'coordinates':[[1,2]],'server_states':[8],
                   'real_candidate':17,'target_endpoint':[9,9]} for t in (0,7,20,28,40)]}
    view,slots=public_view(ex,[20,40])
    assert slots==[0,2] and [e['timestamp_s'] for e in view]==[0,8,20]
    assert all(set(e)=={'timestamp_s','coordinates'} for e in view)
    ex['events'][3]['target_endpoint']=[123,456]
    assert public_view(ex,[20,40])==(view,slots)


def test_set_attack_features_are_permutation_invariant():
    from evaluation.live_comparison_attacks import descriptors
    a=[np.array([[0.,0.],[2.,3.],[5.,1.]]),np.array([[10.,2.],[4.,9.]])]
    x,c=descriptors(a,np.array([0.,20.]))
    y,d=descriptors([p[::-1] for p in a],np.array([0.,20.]))
    np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(c,d)


def test_endpoint_extrapolation_uses_observed_velocity_and_correct_time_direction(monkeypatch):
    from evaluation import live_comparison_endpoint_attacks as endpoint
    # Three visible positions moving east at 2 m/s and north at 1 m/s.
    times=np.array([0.,20.,40.]);points=np.c_[100+2*times,50+times]
    monkeypatch.setattr(endpoint,'public_arrays',lambda *_:([p[None,:] for p in points],times))
    monkeypatch.setattr(endpoint,'viterbi',lambda *_:points)
    start=endpoint.extrapolations([], 'S9', None, None)
    end=endpoint.extrapolations([], 'S10', None, None)
    assert len(start)==len(end)==36
    for name,estimate in start.items():
        seconds=int(name.split('_')[-1][:-1])
        np.testing.assert_allclose(estimate,[[100-2*seconds,50-seconds]],atol=1e-10)
        np.testing.assert_allclose(end[name],[[180+2*seconds,90+seconds]],atol=1e-10)
