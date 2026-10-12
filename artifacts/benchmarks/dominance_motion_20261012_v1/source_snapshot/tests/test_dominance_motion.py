import itertools
from fractions import Fraction
import numpy as np

from benchmark.dominance_motion import DominanceMotionLibrary,StaticInitialPolicy
from benchmark.public_pool_portfolio import FiniteSegmentPolicy
from benchmark.finite_query_kernel import FiniteCenteredKernel
from evaluation.lane_travel import SparseTravel
from tests.test_public_pool_portfolio import objects


def test_every_candidate_preserves_POI_pool_and_directed_travel():
    rn,c,table,pool,old=objects()
    lib=DominanceMotionLibrary(rn,old.goals,pool)
    previous=old.goals[0]
    actions=lib.build(previous,3);travel=SparseTravel(rn)
    for a in actions:
        assert lib.dominates(previous,a)
        before=previous
        for frame in a.frames:
            assert all(y in travel.reachable(x,20.) for x,y in zip(before,frame))
            assert pool.pool(previous)&~pool.pool(frame)==0
            before=frame


def test_shared_initial_action_and_universal_reference_dominance():
    rn,c,t,pool,old=objects();lib=DominanceMotionLibrary(rn,old.goals,pool)
    original=FiniteSegmentPolicy(old,t,pool,use_simulator=True)
    motion=FiniteSegmentPolicy(lib,t,pool,use_simulator=True)
    static=StaticInitialPolicy(old,t,pool,use_simulator=True)
    f=lambda p:p.select(None,Fraction(1,2),np.random.default_rng(17),belief=[.2,.3,.5])
    a,b,d=f(original),f(motion),f(static)
    assert a['frames']==b['frames']==d['frames']
    first=tuple(a['frames'][0])
    for action in lib.build(first,3):
        for frame in action.frames:
            initial=pool.mask(pool.pool(first));current=pool.mask(pool.pool(frame))
            for reference in itertools.combinations(range(len(c.pois)),min(3,len(c.pois))):
                assert sum(current[i] for i in reference)>=sum(initial[i] for i in reference)
    assert static.select(first,Fraction(1,6),None,belief=[np.nan]*3)['certificate']['epsilon_Q_upper']==0


def test_hidden_lane_labels_need_augmented_transcript_composition():
    # Two first-round actions have the SAME wire coordinate but distinct lane
    # labels; their next public libraries can have disjoint support. Publish
    # labels analytically, prove privacy there, then project to coordinates.
    k1=FiniteCenteredKernel([1,1],Fraction(1,2),resolution=1)
    k2=FiniteCenteredKernel([1,1],Fraction(1,6),resolution=1)
    p1,q1=k1.probabilities([0,1]),k1.probabilities([1,0])
    p2,q2=k2.probabilities([1,0]),k2.probabilities([0,1])
    for lane,next_action in itertools.product(range(2),repeat=2):
        likelihood=p1[lane]*p2[next_action]/(q1[lane]*q2[next_action])
        assert likelihood<=(1+k1.epsilon)*(1+k2.epsilon)
