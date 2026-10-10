"""Independent moving-state floor and kernel edge cases, after source freeze."""
from fractions import Fraction
import numpy as np

from benchmark.public_segment_scores import FixedPublicPurposeTable
from benchmark.query_segments import QuerySegment, oscillation_upper, SegmentPqbPolicy, PublicSegmentLibrary
from core.query_budget import allowance
from benchmark.probabilistic_query_bundle import QueryBundle, ProbabilisticQueryBundle
from benchmark.query_bundle_bounds import calibrated_beta
from tests.test_lane_comparison import road


def test_frame_floor_does_not_confuse_average_with_moving_user():
    table=object.__new__(FixedPublicPurposeTable)
    table.frame_table=lambda f: np.array([[1.]*4,[0.]*4]) if f[0]==0 else np.array([[0.]*4,[1.]*4])
    segment=QuerySegment(((0,)*5,(1,)*5,(0,)*5))
    proxy=table([segment]);floor=table.floor_table([segment])
    assert proxy.min()>0 and floor.min()==0
    # Moving true states (1,0,1) receive zero service at all three frames.
    assert sum(table.frame_table(f)[x].mean() for f,x in zip(segment.frames,(1,0,1)))==0


def test_hidden_tiny_oscillation_is_not_rounded_to_zero_in_segment_policy():
    g=np.array([[1e-20,1.],[0.,1.]])
    assert np.ptp(g[:,0]-g[:,1])==0  # naive float differences lose the signal
    assert oscillation_upper(g)>0
    legacy=ProbabilisticQueryBundle([QueryBundle((0,)),QueryBundle((1,))],g,cost_weight=0.)
    assert calibrated_beta(legacy,epsilon_target=0.)==0
    rn=road();lib=PublicSegmentLibrary(rn,[(0,)*5,(1,)*5])
    p=SegmentPqbPolicy(lib,lambda actions:g,minimum_coverage=0.)
    choice=p.select(None,Fraction(0),np.random.default_rng(1),belief=[1.,0.])
    assert choice['certificate']['beta']==0
    assert choice['certificate']['epsilon_Q_upper']==0


def test_infinite_schedule_and_partial_block_are_exact():
    for n in (1,2,12,1000):
        assert sum((allowance(1.,j) for j in range(1,n+1)),Fraction(0))==Fraction(n,n+1)
    # At600s joint reserves all of block11; step has emitted only its first
    # third. They share the whole-session cap and complete-block schedule.
    assert Fraction(10,11)+allowance(1.,11)==Fraction(11,12)
    assert Fraction(10,11)+allowance(1.,11,3)<Fraction(11,12)
