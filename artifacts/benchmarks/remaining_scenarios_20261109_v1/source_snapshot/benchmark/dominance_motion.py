"""Protected-belief motion in a PUBLIC library that never loses retrieved POIs.

Selected lane IDs are randomized OUTPUTS, not true GPS lanes. For privacy use
the augmented action transcript (IDs/full segments), then project to wire coords:
coincident road coordinates need not expose those IDs to the actual server.
"""
from functools import lru_cache
import math
import numpy as np

from benchmark.query_segments import QuerySegment,PublicSegmentLibrary
from benchmark.public_pool_portfolio import FiniteSegmentPolicy
from benchmark.finite_query_kernel import FiniteCenteredKernel
from evaluation.lane_travel import SparseTravel


class DominanceMotionLibrary(PublicSegmentLibrary):
    def __init__(self,rn,goal_bundles,portfolio):
        super().__init__(rn,goal_bundles)
        self.portfolio=portfolio
        self.reach=SparseTravel(rn,cache_limit=4096)
        self.headings=tuple((math.cos(j*math.pi/4),math.sin(j*math.pi/4)) for j in range(8))

    @lru_cache(maxsize=32768)
    def move_one(self,current,track,heading):
        before=self.portfolio.pool(current)
        others=0
        for j,s in enumerate(current):
            if j!=track:others|=self.portfolio.state_pool[s]
        old=current[track];direction=np.array(self.headings[heading])
        candidates=[]
        for state,time in self.reach.reachable(old,20.).items():
            union=others|self.portfolio.state_pool[state]
            if before&~union:continue
            delta=self.rn.xy[state]-self.rn.xy[old]
            progress=float(np.einsum('i,i->',delta,direction,optimize=False))
            candidates.append((-(union&self.portfolio.attainable).bit_count(),-progress,float(time),state))
        assert candidates  # old is reachable and retains the entire pool
        state=min(candidates)[-1]
        result=list(current);result[track]=int(state)
        return tuple(result)

    def dominates(self,previous,action):
        before=self.portfolio.pool(previous)
        for frame in action.frames:
            after=self.portfolio.pool(frame)
            if before&~after:return False
            before=after
        return True

    @lru_cache(maxsize=4096)
    def build(self,previous=None,frames=3):
        if previous is None:return super().build(None,frames)
        candidates={a for a in super().build(previous,frames) if self.dominates(previous,a)}
        for heading in range(8):
            for track in range(5):
                current=previous;path=[]
                for _ in range(frames):
                    current=self.move_one(current,track,heading);path.append(current)
                candidates.add(QuerySegment(tuple(path)))
            current=previous;path=[]
            for _ in range(frames):
                # Each track moves once per public20s frame. Others' moves only
                # update a PUBLIC pool constraint, never another track's time.
                for track in range(5):current=self.move_one(current,track,heading)
                path.append(current)
            candidates.add(QuerySegment(tuple(path)))
        assert all(self.dominates(previous,a) for a in candidates)
        return tuple(sorted(candidates,key=lambda a:a.frames))


class StaticInitialPolicy(FiniteSegmentPolicy):
    """Matched initial Q action, then deterministic replay; experimental control."""
    def select(self,previous,epsilon,rng=None,*,belief,frames=3):
        if previous is None:return super().select(None,epsilon,rng,belief=belief,frames=frames)
        k=FiniteCenteredKernel([1],0,resolution=self.resolution)
        bits=self.portfolio.pool(previous)
        score=k.utility_certificate([0],planning_scores=[0.])
        cert=dict(epsilon_Q_upper=0.,exact_likelihood_ratio_upper='1',alpha='0',
            integer_scores=[0],planning_scores_binary64=[0.],public_draw_total=str(k.public_total),
            public_library_size=1,public_score_resolution=self.resolution,
            public_best_missing_attainable=(self.portfolio.attainable&~bits).bit_count(),
            selected_missing_attainable=(self.portfolio.attainable&~bits).bit_count(),
            invariant_public_reference_mass_floor=self.quality_floor,
            exact_public_reference_mass_floor=str(self.quality_floor_exact),
            selected_public_reference_mass=self.portfolio.quality(bits),
            exact_selected_public_reference_mass=str(self.portfolio.quality_exact(bits)),
            selected_pool_attainable_fraction=bits.bit_count()/self.portfolio.attainable_count,
            public_attainable_POIs=self.portfolio.attainable_count,
            privacy_scope='deterministic postprocessing of initial sampled action',
            random_source='public deterministic replay; no fresh private selection',
            actual_expected_utility_lower=None,utility_status='pool/reference certificate; proxy planning score separate',
            protected_score_range=0.,**score)
        return dict(frames=[list(previous)]*frames,certificate=cert)
