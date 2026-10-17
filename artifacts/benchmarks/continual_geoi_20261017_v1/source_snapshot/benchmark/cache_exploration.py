"""Public exploration plus finite bounded Geo-I selection, with a local cache.

The candidate library/base masses see only prior randomized Q and server replies.
No GPS, private purpose, Z, protected belief or empirical labels select support.
"""
from fractions import Fraction
from functools import lru_cache
import math
import numpy as np
from benchmark.query_segments import PublicSegmentLibrary,QuerySegment
from benchmark.public_pool_portfolio import FiniteSegmentPolicy
from benchmark.finite_query_kernel import FiniteCenteredKernel,quantize_scores,simulator_randbelow
from evaluation.lane_travel import SparseTravel


def mask_bits(mask):return int.from_bytes(np.packbits(np.asarray(mask,dtype=bool)).tobytes(),'big')


class ExploreMotionLibrary(PublicSegmentLibrary):
    def __init__(self,rn,goals,portfolio):
        super().__init__(rn,goals);self.portfolio=portfolio
        self.reach=SparseTravel(rn,cache_limit=4096)
        self.headings=tuple(np.array([math.cos(j*math.pi/4),math.sin(j*math.pi/4)]) for j in range(8))

    @lru_cache(maxsize=32768)
    def move_one(self,current,track,heading,known):
        others=0
        for j,s in enumerate(current):
            if j!=track:others|=self.portfolio.state_pool[s]
        old=current[track];candidates=[]
        for state,time in self.reach.reachable(old,20.).items():
            union=others|self.portfolio.state_pool[state]
            delta=self.rn.xy[state]-self.rn.xy[old]
            progress=float(np.einsum('i,i->',delta,self.headings[heading],optimize=False))
            candidates.append((-(union&~known).bit_count(),-progress,float(time),state))
        state=min(candidates)[-1];result=list(current);result[track]=int(state)
        return tuple(result)

    @lru_cache(maxsize=4096)
    def build_cached(self,previous,known,frames=3):
        candidates=set(super().build(previous,frames))
        if previous is None:return tuple(sorted(candidates,key=lambda a:a.frames))
        for heading in range(8):
            for track in range(5):
                current=previous;path=[];seen=known
                for _ in range(frames):
                    current=self.move_one(current,track,heading,seen);path.append(current);seen|=self.portfolio.pool(current)
                candidates.add(QuerySegment(tuple(path)))
            current=previous;path=[];seen=known
            for _ in range(frames):
                for track in range(5):current=self.move_one(current,track,heading,seen)
                path.append(current);seen|=self.portfolio.pool(current)
            candidates.add(QuerySegment(tuple(path)))
        return tuple(sorted(candidates,key=lambda a:a.frames))


class CacheExplorePolicy:
    def __init__(self,library,table,portfolio,*,use_simulator=False,resolution=65536,novelty_weight=4):
        if isinstance(novelty_weight,bool) or not isinstance(novelty_weight,int) or novelty_weight<0:raise ValueError('Public nonnegative integer novelty weight required')
        self.library,self.table,self.portfolio=library,table,portfolio
        self.use_simulator,self.resolution,self.novelty_weight=use_simulator,resolution,novelty_weight
        self.initial=FiniteSegmentPolicy(library,table,portfolio,use_simulator=use_simulator,resolution=resolution,proximity_weight=.1)

    @lru_cache(maxsize=4096)
    def public_tables(self,previous,known,frames=3):
        actions=self.library.build_cached(previous,known,frames)
        gains=[];columns=[]
        latent=self.library.rn.xy[list(self.table.states)]
        for action in actions:
            seen=known;score=[]
            for frame in action.frames:
                seen|=self.portfolio.pool(frame);mask=self.portfolio.mask(seen)
                coverage=self.table.weights[:,:,mask].sum(axis=2).sum(axis=1)/self.table.defined_count
                proximity=1./(1.+np.linalg.norm(latent[:,None,:]-self.library.rn.xy[list(frame)][None,:,:],axis=2).min(axis=1)/1000.)
                score.append(.9*coverage+.1*proximity)
            columns.append(np.mean(score,axis=0));gains.append((seen&~known).bit_count())
        masses=tuple(1+self.novelty_weight*d for d in gains)
        return actions,np.column_stack(columns),tuple(gains),masses

    def select(self,previous,epsilon,rng=None,*,belief,known,frames=3):
        if previous is None:
            if known:raise ValueError('Initial session cache must be empty')
            choice=self.initial.select(None,epsilon,rng,belief=belief,frames=frames)
            choice['certificate']['cache_stage']='initial public portfolio'
            return choice
        actions,g,gains,masses=self.public_tables(previous,int(known),frames)
        b=np.asarray(belief,dtype=float)
        if b.shape!=(g.shape[0],) or not np.isfinite(b).all() or np.any(b<0) or not np.isclose(b.sum(),1.,rtol=0,atol=1e-12):raise ValueError('Normalized protected belief required')
        values=np.einsum('i,ij->j',b,g,optimize=False);f=quantize_scores(values,self.resolution)
        limit=Fraction(epsilon);value=float(limit)
        if Fraction.from_float(value)>limit:value=float(np.nextafter(value,0.))
        k=FiniteCenteredKernel(masses,Fraction.from_float(value),resolution=self.resolution)
        selected=k.sample(f,randbelow=simulator_randbelow(rng) if self.use_simulator else None)
        score=k.utility_certificate(f,planning_scores=values)
        # These are local diagnostic values, NOT wire fields.
        seen=known;missing=[]
        for frame in actions[selected].frames:
            seen|=self.portfolio.pool(frame);missing.append((self.portfolio.attainable&~seen).bit_count())
        cert=dict(cache_stage='public novelty base plus bounded protected score',epsilon_Q_upper=value,
            exact_likelihood_ratio_upper=str(1+k.epsilon),alpha=str(k.alpha),integer_scores=list(f),
            planning_scores_binary64=values.tolist(),public_draw_total=str(k.public_total),public_library_size=len(actions),
            public_base_masses=list(masses),public_novel_POI_counts=list(gains),public_score_resolution=self.resolution,
            selected_novel_POIs=gains[selected],selected_missing_attainable=max(missing),
            invariant_public_reference_mass_floor=self.initial.quality_floor,
            exact_public_reference_mass_floor=str(self.initial.quality_floor_exact),
            actual_expected_utility_lower=None,utility_status='static session-cache monotonicity relative to identical initial pool',
            random_source='private PCG simulator; not certified entropy' if self.use_simulator else 'fresh secrets.randbelow; uniform OS bits assumed',**score)
        return dict(frames=[list(frame) for frame in actions[selected].frames],certificate=cert)
