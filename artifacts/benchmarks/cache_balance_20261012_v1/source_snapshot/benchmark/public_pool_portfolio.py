"""Public POI-pool portfolio and attainable-service certificates.

Only map/catalogue/reply signatures enter construction. No trajectory, GPS,
private purpose, split or attacker scores. Covering a small static catalogue is
a structural control, not a claimed new Geo-I contribution.
"""
from functools import lru_cache
import numpy as np

from benchmark.finite_query_kernel import FiniteCenteredKernel, quantize_scores, simulator_randbelow
from benchmark.query_segments import PublicSegmentLibrary
from fractions import Fraction


class PublicPoolPortfolio:
    def __init__(self, reply, *, public_poi_weights=None):
        self.reply=reply
        signatures=reply.signatures[reply.access].reshape(len(reply.rn),-1)
        mask=np.zeros((len(reply.rn),len(reply.pois)),dtype=np.uint8)
        rows,slots=np.where(signatures>=0);mask[rows,signatures[rows,slots]]=1
        packed=np.packbits(mask,axis=1)
        unique,representatives=np.unique(packed,axis=0,return_index=True)
        order=np.argsort(representatives)
        self.representatives=tuple(map(int,representatives[order]))
        self.pools=tuple(int.from_bytes(row.tobytes(),'big') for row in unique[order])
        self.state_pool=tuple(int.from_bytes(row.tobytes(),'big') for row in packed)
        # Tiny public maps can have fewer than K distinct reply signatures.
        # Add fixed coordinate representatives, preserving K without pretending
        # they contribute new POIs or anonymity. Never choose from current Z.
        if len(self.representatives)<5:
            extra=tuple(i for i in range(len(reply.rn)) if i not in self.representatives)[:5-len(self.representatives)]
            self.representatives+=extra
            self.pools+=tuple(self.state_pool[i] for i in extra)
        if len(self.representatives)<5:raise ValueError('K5 requires at least five public road states')
        self.attainable=0
        for bits in self.pools:self.attainable|=bits
        self.attainable_count=self.attainable.bit_count()
        self.byte_count=packed.shape[1]
        weights=np.ones(len(reply.pois)) if public_poi_weights is None else np.asarray(public_poi_weights,dtype=float).copy()
        if weights.shape!=(len(reply.pois),) or not np.isfinite(weights).all() or np.any(weights<0) or weights.sum()<=0:
            raise ValueError('Fixed public nonnegative POI-reference masses required')
        self.weights=weights/weights.sum()
        self.weights.flags.writeable=False
        exact=tuple(Fraction.from_float(float(w)) for w in self.weights)
        total=sum(exact)
        self.exact_weights=tuple(w/total for w in exact)

    @lru_cache(maxsize=200000)
    def quality(self,bits):
        return float(np.sum(self.weights[self.mask(bits)]))

    @lru_cache(maxsize=32768)
    def quality_exact(self,bits):
        return sum((w for w,visible in zip(self.exact_weights,self.mask(bits)) if visible),Fraction(0))

    def goals(self, *, width=24):
        """Beam over public reply-equivalence representatives, fixed K5.

        All max-coverage beam members retained (stable lexical ties); no
        location-secret candidate pruning. Alternate pools preserve diversity.
        """
        beam=[((),0)]
        for depth in range(5):
            proposals={}
            for selected,bits in beam:
                for state,pool in zip(self.representatives,self.pools):
                    if state in selected:continue
                    states=tuple(sorted(selected+(state,)))
                    proposals[states]=bits|pool
            ordered=sorted(proposals.items(),key=lambda pair:(-self.quality(pair[1]),-pair[1].bit_count(),pair[0]))
            beam=ordered[:width]
        return tuple(states for states,bits in beam)

    @lru_cache(maxsize=32768)
    def pool(self,frame):
        bits=0
        for state in frame:bits|=self.state_pool[state]
        return bits

    def mask(self,bits):
        packed=np.frombuffer(bits.to_bytes(self.byte_count,'big'),dtype=np.uint8)
        return np.unpackbits(packed)[:len(self.reply.pois)].astype(bool)

    def segment_missing(self,action):
        return max((self.attainable&~self.pool(frame)).bit_count() for frame in action.frames)

    def reference_certificate(self, reference_indices, frame):
        """Exact true-reference bridge for ANY purpose/radius/destination.

        Local-only if reference uses GPS/private purpose. Empty references N/A.
        Achievability ceiling is separate from missing attainable POIs. No
        posterior calibration or surrogate-purpose assumption is needed.
        """
        ids=tuple(sorted(set(map(int,reference_indices))))
        if any(i<0 or i>=len(self.reply.pois) for i in ids):raise ValueError('Valid public POI IDs required')
        if not ids:return dict(status='NA',recall=None,achievable_ceiling=None,missing_attainable=None)
        available=self.mask(self.pool(tuple(frame)));possible=self.mask(self.attainable)
        attainable=sum(bool(possible[i]) for i in ids)
        hit=sum(bool(available[i]) for i in ids)
        missing=sum(bool(possible[i] and not available[i]) for i in ids)
        assert hit==attainable-missing
        return dict(status='exact_reference_on_static_catalogue',recall=hit/len(ids),
            exact_recall=str(Fraction(hit,len(ids))),exact_achievable_ceiling=str(Fraction(attainable,len(ids))),
            achievable_ceiling=attainable/len(ids),missing_attainable=missing,
            reference_size=len(ids),pool_attainable_missing=(self.attainable&~self.pool(tuple(frame))).bit_count(),
            lower_recall=max(0.,(attainable-(self.attainable&~self.pool(tuple(frame))).bit_count())/len(ids)))


class FiniteSegmentPolicy:
    """Bounded integer postprocessing of protected Geo-I belief.

    Public filter fixes a reference-mass floor .005 below the best INITIAL
    public goal at EACH segment frame. Holding supplies an inductive floor;
    the target is never loosened with time or private history.
    Geometric proximity is a secondary protected planning score (future query
    placement), not a fabricated actual-Recall metric. Deployment uses fresh OS
    randomness; the study explicitly opts into private seeded simulator streams.
    """
    def __init__(self, library, table, portfolio, *, broad=True, resolution=65536,
                 use_simulator=False, proximity_weight=.1):
        self.library,self.table,self.portfolio=library,table,portfolio
        self.broad,self.resolution,self.use_simulator=broad,resolution,use_simulator
        self.quality_floor_exact=max(portfolio.quality_exact(portfolio.pool(tuple(goal))) for goal in library.goals)-Fraction(1,200)
        self.quality_floor=float(self.quality_floor_exact)
        if not 0<=proximity_weight<=1:raise ValueError('Public proximity weight in [0,1] required')
        self.proximity_weight=proximity_weight

    @lru_cache(maxsize=4096)
    def public_tables(self,previous,frames):
        actions=self.library.build(previous,frames)
        missing=[self.portfolio.segment_missing(a) for a in actions]
        if self.broad:
            keep=[i for i,a in enumerate(actions) if min(self.portfolio.quality_exact(self.portfolio.pool(f)) for f in a.frames)>=self.quality_floor_exact]
            if not keep:raise ValueError('Public prefix outside the invariant pool domain; no private fallback')
            actions=tuple(actions[i] for i in keep);missing=[missing[i] for i in keep]
        g=self.table(actions)
        if self.broad:
            # Same public geography underlying Geo-I's distance metric;
            # directed feasibility is independently enforced by library.
            latent=self.library.rn.xy[list(self.table.states)]
            proximity=np.column_stack([np.mean([1./(1.+np.linalg.norm(latent[:,None,:]-self.library.rn.xy[list(f)][None,:,:],axis=2).min(axis=1)/1000.) for f in a.frames],axis=0) for a in actions])
            g=(1-self.proximity_weight)*g+self.proximity_weight*proximity
        return actions,g,missing

    def select(self,previous,epsilon,rng=None,*,belief,frames=3):
        actions,g,missing=self.public_tables(previous,frames)
        b=np.asarray(belief,dtype=float)
        if b.shape!=(g.shape[0],) or not np.isfinite(b).all() or np.any(b<0) or not np.isclose(b.sum(),1.,rtol=0,atol=1e-12):
            raise ValueError('Normalized protected belief required')
        scores=np.einsum('i,ij->j',b,g,optimize=False)
        f=quantize_scores(scores,self.resolution)
        # Conservative representable reservation label; integer alpha retains
        # an EXACT bound even when the original fraction rounds upward.
        limit=Fraction(epsilon);value=float(limit)
        if Fraction.from_float(value)>limit:value=float(np.nextafter(value,0.))
        safe=Fraction.from_float(value)
        kernel=FiniteCenteredKernel([1]*len(actions),safe,resolution=self.resolution)
        draw=simulator_randbelow(rng) if self.use_simulator else None
        selected=kernel.sample(f,randbelow=draw)
        utility=kernel.utility_certificate(f,planning_scores=scores)
        cert=dict(epsilon_Q_upper=value,exact_likelihood_ratio_upper=str(1+safe),
            integer_scores=list(f),planning_scores_binary64=scores.tolist(),
            alpha=str(kernel.alpha),public_draw_total=str(kernel.public_total),
            public_library_size=len(actions),public_score_resolution=self.resolution,
            public_best_missing_attainable=min(missing),selected_missing_attainable=missing[selected],
            invariant_public_reference_mass_floor=self.quality_floor if self.broad else None,
            exact_public_reference_mass_floor=str(self.quality_floor_exact) if self.broad else None,
            selected_public_reference_mass=min(self.portfolio.quality(self.portfolio.pool(f)) for f in actions[selected].frames),
            exact_selected_public_reference_mass=str(min(self.portfolio.quality_exact(self.portfolio.pool(f)) for f in actions[selected].frames)),
            selected_pool_attainable_fraction=1-missing[selected]/self.portfolio.attainable_count,
            public_attainable_POIs=self.portfolio.attainable_count,
            privacy_scope='exact finite Q probabilities under independent uniform bits; coordinate/logical-request stream only',
            random_source='private seeded PCG simulator; not information-theoretic RNG certified' if self.use_simulator else 'fresh secrets.randbelow; uniform OS bits assumed',
            actual_expected_utility_lower=None,utility_status='pool/reference certificate; proxy planning score separate',
            protected_score_range=float(scores.max()-scores.min()),**utility)
        return dict(frames=[list(f) for f in actions[selected].frames],certificate=cert)
