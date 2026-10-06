"""Opt-in purpose-normalized public profiles; no new location mechanism.

For each public latent/purpose/destination case, average its valid categories
BEFORE mixing cases. Within each purpose, integrate the protected belief and
uniform destination prior, condition on valid cases, then equally mix defined
purposes. Empty categories/cases/purposes are recorded, never fabricated scores.
"""
import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from benchmark.public_service_profiles import PublicServiceProfiles, VectorizedProfileObjective, _integer
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from evaluation.lane_travel import matrix


PURPOSES=('nearest_distance','fastest_travel','within_radius','minimum_detour')


class _ProtectedMassView:
    """Only an objective adapter; its mass already follows protected belief."""
    def __init__(self,profiles,mass):
        self.base=profiles;self.state_ids=np.array([0]);self.profile_mass=csr_matrix(mass[None,:])

    def __getattr__(self,name):return getattr(self.base,name)


class PurposeNormalizedServiceProfiles(PublicServiceProfiles):
    def __init__(self,base,context,references,purpose_masses,case_valid_categories,metadata):
        if tuple(purpose_masses)!=PURPOSES:
            raise ValueError('Four declared public purposes required')
        self.purpose_masses={p:csr_matrix(purpose_masses[p]) for p in PURPOSES}
        self.case_valid_category_counts={p:np.asarray(case_valid_categories[p],dtype=np.int16).copy() for p in PURPOSES}
        shape=(len(base.state_ids),len(references))
        for p in PURPOSES:
            m=self.purpose_masses[p];counts=self.case_valid_category_counts[p]
            if (m.shape!=shape or not np.isfinite(m.data).all() or np.any(m.data<0)
                    or not np.allclose(np.asarray(m.sum(axis=1)).ravel(),1.,rtol=0,atol=1e-12)
                    or counts.ndim!=2 or counts.shape[0]!=shape[0] or np.any(counts<0)):
                raise ValueError('Matched public per-purpose case masses/counts required')
            counts.setflags(write=False)
        combined=sum(self.purpose_masses.values())/len(PURPOSES)
        super().__init__(base,context,references,combined,metadata)
        digest=hashlib.sha256(self.sha256.encode())
        for p in PURPOSES:
            m=self.purpose_masses[p]
            for a in (m.data.astype('<f8'),m.indices.astype('<i8'),m.indptr.astype('<i8'),
                      self.case_valid_category_counts[p].astype('<i4')):
                digest.update(a.tobytes())
        self.sha256=digest.hexdigest()

    def normalized_mass(self,belief_weights):
        belief=np.asarray(belief_weights,dtype=float)
        if (belief.shape!=(len(self.state_ids),) or not np.isfinite(belief).all()
                or np.any(belief<0) or belief.sum()<=0):
            raise ValueError('Matched finite protected belief required')
        belief=belief/belief.sum();mass=np.zeros(len(self.reference_profiles))
        nonempty=self.lengths>0;valid_mass={};conditional={};undefined={}
        for p in PURPOSES:
            values=np.asarray(belief@self.purpose_masses[p]).ravel()
            denominator=float(values[nonempty].sum())
            valid_mass[p]=denominator;undefined[p]=float(values[~nonempty].sum())
            if denominator>0:
                values[~nonempty]=0.;conditional[p]=values/denominator
        for values in conditional.values():mass+=values/len(conditional)
        diagnostics=dict(normalization='valid_categories_per_case_then_valid_cases_per_purpose_then_equal_defined_purposes',
            purpose_valid_case_mass=valid_mass,purpose_undefined_case_mass=undefined,
            defined_purposes=list(conditional),undefined_purposes=[p for p in PURPOSES if p not in conditional],
            effective_purpose_weights={p:1./len(conditional) if p in conditional else 0. for p in PURPOSES},
            empty_profile_mass=float(np.mean(list(undefined.values()))),
            empty_mass_definition='equal-prior probability of ALL categories undefined in a latent/destination case; partial category N/A normalized inside case')
        return mass,diagnostics

    def objective(self,belief_weights,*,tail_mass=.25,risk_weight=.5):
        mass,diagnostics=self.normalized_mass(belief_weights)
        if not diagnostics['defined_purposes']:
            raise ValueError('All positive-mass reference profiles are undefined')
        result=VectorizedProfileObjective(_ProtectedMassView(self,mass),[1.],tail_mass=tail_mass,risk_weight=risk_weight)
        result.empty_profile_mass=diagnostics['empty_profile_mass']
        result.normalization_diagnostics=diagnostics
        return result


def build_public_service_profiles_v2(base,reply20,*,public_destination_states,public_radius_m=1000.,reference_k=5):
    """Use fixed public graph prototypes; do not supply private trip endpoints."""
    reference_k=_integer(reference_k,'reference top-k')
    if base.context.k!=reference_k:raise ValueError('Reference top-k must match unchanged base context')
    ResponseAwareAnchorModel(base,reply20)
    if isinstance(public_radius_m,bool) or not np.isfinite(public_radius_m) or public_radius_m<=0:
        raise ValueError('Positive fixed public radius required')
    raw=list(public_destination_states)
    if (not raw or any(isinstance(x,bool) or not isinstance(x,(int,np.integer)) for x in raw)
            or len(set(raw))!=len(raw) or min(raw)<0 or max(raw)>=len(base.rn)):
        raise ValueError('Distinct valid PUBLIC destination states required')
    destinations=np.asarray(sorted(raw),dtype=int)
    rn,pois,categories=base.rn,reply20.pois,reply20.categories
    if not categories or not pois:raise ValueError('Nonempty public catalogue required')
    states=base.context.access[base.state_ids];unique,inverse=np.unique(states,return_inverse=True)
    vertices=np.asarray([p['vertex'] for p in pois],dtype=int)
    distances=np.empty((len(unique),len(pois)));times=np.empty_like(distances)
    direct=np.empty((len(unique),len(destinations)))
    distance_graph,time_graph=matrix(rn),matrix(rn,time=True)
    for start in range(0,len(unique),32):
        costs=dijkstra(distance_graph,directed=True,indices=unique[start:start+32])
        distances[start:start+len(costs)]=costs[:,vertices];direct[start:start+len(costs)]=costs[:,destinations]
        elapsed=dijkstra(time_graph,directed=True,indices=unique[start:start+32])
        times[start:start+len(elapsed)]=elapsed[:,vertices]
    to_destination=dijkstra(distance_graph.transpose().tocsr(),directed=True,indices=destinations)
    radius=distances.copy();radius[radius>public_radius_m]=np.inf
    prototypes={PURPOSES[0]:[distances],PURPOSES[1]:[times],PURPOSES[2]:[radius],PURPOSES[3]:[]}
    for j,destination in enumerate(destinations):
        scores=np.full_like(distances,np.inf);valid=np.isfinite(direct[:,j])
        scores[valid]=distances[valid]+to_destination[j,vertices]-direct[valid,j,None]
        finite=np.isfinite(scores);scores[finite]=np.maximum(0.,scores[finite])
        prototypes[PURPOSES[3]].append(scores)
    category_ids=[np.asarray([i for i,p in enumerate(pois) if p['category']==c],dtype=int) for c in categories]
    references,indexes=[],{}
    def index(profile):
        if profile not in indexes:indexes[profile]=len(references);references.append(profile)
        return indexes[profile]
    sparse_entries={};case_counts={};empty_categories=0
    for purpose,cost_tables in prototypes.items():
        entries=[];counts=np.zeros((len(states),len(cost_tables)),dtype=np.int16)
        for d,costs in enumerate(cost_tables):
            category_refs=[]
            for ids in category_ids:
                order=np.argsort(costs[:,ids],axis=1,kind='stable')[:,:reference_k]
                selected=ids[order];finite=np.isfinite(np.take_along_axis(costs[:,ids],order,axis=1))
                category_refs.append([tuple(sorted(map(int,row[valid]))) for row,valid in zip(selected,finite)])
            for latent,at in enumerate(inverse):
                valid_refs=[refs[at] for refs in category_refs if refs[at]]
                counts[latent,d]=len(valid_refs);empty_categories+=len(categories)-len(valid_refs)
                # Normalize each case before any posterior/prototype mixing.
                if valid_refs:
                    for ref in valid_refs:entries.append((latent,index(ref),1./(len(cost_tables)*len(valid_refs))))
                else:entries.append((latent,index(()),1./len(cost_tables)))
        sparse_entries[purpose]=entries;case_counts[purpose]=counts
    masses={}
    for purpose,entries in sparse_entries.items():
        rows,cols,values=zip(*entries)
        masses[purpose]=csr_matrix((values,(rows,cols)),shape=(len(states),len(references)))
    metadata=dict(schema='public-service-profiles-purpose-normalized-v2',reference_k=reference_k,
        reply_l=reply20.k,reference_context_sha256=base.context.sha256,reply_context_sha256=reply20.sha256,
        catalogue_sha256=rn.catalogue_sha256,state_ids=base.state_ids.tolist(),
        public_destination_states=destinations.tolist(),public_radius_m=float(public_radius_m),
        purpose_prior='equal defined purposes after within-purpose protected valid-case conditioning',
        destination_prior='uniform fixed public graph states BEFORE within-purpose N/A conditioning',
        category_prior='equal VALID categories separately within each latent/purpose/destination case',
        normalization_denominator='D_p(b)=sum_x,d b_x*pi_p(d)*1[n_valid_categories(x,p,d)>0]',
        empty_reference='category N/A excluded inside case; whole empty case retained; entirely undefined purpose N/A',
        input_profile_count=len(states)*sum(map(len,prototypes.values()))*len(categories),
        empty_input_category_profile_count=empty_categories,
        all_empty_case_count_by_purpose={p:int(np.sum(v==0)) for p,v in case_counts.items()},
        coalesced_profile_count=len(references),server_service='nearest-distance top-L per ALL public categories',
        access_rule='public coordinate-to-state mapping, including duplicate-coordinate ambiguity',
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return PurposeNormalizedServiceProfiles(base,reply20,references,masses,case_counts,metadata)
