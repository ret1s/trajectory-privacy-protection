"""Finite-grid filtering of protected anchors; never accepts private GPS.

The emission is the ideal noisy-reuse/REM kernel, including the full original
output normalizer and coordinate multiplicity. The latent grid, occupancy prior
and isotropic transition are approximations, not a calibrated user model.
"""
from collections import Counter, OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from scipy.stats import laplace


class PublicAnchorModel:
    def __init__(self,rn,context,prior,*,spacing_m=120.,epsilon_release=.01,
                 epsilon_test=.01,theta_m=200.,cache_path=None):
        if context.rn is not rn:
            raise ValueError('Matching public map/POI context required')
        if not all(np.isfinite(x) and x>0 for x in (spacing_m,epsilon_release,epsilon_test)) or not np.isfinite(theta_m) or theta_m<0:
            raise ValueError('Invalid public belief parameters')
        prior=np.asarray(prior,dtype=float)
        if prior.shape!=(len(rn),) or not np.isfinite(prior).all() or np.any(prior<0) or prior.sum()<=0:
            raise ValueError('A normalized finite public spatial prior is required')
        self.rn,self.context=rn,context
        self.spacing_m,self.epsilon_release,self.epsilon_test,self.theta_m=spacing_m,epsilon_release,epsilon_test,theta_m
        # One observed lane state nearest the geometric center of each fixed
        # public cell, independent of any current user, anchor or scenario.
        cells,inverse=np.unique(np.floor(rn.xy/spacing_m).astype(np.int64),axis=0,return_inverse=True)
        cell_centers=(cells+.5)*spacing_m
        errors=np.linalg.norm(rn.xy-cell_centers[inverse],axis=1)
        order=np.lexsort((np.arange(len(rn)),errors,inverse))
        starts=np.r_[True,inverse[order[1:]]!=inverse[order[:-1]]]
        self.state_ids=order[starts]
        self.xy=rn.xy[self.state_ids]
        self.prior=np.bincount(inverse,weights=prior,minlength=len(cells))
        self.prior/=self.prior.sum()
        self.tree=cKDTree(self.xy)
        counts=Counter(zip(rn.lats,rn.lons))
        self.coordinate_index={rn.latlon(i):(i,counts[rn.latlon(i)]) for i in range(len(rn))}
        metadata={'schema':'public-anchor-model-v1','catalogue':rn.catalogue_sha256,'context':context.sha256,
                  'spacing_m':spacing_m,'epsilon_release':epsilon_release,'epsilon_test':epsilon_test,
                  'theta_m':theta_m,'prior_sha256':hashlib.sha256(prior.astype('<f8').tobytes()).hexdigest(),
                  'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.metadata_json=json.dumps(metadata,sort_keys=True,separators=(',',':'))
        cache=Path(cache_path) if cache_path else None
        if cache and cache.exists():
            with np.load(cache,allow_pickle=False) as a:
                if str(a['metadata'])!=self.metadata_json or not np.array_equal(a['state_ids'],self.state_ids):
                    raise ValueError('Stale belief cache; choose a fresh path')
                self.log_normalizers=a['log_normalizers'].copy()
        else:
            self.log_normalizers=np.empty(len(self.xy))
            for start in range(0,len(self.xy),64):
                logits=-.5*epsilon_release*cdist(self.xy[start:start+64],rn.xy)
                self.log_normalizers[start:start+64]=logsumexp(logits,axis=1)
            if cache:
                cache.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(cache,metadata=self.metadata_json,state_ids=self.state_ids,
                                    log_normalizers=self.log_normalizers)
        if self.log_normalizers.shape!=(len(self.xy),) or not np.isfinite(self.log_normalizers).all():
            raise ValueError('Invalid REM normalizer cache')
        digest=hashlib.sha256(self.metadata_json.encode())
        digest.update(self.state_ids.astype('<i8').tobytes())
        digest.update(self.log_normalizers.astype('<f8').tobytes())
        self.sha256=digest.hexdigest()
        self.transitions=OrderedDict()
        self.poi_weights=self._poi_weights()

    def _poi_weights(self):
        rows,cols,values=[],[],[]
        for i,state in enumerate(self.state_ids):
            signatures=self.context.query_indices(state)
            nonempty=sum(np.any(s>=0) for s in signatures)
            for s in signatures:
                valid=s[s>=0]
                if len(valid):
                    rows.extend([i]*len(valid)); cols.extend(valid.tolist())
                    values.extend([1./(nonempty*len(valid))]*len(valid))
        return csr_matrix((values,(rows,cols)),shape=(len(self.xy),len(self.context.pois)+1))

    def emission(self,anchor,previous=None):
        """Probability of a coordinate output, not a hidden lane-state index."""
        state,multiplicity=self.coordinate_index[tuple(anchor)]
        dist=np.linalg.norm(self.xy-self.rn.xy[state],axis=1)
        fresh=np.exp(np.log(multiplicity)-.5*self.epsilon_release*dist-self.log_normalizers)
        if previous is None:
            return fresh
        last,_=self.coordinate_index[tuple(previous)]
        distance=np.linalg.norm(self.xy-self.rn.xy[last],axis=1)
        reuse=laplace.cdf(self.theta_m-distance,scale=1./self.epsilon_test)
        refresh=laplace.sf(self.theta_m-distance,scale=1./self.epsilon_test)
        return refresh*fresh + (reuse if tuple(anchor)==tuple(previous) else 0.)

    def predict(self,weights,dt):
        if not np.isfinite(dt) or dt<=0:
            raise ValueError('Positive elapsed public time required')
        if dt>120:
            # Long gaps do not justify a confident local-motion extrapolation.
            # Explicit approximate reset, not a directed mobility likelihood.
            persistence=.6**(dt/20.)
            return persistence*weights+(1-persistence)*self.prior
        return np.asarray(weights @ self.transition(dt)).ravel()

    def transition(self,dt):
        if not np.isfinite(dt) or not 0<dt<=120:
            raise ValueError('Local transition requires 0 < dt <= 120; use predict for longer gaps')
        dt=float(dt)
        if dt not in self.transitions:
            # A coarse latent model, NOT a proof of directed motion feasibility.
            # The output generator separately enforces actual directed paths.
            sigma=max(self.spacing_m,.5*8.*dt)
            radius=8.*dt+2*self.spacing_m
            neighbors=self.tree.query_ball_tree(self.tree,radius)
            rows,cols,values=[],[],[]
            for i,ids in enumerate(neighbors):
                d=np.linalg.norm(self.xy[ids]-self.xy[i],axis=1)
                w=np.exp(-.5*(d/sigma)**2)*self.prior[ids]
                if w.sum()==0:
                    ids,w=[i],np.ones(1)
                w/=w.sum()
                rows.extend([i]*len(ids)); cols.extend(ids); values.extend(w.tolist())
            move=csr_matrix((values,(rows,cols)),shape=(len(self.xy),len(self.xy)))
            self.transitions[dt]=.6*eye(len(self.xy),format='csr')+.4*move
            if len(self.transitions)>4:
                self.transitions.popitem(last=False)
        self.transitions.move_to_end(dt)
        return self.transitions[dt]


class AnchorBelief:
    def __init__(self,model):
        self.model=model
        self.weights=model.prior.copy()
        self.previous=None
        self.timestamp=None

    def update(self,anchor,timestamp,*,observed=True):
        if not np.isfinite(timestamp) or (self.timestamp is not None and timestamp<=self.timestamp):
            raise ValueError('Strictly increasing finite public times required')
        if self.timestamp is not None:
            self.weights=self.model.predict(self.weights,timestamp-self.timestamp)
        if observed:
            likelihood=self.model.emission(anchor,self.previous)
            updated=self.weights*likelihood
            if not np.isfinite(updated).all() or updated.sum()<=0:
                raise ValueError('Belief update lost all mass')
            self.weights=updated/updated.sum()
            self.previous=tuple(anchor)
        self.timestamp=timestamp
        return self.weights.copy()

    def poi_weights(self):
        return np.asarray(self.weights @ self.model.poi_weights).ravel()

    def mean_xy(self):
        return self.weights @ self.model.xy
