"""Enhanced-DLS (Niu et al., INFOCOM 2014), explicit road-catalogue adaptation.

Section IV-B / Algorithm 2: probability-neighbor redundancy, entropy screening,
then sequential sampling proportional to products of distances (NOT greedy
farthest-point selection). The paper's intermediate-set convention is not fully
explicit about including truth: we use 2K dummy candidates, scoring their union
with truth. This choice, tie handling and m must remain disclosed.
"""
from dataclasses import replace

import numpy as np

from benchmark.engines.dls import DLSGraph, entropy


def distance_product_probabilities(xy,candidates,chosen):
    distances=np.linalg.norm(xy[np.asarray(candidates)][:,None,:]-xy[np.asarray(chosen)][None,:,:],axis=2)
    with np.errstate(divide='ignore'):
        logits=np.log(distances).sum(axis=1)
    if not np.isfinite(logits).any():
        # Coincident catalogue states can occur at junction boundaries.
        # Explicit local degenerate case; never pretend spatial spread exists.
        return np.full(len(candidates),1/len(candidates))
    weights=np.exp(logits-np.max(logits))
    return weights/weights.sum()


class EnhancedDLSGraph(DLSGraph):
    name='enhanced_dls_graph_adaptation'

    def __init__(self,rn,probabilities,**kwargs):
        super().__init__(rn,probabilities,**kwargs)
        if len(rn)<4*self.k+1:
            raise ValueError('Enhanced-DLS requires at least 4K+1 catalogue states')
        self.screen=DLSGraph(rn,self.q,k=2*self.k,subset_trials=self.subset_trials,rng=self.rng)

    def redundant_candidates(self,true_index):
        pool=self.screen.candidate_pool(true_index)  # 4K neighbors, excluding truth
        best,best_entropy=None,-np.inf
        for _ in range(self.subset_trials):
            candidates=self.rng.choice(pool,2*self.k,replace=False)
            score=entropy(self.q[np.r_[true_index,candidates]])
            if score>best_entropy:
                best,best_entropy=candidates,score
        return list(map(int,best))

    def select(self,true_index):
        remaining=self.redundant_candidates(true_index)
        selected=[int(true_index)]
        for _ in range(self.k-1):
            p=distance_product_probabilities(self.rn.xy,remaining,selected)
            chosen=int(self.rng.choice(remaining,p=p))
            selected.append(chosen)
            remaining.remove(chosen)
        return np.asarray(selected),entropy(self.q[selected])

    def protect_run(self,points):
        run=super().protect_run(points)
        return replace(run,transcript=replace(run.transcript,public_parameters={**dict(run.transcript.public_parameters),
            'algorithm':'Niu2014_Algorithm2_road_adaptation',
            'redundant_dummy_count':2*self.k,'probability_neighbor_count':4*self.k,
            'intermediate_entropy':'2K_dummies_union_truth',
            'spatial_selection':'sequential_distance_product_sampling'}))
