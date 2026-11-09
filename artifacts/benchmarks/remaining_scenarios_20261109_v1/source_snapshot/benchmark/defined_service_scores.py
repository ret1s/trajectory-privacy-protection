"""Public-case normalization BEFORE protected belief, never by private intent.

Undefined purposes/cases remain N/A. Defined public references carry unit mass
within each public state, then the fixed state prior is mixed. This is linear
in belief, unlike normalizing a denominator after mixing private belief.
"""
import numpy as np
from benchmark.public_segment_scores import FixedPublicPurposeTable


class DefinedPublicPurposeTable(FixedPublicPurposeTable):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.defined=self.validity>0
        self.defined_count=self.defined.sum(axis=1)
        if np.any(self.defined_count==0):raise ValueError('Public state has no defined service reference')
        weights=self.weights.copy()
        for x,p in zip(*np.where(self.defined)):
            weights[x,p]/=self.validity[x,p]
        self.weights=weights
        self.weights.flags.writeable=False
        self.public_poi_weights=np.mean(self.weights.sum(axis=1)/self.defined_count[:,None],axis=0)
        self.public_poi_weights/=self.public_poi_weights.sum()
        self.public_poi_weights.flags.writeable=False

    def state_frame_scores(self,frame):
        return self.frame_table(frame).sum(axis=1)/self.defined_count

    def __call__(self,actions):
        return np.clip(np.column_stack([np.mean([self.state_frame_scores(f) for f in a.frames],axis=0) for a in actions]),0.,1.)

    def floor_table(self,actions):
        # State-average defined-purpose score. A separate reference certificate
        # is required for one actual purpose; don't call this a per-purpose floor.
        return np.clip(np.column_stack([np.min([self.state_frame_scores(f) for f in a.frames],axis=0) for a in actions]),0.,1.)
