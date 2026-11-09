"""Separate the client's top-k target from the server's deeper reply support.

This public view changes the planner objective, never the private mechanism or
filter dynamics. It is valid only when the service really returns this depth.
"""
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class ResponseAwareAnchorModel:
    base: object
    context: object

    def __post_init__(self):
        reference = self.base.context
        if (self.context.rn is not self.base.rn
                or self.context.categories != reference.categories
                or self.context.pois != reference.pois
                or self.context.k < reference.k
                or not np.array_equal(self.context.access, reference.access)
                or not np.array_equal(self.context.signatures[:, :, :reference.k], reference.signatures)):
            raise ValueError('Reply context must extend the same ordered reference service')

    @property
    def sha256(self):
        return hashlib.sha256(('response-aware-v1/'+self.base.sha256+'/'+self.context.sha256).encode()).hexdigest()

    @property
    def poi_weights(self):
        # Deliberately keep expected membership in the client's reference top-k.
        # Recomputing these on top-L would silently change the utility target.
        return self.base.poi_weights

    def __getattr__(self, name):
        return getattr(self.base, name)
