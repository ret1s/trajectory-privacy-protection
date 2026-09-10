"""Two-mode finite forward filter over already-private anchors only.

This is approximate public mobility inference, not a calibrated traffic model.
The fixed constants are frozen in fresh_switching_protocol.md. No raw speed,
trajectory labels or future observations are accepted by this module.
"""
import numpy as np
from scipy.sparse import eye

from benchmark.anchor_belief import AnchorBelief


class SwitchingAnchorBelief(AnchorBelief):
    parameters = {'mode_names': ['stopped', 'moving'], 'mode_time_constant_s': 100.,
                  'spatial_self_weights': [.95, .10], 'long_gap_s': 120.}

    def __init__(self, model):
        super().__init__(model)
        self.joint = np.stack([.5*self.weights, .5*self.weights])

    def update(self, anchor, timestamp, *, observed=True):
        if not np.isfinite(timestamp) or (self.timestamp is not None and timestamp <= self.timestamp):
            raise ValueError('Strictly increasing finite public times required')
        if self.timestamp is not None:
            dt = timestamp-self.timestamp
            stay = .5*(1+np.exp(-dt/100.))
            mixed = np.array([[stay, 1-stay], [1-stay, stay]]).T @ self.joint
            if dt > 120:
                # model.predict expects unit mass; preserve each mode's mass.
                predicted = np.array([self.model.predict(w/w.sum(), dt)*w.sum()
                    if w.sum() > 0 else w for w in mixed])
            else:
                diffusion = (self.model.transition(dt)-.6*eye(len(self.weights), format='csr'))/.4
                if diffusion.data.size and diffusion.data.min() < -1e-12:
                    raise ValueError('Unexpected negative public transition')
                diffusion.data = np.maximum(diffusion.data, 0.)
                predicted = np.array([a*w+(1-a)*np.asarray(w @ diffusion).ravel()
                    for a, w in zip((.95, .10), mixed)])
            self.joint = predicted/self._mass(predicted)
        if observed:
            updated = self.joint*self.model.emission(anchor, self.previous)[None, :]
            self.joint = updated/self._mass(updated)
            self.previous = tuple(anchor)
        self.weights = self.joint.sum(axis=0)
        self.timestamp = timestamp
        return self.weights.copy()

    @staticmethod
    def _mass(value):
        if not np.isfinite(value).all() or np.any(value < 0) or value.sum() <= 0:
            raise ValueError('Invalid joint belief mass')
        return value.sum()

    @property
    def mode_probabilities(self):
        return self.joint.sum(axis=1).copy()
