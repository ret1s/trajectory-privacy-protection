"""Standard full-plane Laplace anchor with predictive private reuse.

An ablation of existing Geo-I primitives (CCS2013, PETS2014), not a new noise
mechanism. The metric is Euclidean distance in the fixed public XY projection.
Floating-point sampling is an approximation to the ideal continuous kernel.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.stats import laplace
from core.mechanisms import _require_positive_finite


class PredictivePlanarLaplace:
    name = 'predictive_planar_laplace'

    def __init__(self, rn, *, epsilon_release, epsilon_test, theta=200., rng=None):
        _require_positive_finite(epsilon_release, 'epsilon_release')
        _require_positive_finite(epsilon_test, 'epsilon_test')
        if not np.isfinite(theta) or theta < 0:
            raise ValueError('Public nonnegative finite reuse threshold required')
        self.rn, self.epsilon, self.eps_test = rn, float(epsilon_release), float(epsilon_test)
        self.theta = float(theta)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.privacy_cost_per_step_max = self.epsilon+self.eps_test
        _require_positive_finite(self.privacy_cost_per_step_max, 'step privacy cap')
        self.reset()

    def reset(self):
        self.previous = None
        self.n_test = self.n_resample = 0

    def perturb(self, lat, lon, t=None):
        if not np.isfinite([lat, lon]).all() or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError('Valid private coordinates required')
        xy = np.asarray(self.rn.point_xy(lat, lon))
        self.n_test += 1
        if self.previous is not None:
            distance = np.linalg.norm(xy-self.rn.point_xy(*self.previous))
            if distance+self.rng.laplace(scale=1/self.eps_test) <= self.theta:
                return self.previous
        radius = self.rng.gamma(shape=2., scale=1/self.epsilon)
        angle = self.rng.uniform(0., 2*np.pi)
        noisy = xy+radius*np.array([np.cos(angle), np.sin(angle)])
        self.previous = tuple(map(float, self.rn.proj.to_latlon(*noisy)))
        self.n_resample += 1
        return self.previous


@dataclass(frozen=True)
class PlanarAnchorModel:
    """Same public latent approximation, emission matched to the continuous kernel."""
    base: object

    @property
    def sha256(self):
        return hashlib.sha256(('full-plane-predictive-laplace-v1/'+self.base.sha256).encode()).hexdigest()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def emission(self, anchor, previous=None):
        xy = np.asarray(self.rn.point_xy(*anchor))
        distance = np.linalg.norm(self.xy-xy, axis=1)
        fresh = self.epsilon_release**2/(2*np.pi)*np.exp(-self.epsilon_release*distance)
        if previous is None:
            return fresh
        old_distance = np.linalg.norm(self.xy-self.rn.point_xy(*previous), axis=1)
        threshold = self.theta_m-old_distance
        if tuple(anchor) == tuple(previous):
            # The continuous fresh draw has zero mass at this exact point.
            # Do NOT add a density to an atom or import REM's collision term.
            return laplace.cdf(threshold, scale=1/self.epsilon_test)
        return laplace.sf(threshold, scale=1/self.epsilon_test)*fresh
