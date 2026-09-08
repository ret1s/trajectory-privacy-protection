"""Separate initial spatial mass from approximate motion/reset mass."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark.engines.service_cover import ServiceCoverLaneDummy


def cell_balanced_prior(rn, prior, spacing_m):
    prior = np.asarray(prior, dtype=float)
    if (prior.shape != (len(rn),) or not np.isfinite(prior).all()
            or np.any(prior < 0) or prior.sum() <= 0
            or not np.isfinite(spacing_m) or spacing_m <= 0):
        raise ValueError('Finite nonnegative prior and positive cell spacing required')
    _, inv, counts = np.unique(np.floor(rn.xy / spacing_m).astype(np.int64),
                              axis=0, return_inverse=True, return_counts=True)
    weights = prior / counts[inv]
    return weights / weights.sum()


class FactorizedAnchorModel:
    """Use initial's emission/grid/context but motion's prediction/reset.

    Immutable by convention. Neither wrapped model is mutated, allowing exact
    reproduction of the original (L,L) and (U,U) controls.
    """
    def __init__(self, initial, motion):
        for name in ('rn', 'context'):
            if getattr(initial, name) is not getattr(motion, name):
                raise ValueError('Identical public map/context required')
        for name in ('spacing_m', 'epsilon_release', 'epsilon_test', 'theta_m'):
            if getattr(initial, name) != getattr(motion, name):
                raise ValueError('Identical grid/emission parameters required')
        if (not np.array_equal(initial.state_ids, motion.state_ids)
                or not np.array_equal(initial.log_normalizers, motion.log_normalizers)):
            raise ValueError('Identical grid and emission normalizers required')
        self.initial, self.motion = initial, motion
        self.prior = initial.prior.copy()
        self.sha256 = hashlib.sha256(json.dumps({
            'schema': 'factorized-anchor-v1', 'initial': initial.sha256,
            'motion': motion.sha256,
            'source': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        }, sort_keys=True).encode()).hexdigest()

    def __getattr__(self, name):
        return getattr(self.initial, name)

    def predict(self, weights, dt):
        return self.motion.predict(weights, dt)

    def transition(self, dt):
        return self.motion.transition(dt)


class PriorFactorCover(ServiceCoverLaneDummy):
    name = 'prior_factor_cover'

    def protect_run(self, points):
        run = super().protect_run(points)
        params = dict(run.transcript.public_parameters)
        for key in ('offset_m', 'temperature_m', 'route_weight', 'coverage_weight', 'center_mode'):
            params.pop(key, None)
        model = self.belief_model
        params.update(initial_prior_model_sha256=model.initial.sha256,
                      motion_prior_model_sha256=model.motion.sha256,
                      belief_transition='dt<=120:0.6_stay_0.4_motion_prior_diffusion;dt>120:reset_towards_motion_prior',
                      coverage_target='approximate_protected_history_belief')
        return replace(run, transcript=replace(run.transcript, public_parameters=params))
