"""Auxiliary-fitted continuous-time mobility as protected-history postprocessing.

The state grid, spatial prior, anchor kernel and service weights are delegated
unchanged. Training must be public/authorized auxiliary data, never the private
trajectory being protected. This approximate grid model does not prove that
latent paths follow lane geometry; output feasibility is enforced separately.
"""
import hashlib
import json
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import expm_multiply


def smoothed_generator(counts, exposures, prior_transition, pseudo_exposure_s):
    counts = csr_matrix(counts, dtype=float, copy=True)
    exposures = np.asarray(exposures, dtype=float)
    prior = csr_matrix(prior_transition, dtype=float, copy=True)
    n = counts.shape[0]
    if (counts.shape != (n, n) or prior.shape != counts.shape or exposures.shape != (n,)
            or not np.isfinite(exposures).all() or np.any(exposures < 0)
            or not np.isfinite(counts.data).all() or np.any(counts.data < 0)
            or not np.isfinite(prior.data).all() or np.any(prior.data < 0)
            or not np.allclose(np.asarray(prior.sum(axis=1)).ravel(), 1., atol=1e-12)
            or not np.isfinite(pseudo_exposure_s) or pseudo_exposure_s <= 0):
        raise ValueError('Finite nonnegative counts/exposures and stochastic public P20 required')
    counts.setdiag(0.); counts.eliminate_zeros()
    prior.setdiag(0.); prior.eliminate_zeros()
    rates = diags(1./(exposures+pseudo_exposure_s)) @ (counts+pseudo_exposure_s*prior/20.)
    return (rates-diags(np.asarray(rates.sum(axis=1)).ravel())).tocsr()


class EmpiricalMobilityModel:
    def __init__(self, base, generator, metadata):
        self.base = base
        self.generator = csr_matrix(generator, dtype=float, copy=True)
        self.generator.sort_indices()
        n = len(base.xy)
        off = self.generator.copy(); off.setdiag(0.); off.eliminate_zeros()
        if (self.generator.shape != (n, n) or not np.isfinite(self.generator.data).all()
                or np.any(off.data < 0) or np.any(self.generator.diagonal() > 1e-12)
                or not np.allclose(np.asarray(self.generator.sum(axis=1)).ravel(), 0., atol=1e-10)):
            raise ValueError('Valid finite Markov generator on the fixed public grid required')
        self.mobility_metadata = json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False))
        self.generator_transpose = self.generator.transpose().tocsr()
        self.trace = float(self.generator.diagonal().sum())
        digest = hashlib.sha256(base.sha256.encode())
        digest.update(json.dumps(self.mobility_metadata, sort_keys=True).encode())
        for array, dtype in ((self.generator.data, '<f8'), (self.generator.indices, '<i8'),
                             (self.generator.indptr, '<i8')):
            digest.update(array.astype(dtype).tobytes())
        self.sha256 = digest.hexdigest()

    def __getattr__(self, name):
        return getattr(self.base, name)

    def predict(self, weights, dt):
        weights = np.asarray(weights, dtype=float)
        if (weights.shape != (len(self.xy),) or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1., atol=1e-10)
                or not np.isfinite(dt) or dt <= 0):
            raise ValueError('Normalized protected belief and positive finite public time required')
        result = np.asarray(expm_multiply(self.generator_transpose*dt, weights,
                                         traceA=self.trace*dt)).ravel()
        if not np.isfinite(result).all() or result.min() < -1e-10 or result.sum() <= 0:
            raise RuntimeError('Invalid numerical continuous-time prediction')
        result = np.maximum(result, 0.)
        return result/result.sum()
