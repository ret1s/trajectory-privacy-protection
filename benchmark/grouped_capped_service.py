"""Group identical public reference-utility rows before nonlinear saturation.

F=sum_i b_i phi(W_i covered) equals sum_g (sum_{i in g} b_i)
phi(W_g covered). No low-probability location is dropped. This is equality in
real arithmetic; floating-point reassociation can affect exact selector ties.
"""
import numpy as np
from dataclasses import replace
from scipy.sparse import csr_matrix
from benchmark.capped_service_objective import CappedServiceIndex, CappedServiceObjective
from benchmark.engines.capped_service import CappedServiceLaneDummy


class GroupedCappedServiceIndex(CappedServiceIndex):
    def __init__(self, signatures, access, reference_weights):
        reference = csr_matrix(reference_weights, dtype=float)
        self.original_latent_count = reference.shape[0]
        unique, self.latent_groups = np.unique(reference.toarray(), axis=0, return_inverse=True)
        super().__init__(signatures, access, csr_matrix(unique))


class GroupedCappedServiceObjective(CappedServiceObjective):
    def __init__(self, index, belief, cap=.9):
        b = np.asarray(belief, dtype=float)
        if b.shape != (index.original_latent_count,) or not np.isfinite(b).all() or np.any(b < 0):
            raise ValueError('Finite nonnegative belief over original latent states required')
        grouped = np.bincount(index.latent_groups, weights=b, minlength=index.reference.shape[0])
        super().__init__(index, grouped, cap)


class GroupedCappedServiceLaneDummy(CappedServiceLaneDummy):
    name = 'grouped_capped_service_lane_dummy'
    index_class = GroupedCappedServiceIndex
    objective_class = GroupedCappedServiceObjective

    def protect_run(self, points):
        run = super().protect_run(points); params = dict(run.transcript.public_parameters)
        if self.capped_index is not None:
            params.update(reference_utility_groups=self.capped_index.reference.shape[0],
                          original_latent_states=self.capped_index.original_latent_count,
                          belief_mass_pruned=False)
        return replace(run, transcript=replace(run.transcript, public_parameters=params))
