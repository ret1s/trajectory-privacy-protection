"""Isolate anchor-kernel choice while retaining the paced reachable planner."""
from dataclasses import replace
from benchmark.planar_anchor import PredictivePlanarLaplace, PlanarAnchorModel
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy


class PlanarPacedLaneDummy(PacedSlackProgressLaneDummy):
    name = 'planar_paced_lane_dummy'

    def __init__(self, rn, *, belief_model, **kwargs):
        if not isinstance(belief_model, PlanarAnchorModel):
            raise ValueError('Matching continuous planar emission required')
        super().__init__(rn, belief_model=belief_model, **kwargs)

    def reset(self):
        super().reset()
        self.anchor = PredictivePlanarLaplace(self.rn,
            epsilon_release=self.unit_epsilon, epsilon_test=self.unit_epsilon,
            theta=self.theta_m, rng=self.anchor_rng)

    def protect_run(self, points):
        run = super().protect_run(points)
        params = dict(run.transcript.public_parameters)
        params.update(anchor_kernel='full_plane_planar_laplace_with_private_reuse',
                      support='full_plane_anchors__public_largest_SCC_dummies',
                      privacy_metric='Euclidean_in_fixed_public_XY_projection',
                      kernel_claim='ideal_continuous_kernel_only_float_sampler_is_approximate',
                      comparison_scope='primitive_ablation_not_full_CCS2013_or_PETS2014_reproduction')
        return replace(run, transcript=replace(run.transcript, public_parameters=params))
