"""Causal, zero-publication-delay endpoint-noise configuration of GeoI-Slack.

The end of an online trip is unknown. Consequently the tighter epsilon applies
to every protected read, not to a secretly detected last window. This is an
allocation candidate of the existing mechanism, not a new privacy primitive.
Session timing, account identifiers and transport metadata remain observable.
"""
from dataclasses import replace
import math

from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy


class EndpointNoiseProgressLaneDummy(PacedSlackProgressLaneDummy):
    name = 'endpoint_noise_progress_lane_dummy'

    def __init__(self, rn, *, privacy_scale=.5, budget=.24, **kwargs):
        if (isinstance(privacy_scale, bool) or not math.isfinite(privacy_scale)
                or not 0 < privacy_scale <= 1):
            raise ValueError('Public privacy_scale must lie in (0,1]')
        self.privacy_scale = float(privacy_scale)
        self.requested_budget = float(budget)
        # The parent's constructor verifies that the belief emission agrees
        # with these actual mechanism parameters. Do not silently reuse a
        # belief built at the original epsilon.
        super().__init__(rn, budget=budget*self.privacy_scale, **kwargs)

    def protect_run(self, points):
        run = super().protect_run(points)
        parameters = dict(run.transcript.public_parameters)
        parameters.update(
            requested_budget_per_m=self.requested_budget,
            endpoint_noise_scale=self.privacy_scale,
            publication_delay_s=0., head_suppression_s=0.,
            endpoint_policy='same_stronger_noise_every_read_unknown_online_close',
            endpoint_scope='coordinates_only_session_start_and_close_visible',
            future_private_gps_used=False,
        )
        return replace(run, transcript=replace(run.transcript, public_parameters=parameters))
