"""No-delay first-read protection with existing public phase allocation.

Only the initial public phase gets quarter epsilon. Later locations, including
an unforeseeable final location, retain the ordinary accounted Geo-I mechanism.
There is no future-route/close oracle and no separate destination guarantee.
"""
from dataclasses import replace

from benchmark.engines.origin_guard import OriginGuardProgressLaneDummy
from benchmark.engines.paced_guard import _PublicPacing
from benchmark.engines.slack_progress import SlackProgressCoverLaneDummy


class EndpointPhaseNoiseProgressLaneDummy(
        _PublicPacing, OriginGuardProgressLaneDummy, SlackProgressCoverLaneDummy):
    name = 'endpoint_phase_noise_progress_lane_dummy'

    def protect_run(self, points):
        run = super().protect_run(points)
        parameters = dict(run.transcript.public_parameters)
        parameters.update(publication_delay_s=0., head_suppression_s=0.,
            endpoint_policy='public_initial_phase_quarter_epsilon_then_regular_GeoI',
            endpoint_scope='initial_coordinates_and_accounted_session_noise_not_timing',
            destination_guard='ordinary_GeoI_no_future_end_window_detection')
        return replace(run, transcript=replace(run.transcript, public_parameters=parameters))
