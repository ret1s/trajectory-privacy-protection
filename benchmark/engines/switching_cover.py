"""Causal switching-belief reachable mean-coverage candidate."""
from dataclasses import replace
import json

from benchmark.engines.fair_cover import FairCoverLaneDummy
from benchmark.switching_belief import SwitchingAnchorBelief


class SwitchingCoverLaneDummy(FairCoverLaneDummy):
    name = 'switching_cover_lane_dummy'

    def reset(self):
        super().reset()
        self.belief = SwitchingAnchorBelief(self.belief_model)

    def protect_run(self, points):
        run = super().protect_run(points)
        parameters = {**dict(run.transcript.public_parameters),
            'belief_transition': 'two_mode_finite_forward_filter_v1',
            'switching_belief': json.dumps(SwitchingAnchorBelief.parameters, sort_keys=True)}
        return replace(run, transcript=replace(run.transcript, public_parameters=parameters))
