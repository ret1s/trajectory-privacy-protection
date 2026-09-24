"""Public auxiliary mobility in the existing response-aware paced planner."""
from dataclasses import replace
import json
from benchmark.engines.paced_guard import PacedProgressLaneDummy
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy


class _EmpiricalMetadata:
    def __init__(self, rn, *, belief_model, **kwargs):
        if not hasattr(belief_model, 'mobility_metadata'):
            raise ValueError('An explicit auxiliary mobility model is required')
        super().__init__(rn, belief_model=belief_model, **kwargs)

    def protect_run(self, points):
        run = super().protect_run(points)
        params = dict(run.transcript.public_parameters)
        params.update(belief_transition='auxiliary_fitted_smoothed_continuous_time_grid_Markov',
                      mobility_model_sha256=self.belief_model.sha256,
                      mobility_training_json=json.dumps(self.belief_model.mobility_metadata, sort_keys=True))
        return replace(run, transcript=replace(run.transcript, public_parameters=params))


class EmpiricalPacedProgressLaneDummy(_EmpiricalMetadata, PacedProgressLaneDummy):
    name = 'empirical_paced_progress_lane_dummy'


class EmpiricalPacedSlackProgressLaneDummy(_EmpiricalMetadata, PacedSlackProgressLaneDummy):
    name = 'empirical_paced_slack_progress_lane_dummy'
