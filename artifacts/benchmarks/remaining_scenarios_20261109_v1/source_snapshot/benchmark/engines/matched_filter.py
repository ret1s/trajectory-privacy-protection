"""Match the fixed-H controller's tighter (2H-1)*B/(2H) bound exactly."""
from dataclasses import replace
from benchmark.engines.filtered_cover import FilteredCoverLaneDummy
from benchmark.engines.progress_cover import FilteredProgressCoverLaneDummy
from benchmark.engines.slack_progress import FilteredSlackProgressCoverLaneDummy

class _MatchFixedHBound:
    def reset(self):
        super().reset()
        # Keep test/fresh unit epsilon unchanged. Original fixed-H costs one
        # first release plus H-1 two-unit steps, leaving one nominal unit unused.
        self.max_units-=1

    def protect_run(self,points):
        run=super().protect_run(points)
        params=dict(run.transcript.public_parameters)
        params['tight_session_bound_per_m']=self.max_units*self.unit_epsilon
        params['matched_fixed_H_bound']=True
        return replace(run,transcript=replace(run.transcript,public_parameters=params))

class MatchedFilteredCoverLaneDummy(_MatchFixedHBound,FilteredCoverLaneDummy):
    name='matched_filtered_cover_lane_dummy'

class MatchedFilteredProgressCoverLaneDummy(_MatchFixedHBound,FilteredProgressCoverLaneDummy):
    name='matched_filtered_progress_cover_lane_dummy'

class MatchedFilteredSlackProgressCoverLaneDummy(_MatchFixedHBound,FilteredSlackProgressCoverLaneDummy):
    name='matched_filtered_slack_progress_cover_lane_dummy'
