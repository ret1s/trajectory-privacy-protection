"""Public-time pacing with the existing bounded-loss protected-goal planner."""
from benchmark.engines.paced_guard import _PublicPacing
from benchmark.engines.matched_filter import MatchedFilteredSlackProgressCoverLaneDummy


class PacedSlackProgressLaneDummy(_PublicPacing, MatchedFilteredSlackProgressCoverLaneDummy):
    name = 'paced_slack_progress_lane_dummy'
