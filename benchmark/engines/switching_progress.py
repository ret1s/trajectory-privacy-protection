"""Apply the new selector/filter to the existing stopped/moving belief.

This is an ablation integration, not a new mobility model. Privacy accounting
does not depend on calibration of this approximate belief.
"""
from benchmark.switching_belief import SwitchingAnchorBelief
from benchmark.engines.quotient_cover import QuotientCoverLaneDummy
from benchmark.engines.matched_filter import MatchedFilteredProgressCoverLaneDummy


class SwitchingQuotientCoverLaneDummy(QuotientCoverLaneDummy):
    name = 'switching_quotient_cover_lane_dummy'

    def reset(self):
        super().reset()
        self.belief = SwitchingAnchorBelief(self.belief_model)


class _FilteredSwitchingBelief(SwitchingAnchorBelief):
    def __init__(self, model, owner):
        super().__init__(model)
        self.owner = owner

    def update(self, anchor, timestamp, *, observed=True):
        return super().update(anchor, timestamp,
                              observed=self.owner.privacy_read_this_step)


class MatchedSwitchingProgressCoverLaneDummy(MatchedFilteredProgressCoverLaneDummy):
    name = 'matched_switching_progress_cover_lane_dummy'

    def reset(self):
        super().reset()
        self.belief = _FilteredSwitchingBelief(self.belief_model, self)
