"""Fixed-cap privacy filter over an extended private-test/anchor transcript.

Predictive budget management is established prior work (Chatzikokolakis et al.,
PETS 2014). This separate candidate spends integer units prospectively. It does
NOT reinterpret old worst-case-ledger results using diagnostic resample counts.
See docs/research/predictive_filter_argument.md for the ideal-kernel argument.
"""
from dataclasses import replace
import math,time
from benchmark.anchor_belief import AnchorBelief
from benchmark.engines.quotient_cover import QuotientCoverLaneDummy

class _FilterBelief(AnchorBelief):
    def __init__(self,model,owner):
        super().__init__(model);self.owner=owner
    def update(self,anchor,timestamp,*,observed=True):
        # Fixed emission parameters still match the primitive. This approximate
        # filter does not model all information in the internal branch history.
        return super().update(anchor,timestamp,observed=self.owner.privacy_read_this_step)

class FilteredCoverLaneDummy(QuotientCoverLaneDummy):
    name='filtered_cover_lane_dummy'

    def reset(self):
        super().reset()
        self.spent_units=0
        self.max_units=2*self.horizon
        self.unit_epsilon=self.budget/self.max_units
        self.privacy_read_this_step=False
        self.evaluator_ledger=[]
        self.belief=_FilterBelief(self.belief_model,self)

    def protect_step(self,lat,lon,timestamp_s):
        started=time.perf_counter()
        if not math.isfinite(timestamp_s) or (self.last_t is not None and timestamp_s<=self.last_t):
            raise ValueError('Strictly increasing finite public times required')
        first=self.last_anchor is None
        reserve=1 if first else 2
        self.privacy_read_this_step=self.spent_units+reserve<=self.max_units
        cost=0;branch='postprocess'
        if self.privacy_read_this_step:
            if not math.isfinite(lat) or not math.isfinite(lon) or not -90<=lat<=90 or not -180<=lon<=180:
                raise ValueError('Valid private coordinates required when read')
            before=self.anchor.n_resample
            self.last_anchor=self.anchor.perturb(lat,lon,t=timestamp_s)
            fresh=self.anchor.n_resample>before
            cost=(0 if first else 1)+int(fresh)
            if first and not fresh:raise RuntimeError('First anchor must be fresh')
            self.spent_units+=cost
            branch='fresh' if fresh else 'reuse'
        assert self.spent_units<=self.max_units
        self.spent_bound=self.spent_units*self.unit_epsilon
        output=self.postprocess(self.last_anchor,timestamp_s)
        self.evaluator_anchors.append(list(self.last_anchor))
        self.evaluator_ledger.append({'cost_units':cost,'spent_units':self.spent_units,
                                     'branch':branch,'private_read':self.privacy_read_this_step})
        self.n+=1
        self.step_ms.append((time.perf_counter()-started)*1000)
        return output

    def protect_run(self,points):
        run=super().protect_run(points)
        params=dict(run.transcript.public_parameters)
        params.pop('horizon_events',None)
        params.pop('after_horizon',None)
        params.update(budget_filter='reserve_worst_next_step_then_charge_extended_transcript_branch',
                      privacy_unit_epsilon=self.unit_epsilon,max_privacy_units=self.max_units,
                      after_filter_stop='public_postprocessing_no_new_private_read',
                      ideal_guarantee='fixed_clock_B_times_D_infinity_extended_transcript',
                      branch_history_public=False)
        return replace(run,transcript=replace(run.transcript,public_parameters=params))
