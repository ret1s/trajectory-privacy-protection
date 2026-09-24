"""Bounded loss of approximate current cover to escape service-region plateaus.

The bound concerns the protected-belief objective, not actual user Recall. It
uses no extra private data and therefore adds no privacy expenditure.
"""
from dataclasses import replace
import numpy as np
from benchmark.engines.progress_cover import ProgressCoverLaneDummy
from benchmark.engines.filtered_cover import FilteredCoverLaneDummy
from benchmark.engines.fair_cover import CoverageObjective

class SlackProgressCoverLaneDummy(ProgressCoverLaneDummy):
    name='slack_progress_cover_lane_dummy'
    def __init__(self,rn,*,utility_slack=.01,**kwargs):
        if not np.isfinite(utility_slack) or not 0<=utility_slack<=.1:
            raise ValueError('Public utility slack must lie in [0,.1]')
        self.utility_slack=float(utility_slack)
        super().__init__(rn,**kwargs)
        if self.category_cap is not None:raise ValueError('Slack currently defined for the mean coverage objective')

    def postprocess(self,anchor,timestamp_s):
        previous=self.previous;previous_t=self.last_t
        original=super().postprocess(anchor,timestamp_s)
        if previous is None or self.utility_slack==0:return original
        selected=list(self.previous);before=list(selected)
        context=self.belief_model.context
        weights=np.asarray(self.belief.weights @ self.belief_model.poi_weights).ravel()
        objective=CoverageObjective(context.signatures,context.access,weights,None,self.category_ids)
        base_value=objective.value(selected);floor=base_value-self.utility_slack
        goals=self.evaluator_objective[-1]['progress_goals']
        for j,prior in enumerate(previous):
            reached=self.travel.reachable(prior,timestamp_s-previous_t)
            ids=np.array(sorted(i for i in reached if self.viable[i]),dtype=int)
            others=selected[:j]+selected[j+1:]
            values=objective.value(others)+objective.marginal(ids,others)
            ids=ids[values>=floor-1e-12]
            assert len(ids)>0
            d=self._to_goal(goals[j])
            movement=np.linalg.norm(self.rn.xy[ids]-self.rn.xy[prior],axis=1)
            proposal=int(ids[np.lexsort((ids,movement,d[ids]))[0]])
            if d[proposal]<d[selected[j]]-1e-12:
                candidate=selected.copy();candidate[j]=proposal
                if objective.value(candidate)>=floor-1e-12:selected=candidate
        value=objective.value(selected)
        assert value>=floor-1e-12
        self.evaluator_objective[-1].update(slack_states_before=before,slack_states_after=selected,
            objective_before_slack=base_value,objective_after_slack=value,
            objective_loss=base_value-value,utility_slack=self.utility_slack,
            same_step_service_preserved=bool(np.array_equal(context.signatures[context.access[before]],context.signatures[context.access[selected]])))
        self.previous=selected;self.evaluator_states[-1]=list(selected)
        return tuple(self.rn.latlon(i) for i in selected)

    def protect_run(self,points):
        run=super().protect_run(points);params=dict(run.transcript.public_parameters)
        params['protected_belief_utility_slack']=self.utility_slack
        return replace(run,transcript=replace(run.transcript,public_parameters=params))

class FilteredSlackProgressCoverLaneDummy(FilteredCoverLaneDummy,SlackProgressCoverLaneDummy):
    name='filtered_slack_progress_cover_lane_dummy'
