"""Service-equivalent motion towards a public protected-belief cover plan.

Preserves each step's selected POI signatures conditional on that step's history;
future performance and empirical privacy require evaluation. No trajectory-wide
utility dominance or planning optimality is claimed.
"""
from collections import OrderedDict
from dataclasses import replace
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import dijkstra
from benchmark.engines.quotient_cover import QuotientCoverLaneDummy,reduce_groups
from benchmark.engines.filtered_cover import FilteredCoverLaneDummy
from benchmark.engines.fair_cover import CoverageObjective,exchange_refine
from benchmark.engines.service_cover import greedy_cover
from evaluation.lane_travel import matrix

class ProgressCoverLaneDummy(QuotientCoverLaneDummy):
    name='progress_cover_lane_dummy'
    def __init__(self,rn,**kwargs):
        super().__init__(rn,**kwargs)
        self.progress_reverse=matrix(rn,time=True).transpose().tocsr()
        self.progress_cache=OrderedDict()

    def _to_goal(self,goal):
        goal=int(goal)
        if goal not in self.progress_cache:
            self.progress_cache[goal]=dijkstra(self.progress_reverse,directed=True,indices=goal)
            if len(self.progress_cache)>32:self.progress_cache.popitem(last=False)
        self.progress_cache.move_to_end(goal)
        return self.progress_cache[goal]

    def postprocess(self,anchor,timestamp_s):
        previous=self.previous;previous_t=self.last_t
        original=super().postprocess(anchor,timestamp_s)
        if previous is None:return original
        base=list(self.previous)
        weights=np.asarray(self.belief.weights @ self.belief_model.poi_weights).ravel()
        context=self.belief_model.context
        objective=CoverageObjective(context.signatures,context.access,weights,self.category_cap,self.category_ids)
        center=self.belief.weights @ self.belief_model.xy
        def global_ties(j,ids):
            return np.linalg.norm(self.rn.xy[ids]-center,axis=1),np.zeros(len(ids))
        distinct=reduce_groups([self.viable_ids],self.service_profiles,global_ties)[0]
        groups=[distinct]*self.k
        goals,_=greedy_cover(groups,objective.marginal,global_ties)
        goals,_=exchange_refine(groups,goals,objective,global_ties,self.max_exchanges)
        distances=[self._to_goal(goal) for goal in goals]
        costs=np.array([[d[state] for d in distances] for state in previous])
        rr,cc=linear_sum_assignment(costs)
        assigned=dict(zip(rr,cc));selected=[]
        for j,(prior,selected_base) in enumerate(zip(previous,base)):
            reached=self.travel.reachable(prior,timestamp_s-previous_t)
            ids=np.array(sorted(i for i in reached if self.viable[i] and
                self.service_profiles[i]==self.service_profiles[selected_base]),dtype=int)
            assert len(ids)>0
            d=distances[assigned[j]][ids]
            movement=np.linalg.norm(self.rn.xy[ids]-self.rn.xy[prior],axis=1)
            selected.append(int(ids[np.lexsort((ids,movement,d))[0]]))
        assert np.array_equal(context.signatures[context.access[base]],context.signatures[context.access[selected]])
        self.evaluator_objective[-1].update(pre_progress_states=base,progress_states=selected,
            progress_goals=[goals[assigned[j]] for j in range(self.k)],
            progress_replaced=sum(a!=b for a,b in zip(base,selected)),same_step_service_preserved=True)
        self.previous=selected
        self.evaluator_states[-1]=list(selected)
        return tuple(self.rn.latlon(i) for i in selected)

    def protect_run(self,points):
        run=super().protect_run(points)
        params=dict(run.transcript.public_parameters)
        params['service_equivalent_motion']='directed_progress_to_protected_belief_global_cover'
        return replace(run,transcript=replace(run.transcript,public_parameters=params))

class FilteredProgressCoverLaneDummy(FilteredCoverLaneDummy,ProgressCoverLaneDummy):
    name='filtered_progress_cover_lane_dummy'
