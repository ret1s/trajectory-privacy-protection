"""Exact service-equivalence reduction for reachable POI coverage.

Within each track's feasible set, keep the best public tie-break representative
of each identical ordered POI signature. This preserves the greedy/exchange
selector, not just its utility value. It is a computational improvement, not a
new privacy guarantee or a change to the anchor mechanism.
"""
from dataclasses import replace
import numpy as np
from benchmark.engines.fair_cover import FairCoverLaneDummy, CoverageObjective, exchange_refine
from benchmark.engines.service_cover import greedy_cover


def reduce_groups(groups, profiles, tie_cost):
    """Preserve objective-equivalent states' minimum (primary, secondary, id)."""
    result=[]
    for j,group in enumerate(groups):
        ids=np.asarray(group,dtype=int)
        if ids.ndim!=1 or not len(ids):
            raise ValueError('Nonempty feasible groups required')
        primary,secondary=tie_cost(j,ids)
        if not np.isfinite(primary).all() or not np.isfinite(secondary).all():
            raise ValueError('Finite public tie costs required')
        order=np.lexsort((ids,secondary,primary,profiles[ids]))
        ordered=ids[order]; p=profiles[ordered]
        keep=np.r_[True,p[1:]!=p[:-1]]
        result.append(np.sort(ordered[keep]))
    return result


class QuotientCoverLaneDummy(FairCoverLaneDummy):
    name='quotient_cover_lane_dummy'

    def __init__(self,rn,**kwargs):
        super().__init__(rn,**kwargs)
        ctx=self.belief_model.context
        # Identical ordered signatures imply bitwise-identical marginal sums.
        # Set equality alone could reorder floats and change exact ties.
        _,ids=np.unique(ctx.signatures.reshape(len(ctx.signatures),-1),axis=0,return_inverse=True)
        self.service_profiles=ids[ctx.access]

    def postprocess(self,anchor,timestamp_s):
        belief=self.belief.update(anchor,timestamp_s,observed=self.n<self.horizon)
        weights=np.asarray(belief @ self.belief_model.poi_weights).ravel()
        context=self.belief_model.context
        objective=CoverageObjective(context.signatures,context.access,weights,
                                    self.category_cap,self.category_ids)
        groups=[]
        for j in range(self.k):
            if self.previous is None:ids=self.viable_ids
            else:
                reached=self.travel.reachable(self.previous[j],timestamp_s-self.last_t)
                ids=np.array(sorted(i for i in reached if self.viable[i]),dtype=int)
            groups.append(ids)
        def ties(j,ids):
            movement=(np.zeros(len(ids)) if self.previous is None else
                      np.linalg.norm(self.rn.xy[ids]-self.rn.xy[self.previous[j]],axis=1))
            return movement,np.linalg.norm(self.rn.xy[ids]-self.public_center,axis=1)
        reduced=reduce_groups(groups,self.service_profiles,ties)
        greedy,gains=greedy_cover(reduced,objective.marginal,ties)
        selected,history=exchange_refine(reduced,greedy,objective,ties,self.max_exchanges)
        self.evaluator_objective.append({'greedy_states':greedy,'greedy_gains':gains,
            'objective_history':history,'value':objective.value(selected),
            'category_values':objective.category_values(selected).tolist(),
            'category_mass':objective.mass.tolist(),'reachable_counts':list(map(len,groups)),
            'quotient_counts':list(map(len,reduced))})
        self.previous,self.last_t=selected,timestamp_s
        self.evaluator_states.append(list(selected))
        return tuple(self.rn.latlon(i) for i in selected)

    def protect_run(self,points):
        run=super().protect_run(points)
        params=dict(run.transcript.public_parameters)
        params['candidate_reduction']='exact_ordered_service_signature_with_public_tie_representative'
        return replace(run,transcript=replace(run.transcript,public_parameters=params))
