"""Belief-weighted public POI coverage as BR-lane postprocessing."""
from dataclasses import replace

import numpy as np

from benchmark.anchor_belief import AnchorBelief
from benchmark.engines.contextual_lane import ContextualLaneDummy


class _BeliefContext:
    def __init__(self,base,owner):
        self.base,self.owner=base,owner
        self.rn,self.sha256=base.rn,base.sha256

    def reference_weights(self,anchor):
        return self.owner.belief.poi_weights()

    def marginal_gain(self,states,weights,selected):
        return self.base.marginal_gain(states,weights,selected)


class BeliefLaneDummy(ContextualLaneDummy):
    name='belief_lane_dummy'

    def __init__(self,rn,*,belief_model,center_mode='anchor',**kwargs):
        if belief_model.rn is not rn or center_mode not in {'anchor','belief_mean'}:
            raise ValueError('Matching belief context and known center mode required')
        self.belief_model,self.center_mode=belief_model,center_mode
        context=_BeliefContext(belief_model.context,self)
        super().__init__(rn,context=context,**kwargs)
        cap=self.budget/self.horizon
        if self.anchor_mode!='private_reuse' or not np.isclose(cap/2,belief_model.epsilon_release,rtol=1e-12,atol=0) or not np.isclose(cap/2,belief_model.epsilon_test,rtol=1e-12,atol=0) or self.theta_m!=belief_model.theta_m:
            raise ValueError('Belief emission must match the actual private-anchor mechanism')

    def reset(self):
        super().reset()
        self.belief=AnchorBelief(self.belief_model)
        self.evaluator_belief=[]

    def postprocess(self,anchor,timestamp_s):
        weights=self.belief.update(anchor,timestamp_s,observed=self.n<self.horizon)
        center=self.belief.mean_xy()
        self.evaluator_belief.append({'mean_xy':center.tolist(),
            'entropy_nats':float(-np.sum(weights[weights>0]*np.log(weights[weights>0]))),
            'effective_states':float(1/np.sum(weights**2))})
        target=tuple(float(v) for v in self.rn.proj.to_latlon(*center)) if self.center_mode=='belief_mean' else anchor
        return super().postprocess(target,timestamp_s)

    def protect_run(self,points):
        run=super().protect_run(points)
        params={**dict(run.transcript.public_parameters),'belief_model_sha256':self.belief_model.sha256,
                'belief_grid_spacing_m':self.belief_model.spacing_m,'center_mode':self.center_mode,
                'coverage_target':'expected_POI_recall_under_approximate_protected_history_belief',
                'belief_transition':'dt<=120:0.6_stay_0.4_public_grid_diffusion;dt>120:0.6**(dt/20)_persistence_plus_training_prior'}
        return replace(run,transcript=replace(run.transcript,public_parameters=params))
