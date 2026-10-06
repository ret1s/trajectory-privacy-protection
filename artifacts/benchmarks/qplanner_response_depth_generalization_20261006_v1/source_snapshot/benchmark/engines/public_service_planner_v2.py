"""Additive normalized planner controls using the unchanged Geo-I inheritance."""
from dataclasses import replace

from benchmark.engines.public_service_planner import PublicPurposePacedLaneDummy, make_service_planner_engine
from benchmark.public_service_profiles import PublicProfileAnchorModel
from benchmark.public_service_profiles_v2 import PurposeNormalizedServiceProfiles


V2_MODES=('legacy_l10','aligned_nearest','normalized_mean','normalized_tight','normalized_tail')


class PurposeNormalizedPacedLaneDummy(PublicPurposePacedLaneDummy):
    name='purpose_normalized_paced_lane_dummy'

    def __init__(self,rn,*,normalization_mode,**kwargs):
        self.normalization_mode=normalization_mode
        super().__init__(rn,**kwargs)

    def postprocess(self,anchor,timestamp_s):
        output=super().postprocess(anchor,timestamp_s)
        _,diagnostics=self.belief_model.profiles.normalized_mass(self.belief.weights)
        self.evaluator_objective[-1].update(normalization_diagnostics=diagnostics,
                                           normalization_mode=self.normalization_mode)
        return output

    def protect_run(self,points):
        run=super().protect_run(points);params=dict(run.transcript.public_parameters)
        params.update(planner_mode=self.normalization_mode,profile_normalization_schema='purpose-normalized-v2',
            normalization='valid_categories_per_case_then_valid_cases_per_purpose_then_equal_defined_purposes',
            destination_prior='public_uniform_before_valid_case_conditioning',
            contribution_scope='public/protected_Q_postprocessing_only; no privacy_equivalence_or_trajectory_utility_claim')
        return replace(run,transcript=replace(run.transcript,public_parameters=params))


def make_service_planner_engine_v2(mode,rn,base,reply20,profiles_v2=None,*,legacy_context=None,
                                  risk_weight=None,tail_mass=.25,mean_slack=.01,max_risk_exchanges=3,
                                  **old_engine_kwargs):
    """Fixed public arms: mean slack.03, tight slack0, tail slack0/lambda.25.

Control modes delegate the sealed factory. Normalized modes share its exact
Geo-I/private filter/pacing and motion algorithm; only profile weighting and
the explicitly named public motion/risk setting differ.
    """
    if mode not in V2_MODES or rn is not base.rn or rn is not reply20.rn:
        raise ValueError('Known normalized/control mode and same public map required')
    if mode in V2_MODES[:2]:
        return make_service_planner_engine(mode,rn,base,reply20,legacy_context=legacy_context,**old_engine_kwargs)
    if not isinstance(profiles_v2,PurposeNormalizedServiceProfiles) or profiles_v2.context is not reply20:
        raise ValueError('Matched purpose-normalized public profiles required')
    declared_slack=.03 if mode=='normalized_mean' else 0.
    declared_risk=.25 if mode=='normalized_tail' else 0.
    if risk_weight is None:risk_weight=declared_risk
    if (risk_weight!=declared_risk or tail_mass!=.25 or mean_slack!=.01 or max_risk_exchanges!=3):
        raise ValueError('Risk/tail/floor/exchange parameters differ from the named fixed normalized arm')
    if 'utility_slack' in old_engine_kwargs and old_engine_kwargs['utility_slack']!=declared_slack:
        raise ValueError('Motion slack differs from the named fixed normalized arm')
    old_engine_kwargs['utility_slack']=declared_slack
    model=PublicProfileAnchorModel(base,profiles_v2)
    return PurposeNormalizedPacedLaneDummy(rn,normalization_mode=mode,belief_model=model,
        planner_mode='risk_multi' if mode=='normalized_tail' else 'aligned_mean_multi',
        risk_weight=risk_weight,tail_mass=tail_mass,
        mean_slack=mean_slack,max_risk_exchanges=max_risk_exchanges,**old_engine_kwargs)
