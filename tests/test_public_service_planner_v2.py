"""Normalization contracts and unchanged Geo-I on independent tiny maps."""
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.public_service_planner_v2 import V2_MODES,make_service_planner_engine_v2
from benchmark.public_poi_context import PublicPoiContext
from benchmark.public_service_profiles import VectorizedProfileObjective,build_public_service_profiles
from benchmark.public_service_profiles_v2 import PURPOSES,PurposeNormalizedServiceProfiles,build_public_service_profiles_v2
from core.demo_protocol import TrajectoryPoint
from evaluation.lane_travel import LanePoiService
from tests.test_lane_comparison import road
from tests.test_public_service_planner import fixture


def toy_masses(*,all_empty=False):
    context=SimpleNamespace(rn=[0,1],pois=[0,1],access=np.arange(2),
        signatures=np.array([[[1]],[[0]]]))
    references=[(),(0,),(1,)]
    regular=np.array([[0.,1.,0.],[0.,1.,0.]])
    radius=np.array([[1.,0.,0.],[0.,0.,1.]])
    masses={p:regular.copy() for p in PURPOSES};masses['within_radius']=radius
    counts={p:np.ones((2,1),int) for p in PURPOSES};counts['within_radius']=np.array([[0],[1]])
    if all_empty:
        masses={p:np.array([[1.,0.,0.],[1.,0.,0.]]) for p in PURPOSES}
        counts={p:np.zeros((2,1),int) for p in PURPOSES}
    return PurposeNormalizedServiceProfiles(SimpleNamespace(state_ids=np.array([0,1])),context,references,
        masses,counts,{'reference_k':1})


def test_radius_gets_equal_purpose_weight_after_conditioning_its_valid_cases():
    profiles=toy_masses()
    normalized=profiles.objective([.5,.5],risk_weight=0.)
    pooled=VectorizedProfileObjective(profiles,[.5,.5],risk_weight=0.)
    assert normalized.score([0]).mean==pytest.approx(.25)
    assert pooled.score([0]).mean==pytest.approx(1/7)
    diagnostics=normalized.normalization_diagnostics
    assert diagnostics['purpose_valid_case_mass']['within_radius']==.5
    assert diagnostics['effective_purpose_weights']=={p:.25 for p in PURPOSES}
    assert diagnostics['empty_profile_mass']==.125


def test_wholly_undefined_purpose_remains_na_and_other_purposes_share_mass_equally():
    profiles=toy_masses()
    normalized=profiles.objective([1.,0.],risk_weight=0.)
    assert normalized.score([0]).mean==0.
    diagnostics=normalized.normalization_diagnostics
    assert diagnostics['undefined_purposes']==['within_radius']
    assert diagnostics['effective_purpose_weights']['within_radius']==0.
    assert all(diagnostics['effective_purpose_weights'][p]==pytest.approx(1/3) for p in PURPOSES if p!='within_radius')
    with pytest.raises(ValueError,match='undefined'):
        toy_masses(all_empty=True).objective([.5,.5])


def test_actual_public_graph_averages_one_vs_six_valid_categories_inside_each_case():
    rn=road()
    pois=[dict(id='a-only',category='c0',lat=0.,lon=.0002)]
    pois.extend(dict(id=f'b{i}',category=f'c{i}',lat=0.,lon=.002) for i in range(6))
    ref=PublicPoiContext(LanePoiService(rn,pois,k=5));reply=PublicPoiContext(LanePoiService(rn,pois,k=20))
    base=PublicAnchorModel(rn,ref,np.ones(len(rn)),spacing_m=.5)
    profiles=build_public_service_profiles_v2(base,reply,public_destination_states=[2,20],public_radius_m=5.)
    a=next(i for i,state in enumerate(base.state_ids) if ref.access[state]==2)
    b=next(i for i,state in enumerate(base.state_ids) if ref.access[state]==20)
    assert profiles.case_valid_category_counts['within_radius'][a,0]==1
    assert profiles.case_valid_category_counts['within_radius'][b,0]==6
    belief=np.zeros(len(base.state_ids));belief[a]=belief[b]=.5
    radius=np.asarray(belief@profiles.purpose_masses['within_radius']).ravel()
    poi=next(i for i,p in enumerate(reply.pois) if p['id']=='a-only')
    target=profiles.reference_profiles.index((poi,))
    assert radius[target]==pytest.approx(.5)  # Not1/7 from pooling valid categories.
    assert radius.sum()==pytest.approx(1.)


def test_all_valid_equal_category_cases_reproduce_original_mean_and_discrete_tail():
    rn,base,legacy,reply,_=fixture()
    old=build_public_service_profiles(base,reply,public_destination_states=[2,27],public_radius_m=1e9)
    new=build_public_service_profiles_v2(base,reply,public_destination_states=[2,27],public_radius_m=1e9)
    belief=np.arange(1,len(base.state_ids)+1,dtype=float);belief/=belief.sum()
    left=old.objective(belief,tail_mass=.25,risk_weight=.25)
    right=new.objective(belief,tail_mass=.25,risk_weight=.25)
    for action in ([0],[0,15],[14,29]):
        a,b=left.score(action),right.score(action)
        assert (a.mean,a.lower_tail_cvar,a.objective)==pytest.approx((b.mean,b.lower_tail_cvar,b.objective),abs=1e-12)


def test_five_modes_keep_all_anchor_read_flags_and_ledger_branches_identical():
    rn,base,legacy,reply,_=fixture()
    profiles=build_public_service_profiles_v2(base,reply,public_destination_states=[2,27],public_radius_m=15.)
    engines={}
    for mode in V2_MODES:
        engine=make_service_planner_engine_v2(mode,rn,base,reply,profiles,legacy_context=legacy,
            k=2,budget=.24,horizon=12,theta_m=200.,read_interval_s=60.,rng=np.random.default_rng(41))
        engine.anchor_rng=np.random.default_rng(73);engine.dummy_rng=np.random.default_rng(97);engine.reset()
        engines[mode]=engine
    points=[TrajectoryPoint(i*20.,0.,.0001+(i%20)*.0001) for i in range(45)]
    for point in points:
        for engine in engines.values():engine.protect_step(point.lat,point.lon,point.timestamp_s)
    baseline=engines['legacy_l10']
    for engine in engines.values():
        assert engine.evaluator_anchors==baseline.evaluator_anchors
        assert engine.evaluator_ledger==baseline.evaluator_ledger
        assert engine.spent_bound==baseline.spent_bound
    assert engines['normalized_mean'].utility_slack==.03
    assert engines['normalized_tight'].utility_slack==engines['normalized_tail'].utility_slack==0.
    assert engines['normalized_mean'].risk_weight==engines['normalized_tight'].risk_weight==0.
    assert engines['normalized_tail'].risk_weight==.25
    for mode in V2_MODES[2:]:
        for earlier,later in zip(engines[mode].evaluator_states,engines[mode].evaluator_states[1:]):
            assert all(b in engines[mode].travel.reachable(a,20.) for a,b in zip(earlier,later))


def test_fixed_named_arms_reject_mislabeled_config_and_preserve_public_only_parameters():
    rn,base,legacy,reply,_=fixture()
    profiles=build_public_service_profiles_v2(base,reply,public_destination_states=[2,27])
    with pytest.raises(ValueError,match='Motion slack'):
        make_service_planner_engine_v2('normalized_tight',rn,base,reply,profiles,utility_slack=.03)
    with pytest.raises(ValueError,match='Risk/tail'):
        make_service_planner_engine_v2('normalized_mean',rn,base,reply,profiles,risk_weight=.5)
    engine=make_service_planner_engine_v2('normalized_mean',rn,base,reply,profiles,k=2,
        budget=.24,horizon=12,rng=np.random.default_rng(1))
    public=engine.protect_run([TrajectoryPoint(0.,0.,.0001)]).to_attacker_dict()
    assert public['public_parameters']['planner_mode']=='normalized_mean'
    assert 'normalization_diagnostics' not in str(public)
    assert 'evaluator_anchors' not in str(public)
