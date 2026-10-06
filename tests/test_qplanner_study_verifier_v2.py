"""Independent normalized profiles, objective arithmetic, and control pairing."""
from copy import deepcopy
import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from benchmark.engines.public_service_planner_v2 import make_service_planner_engine_v2
from benchmark.public_service_profiles_v2 import build_public_service_profiles_v2
from experiments.verify_qplanner_study_20261006_v2 import (
    PROFILE_BUILDER, PURPOSES, ROOT, control_pairing_bundle, direct_score,
    normalized_mass_oracle, normalized_objective_checks, profile_resource_digest,
    rebuild_profiles, sha, PAIRED_CONFIGURATION_FIELDS, PAIRED_PRIVACY_SOURCES,
    paired_development_check,
)
from evaluation.lane_travel import SparseTravel
from tests.test_public_service_planner import fixture


@pytest.mark.parametrize('radius',(5.,15.,1e9))
def test_independent_local_oracle_rebuilds_all_profiles_counts_and_serialized_hash(radius):
    rn,base,legacy,reply,_=fixture()
    built=build_public_service_profiles_v2(base,reply,public_destination_states=[2,27],public_radius_m=radius)
    oracle=rebuild_profiles(rn,base.context,reply,base.state_ids,[2,27],built.metadata,
                            {'source_sha256':{PROFILE_BUILDER:sha(ROOT/PROFILE_BUILDER)}})
    assert oracle.reference_profiles==built.reference_profiles and oracle.sha256==built.sha256
    for purpose in PURPOSES:
        assert np.array_equal(oracle.case_valid_category_counts[purpose],built.case_valid_category_counts[purpose])
        assert np.array_equal(oracle.purpose_masses[purpose].toarray(),built.purpose_masses[purpose].toarray())
    belief=np.arange(1,len(base.state_ids)+1,dtype=float)
    mass,diagnostics=normalized_mass_oracle(oracle,belief)
    expected,actual=built.normalized_mass(belief)
    assert np.allclose(mass,expected,rtol=0.,atol=1e-12)
    assert diagnostics['effective_purpose_weights']==actual['effective_purpose_weights']
    assert diagnostics['undefined_purposes']==actual['undefined_purposes']
    for action in ([0],[0,15],[14,29]):
        score=direct_score(oracle,mass,action,tail_mass=.25,risk_weight=.25)
        expected=built.objective(belief,tail_mass=.25,risk_weight=.25).score(action)
        assert (score['mean'],score['lower_tail_cvar'],score['objective'])==pytest.approx(
            (expected.mean,expected.lower_tail_cvar,expected.objective),abs=1e-12)


def test_independent_profile_metadata_and_mass_hash_catch_weighting_corruption():
    rn,base,legacy,reply,_=fixture()
    built=build_public_service_profiles_v2(base,reply,public_destination_states=[2,27],public_radius_m=15.)
    metadata=deepcopy(built.metadata);metadata['empty_input_category_profile_count']+=1
    with pytest.raises(AssertionError):
        rebuild_profiles(rn,base.context,reply,base.state_ids,[2,27],metadata,
                         {'source_sha256':{PROFILE_BUILDER:sha(ROOT/PROFILE_BUILDER)}})
    masses={p:m.copy() for p,m in built.purpose_masses.items()}
    masses['within_radius'].data[0]*=.5
    assert profile_resource_digest(built.reference_profiles,masses,built.case_valid_category_counts,
                                   built.metadata,len(reply.pois))!=built.sha256


def tiny_profiles():
    references=((),(0,),(1,))
    regular=np.array([[0.,1.,0.],[0.,1.,0.]])
    radius=np.array([[1.,0.,0.],[0.,0.,1.]])
    masses={p:csr_matrix(radius if p=='within_radius' else regular) for p in PURPOSES}
    return SimpleNamespace(reference_profiles=references,purpose_masses=masses,state_ids=[0,1],lengths=np.array([0,1,1]))


def test_direct_mass_oracle_conditions_each_purpose_and_never_pools_valid_categories():
    profiles=tiny_profiles();mass,diagnostics=normalized_mass_oracle(profiles,[.5,.5])
    assert mass==pytest.approx([0.,.75,.25])
    assert diagnostics['purpose_valid_case_mass']['within_radius']==.5
    assert diagnostics['empty_profile_mass']==.125
    assert diagnostics['effective_purpose_weights']=={p:.25 for p in PURPOSES}
    mass,diagnostics=normalized_mass_oracle(profiles,[1.,0.])
    assert mass==pytest.approx([0.,1.,0.]) and diagnostics['undefined_purposes']==['within_radius']
    assert diagnostics['effective_purpose_weights']['within_radius']==0.
    assert all(diagnostics['effective_purpose_weights'][p]==pytest.approx(1/3) for p in PURPOSES if p!='within_radius')
    for p in PURPOSES:profiles.purpose_masses[p]=csr_matrix([[1.,0.,0.],[1.,0.,0.]])
    mass,diagnostics=normalized_mass_oracle(profiles,[.5,.5])
    assert np.all(mass==0.) and diagnostics['undefined_purposes']==list(PURPOSES)


def test_direct_fractional_tail_uses_partial_probability_mass_and_unique_reply_union():
    class Context:
        def query_indices(self,state):return np.array([[state,-1,state]])
    profiles=SimpleNamespace(context=Context(),reference_profiles=((0,),(1,),(0,1)))
    mass=np.array([.3,.4,.3])
    score=direct_score(profiles,mass,[0,0],tail_mass=.5,risk_weight=.5)
    assert score==pytest.approx(dict(mean=.45,lower_tail_cvar=.1,objective=.275))
    assert direct_score(profiles,mass,[0,1],tail_mass=.5,risk_weight=.5)==pytest.approx(
        dict(mean=1.,lower_tail_cvar=1.,objective=1.))
    assert direct_score(profiles,np.zeros(3),[0],tail_mass=.5,risk_weight=.5) is None


def test_actual_engine_records_match_independent_protected_filter_and_objective_oracle():
    rn,base,legacy,reply,_=fixture()
    profiles=build_public_service_profiles_v2(base,reply,public_destination_states=[2,27],public_radius_m=15.)
    oracle=rebuild_profiles(rn,base.context,reply,base.state_ids,[2,27],profiles.metadata,
                            {'source_sha256':{PROFILE_BUILDER:sha(ROOT/PROFILE_BUILDER)}})
    methods={'legacy_l10':{'mode':'legacy_l10'},'aligned_nearest':{'mode':'aligned_nearest'},
             'normalized_mean':{'mode':'normalized_mean','risk_weight':0.},
             'normalized_tight':{'mode':'normalized_tight','risk_weight':0.},
             'normalized_tail':{'mode':'normalized_tail','risk_weight':.25}}
    engines={}
    for mode in methods:
        engine=make_service_planner_engine_v2(mode,rn,base,reply,profiles,legacy_context=legacy,
            k=2,budget=.24,horizon=12,theta_m=200.,read_interval_s=60.,rng=np.random.default_rng(41))
        engine.anchor_rng=np.random.default_rng(73);engine.dummy_rng=np.random.default_rng(97);engine.reset()
        engines[mode]=engine
    times=list(range(0,121,20))
    for t in times:
        for engine in engines.values():engine.protect_step(0.,.0001+(t//20)*.0001,t)
    ledger={m:dict(anchors=e.evaluator_anchors,ledger=e.evaluator_ledger,states=e.evaluator_states,
                   planner_objectives=e.evaluator_objective) for m,e in engines.items()}
    bundle=dict(public={'streams':{'raw':[{'events':[{'timestamp_s':t} for t in times]}]}},
                evaluator_only={'sessions':[{'ledger':ledger}]})
    family={'evaluator_only':{'sessions':[{'depart_s':0}]}}
    counters=dict(protected_belief_steps=0,normalized_records=0,action_scores=0)
    normalized_objective_checks(bundle,family,base,oracle,{'configuration':{'methods':methods}},SparseTravel(rn),counters)
    assert counters['protected_belief_steps']==7 and counters['normalized_records']==21
    broken=deepcopy(bundle);broken['evaluator_only']['sessions'][0]['ledger']['normalized_tail']['planner_objectives'][0]['value']+=.1
    with pytest.raises(AssertionError):
        normalized_objective_checks(broken,family,base,oracle,{'configuration':{'methods':methods}},SparseTravel(rn),
                                    dict(protected_belief_steps=0,normalized_records=0,action_scores=0))


def paired_bundle():
    controls=['legacy_l10','aligned_nearest'];ledger={}
    for method in controls:
        ledger[method]=dict(anchors=[[0.,0.]],ledger=[{'private_read':True}],supplier_times_s=[0],private_reads=1,
            spent_per_m=.01,allocation={'slot':0},states=[[1]],step_ms=[.1],planner_objectives=[])
    target=dict(slot=0,choice_index=0,destination_role='routine',origin_xy=[0.,0.],destination_xy=[1.,0.],
                next_edge_labels={},ledger=ledger)
    return dict(public={'streams':{m:[{'events':[{'point':[0.,0.]}]}] for m in controls}},
        evaluator_only={'sessions':[target],'epoch_accounting':{m:{'spent_per_m':.01} for m in controls}},
        utility=[{'method':m,'value':.9} for m in controls],wire=[{'method':m,'reply_bytes':100} for m in controls])


@pytest.mark.parametrize('fault',(None,'point','utility','wire','anchor','private_read','control_state'))
def test_immutable_control_pairing_detects_output_or_private_tape_changes(tmp_path,fault):
    old=paired_bundle();path=tmp_path/'families/sample.json.gz';path.parent.mkdir()
    path.write_bytes(gzip.compress(json.dumps(old).encode()));new=deepcopy(old)
    ledger=new['evaluator_only']['sessions'][0]['ledger']
    ledger['normalized_mean']=deepcopy(ledger['legacy_l10']);ledger['normalized_mean']['states']=[[8]]
    if fault=='point':new['public']['streams']['aligned_nearest'][0]['events'][0]['point']=[9.,9.]
    elif fault=='utility':new['utility'][0]['value']=1.
    elif fault=='wire':new['wire'][0]['reply_bytes']=101
    elif fault=='anchor':ledger['normalized_mean']['anchors']=[[1.,1.]]
    elif fault=='private_read':ledger['normalized_mean']['ledger'][0]['private_read']=False
    elif fault=='control_state':ledger['aligned_nearest']['states']=[[8]]
    receipt={'path':tmp_path,'controls':['legacy_l10','aligned_nearest'],'blocks':0}
    if fault:
        with pytest.raises(AssertionError):control_pairing_bundle(new,'sample.json.gz',receipt,list(ledger))
    else:
        control_pairing_bundle(new,'sample.json.gz',receipt,list(ledger));assert receipt['blocks']==1


def json_file(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def pairing_plan_fixture(tmp_path):
    old=tmp_path/'artifacts/old';out=tmp_path/'artifacts/new'
    pins={}
    for name in PAIRED_PRIVACY_SOURCES:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('unchanged=1\n');pins[name]=sha(path)
        for snapshot in ('source_snapshot','execution_source_snapshot'):
            copied=old/snapshot/name;copied.parent.mkdir(parents=True,exist_ok=True);copied.write_bytes(path.read_bytes())
    configuration={field:1 for field in PAIRED_CONFIGURATION_FIELDS}
    configuration.update(methods={'legacy_l10':{'mode':'legacy_l10'},'aligned_nearest':{'mode':'aligned_nearest'}},utility_slack=.03)
    prior=dict(dataset_path='data.gz',dataset_sha256='a'*64,draws_by_split={'selection':1,'train':1},
               splits=['selection','train'],public_inputs_sha256={'map.gz':'b'*64},configuration=configuration,source_sha256=pins)
    json_file(old/'protocol.json',prior);(old/'protocol.sha256').write_text(sha(old/'protocol.json')+'\n')
    data={'families':[dict(family_id='a',split='selection'),dict(family_id='b',split='train')]}
    jobs=[dict(name=f'{name}--draw1.json.gz',family_id=name,split=split,draw=1) for name,split in (('a','selection'),('b','train'))]
    hashes={}
    for i,job in enumerate(jobs):
        path=old/'families'/job['name'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(bytes([i]));hashes[job['name']]=sha(path)
    json_file(old/'generation.json',dict(family_files_sha256=hashes))
    for name in ('generation_started.json','resources.json','execution_environment.json'):json_file(old/name,{})
    json_file(old/'execution_protocol.json',dict(common_protocol_sha256=sha(old/'protocol.json'),source_sha256=pins))
    (old/'execution_protocol.sha256').write_text(sha(old/'execution_protocol.json')+'\n')
    plan=dict(schema='qplanner-matched-development-private-realization-v1',source_output=str(old),
        source_files_sha256={n:sha(old/n) for n in ('protocol.json','protocol.sha256','generation.json','generation_started.json',
            'resources.json','execution_protocol.json','execution_protocol.sha256','execution_environment.json')},
        family_files_sha256=hashes,privacy_control_source_sha256=pins,planned_jobs=jobs,
        source_private_root='/private/tmp/intentionally-unavailable-qplanner-pairing-keys',
        retained_key_blocks=list(hashes),matched_controls=['legacy_l10','aligned_nearest'],scope='development',key_policy='no keys exported')
    execution=dict(paired_development=plan,transfer=None,private_transfer=None)
    protocol=deepcopy(prior)
    protocol['configuration']['methods']={m:dict(config,utility_slack=.03) for m,config in configuration['methods'].items()}
    protocol['configuration']['utility_slack']='per_method_declared_above'
    generation=dict(imported_completed_block_count=0,generated_block_count=2,paired_development_private_realization=True)
    receipt=dict(paired_development_private_key_block_count=2,copied_family_files_sha256={},
                 retained_private_key_block_count=0,private_key_bytes_or_hashes_exported=False)
    json_file(out/'transfer_receipt.json',receipt)
    return old,out,plan,execution,protocol,generation,data


def test_pre_generation_pairing_contract_checks_public_metadata_without_private_keys(tmp_path):
    old,out,plan,execution,protocol,generation,data=pairing_plan_fixture(tmp_path)
    result=paired_development_check(out,execution,protocol,generation,data,root=tmp_path)
    assert result['paired_blocks']==2 and result['source_files_verified']==8
    assert result['unchanged_privacy_control_sources']==17
    assert result['retained_private_key_block_count_metadata']==2
    assert result['private_key_material_verified'] is False and result['fresh_test_pairing_permitted'] is False


@pytest.mark.parametrize('fault',('service_config','control_motion','source_pin','source_family',
                                  'plan_order','retained_name','receipt_count','test_scope','key_export_field','generation_assertion'))
def test_pre_generation_pairing_rejects_contract_or_provenance_faults(tmp_path,fault):
    old,out,plan,execution,protocol,generation,data=pairing_plan_fixture(tmp_path)
    if fault=='service_config':protocol['configuration']['K']=6
    elif fault=='control_motion':protocol['configuration']['methods']['aligned_nearest']['utility_slack']=0.
    elif fault=='source_pin':(tmp_path/'core/mechanisms.py').write_text('changed=1\n')
    elif fault=='source_family':(old/'families'/plan['retained_key_blocks'][0]).write_bytes(b'resampled')
    elif fault=='plan_order':plan['planned_jobs']=list(reversed(plan['planned_jobs']))
    elif fault=='retained_name':plan['retained_key_blocks'].pop()
    elif fault=='receipt_count':
        json_file(out/'transfer_receipt.json',dict(paired_development_private_key_block_count=1,
            copied_family_files_sha256={},retained_private_key_block_count=0,private_key_bytes_or_hashes_exported=False))
    elif fault=='test_scope':protocol['splits'].append('test')
    elif fault=='key_export_field':plan['key_sha256']='c'*64
    elif fault=='generation_assertion':generation['paired_development_private_realization']=False
    with pytest.raises(AssertionError):paired_development_check(out,execution,protocol,generation,data,root=tmp_path)
