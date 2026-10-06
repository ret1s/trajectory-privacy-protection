"""Source-only depth freeze and independent whole-family readout contracts."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import qplanner_response_depth_generalization_20261006 as study
from experiments.qplanner_study_20261006_v2 import METHOD_CONFIGS, configuration
from experiments.qplanner_paired_readout_20261006 import paired, draw_diagnostic


CORE='experiments/qplanner_study_20261006_v2.py'
PROFILE='benchmark/public_service_profiles_v2.py'
PUBLIC='artifacts/public/toy.net.xml'
ERRORS=(ValueError,FileNotFoundError,KeyError)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def load(path):return json.loads(Path(path).read_text())


def pin(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(root):
    development=root/'artifacts/benchmarks/depthdev';original=root/'artifacts/benchmarks/original'
    base=root/'artifacts/benchmarks/baseQ';out=root/'artifacts/benchmarks/depthfresh'
    names=[CORE,PROFILE,PUBLIC,study.THIS,study.BASE_VERIFIER,study.DEV_VERIFIER,*study.VERIFIER_HELPERS]
    for name in names:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(('public source fixture '+name).encode())
    sources={name:pin(root/name) for name in (CORE,PROFILE)}
    public={PUBLIC:pin(root/PUBLIC)}
    qprotocol={'configuration':configuration(list(METHOD_CONFIGS)),
        'dataset_path':'artifacts/datasets/old-absent.gz','dataset_sha256':'b'*64,
        'source_sha256':sources,'public_inputs_sha256':public,'splits':['train','selection'],
        'draws_by_split':{'train':1,'selection':3}}
    write(original/'protocol.json',qprotocol)
    dp={'schema':'qplanner-response-depth-development-v1',
        'selection_criterion':study.kernel.CRITERION,'depths':list(study.kernel.DEPTHS),
        'splits':['train','selection'],'draws_by_split':qprotocol['draws_by_split'],
        'source_output':str(original),'source_files_sha256':{'protocol.json':pin(original/'protocol.json')},
        'source_sha256':sources,'fixed_configuration':qprotocol['configuration'],
        'family_files_sha256':{'toy--draw1.json.gz':'d'*64}}
    write(development/'protocol.json',dp);(development/'protocol.sha256').write_text(pin(development/'protocol.json')+'\n')
    resource={'full_reply60_sha256':'c'*64,'frozen_reply20_sha256':'e'*64,
        'depth_context_sha256':{str(d):str(d)*32 for d in (20,30,40,60)},'exact_l20_prefix_asserted':True}
    write(development/'resources.json',resource)
    write(development/'readout.json',{'protocol_sha256':pin(development/'protocol.json'),
        'resources_sha256':pin(development/'resources.json'),'family_files_sha256':dp['family_files_sha256']})
    candidates=[]
    for d,ratio in [(30,1.3),(40,1.6),(60,2.2)]:
        candidates.append({'depth':d,'differences':{'equal_purpose_macro':.025,'nearest_distance':.02},
            'reply_json_byte_ratio':ratio,'gates':{'macro_gain_at_least_2pp':True,'nearest_no_loss':True,
            'reply_byte_ratio_at_most_2_5':True},'eligible':True})
    write(development/'depth_selection.json',{'schema':'qplanner-response-depth-selection-v1',
        'selected_depth':30,'criterion':study.kernel.CRITERION,'candidates':candidates,
        'protocol_sha256':pin(development/'protocol.json'),'readout_sha256':pin(development/'readout.json')})
    write(development/'validation.json',{'status':'pass','protocol_sha256':pin(development/'protocol.json'),
        'readout_sha256':pin(development/'readout.json'),'verifier_sha256':pin(root/study.DEV_VERIFIER),
        'selected_depth':30,'fixed_Q_clock_anchors_ledger_inputs_unchanged':True})
    fresh=deepcopy(qprotocol);fresh.update(configuration=configuration(['legacy_l10']),
        dataset_path='artifacts/datasets/fresh-absent.gz',dataset_sha256='a'*64,
        splits=['train','selection','test'],draws_by_split=study.DRAW_SCHEDULE)
    write(base/'protocol.json',fresh);(base/'protocol.sha256').write_text(pin(base/'protocol.json')+'\n')
    for destination in (base,development):
        for name in sources:
            path=destination/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((root/name).read_bytes())
    return development,base,out


def reseal(out,base):
    """Rehash fresh receipts so tests exercise semantic, not accidental hash faults."""
    p=load(out/'protocol.json');f=load(out/'depth_freeze.json')
    (out/'protocol.sha256').write_text(pin(out/'protocol.json')+'\n')
    f.update(depth_protocol_sha256=pin(out/'protocol.json'),selected_depth=p['selected_depth'])
    write(out/'depth_freeze.json',f)
    b=load(base/'freeze.json');b.update(depth_protocol_sha256=pin(out/'protocol.json'),depth_freeze_sha256=pin(out/'depth_freeze.json'))
    write(base/'freeze.json',b)


def test_valid_freeze_precedes_q_generation_and_never_opens_absent_fresh_data_metrics(tmp_path):
    dev,base,out=fixture(tmp_path)
    p=study.declare_freeze(dev,base,out,root=tmp_path)
    assert p['depths']==[20,30] and p['criterion']==study.CRITERION
    assert not (tmp_path/p['dataset_path']).exists()
    assert not (base/'generation.json').exists() and not (out/'readout.json').exists()
    result=study.depth_contract(out,root=tmp_path)
    assert result['status']=='pass' and result['fresh_dataset_opened_for_contract'] is False
    assert result['fresh_metrics_opened_for_contract'] is False
    assert load(base/'freeze.json')['depth_freeze_sha256']==pin(out/'depth_freeze.json')
    assert study.BASE_VERIFIER in p['source_sha256'] and study.DEV_VERIFIER in p['source_sha256']
    assert all(name in p['source_sha256'] for name in study.VERIFIER_HELPERS)
    with pytest.raises(FileExistsError):study.declare_freeze(dev,base,out,root=tmp_path)


@pytest.mark.parametrize('fault',['no_candidate','not_smallest','claim_bad_gate','missing_validation',
                                 'validation_hash','old_dataset','extra_q_method','budget_change','started_q'])
def test_declaration_refuses_invalid_development_or_changed_fresh_controls(tmp_path,fault):
    dev,base,out=fixture(tmp_path)
    if fault in ('no_candidate','not_smallest','claim_bad_gate'):
        s=load(dev/'depth_selection.json')
        if fault=='no_candidate':s['selected_depth']=None
        elif fault=='not_smallest':s['selected_depth']=40
        else:s['candidates'][0]['differences']['equal_purpose_macro']=.001
        write(dev/'depth_selection.json',s)
    elif fault=='missing_validation':(dev/'validation.json').unlink()
    elif fault=='validation_hash':
        v=load(dev/'validation.json');v['readout_sha256']='0'*64;write(dev/'validation.json',v)
    elif fault=='started_q':write(base/'generation_started.json',{})
    else:
        p=load(base/'protocol.json')
        if fault=='old_dataset':p['dataset_sha256']='b'*64
        elif fault=='extra_q_method':p['configuration']['methods']['normalized_mean']=METHOD_CONFIGS['normalized_mean']
        else:p['configuration']['budget']['total_effective_epsilon_per_m']=.46
        write(base/'protocol.json',p);(base/'protocol.sha256').write_text(pin(base/'protocol.json')+'\n')
    with pytest.raises(ERRORS):study.declare_freeze(dev,base,out,root=tmp_path)


@pytest.mark.parametrize('fault',['depth_change','criterion_relaxed','kernel_change','dataset_change',
                                 'source_change','snapshot_change','base_freeze','depth_freeze','old_pairing'])
def test_contract_rejects_rehashed_configuration_and_pre_generation_binding_faults(tmp_path,fault):
    dev,base,out=fixture(tmp_path);study.declare_freeze(dev,base,out,root=tmp_path)
    p=load(out/'protocol.json')
    if fault=='depth_change':p.update(selected_depth=40,depths=[20,40])
    elif fault=='criterion_relaxed':p['criterion']['minimum_absolute_mean_gain']=.01
    elif fault=='kernel_change':p['service_kernel']['full_reply60_sha256']='0'*64
    elif fault=='dataset_change':p['dataset_sha256']='0'*64
    elif fault=='source_change':(tmp_path/PROFILE).write_bytes(b'changed source')
    elif fault=='snapshot_change':(out/'source_snapshot'/PROFILE).write_bytes(b'changed snapshot')
    write(out/'protocol.json',p);reseal(out,base)
    if fault=='base_freeze':
        b=load(base/'freeze.json');b['depth_freeze_sha256']='0'*64;write(base/'freeze.json',b)
    elif fault=='depth_freeze':
        f=load(out/'depth_freeze.json');f['fresh_depth_scores_viewed']=True;write(out/'depth_freeze.json',f)
        b=load(base/'freeze.json');b['depth_freeze_sha256']=pin(out/'depth_freeze.json');write(base/'freeze.json',b)
    elif fault=='old_pairing':
        e={'common_protocol_sha256':pin(base/'protocol.json'),'paired_development':{'old':True},
            'predeclared_files_sha256':{'freeze.json':pin(base/'freeze.json')},'source_sha256':{}}
        write(base/'execution_protocol.json',e);(base/'execution_protocol.sha256').write_text(pin(base/'execution_protocol.json')+'\n')
    with pytest.raises(ERRORS):study.depth_contract(out,root=tmp_path)


def test_before_score_receipt_gate_and_development_scope_cannot_open_fresh_readout(tmp_path,monkeypatch):
    dev,base,out=fixture(tmp_path);study.declare_freeze(dev,base,out,root=tmp_path)
    with pytest.raises(FileNotFoundError):study.base_q_receipt(out,root=tmp_path)
    opened=[];original=study.read
    def watched(path):
        opened.append(Path(path));return original(path)
    monkeypatch.setattr(study,'read',watched)
    with pytest.raises(ValueError,match='may not open fresh metrics'):
        study.paired_depth_readout(out,scope='development')
    assert out/'readout.json' not in opened


def test_family_bootstrap_and_nested_draw_signs_are_not_tick_independent():
    left={'a':.9,'b':.7,'c':.95};right={'a':.8,'b':.6,'c':.85}
    result=paired(left,right)
    assert result['independent_family_clusters']==3
    assert result['percentile95_family_bootstrap']==pytest.approx([.1,.1])
    ld={f:{str(d):v for d in (1,2,3)} for f,v in left.items()}
    rd={f:{str(d):v for d in (1,2,3)} for f,v in right.items()}
    diagnostic=draw_diagnostic(ld,rd,[1,2,3])
    assert diagnostic['sign_consistency']=={'positive_draws':3,'negative_draws':0,'zero_draws':0,'defined_draws':3}
    assert all(row['family_clusters']==3 for row in diagnostic['draws'].values())
