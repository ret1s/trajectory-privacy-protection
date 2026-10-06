"""New sole-legacy adapter checks adoption before any fresh bytes/scores."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import qplanner_response_depth_generalization_20261006 as depth
from experiments import verify_qplanner_depth_base_q_20261006 as audit


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();base=root/'artifacts/base';fresh=root/'artifacts/depth'
    config={'B':.23,'server_L':20,'methods':{'legacy_l10':{'mode':'legacy_l10','utility_slack':.03}}}
    protocol={'configuration':config,'splits':['train','selection','test'],'draws_by_split':dict(audit.DRAW_SCHEDULE)}
    save(base/'protocol.json',protocol);(base/'protocol.sha256').write_text(audit.sha(base/'protocol.json'))
    pins={}
    for name in ('experiments/verify_qplanner_study_20261006.py','experiments/verify_qplanner_study_20261006_v2.py',
                 'experiments/qplanner_response_depth_generalization_20261006.py','experiments/verify_qplanner_depth_base_q_20261006.py'):
        path=root/name;path.parent.mkdir(exist_ok=True);path.write_text('# fixture dependency\n');pins[name]=audit.sha(path)
    dp={'base_q_protocol_sha256':audit.sha(base/'protocol.json'),'base_q_output':'artifacts/base',
        'base_q_configuration':config,'source_sha256':pins}
    save(fresh/'protocol.json',dp);save(fresh/'depth_freeze.json',{'fixture':'public freeze'})
    freeze={'schema':'qplanner-base-Q-for-selected-depth-freeze-v1','fresh_test_evaluated':False,
        'depth_output':'artifacts/depth','fresh_protocol_sha256':audit.sha(base/'protocol.json'),
        'configuration':config,'depth_protocol_sha256':audit.sha(fresh/'protocol.json'),
        'depth_freeze_sha256':audit.sha(fresh/'depth_freeze.json')}
    save(base/'freeze.json',freeze)
    expected={'status':'pass','base_q_protocol_sha256':dp['base_q_protocol_sha256'],
        'depth_protocol_sha256':freeze['depth_protocol_sha256'],'depth_freeze_sha256':freeze['depth_freeze_sha256'],
        'fresh_dataset_opened_for_contract':False,'fresh_metrics_opened_for_contract':False}
    calls=[]
    def contract(output,*,root):
        assert Path(output)==fresh;calls.append(output);return deepcopy(expected)
    monkeypatch.setattr(depth,'depth_contract',contract)
    return root,base,fresh,protocol,dp,freeze,expected,calls


def test_adoption_adapter_passes_without_any_fresh_dataset_or_metrics(tmp_path,monkeypatch):
    root,base,fresh,_,_,_,_,calls=fixture(tmp_path,monkeypatch)
    result=audit.fresh_contract(base,root=root)
    assert calls==[fresh] and result['sole_legacy_Q_method'] is True
    assert result['no_fresh_depth_scores_opened'] is True
    assert not (base/'generation.json').exists() and not (fresh/'readout.json').exists()


@pytest.mark.parametrize('fault',['extra_arm','mode','config','schema','viewed','base_binding','freeze_hash',
                                 'helper_data_read','helper_score_read','dependency','missing_dependency'])
def test_adoption_adapter_rejects_material_contract_faults(tmp_path,monkeypatch,fault):
    root,base,fresh,p,dp,f,helper,calls=fixture(tmp_path,monkeypatch)
    if fault=='extra_arm':p['configuration']['methods']['new']={'mode':'new'};dp['base_q_configuration']=p['configuration'];f['configuration']=p['configuration']
    elif fault=='mode':p['configuration']['methods']['legacy_l10']['mode']='normalized_mean'
    elif fault=='config':f['configuration']=dict(p['configuration'],B=.24)
    elif fault=='schema':f['schema']='old-three-method-freeze'
    elif fault=='viewed':f['fresh_test_evaluated']=True
    elif fault=='base_binding':dp['base_q_output']='artifacts/other-base'
    elif fault=='freeze_hash':f['depth_freeze_sha256']='wrong'
    elif fault=='helper_data_read':helper['fresh_dataset_opened_for_contract']=True
    elif fault=='helper_score_read':helper['fresh_metrics_opened_for_contract']=True
    elif fault=='dependency':(root/'experiments/verify_qplanner_study_20261006_v2.py').write_text('# changed\n')
    else:dp['source_sha256'].pop('experiments/verify_qplanner_study_20261006.py')
    save(base/'protocol.json',p);(base/'protocol.sha256').write_text(audit.sha(base/'protocol.json'))
    if fault in ('extra_arm','mode'):
        f['fresh_protocol_sha256']=dp['base_q_protocol_sha256']=helper['base_q_protocol_sha256']=audit.sha(base/'protocol.json')
    save(fresh/'protocol.json',dp)
    if fault in ('extra_arm','mode','base_binding','missing_dependency'):
        f['depth_protocol_sha256']=helper['depth_protocol_sha256']=audit.sha(fresh/'protocol.json')
    save(base/'freeze.json',f)
    with pytest.raises(AssertionError):audit.fresh_contract(base,root=root)


def test_verify_calls_adoption_contract_before_generation_or_dataset(tmp_path,monkeypatch):
    root,base,_,_,_,_,_,calls=fixture(tmp_path,monkeypatch)
    original=audit.fresh_contract
    monkeypatch.setattr(audit,'fresh_contract',lambda output:original(output,root=root))
    with pytest.raises(FileNotFoundError,match='generation.json'):audit.verify(base)
    assert len(calls)==1
