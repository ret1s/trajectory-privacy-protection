"""New sole-legacy adapter checks adoption before any fresh bytes/scores."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from experiments import qplanner_response_depth_generalization_20261006_v2 as depth
from experiments import verify_qplanner_depth_base_q_20261006_v2 as audit


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def fixture(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();base=root/'artifacts/base';fresh=root/'artifacts/depth'
    config={'B':.23,'server_L':20,'methods':{'legacy_l10':{'mode':'legacy_l10','utility_slack':.03}}}
    protocol={'configuration':config,'splits':['train','selection','test'],'draws_by_split':dict(audit.DRAW_SCHEDULE)}
    save(base/'protocol.json',protocol);(base/'protocol.sha256').write_text(audit.sha(base/'protocol.json'))
    pins={}
    for name in ('experiments/verify_qplanner_study_20261006.py','experiments/verify_qplanner_study_20261006_v2.py',
                 'experiments/qplanner_response_depth_generalization_20261006_v2.py','experiments/verify_qplanner_depth_base_q_20261006_v2.py'):
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


def alias_fixture(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    old_name='artifacts/datasets/older-network/public_native.net.xml.gz'
    new_name='artifacts/datasets/fresh/public_native.net.xml.gz'
    for name in (old_name,new_name):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(gzip.compress(b'identical public native XML',mtime=0))
    resources='artifacts/benchmarks/public/resources.json';p=root/resources;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'public POI catalogue')
    old={'dataset_path':'artifacts/datasets/development/dataset.json.gz',
         'public_inputs_sha256':{old_name:audit.sha(root/old_name),resources:audit.sha(p)}}
    new={'dataset_path':'artifacts/datasets/fresh/dataset.json.gz',
         'public_inputs_sha256':{new_name:audit.sha(root/new_name),resources:audit.sha(p)}}
    metadata={'network':dict(compressed_path=old_name,compressed_sha256=audit.sha(root/old_name),
        native_sha256=hashlib.sha256(b'identical public native XML').hexdigest())}
    old_data=root/old['dataset_path'];old_data.parent.mkdir(parents=True,exist_ok=True)
    old_data.write_bytes(gzip.compress(json.dumps(metadata).encode(),mtime=0));old['dataset_sha256']=audit.sha(old_data)
    return root,old,new,old_name,new_name,resources


def test_public_network_alias_is_byte_identical_without_opening_fresh_dataset(tmp_path):
    root,old,new,old_name,new_name,_=alias_fixture(tmp_path)
    result=depth.equivalent_public_inputs(new,old,root=root)
    assert result['native_archive_relocated'] is True
    assert result['identical_archive_sha256']==audit.sha(root/old_name)==audit.sha(root/new_name)
    assert not (root/new['dataset_path']).exists()


@pytest.mark.parametrize('fault',['bytes','pin','public_resource','extra_key','nonconventional_path','missing_shared_key'])
def test_public_network_alias_rejects_changed_map_or_other_public_inputs(tmp_path,fault):
    root,old,new,old_name,new_name,resources=alias_fixture(tmp_path)
    if fault=='bytes':(root/new_name).write_bytes(b'changed network')
    elif fault=='pin':new['public_inputs_sha256'][new_name]='a'*64
    elif fault=='public_resource':new['public_inputs_sha256'][resources]='b'*64
    elif fault=='extra_key':new['public_inputs_sha256']['extra-public-file']='c'*64
    elif fault=='nonconventional_path':new['public_inputs_sha256']['different.net.gz']=new['public_inputs_sha256'].pop(new_name)
    else:new['public_inputs_sha256'].pop(resources)
    with pytest.raises(ValueError):depth.equivalent_public_inputs(new,old,root=root)


@pytest.mark.parametrize('fault',['old_dataset_hash','old_network_path','old_compressed_digest','old_native_digest'])
def test_archive_alias_binds_already_inspected_old_metadata_before_fresh_data(tmp_path,fault):
    root,old,new,old_name,new_name,resources=alias_fixture(tmp_path)
    path=root/old['dataset_path'];metadata=json.loads(gzip.decompress(path.read_bytes()))
    if fault=='old_dataset_hash':old['dataset_sha256']='a'*64
    else:
        field={'old_network_path':'compressed_path','old_compressed_digest':'compressed_sha256','old_native_digest':'native_sha256'}[fault]
        metadata['network'][field]='other.xml.gz' if field=='compressed_path' else 'b'*64
        path.write_bytes(gzip.compress(json.dumps(metadata).encode(),mtime=0));old['dataset_sha256']=audit.sha(path)
    with pytest.raises((ValueError,FileNotFoundError)):depth.equivalent_public_inputs(new,old,root=root)
    assert not (root/new['dataset_path']).exists()
