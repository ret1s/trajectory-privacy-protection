"""Source-only depth freeze and independent whole-family readout contracts."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import qplanner_response_depth_generalization_20261006_v2 as study
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
    names=[CORE,PROFILE,PUBLIC,study.THIS,study.PREVIOUS_THIS,study.BASE_VERIFIER,study.DEV_VERIFIER,*study.VERIFIER_HELPERS]
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
    write(development/'paired_protocol.json',{'analysis_source_sha256':pin(root/study.PREVIOUS_THIS)})
    write(development/'paired_readout.json',{'source_readout_sha256':pin(development/'readout.json')})
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


def archive_pair(root):
    fresh={'dataset_path':'artifacts/datasets/fresh/dataset.json.gz',
        'public_inputs_sha256':{PUBLIC:pin(root/PUBLIC)}}
    old={'dataset_path':'artifacts/datasets/old_v2/dataset.json.gz',
        'public_inputs_sha256':{PUBLIC:pin(root/PUBLIC)}}
    for protocol, directory in ((fresh, 'fresh'), (old, 'old_v1')):
        name='artifacts/datasets/'+directory+'/public_native.net.xml.gz'
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(gzip.compress(b'<net>byte-identical public archive fixture</net>',mtime=0))
        protocol['public_inputs_sha256'][name]=pin(path)
    old_archive='artifacts/datasets/old_v1/public_native.net.xml.gz'
    old_path=root/old['dataset_path'];old_path.parent.mkdir(parents=True,exist_ok=True)
    network={'compressed_path':old_archive,'compressed_sha256':pin(root/old_archive),
        'native_sha256':hashlib.sha256(gzip.decompress((root/old_archive).read_bytes())).hexdigest()}
    old_path.write_bytes(gzip.compress(json.dumps({'network':network}).encode(),mtime=0))
    old['dataset_sha256']=pin(old_path)
    return fresh,old


def test_only_same_archive_bytes_may_relocate_beside_declared_dataset(tmp_path):
    fixture(tmp_path);fresh,old=archive_pair(tmp_path)
    result=study.equivalent_public_inputs(fresh,old,root=tmp_path)
    assert result['native_archive_relocated'] is True
    assert result['all_other_public_inputs_identical'] is True
    assert result['identical_archive_sha256']==pin(tmp_path/result['fresh_native_archive_path'])
    assert 'old_v2' in old['dataset_path'] and 'old_v1' in result['development_native_archive_path']
    assert result['development_dataset_sha256']==pin(tmp_path/old['dataset_path'])
    assert not (tmp_path/fresh['dataset_path']).exists()


def test_full_freeze_accepts_only_metadata_path_relocation_with_no_fresh_dataset(tmp_path):
    dev,base,out=fixture(tmp_path);fresh,old=archive_pair(tmp_path)
    original=Path(load(dev/'protocol.json')['source_output'])
    q=load(original/'protocol.json');q.update(old);write(original/'protocol.json',q)
    b=load(base/'protocol.json');b.update(fresh);write(base/'protocol.json',b)
    (base/'protocol.sha256').write_text(pin(base/'protocol.json')+'\n')
    p=load(dev/'protocol.json');p['source_files_sha256']['protocol.json']=pin(original/'protocol.json')
    write(dev/'protocol.json',p);(dev/'protocol.sha256').write_text(pin(dev/'protocol.json')+'\n')
    r=load(dev/'readout.json');r['protocol_sha256']=pin(dev/'protocol.json');write(dev/'readout.json',r)
    s=load(dev/'depth_selection.json');s.update(protocol_sha256=pin(dev/'protocol.json'),readout_sha256=pin(dev/'readout.json'));write(dev/'depth_selection.json',s)
    v=load(dev/'validation.json');v.update(protocol_sha256=pin(dev/'protocol.json'),readout_sha256=pin(dev/'readout.json'));write(dev/'validation.json',v)
    p=study.declare_freeze(dev,base,out,root=tmp_path)
    assert p['public_input_equivalence']['native_archive_relocated'] is True
    assert study.depth_contract(out,root=tmp_path)['status']=='pass'
    assert not (tmp_path/p['dataset_path']).exists()


def test_same_path_same_claimed_digest_still_rejects_changed_archive_bytes(tmp_path):
    fixture(tmp_path);fresh,old=archive_pair(tmp_path)
    fresh['public_inputs_sha256']=deepcopy(old['public_inputs_sha256'])
    path=tmp_path/'artifacts/datasets/old_v1/public_native.net.xml.gz'
    path.write_bytes(b'changed archive after source seal')
    with pytest.raises(ValueError,match='Unrelocated public input bytes'):
        study.equivalent_public_inputs(fresh,old,root=tmp_path)


@pytest.mark.parametrize('fault',['bytes_tampered','new_digest','extra_resource','arbitrary_native_path','same_path_bad_bytes'])
def test_archive_alias_cannot_hide_content_or_other_resource_changes(tmp_path,fault):
    fixture(tmp_path);fresh,old=archive_pair(tmp_path)
    path=str(Path(fresh['dataset_path']).parent/'public_native.net.xml.gz')
    if fault in ('bytes_tampered','new_digest'):
        (tmp_path/path).write_bytes(b'changed public network bytes')
        if fault=='new_digest':fresh['public_inputs_sha256'][path]=pin(tmp_path/path)
    elif fault=='extra_resource':fresh['public_inputs_sha256']['artifacts/public/extra']='0'*64
    elif fault=='arbitrary_native_path':
        digest=fresh['public_inputs_sha256'].pop(path)
        fresh['public_inputs_sha256']['artifacts/unrelated/public_native.net.xml.gz']=digest
    else:
        # Same relocated path assertion with a different pinned digest still fails.
        fresh['public_inputs_sha256'][path]='0'*64
    with pytest.raises(ValueError):study.equivalent_public_inputs(fresh,old,root=tmp_path)


@pytest.mark.parametrize('fault',['dataset_bytes','metadata_archive','compressed_digest','native_digest'])
def test_old_network_metadata_must_be_authenticated_before_relocation(tmp_path,fault):
    fixture(tmp_path);fresh,old=archive_pair(tmp_path);path=tmp_path/old['dataset_path']
    data=study.read(path)
    if fault=='dataset_bytes':path.write_bytes(b'changed before metadata read')
    else:
        field={'metadata_archive':'compressed_path','compressed_digest':'compressed_sha256',
               'native_digest':'native_sha256'}[fault]
        data['network'][field]='artifacts/elsewhere/public_native.net.xml.gz' if fault=='metadata_archive' else '0'*64
        path.write_bytes(gzip.compress(json.dumps(data).encode(),mtime=0));old['dataset_sha256']=pin(path)
    with pytest.raises(ValueError):study.equivalent_public_inputs(fresh,old,root=tmp_path)
