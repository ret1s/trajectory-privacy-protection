"""Paired development integrity on synthetic blocks and isolated private keys."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pytest

from experiments import qplanner_parallel_generation_20261006_v2 as parallel


@pytest.fixture
def paired(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();monkeypatch.setattr(parallel.common,'ROOT',root)
    sources={}
    for name in parallel.PAIRED_PRIVACY_SOURCES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# frozen primitive\n')
        sources[name]=parallel.common.sha(path)
    data={'families':[{'family_id':'f1','split':'selection'},{'family_id':'f2','split':'train'}]}
    (root/'dataset.json').write_text(json.dumps(data))
    config={name:'fixed-public-value' for name in parallel.PAIRED_CONFIGURATION_FIELDS}
    config.update(methods={m:{'mode':m} for m in ('legacy_l10','aligned_nearest')},utility_slack=.03)
    old=dict(configuration=config,dataset_path='dataset.json',dataset_sha256=parallel.common.sha(root/'dataset.json'),
        source_sha256=sources,draw_count=1,draws_by_split={'selection':1,'train':1},
        splits=['train','selection'],public_inputs_sha256={},status='DEVELOPMENT')
    out=root/'old';out.mkdir();parallel.common.save(out/'protocol.json',old)
    (out/'protocol.sha256').write_text(parallel.common.sha(out/'protocol.json')+'\n')
    for name in sources:
        path=out/'source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((root/name).read_bytes())
    jobs=parallel.jobs_for(data,old);hashes={}
    (out/'families').mkdir()
    for job in jobs:
        payload=dict(public={'streams':{m:[{}]*8 for m in ('raw','legacy_l10','aligned_nearest')}},
            evaluator_only={**{k:job[k] for k in ('family_id','split','draw')},'sessions':[{}]*8})
        path=out/'families'/job['name'];parallel.common.compressed_save(path,payload);hashes[job['name']]=parallel.common.sha(path)
    parallel.common.save(out/'generation.json',{'family_files_sha256':hashes})
    target=deepcopy(old);target['configuration']['utility_slack']='per_method_declared_above'
    target['configuration']['methods']={m:{'mode':m,'utility_slack':.03} for m in ('legacy_l10','aligned_nearest')}
    target['configuration']['methods']['normalized_mean']={'mode':'normalized_mean','utility_slack':.03}
    with tempfile.TemporaryDirectory(dir='/private/tmp',prefix='paired-v2-unit-') as local:
        local=Path(local);private=local/'original';private.mkdir()
        for job in jobs:
            key=parallel._paired_key(private,job['name']);key.parent.mkdir();key.write_bytes(b'k'*32)
            key.with_suffix('.sqlite').write_bytes(b'never transfer ledger')
        yield out,target,jobs,private,local/'new'


def test_complete_paired_manifest_allows_new_objective_and_copies_only_private_keys(paired):
    out,target,jobs,private,work=paired
    plan=parallel.audit_paired_development(out,target,jobs,private)
    assert plan['retained_key_blocks']==[j['name'] for j in jobs]
    assert plan['matched_controls']==['legacy_l10','aligned_nearest']
    parallel._checkpoint_paired_keys(plan,work)
    assert parallel.transfer_paired_keys(plan,work)==2
    for job in jobs:
        key=parallel._paired_key(work/'private_state',job['name'])
        assert key.read_bytes()==b'k'*32 and key.stat().st_mode&0o777==0o600
        assert key.parent.stat().st_mode&0o777==0o700
    assert not list(work.rglob('*.sqlite'))
    assert 'k'*32 not in json.dumps(plan)


def test_missing_key_and_changed_checkpoint_are_rejected_without_material(paired):
    out,target,jobs,private,work=paired
    key=parallel._paired_key(private,jobs[0]['name']);key.unlink()
    with pytest.raises(ValueError,match='nested private key'):
        parallel.audit_paired_development(out,target,jobs,private)
    key.write_bytes(b'k'*32);plan=parallel.audit_paired_development(out,target,jobs,private)
    parallel._checkpoint_paired_keys(plan,work);key.write_bytes(b'z'*32)
    with pytest.raises(ValueError,match='private key changed') as error:
        parallel.transfer_paired_keys(plan,work)
    assert 'z'*32 not in str(error.value)


def test_configuration_and_primitive_changes_cannot_be_called_paired(paired):
    out,target,jobs,private,_=paired
    changed=deepcopy(target);changed['configuration']['budget']='different cap'
    with pytest.raises(ValueError,match='configuration differs: budget'):
        parallel.audit_paired_development(out,changed,jobs,private)
    changed=deepcopy(target);changed['source_sha256']['core/mechanisms.py']='0'*64
    with pytest.raises(ValueError,match='privacy/control source'):
        parallel.audit_paired_development(out,changed,jobs,private)
    changed=deepcopy(target);changed['splits'].append('test')
    with pytest.raises(ValueError,match='fresh TEST'):
        parallel.audit_paired_development(out,changed,jobs,private)


def test_old_blocks_and_manifest_are_rechecked_after_declaration(paired):
    out,target,jobs,private,_=paired
    plan=parallel.audit_paired_development(out,target,jobs,private)
    parallel.validate_paired_development(plan,target,jobs)
    block=out/'families'/jobs[0]['name'];block.write_bytes(block.read_bytes()+b'tampered')
    with pytest.raises(ValueError,match='block changed'):
        parallel.validate_paired_development(plan,target,jobs)


def test_old_source_generation_pin_and_extra_blocks_are_rejected(paired):
    out,target,jobs,private,_=paired
    plan=parallel.audit_paired_development(out,target,jobs,private)
    (out/'generation.json').write_text('{}')
    with pytest.raises(ValueError,match='source file changed'):
        parallel.validate_paired_development(plan,target,jobs)
    parallel.common.save(out/'families/extra.json',{})
    (out/'generation.json').write_text(json.dumps({'family_files_sha256':plan['family_files_sha256']}))
    with pytest.raises(ValueError,match='missing/extra'):
        parallel.audit_paired_development(out,target,jobs,private)
