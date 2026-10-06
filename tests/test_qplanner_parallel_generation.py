"""Parallel execution integrity tests using synthetic evidence, never scores."""
import json
import os
from pathlib import Path
import tempfile

import pytest

from experiments import qplanner_parallel_generation_20261006 as parallel


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def sealed_fixture(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir()
    monkeypatch.setattr(parallel.common,'ROOT',root)
    source=root/'engine.py';source.write_text('PUBLIC_CONSTANT=5\n')
    dataset=root/'dataset.json';save(dataset,{'families':[
        {'family_id':'train1','split':'train'}, {'family_id':'selection1','split':'selection'}]})
    protocol=dict(configuration={'methods':{'legacy_l10':{'mode':'legacy_l10'}}},
        dataset_path='dataset.json',dataset_sha256=parallel.common.sha(dataset),
        source_sha256={'engine.py':parallel.common.sha(source)},draw_count=2,
        draws_by_split={'train':1,'selection':3},splits=['train','selection'],
        public_inputs_sha256={},status='DEVELOPMENT')
    out=root/'old';out.mkdir();save(out/'protocol.json',protocol)
    (out/'protocol.sha256').write_text(parallel.common.sha(out/'protocol.json')+'\n')
    (out/'source_snapshot').mkdir();(out/'source_snapshot/engine.py').write_bytes(source.read_bytes())
    (out/'families').mkdir()
    jobs=parallel.jobs_for(parallel.common.read(dataset),protocol)
    return root,out,protocol,jobs


def bundle(job):
    return dict(public={'streams':{'raw':[{}]*8,'legacy_l10':[{}]*8}},
        evaluator_only={**{k:job[k] for k in ('family_id','split','draw')},'sessions':[{}]*8},
        utility=[{'retained':'all original values'}],wire=[{'retained':'all original values'}])


def block(out,job):
    path=out/'families'/job['name'];parallel.common.compressed_save(path,bundle(job));return path


def test_fixed_submission_schedule_preserves_all_draws_and_selection_first(tmp_path,monkeypatch):
    _,_,_,jobs=sealed_fixture(tmp_path,monkeypatch)
    assert [j['name'] for j in jobs]==['selection1--draw1.json.gz','selection1--draw2.json.gz',
        'selection1--draw3.json.gz','train1--draw1.json.gz']


def test_transfer_accepts_only_full_prefix_and_verifies_block_identity(tmp_path,monkeypatch):
    _,out,p,jobs=sealed_fixture(tmp_path,monkeypatch)
    block(out,jobs[0]);block(out,jobs[1])
    receipt=parallel.audit_transfer(out,p,jobs)
    assert list(receipt['family_files_sha256'])==[j['name'] for j in jobs[:2]]
    block(out,jobs[3])
    with pytest.raises(ValueError,match='complete serial'):
        parallel.audit_transfer(out,p,jobs)
    (out/'families'/jobs[3]['name']).unlink()
    path=out/'families'/jobs[0]['name'];value=bundle(jobs[0]);value['evaluator_only']['draw']=99
    path.unlink();parallel.common.compressed_save(path,value)
    with pytest.raises(ValueError,match='identity'):
        parallel.audit_transfer(out,p,jobs)


def test_import_rejects_config_mismatch_source_snapshot_tampering_and_manifest_tampering(tmp_path,monkeypatch):
    _,out,p,jobs=sealed_fixture(tmp_path,monkeypatch)
    block(out,jobs[0]);different={**p,'draws_by_split':{'selection':2,'train':1}}
    with pytest.raises(ValueError,match='draws_by_split'):
        parallel.audit_transfer(out,different,jobs)
    source=out/'source_snapshot/engine.py';source.write_text('PUBLIC_CONSTANT=99\n')
    with pytest.raises(ValueError,match='snapshot'):
        parallel.audit_transfer(out,p,jobs)
    source.write_text('PUBLIC_CONSTANT=5\n')
    save(out/'generation.json',{'family_files_sha256':{jobs[0]['name']:'0'*64}})
    with pytest.raises(ValueError,match='generation manifest'):
        parallel.audit_transfer(out,p,jobs)


def test_audited_transfer_is_byte_identical_and_changed_input_is_rejected(tmp_path,monkeypatch):
    _,source,p,jobs=sealed_fixture(tmp_path,monkeypatch);original=block(source,jobs[0])
    receipt=parallel.audit_transfer(source,p,jobs)
    out=tmp_path/'new';out.mkdir();save(out/'execution_protocol.json',{'public':'nonsecret'})
    plan={'transfer':receipt,'private_transfer':None}
    result=parallel.transfer_completed(out,plan)
    assert (out/'families'/original.name).read_bytes()==original.read_bytes()
    assert result==receipt['family_files_sha256']
    assert not list(out.rglob('*.key'))
    original.write_bytes(original.read_bytes()+b'changed')
    rejected=tmp_path/'rejected';rejected.mkdir();save(rejected/'execution_protocol.json',{})
    with pytest.raises(ValueError,match='changed after declaration'):
        parallel.transfer_completed(rejected,plan)


def test_private_key_transfer_never_copies_ledgers_or_exports_key_material(tmp_path):
    with tempfile.TemporaryDirectory(dir='/private/tmp',prefix='parallel-transfer-test-') as local:
        local=Path(local);old=local/'old';old.mkdir();work=local/'new'
        (old/'selection1--draw1.key').write_bytes(b'q'*32)
        (old/'selection1--draw1.sqlite').write_bytes(b'private-ledger')
        checkpoint=work/'private_transfer_checkpoint';checkpoint.mkdir(parents=True)
        (checkpoint/'selection1--draw1.key').write_bytes(b'q'*32)
        out=tmp_path/'out';out.mkdir();save(out/'execution_protocol.json',{'public':'nonsecret'})
        e=dict(transfer=None,workdir=str(work),private_transfer={
            'source_directory':str(old),'retained_key_blocks':['selection1--draw1.json.gz']})
        parallel.transfer_completed(out,e)
        key=work/'private_state/selection1--draw1/selection1--draw1.key'
        assert key.read_bytes()==b'q'*32 and key.stat().st_mode&0o777==0o600
        assert not list(work.rglob('*.sqlite')) and not list(out.rglob('*.key'))
        public=(out/'transfer_receipt.json').read_text()
        assert 'q'*32 not in public and 'private-ledger' not in public
        assert json.loads(public)['private_key_bytes_or_hashes_exported'] is False


def test_changed_private_key_is_rejected_privately(tmp_path):
    with tempfile.TemporaryDirectory(dir='/private/tmp',prefix='parallel-transfer-guard-') as local:
        local=Path(local);old=local/'old';old.mkdir();work=local/'new'
        (old/'selection1--draw1.key').write_bytes(b'z'*32)
        checkpoint=work/'private_transfer_checkpoint';checkpoint.mkdir(parents=True)
        (checkpoint/'selection1--draw1.key').write_bytes(b'q'*32)
        out=tmp_path/'out';out.mkdir();save(out/'execution_protocol.json',{})
        e=dict(transfer=None,workdir=str(work),private_transfer={
            'source_directory':str(old),'retained_key_blocks':['selection1--draw1.json.gz']})
        with pytest.raises(ValueError,match='private key changed') as error:
            parallel.transfer_completed(out,e)
        assert 'z'*32 not in str(error.value) and 'q'*32 not in str(error.value)


def test_public_cache_whitelist_and_private_paths_exclude_secrets_and_symlinks(tmp_path):
    with tempfile.TemporaryDirectory(dir='/private/tmp',prefix='parallel-cache-test-') as directory:
        cache=Path(directory)
        for name in parallel.PUBLIC_CACHE_FILES:(cache/name).write_bytes(name.encode())
        (cache/'never-copy.key').write_bytes(b'x'*32)
        manifest=parallel.cache_manifest(cache)
        assert set(manifest)==set(parallel.PUBLIC_CACHE_FILES)
        (cache/'reply10.npz').unlink();(cache/'reply10.npz').symlink_to(cache/'never-copy.key')
        with pytest.raises(ValueError,match='regular-file'):
            parallel.cache_manifest(cache)
    with pytest.raises(ValueError,match='/private/tmp'):
        parallel.private_path(Path(__file__).parent)


def test_execution_entrypoint_and_cache_hashes_are_checked_separately_from_common_protocol(tmp_path,monkeypatch):
    root,out,p,_=sealed_fixture(tmp_path,monkeypatch)
    (root/'executor.py').write_text('EXECUTION_VERSION=2\n')
    def pins():return {**p['source_sha256'],'executor.py':parallel.common.sha(root/'executor.py')}
    monkeypatch.setattr(parallel,'source_pins',pins)
    with tempfile.TemporaryDirectory(dir='/private/tmp',prefix='parallel-source-test-') as directory:
        cache=Path(directory)
        for name in parallel.PUBLIC_CACHE_FILES:(cache/name).write_bytes(name.encode())
        e=dict(common_protocol_sha256=parallel.common.sha(out/'protocol.json'),source_sha256=pins(),
            public_cache_source=str(cache),public_cache_files_sha256=parallel.cache_manifest(cache))
        save(out/'execution_protocol.json',e)
        (out/'execution_protocol.sha256').write_text(parallel.common.sha(out/'execution_protocol.json')+'\n')
        for name in e['source_sha256']:
            path=out/'execution_source_snapshot'/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((root/name).read_bytes())
        assert parallel.validate_execution(out)[1]==e
        (root/'executor.py').write_text('EXECUTION_VERSION=99\n')
        with pytest.raises(ValueError,match='source closure'):
            parallel.validate_execution(out)
        (root/'executor.py').write_text('EXECUTION_VERSION=2\n')
        (cache/'reply20.npz').write_bytes(b'different public cache')
        with pytest.raises(ValueError,match='cache changed'):
            parallel.validate_execution(out)


def test_additive_execution_preserves_predeclared_freeze_and_rejects_already_scored_output(tmp_path,monkeypatch):
    root,out,p,_=sealed_fixture(tmp_path,monkeypatch)
    (out/'families').rmdir()
    freeze=dict(configuration=p['configuration'],fresh_protocol_sha256=parallel.common.sha(out/'protocol.json'),
                fresh_test_evaluated=False,selected='legacy_l10')
    save(out/'freeze.json',freeze)
    original_protocol=(out/'protocol.json').read_bytes();original_freeze=(out/'freeze.json').read_bytes()
    monkeypatch.setattr(parallel.common,'declare',lambda *args,**kwargs:parallel.common.validate(out))
    monkeypatch.setattr(parallel,'source_pins',lambda:p['source_sha256'])
    with tempfile.TemporaryDirectory(dir='/private/tmp',prefix='parallel-predeclared-test-') as directory:
        work=Path(directory);cache=work/'public';cache.mkdir()
        for name in parallel.PUBLIC_CACHE_FILES:(cache/name).write_bytes(name.encode())
        result=parallel.declare_execution(out,work/'work',dataset=root/'dataset.json',methods=['legacy_l10'],
            draws=2,splits=p['splits'],status=p['status'],draws_by_split=p['draws_by_split'],public_cache=cache)
        assert result['predeclared_files_sha256']=={'freeze.json':parallel.common.sha(out/'freeze.json')}
        assert (out/'protocol.json').read_bytes()==original_protocol
        assert (out/'freeze.json').read_bytes()==original_freeze
        assert parallel.validate_execution(out)[1]==result
        with pytest.raises(FileExistsError,match='untouched predeclared'):
            parallel.declare_execution(out,work/'second',dataset=root/'dataset.json',methods=['legacy_l10'],
                draws=2,splits=p['splits'],status=p['status'],draws_by_split=p['draws_by_split'],public_cache=cache)
