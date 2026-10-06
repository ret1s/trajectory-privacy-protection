"""Secret matching uses only sanitized counts, including compressed exports."""
import base64
import gzip
import json
import subprocess

import pytest

from experiments import audit_public_evidence_boundary_20261006 as boundary


@pytest.fixture
def private(tmp_path):
    work=tmp_path/'private_work';work.mkdir()
    key=bytes(range(32));(work/'one.key').write_bytes(key)
    copied=work/'copy';copied.mkdir();(copied/'another.key').write_bytes(key)
    patterns,counts=boundary.private_patterns([work])
    assert counts['private_key_file_count']==2 and counts['distinct_32byte_private_key_count']==1
    return key,patterns,counts


@pytest.mark.parametrize('encoding',['raw','hex','HEX','b64','url64','unpadded'])
def test_actual_secret_variants_inside_gzip_cannot_pass(tmp_path,private,encoding):
    key,patterns,_=private
    value={'raw':key,'hex':key.hex().encode(),'HEX':key.hex().upper().encode(),
        'b64':base64.b64encode(key),'url64':base64.urlsafe_b64encode(key),
        'unpadded':base64.b64encode(key).rstrip(b'=')}[encoding]
    (tmp_path/'evidence.json.gz').write_bytes(gzip.compress(b'prefix:'+value+b':end'))
    result=boundary.audit_files(tmp_path,['evidence.json.gz'],patterns)
    assert result['status']=='fail' and result['private_key_matching_gzip_expanded_file_count']==1
    assert result['gates']['no_private_key_bytes_raw_hex_or_base64'] is False
    serialized=json.dumps(result)
    assert key.hex() not in serialized and base64.b64encode(key).decode() not in serialized


def test_raw_binary_export_and_private_state_files_fail_without_exporting_secret(tmp_path,private):
    key,patterns,_=private
    (tmp_path/'binary.bin').write_bytes(b'bytes'+key)
    folder=tmp_path/'private_state';folder.mkdir();(folder/'one.key').write_bytes(key)
    (tmp_path/'unnamed.dat').write_bytes(b'SQLite format 3\0more')
    result=boundary.audit_files(tmp_path,['binary.bin','private_state/one.key','unnamed.dat'],patterns)
    assert result['actual_private_storage_path_count']==1
    assert result['actual_private_key_database_filename_count']==1 and result['sqlite_magic_header_count']==1
    assert result['private_key_matching_file_count']==2
    assert 'private_state/one.key' not in json.dumps(result)


def test_allowed_local_path_provenance_is_counted_separately_and_no_secret(tmp_path,private):
    _,patterns,_=private
    (tmp_path/'protocol.json').write_text('{"source_private_root":"/private/tmp/demo/private_state"}')
    result=boundary.audit_files(tmp_path,['protocol.json'],patterns)
    assert result['status']=='pass' and result['non_secret_private_storage_path_reference_file_count']==1
    assert result['actual_private_storage_path_count']==0


def test_bad_gzip_and_actual_oversize_are_failures(tmp_path,private,monkeypatch):
    _,patterns,_=private;monkeypatch.setattr(boundary,'GIT_FILE_LIMIT',100)
    (tmp_path/'bad.gz').write_bytes(b'not gzip');(tmp_path/'large.bin').write_bytes(b'z'*101)
    result=boundary.audit_files(tmp_path,['bad.gz','large.bin'],patterns)
    assert result['status']=='fail' and result['decoding_or_nonregular_errors']==1
    assert result['files_over_git_100MiB_limit']==1
    assert result['largest_new_files'][0]['bytes']==101


def test_git_inventory_excludes_ignored_private_files_and_tracked_modifications(tmp_path):
    subprocess.run(['git','init','-q',str(tmp_path)],check=True)
    (tmp_path/'.gitignore').write_text('ignored/\n')
    (tmp_path/'tracked.txt').write_text('first')
    subprocess.run(['git','add','.gitignore','tracked.txt'],cwd=tmp_path,check=True)
    subprocess.run(['git','-c','user.email=test@example.invalid','-c','user.name=test',
        'commit','-qm','fixture'],cwd=tmp_path,check=True)
    (tmp_path/'tracked.txt').write_text('modified')
    ignored=tmp_path/'ignored';ignored.mkdir();(ignored/'private.key').write_bytes(b'x'*32)
    (tmp_path/'new.txt').write_text('public')
    (tmp_path/'added.txt').write_text('public staged')
    subprocess.run(['git','add','added.txt'],cwd=tmp_path,check=True)
    assert boundary.git_new_files(tmp_path)==['added.txt','new.txt']
