"""Private-key digests are also banned, independently of raw key exports."""
import base64
import gzip
import hashlib
import json

import pytest

from experiments import audit_public_evidence_boundary_20261006_v2 as boundary


@pytest.mark.parametrize('encoding',['hex','HEX','base64','urlsafe','unpadded','raw'])
def test_digest_only_gzip_export_is_detected_without_printing_digest(tmp_path,encoding):
    private=tmp_path/'private';private.mkdir();key=bytes(range(32))
    (private/'one.key').write_bytes(key);digest=hashlib.sha256(key).digest()
    encoded={'hex':digest.hex().encode(),'HEX':digest.hex().upper().encode(),
        'base64':base64.b64encode(digest),'urlsafe':base64.urlsafe_b64encode(digest),
        'unpadded':base64.b64encode(digest).rstrip(b'='),'raw':digest}[encoding]
    patterns,counts=boundary.private_patterns([private])
    assert counts['sha256_digest_encoding_variant_count']>0
    (tmp_path/'evidence.json.gz').write_bytes(gzip.compress(b'prefix '+encoded+b' suffix'))
    value=boundary.original.audit_files(tmp_path,['evidence.json.gz'],patterns)
    assert value['status']=='fail' and value['private_key_matching_gzip_expanded_file_count']==1
    report=json.dumps(value)
    assert digest.hex() not in report and base64.b64encode(digest).decode() not in report


def test_nonsecret_source_digest_is_allowed_and_original_source_unchanged(tmp_path):
    private=tmp_path/'private';private.mkdir();(private/'one.key').write_bytes(bytes(range(32)))
    patterns,_=boundary.private_patterns([private])
    public=hashlib.sha256(b'nonsecret source file').hexdigest()
    (tmp_path/'source.json').write_text(json.dumps({'source_sha256':public}))
    value=boundary.original.audit_files(tmp_path,['source.json'],patterns)
    assert value['status']=='pass' and value['private_key_matching_file_count']==0
