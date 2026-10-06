"""Source tamper checks for the independent native verifier."""
import gzip
import hashlib
import pytest

pytest.importorskip('sumolib')

from experiments.verify_qplanner_fresh_native_20261006 import archived_native, public_overlap


def test_native_archive_checks_both_compressed_and_original_source(tmp_path):
    original = b'<fcd-export>native source</fcd-export>\n'
    archive = tmp_path/'fcd.xml.gz'
    archive.write_bytes(gzip.compress(original, mtime=0))
    target = tmp_path/'restore'
    target.mkdir()
    metadata = {'fcd.xml': {'compressed_path': str(archive),
        'compressed_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'native_sha256': hashlib.sha256(original).hexdigest()}}
    restored = archived_native(metadata, target)
    assert restored['fcd.xml'].read_bytes() == original
    metadata['fcd.xml']['native_sha256'] = '0'*64
    with pytest.raises(AssertionError, match='native_source_changed'):
        archived_native(metadata, target)
    archive.write_bytes(gzip.compress(b'different valid gzip', mtime=0))
    with pytest.raises(AssertionError, match='archive_changed'):
        archived_native(metadata, target)


def test_independent_near_duplicate_audit_is_set_based():
    assert public_overlap(['a', 'b', 'b'], ['b', 'a']) == 1.
    assert public_overlap(['a', 'b'], ['b', 'c']) == pytest.approx(1/3)
    assert public_overlap(['a'], ['z']) == 0.
