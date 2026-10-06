import copy
import gzip
import hashlib
import json

import pytest

from experiments import export_endpoint_public_evidence_20261006 as exporter


def test_nested_private_fields_removed_without_changing_input_or_public_values():
    original = {'rows': [{'rng_seed_evaluator_only': 42,
        'events': [{'Q': [[1.2, 3.4]], 'time': 60}],
        'truth': [5.6, 7.8], 'other_seed_label': 'public configuration'}],
        'nested': {'master_hex': 'test-only', 'value': None}}
    before = copy.deepcopy(original)
    clean, count = exporter.strip_seeds(original)
    assert count == 2 and original == before
    assert clean['rows'][0]['events'] == original['rows'][0]['events']
    assert clean['rows'][0]['truth'] == original['rows'][0]['truth']
    assert clean['rows'][0]['other_seed_label'] == 'public configuration'
    assert clean['nested'] == {'value': None}
    assert exporter.strip_seeds(clean)[1] == 0


def create_sources(path):
    path.mkdir()
    originals = {}
    for split in exporter.SPLITS:
        for method in exporter.METHODS:
            name = f'{split}-{method}.json.gz'
            payload = [{'rng_seed_evaluator_only': 42,
                        'events': [{'time': 60, 'Q': [[1.2, 3.4]]}],
                        'truth': [5.6, 7.8], 'method': method}]
            raw = gzip.compress(json.dumps(payload).encode(), mtime=0)
            (path/name).write_bytes(raw)
            originals[name] = raw, payload
    return originals


def test_export_roundtrip_provenance_and_write_once(monkeypatch, tmp_path):
    monkeypatch.setattr(exporter, 'ROOT', tmp_path)
    source, out = tmp_path/'source', tmp_path/'public'
    originals = create_sources(source)
    manifest = exporter.export(source, out)
    assert len(manifest['files']) == 9
    for name, (raw, original) in originals.items():
        record = manifest['files'][name]
        exported = (out/name).read_bytes()
        decoded = json.loads(gzip.decompress(exported))
        assert (source/name).read_bytes() == raw
        assert record['source_sha256'] == hashlib.sha256(raw).hexdigest()
        assert record['exported_sha256'] == hashlib.sha256(exported).hexdigest()
        assert record['removed_private_fields'] == 1
        assert decoded == exporter.strip_seeds(original)[0]
    before = (out/'manifest.json').read_bytes()
    with pytest.raises(FileExistsError):
        exporter.export(source, out)
    assert (out/'manifest.json').read_bytes() == before


def test_missing_private_field_rejects_before_creating_output(monkeypatch, tmp_path):
    monkeypatch.setattr(exporter, 'ROOT', tmp_path)
    source, out = tmp_path/'source', tmp_path/'public'
    create_sources(source)
    (source/'fit-scale025_L20.json.gz').write_bytes(gzip.compress(b'[{"events": []}]', mtime=0))
    with pytest.raises(ValueError, match='private sampler'):
        exporter.export(source, out)
    assert not out.exists()
