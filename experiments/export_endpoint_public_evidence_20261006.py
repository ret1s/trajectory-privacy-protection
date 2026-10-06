"""Export NEW evaluator bundles without sampler seeds; never alter originals.

Outputs still contain synthetic evaluator truth, not just the attacker view.
They are not substitutes for private exact-RNG verification. Original protocols,
outputs and receipts retain their original hashes.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'artifacts/benchmarks/endpoint_generalization_20261005'
METHODS = ('scale025_L20', 'scale100_L10', 'scale100_L20')
SPLITS = ('fit', 'selection', 'test')
PRIVATE_FIELDS = frozenset(('rng_seed_evaluator_only', 'master_hex'))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strip_seeds(value):
    """Return an independent JSON copy and count removed private fields."""
    removed = 0
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in PRIVATE_FIELDS:
                removed += 1
                continue
            result[key], count = strip_seeds(item)
            removed += count
        return result, removed
    if isinstance(value, list):
        result = []
        for item in value:
            clean, count = strip_seeds(item)
            result.append(clean)
            removed += count
        return result, removed
    return value, 0


def export(source, out):
    source, out = Path(source), Path(out)
    inputs = []
    for split in SPLITS:
        for method in METHODS:
            name = f'{split}-{method}.json.gz'
            raw = (source/name).read_bytes()
            decoded = json.loads(gzip.decompress(raw))
            clean, removed = strip_seeds(decoded)
            if not removed:
                raise ValueError('Expected private sampler field missing: '+name)
            encoded = gzip.compress((json.dumps(clean, ensure_ascii=False, indent=2)+'\n').encode(), mtime=0)
            inputs.append((name, raw, decoded, clean, encoded, removed))
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema='endpoint-public-evaluator-export-v1',
        exported_on='2026-10-06', source_directory=str(source.relative_to(ROOT)),
        exporter_sha256=digest(Path(__file__).read_bytes()),
        transformation='Only rng_seed_evaluator_only/master_hex fields omitted recursively; every other JSON value including Q, labels and predictions unchanged.',
        scope='Synthetic evaluator bundles; only events represent simulated attacker-visible data. NOT exact private-RNG replay or production-safe sampler evidence.',
        files={})
    for name, raw, original, clean, encoded, removed in inputs:
        with (out/name).open('xb') as stream:
            stream.write(encoded)
        reread = json.loads(gzip.decompress((out/name).read_bytes()))
        assert reread == clean
        assert strip_seeds(original)[0] == reread
        assert strip_seeds(reread)[1] == 0
        assert (source/name).read_bytes() == raw
        manifest['files'][name] = dict(source_sha256=digest(raw),
            exported_sha256=digest(encoded), removed_private_fields=removed,
            all_other_values_identical=True, original_unchanged=True)
    with (out/'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=SOURCE/'public_release_20261006')
    args = parser.parse_args()
    manifest = export(args.source, args.output)
    print('Exported', len(manifest['files']), 'new seed-free bundles; originals unchanged.')
