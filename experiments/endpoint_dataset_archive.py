"""Lossless archive for the 32-family dataset, whose JSON exceeds GitHub's file limit."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

from experiments.research_loop_resources import ROOT, sha

DATA = ROOT / 'artifacts/datasets/endpoint_holdout_expanded_v1'


def main(mode):
    raw, archive, manifest = (DATA / name for name in ('dataset.json', 'dataset.json.gz', 'archive.json'))
    if mode == 'pack':
        with raw.open('rb') as source, archive.open('wb') as target:
            with gzip.GzipFile(filename='', fileobj=target, mode='wb', mtime=0) as zipped:
                shutil.copyfileobj(source, zipped)
        meta = {'raw_sha256': sha(raw), 'raw_bytes': raw.stat().st_size,
                'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
                'restore_command': 'python -m experiments.endpoint_dataset_archive unpack'}
        manifest.write_text(json.dumps(meta, indent=2) + '\n')
        import hashlib
        digest = hashlib.sha256()
        with gzip.open(archive, 'rb') as source:
            for block in iter(lambda: source.read(1024*1024), b''): digest.update(block)
        assert digest.hexdigest() == meta['raw_sha256']
    else:
        meta = json.loads(manifest.read_text())
        assert sha(archive) == meta['archive_sha256']
        if raw.exists():
            assert sha(raw) == meta['raw_sha256'], 'Existing dataset differs; preserve and inspect it'
        else:
            temporary = raw.with_suffix('.unpacking')
            with gzip.open(archive, 'rb') as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target)
            assert sha(temporary) == meta['raw_sha256']
            temporary.rename(raw)
    print('Lossless dataset archive verified:', meta['raw_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['pack', 'unpack'])
    main(parser.parse_args().mode)
