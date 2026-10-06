"""Lossless local model archives for the two reviewed endpoint development loops."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def unpack(directory):
    directory = Path(directory)
    metadata = json.loads((directory/'model_archive.json').read_text())
    packed = directory/metadata['compressed_path']
    assert hashlib.sha256(packed.read_bytes()).hexdigest() == metadata['compressed_sha256']
    content = gzip.decompress(packed.read_bytes())
    assert hashlib.sha256(content).hexdigest() == metadata['original_sha256']
    target = directory/metadata['original_path']
    if target.exists():
        assert target.read_bytes() == content, 'Refuse to replace another model'
    else:
        target.write_bytes(content)
    print('Verified and restored', target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    unpack(parser.parse_args().directory)
