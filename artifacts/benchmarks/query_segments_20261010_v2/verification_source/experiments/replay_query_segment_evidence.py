"""Verify published segment evidence using its immutable source snapshot.

This avoids treating later maintenance of a shared helper as the source of an
older experiment. Public inputs are hash-checked, private keys are never needed.
The temporary root contains frozen Python sources and read-only-use artifact
links; verification reuses existing receipts instead of producing new results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'artifacts/benchmarks/query_segments_20261010_v2'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(root,name):
    path=Path(name)
    if path.is_absolute() or not (root/path).resolve().is_relative_to(root.resolve()):
        raise ValueError('Safe root-relative artifact/source path required')
    return root/path


def replay(output=DEFAULT):
    output=Path(output).resolve()
    output.relative_to(ROOT)
    p=json.loads((output/'protocol.json').read_text())
    generation=json.loads((output/'generation.json').read_text())
    if not (output/'independent_validation.json').is_file():
        raise ValueError('Completed independently reviewed evidence required')
    with tempfile.TemporaryDirectory(prefix='frozen-query-segments-') as td:
        root=Path(td)
        for name,digest in p['source_sha256'].items():
            source=relative(output/'source_snapshot',name)
            if sha(source)!=digest:raise ValueError('Frozen source snapshot changed: '+name)
            target=relative(root,name);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
        for name,digest in p['public_inputs_sha256'].items():
            source=relative(ROOT,name)
            if sha(source)!=digest:raise ValueError('Frozen public input changed: '+name)
            target=relative(root,name);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
        # Resolve the dataset from the FROZEN generator's literal public path.
        generator=root/'experiments/build_query_segment_fresh_native_20261010.py'
        match=re.search(r"^OUT = ROOT/'([^']+)'",generator.read_text(),re.M)
        if match is None:raise ValueError('No frozen public dataset path')
        dataset=relative(ROOT,match[1])
        if sha(dataset/'dataset.json.gz')!=generation['dataset_sha256']:
            raise ValueError('Frozen cohort bytes changed')
        for source,name in ((dataset,match[1]),(output,str(output.relative_to(ROOT)))):
            target=relative(root,name);target.parent.mkdir(parents=True,exist_ok=True)
            target.symlink_to(source,target_is_directory=True)
        checker=ROOT/'experiments/verify_query_segments_20261010.py'
        receipt=json.loads((output/'independent_validation.json').read_text())
        if sha(checker)!=receipt['verifier_sha256']:raise ValueError('Independent verifier changed; restore its verification snapshot')
        shutil.copyfile(checker,root/'experiments'/checker.name)
        env=dict(os.environ,PYTHONPATH=str(root))
        subprocess.run([sys.executable,'-m','experiments.verify_query_segments_20261010',
                        '--output',str(root/output.relative_to(ROOT))],cwd=root,env=env,check=True)
    print('PASS: frozen-source replay; no private keys, refit, resampling or evidence replacement')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    replay(parser.parse_args().output)
