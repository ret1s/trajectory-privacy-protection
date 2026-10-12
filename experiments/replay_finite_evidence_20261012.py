"""Replay either retained12/10 study with frozen sources and no private keys."""
import argparse,hashlib,json,os,re,shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def safe(root,name):
    p=root/name
    if Path(name).is_absolute() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Safe root-relative path required')
    return p

def replay(output):
    output=output.resolve();relative=output.relative_to(ROOT)
    protocol=json.loads((output/'protocol.json').read_text());receipt=json.loads((output/'independent_validation.json').read_text())
    pair={'finite_pool_20261012_v1':('finite_pool','build_finite_pool_fresh_native_20261012'),
          'dominance_motion_20261012_v1':('dominance','build_dominance_fresh_native_20261012')}
    prefix,generator=pair[output.name];module=prefix+'_study_20261012';checker='verify_'+prefix+'_20261012'
    assert sha(output/'source_snapshot'/('experiments/'+checker+'.py'))==receipt['verifier_sha256']
    with tempfile.TemporaryDirectory(prefix='frozen-finite-Q-') as td:
        root=Path(td)
        for name,digest in protocol['source_sha256'].items():
            source=safe(output/'source_snapshot',name);assert sha(source)==digest
            target=safe(root,name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        for name,digest in protocol['public_inputs_sha256'].items():
            source=safe(ROOT,name);assert sha(source)==digest
            target=safe(root,name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        content=(root/('experiments/'+generator+'.py')).read_text()
        dataset_name=re.search(r"^OUT = ROOT/'([^']+)'",content,re.M).group(1)
        gen=json.loads((output/'generation.json').read_text());assert sha(ROOT/dataset_name/'dataset.json.gz')==gen['dataset_sha256']
        links=[(ROOT/dataset_name,dataset_name),(output,str(relative))]
        development=re.search(r"^DEVELOPMENT = ROOT/'([^']+)'",content,re.M)
        if development:
            name=development.group(1);links.append(((ROOT/name).parent,str(Path(name).parent)))
        for source,name in links:
            target=safe(root,name);target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source,target_is_directory=True)
        cache=root/'public_cache';cache.mkdir()
        # Optional immutable public resource acceleration; no keys/SQLite/RNG.
        canonical_cache=Path('/private/tmp/trajectory-'+('finite-pool' if prefix=='finite_pool' else 'dominance')+'-study-20261012-v1/public_resources')
        for name in ('native.net.xml','reference5.npz','reply30.npz','belief01.npz'):
            if (canonical_cache/name).exists():shutil.copyfile(canonical_cache/name,cache/name)
        launcher=f"from pathlib import Path; from experiments import {module} as study; study.WORK=Path({str(root)!r}); from experiments.{checker} import verify; verify(Path({str(root/relative)!r}))"
        # resources() appends public_resources to WORK.
        cache.rename(root/'public_resources')
        subprocess.run([sys.executable,'-c',launcher],cwd=root,env=dict(os.environ,PYTHONPATH=str(root)),check=True)
    print('PASS: frozen source replay without private keys, resampling or refitting')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    replay(parser.parse_args().output)
