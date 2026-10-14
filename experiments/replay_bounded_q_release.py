"""Replay exact evidence audits from immutable sources, without private keys."""
import argparse,hashlib,json,os,re,shutil,subprocess,sys,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TRIALS={
    'targeted_cache_20261014_v1':('targeted_cache_study_20261014','verify_targeted_cache_20261014'),
    'main_depth_20261014_v1':('main_depth_study_20261014','verify_main_depth_20261014'),
}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def safe(root,name):
    p=root/name
    if Path(name).is_absolute() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Safe root-relative path required')
    return p

def replay(output):
    output=output.resolve();relative=output.relative_to(ROOT)
    module,checker=TRIALS[output.name]
    protocol=json.loads((output/'protocol.json').read_text())
    receipt=json.loads((output/'independent_validation.json').read_text())
    assert sha(output/'protocol.json')==receipt['protocol_sha256']
    assert sha(output/'results.json')==receipt['results_sha256']
    assert sha(output/'source_snapshot'/('experiments/'+checker+'.py'))==receipt['verifier_sha256']
    with tempfile.TemporaryDirectory(prefix='bounded-Q-release-replay-') as td:
        root=Path(td)
        for name,digest in protocol['source_sha256'].items():
            source=safe(output/'source_snapshot',name);assert sha(source)==digest
            target=safe(root,name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        for name,digest in protocol['public_inputs_sha256'].items():
            source=safe(ROOT,name);assert sha(source)==digest
            target=safe(root,name);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        # The native input is intentionally the same previously archived cohort.
        dataset='artifacts/datasets/cache_fresh_native_20261012_v2'
        assert sha(ROOT/dataset/'dataset.json.gz')==protocol['pinned_reused_dataset_sha256']
        links=[(ROOT/dataset,dataset),(output,str(relative))]
        generator=(root/'experiments/build_cache_fresh_native_20261012_v2.py').read_text()
        for name in re.findall(r"^DEVELOPMENT(?:_2)? = ROOT/'([^']+)'",generator,re.M):
            links.append(((ROOT/name).parent,str(Path(name).parent)))
        for source,name in links:
            target=safe(root,name);target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(source,target_is_directory=True)
        frozen=(root/('experiments/'+module+'.py')).read_text()
        prior_work=Path(re.search(r"^WORK=Path\('([^']+)'\)",frozen,re.M).group(1))/'public_resources'
        cache=root/'public_resources';cache.mkdir()
        for name in ('native.net.xml','reference5.npz','reply30.npz','reply60.npz','belief01.npz'):
            if (prior_work/name).exists():shutil.copyfile(prior_work/name,cache/name)
        launch=f"from pathlib import Path; from experiments import {module} as study; study.WORK=Path({str(root)!r}); from experiments.{checker} import verify; verify(Path({str(root/relative)!r}))"
        subprocess.run([sys.executable,'-c',launch],cwd=root,env=dict(os.environ,PYTHONPATH=str(root)),check=True)
    print('PASS: immutable-source audit replay; no private keys/resampling/refitting')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    replay(p.parse_args().output)
