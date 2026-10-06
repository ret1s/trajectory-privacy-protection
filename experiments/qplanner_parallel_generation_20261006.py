"""Write-once parallel execution of the unchanged frozen Q-planner study.

Whole family/draw blocks are submitted in the original fixed order. Completion
can be out of order; receipts and the final manifest retain submission order.
Completed serial-prefix files are imported byte-for-byte, never resampled.
Only private keys, never SQLite ledgers, may be transferred outside the repo.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import time
import traceback

from threadpoolctl import threadpool_info

from experiments import qplanner_study_20261006 as common


PUBLIC_CACHE_FILES = ('native.net.xml', 'reference5.npz', 'reply10.npz',
                      'reply20.npz', 'belief-0.01.npz', 'belief-0.00125.npz')
_WORKER = None


def now():
    return datetime.now(timezone.utc).isoformat()


def jobs_for(data, protocol):
    families = [f for f in data['families'] if f['split'] in protocol['splits']]
    families.sort(key=lambda f: (('selection', 'train', 'test').index(f['split']), f['family_id']))
    if not families or any(not any(f['split'] == s for f in families) for s in protocol['splits']):
        raise ValueError('Every declared split must contain families')
    jobs = [dict(name=f'{f["family_id"]}--draw{d}.json.gz', family_id=f['family_id'],
                 split=f['split'], draw=d)
            for f in families for d in range(1, protocol['draws_by_split'][f['split']]+1)]
    if len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate family/draw identity')
    return jobs


def private_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(Path('/private/tmp').resolve()) or path.is_relative_to(common.ROOT.resolve()):
        raise ValueError('Private execution paths must be exclusively under /private/tmp outside repository')
    return path


def source_pins():
    names = set(common.source_closure()) | {str(Path(__file__).resolve().relative_to(common.ROOT))}
    return {p: common.sha(common.ROOT/p) for p in sorted(names)}


def cache_manifest(cache):
    cache = private_path(cache)
    if any(not (cache/n).is_file() or (cache/n).is_symlink() for n in PUBLIC_CACHE_FILES):
        raise ValueError('Complete regular-file public cache required; no private files are copied')
    return {n: common.sha(cache/n) for n in PUBLIC_CACHE_FILES}


def audit_transfer(source, target, jobs):
    """Audit a serial completed PREFIX without looking at or selecting scores."""
    source = Path(source).resolve()
    p = common.validate(source)
    for field in ('configuration', 'dataset_path', 'dataset_sha256', 'source_sha256',
                  'draw_count', 'draws_by_split', 'splits', 'public_inputs_sha256', 'status'):
        if p[field] != target[field]:
            raise ValueError('Import protocol differs: '+field)
    for name, digest in p['source_sha256'].items():
        snapshot = source/'source_snapshot'/name
        if not snapshot.is_file() or common.sha(snapshot) != digest:
            raise ValueError('Imported source snapshot differs: '+name)
    directory = source/'families'
    names = sorted(x.name for x in directory.iterdir()) if directory.exists() else []
    expected = [j['name'] for j in jobs]
    if set(names) != set(expected[:len(names)]):
        raise ValueError('Import must contain the complete serial submission prefix')
    known = {j['name']: j for j in jobs}
    hashes = {}
    for name in expected[:len(names)]:
        path = directory/name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Regular completed block required')
        payload = common.read(path)
        truth, job = payload['evaluator_only'], known[name]
        if any(truth[k] != job[k] for k in ('family_id', 'split', 'draw')):
            raise ValueError('Imported block identity differs: '+name)
        if set(payload['public']['streams']) != {'raw', *target['configuration']['methods']}:
            raise ValueError('Imported method set differs: '+name)
        if len(truth['sessions']) != 8 or any(len(s) != 8 for s in payload['public']['streams'].values()):
            raise ValueError('Imported block is incomplete: '+name)
        hashes[name] = common.sha(path)
    if (source/'generation.json').exists():
        recorded = common.read(source/'generation.json')['family_files_sha256']
        if recorded != hashes:
            raise ValueError('Completed source generation manifest differs')
    provenance = {n: common.sha(source/n) for n in ('protocol.json', 'protocol.sha256',
                  'generation_started.json', 'execution_environment.json', 'resources.json',
                  'generation.json', 'interruption.json') if (source/n).is_file()}
    return dict(source_output=str(source), source_files_sha256=provenance,
                family_files_sha256=hashes, import_rule='byte-identical complete prefix, all methods and all scores retained')


def declare_execution(out, work, *, dataset, methods, draws, splits, status,
                      draws_by_split, processes=3, public_cache, import_output=None,
                      import_private_root=None):
    out, work = Path(out).resolve(), private_path(work)
    if isinstance(processes, bool) or not isinstance(processes, int) or not 1 <= processes <= 3:
        raise ValueError('One to three declared processes required')
    predeclared = {}
    if out.exists() and any(out.iterdir()):
        allowed = {'protocol.json','protocol.sha256','source_snapshot','freeze.json'}
        if not (out/'protocol.json').is_file() or not {x.name for x in out.iterdir()} <= allowed:
            raise FileExistsError('New or untouched predeclared common-protocol output required; retain prior execution')
        prior = common.validate(out)
        for name,digest in prior['source_sha256'].items():
            if common.sha(out/'source_snapshot'/name) != digest:
                raise ValueError('Predeclared source snapshot changed: '+name)
        if (out/'freeze.json').exists():
            freeze = common.read(out/'freeze.json')
            if (freeze.get('fresh_protocol_sha256') != common.sha(out/'protocol.json')
                    or freeze.get('configuration') != prior['configuration']
                    or freeze.get('fresh_test_evaluated') is not False):
                raise ValueError('Untouched pre-test freeze matched to this common protocol required')
            predeclared['freeze.json'] = common.sha(out/'freeze.json')
    if (import_output is None) != (import_private_root is None):
        raise ValueError('Import output and original private-key directory must be provided together')
    cache = private_path(public_cache)
    cached = cache_manifest(cache)
    p = common.declare(out, Path(dataset), methods, draws, splits, status, draws_by_split)
    jobs = jobs_for(common.read(common.ROOT/p['dataset_path']), p)
    transfer = audit_transfer(import_output, p, jobs) if import_output else None
    key_names = []
    if import_private_root:
        key_root = private_path(import_private_root)
        for job in jobs:
            key = key_root/(job['name'].removesuffix('.json.gz')+'.key')
            if key.exists():
                if key.is_symlink() or not key.is_file() or len(key.read_bytes()) != 32:
                    raise ValueError('Invalid original private key; material is never exported')
                key_names.append(job['name'])
        if not set(transfer['family_files_sha256']) <= set(key_names):
            raise ValueError('Every imported completed block requires its original private key')
        # Guard material stays local; publishing a key digest would reveal
        # key-derived material. Pin the actual original bytes in a private
        # checkpoint instead, and compare them privately before transfer.
        checkpoint = work/'private_transfer_checkpoint'
        checkpoint.mkdir(parents=True, mode=0o700, exist_ok=False)
        for name in key_names:
            stem = name.removesuffix('.json.gz')
            fd = os.open(checkpoint/(stem+'.key'),os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as stream: stream.write((key_root/(stem+'.key')).read_bytes())
    value = dict(schema='qplanner-parallel-execution-v1', created_utc=now(),
        common_protocol_sha256=common.sha(out/'protocol.json'), source_sha256=source_pins(),
        predeclared_files_sha256=predeclared,
        processes=processes, start_method='spawn', native_threads_per_worker=1,
        workdir=str(work), public_cache_source=str(cache), public_cache_files_sha256=cached,
        jobs=jobs, transfer=transfer,
        private_transfer=None if import_private_root is None else dict(
            source_directory=str(private_path(import_private_root)), retained_key_blocks=key_names,
            policy='copy matching32byte keys to private isolated block directories, mode0600; do not copy ledgers, bytes or key hashes'),
        cache_policy='directed-cost memoization limit256, cleared between blocks; exact recomputation only',
        completion_order='may differ; final manifest and job receipts indexed by original submission order',
        timing_scope='parallel execution diagnostic; imported serial and new parallel timings are not pooled into an unbiased benchmark',
        failure_policy='retain completed blocks and failure receipts; no retry, seed replacement, or resume of this execution')
    common.save(out/'execution_protocol.json', value)
    (out/'execution_protocol.sha256').write_text(common.sha(out/'execution_protocol.json')+'\n')
    for name, digest in value['source_sha256'].items():
        target = out/'execution_source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((common.ROOT/name).read_bytes())
        if common.sha(target) != digest:
            raise ValueError('Execution source changed during snapshot')
    return value


def validate_execution(out):
    out = Path(out)
    p = common.validate(out)
    e = common.read(out/'execution_protocol.json')
    if common.sha(out/'execution_protocol.json') != (out/'execution_protocol.sha256').read_text().strip():
        raise ValueError('Execution protocol changed')
    if common.sha(out/'protocol.json') != e['common_protocol_sha256']:
        raise ValueError('Common protocol changed')
    for name,digest in e.get('predeclared_files_sha256',{}).items():
        if common.sha(out/name) != digest: raise ValueError('Predeclared freeze changed')
    if e['source_sha256'] != source_pins():
        raise ValueError('Execution source closure changed')
    for name, digest in e['source_sha256'].items():
        if common.sha(common.ROOT/name) != digest or common.sha(out/'execution_source_snapshot'/name) != digest:
            raise ValueError('Execution source changed: '+name)
    if cache_manifest(e['public_cache_source']) != e['public_cache_files_sha256']:
        raise ValueError('Public execution cache changed')
    return p, e


def transfer_completed(out, e):
    """Copy audited public/evaluator evidence and LOCAL keys before submission."""
    out = Path(out); directory = out/'families'; directory.mkdir()
    hashes = {}
    if e['transfer']:
        source = Path(e['transfer']['source_output'])
        for name, digest in e['transfer']['family_files_sha256'].items():
            original = source/'families'/name
            if common.sha(original) != digest:
                raise ValueError('Import file changed after declaration: '+name)
            with (directory/name).open('xb') as stream: stream.write(original.read_bytes())
            hashes[name] = common.sha(directory/name)
            if hashes[name] != digest: raise ValueError('Import copy changed')
    keys = e['private_transfer']
    if keys:
        for name in keys['retained_key_blocks']:
            stem = name.removesuffix('.json.gz')
            original = Path(keys['source_directory'])/(stem+'.key')
            data = original.read_bytes()
            checkpoint = private_path(e['workdir'])/'private_transfer_checkpoint'/(stem+'.key')
            if len(data) != 32 or original.is_symlink() or data != checkpoint.read_bytes():
                raise ValueError('Original private key changed or invalid; material withheld')
            target = private_path(e['workdir'])/'private_state'/stem
            target.mkdir(parents=True, mode=0o700, exist_ok=False)
            fd = os.open(target/(stem+'.key'), os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
            with os.fdopen(fd, 'wb') as stream: stream.write(data)
    common.save(out/'transfer_receipt.json', dict(execution_protocol_sha256=common.sha(out/'execution_protocol.json'),
        copied_family_files_sha256=hashes, retained_private_key_block_count=0 if not keys else len(keys['retained_key_blocks']),
        private_key_bytes_or_hashes_exported=False, completed_at_utc=now()))
    return hashes


def _initialize_worker(out):
    global _WORKER
    p, e = validate_execution(Path(out))
    worker = private_path(e['workdir'])/'workers'/str(os.getpid())
    cache = worker/'public_resources'; cache.mkdir(parents=True, exist_ok=False)
    for name, digest in e['public_cache_files_sha256'].items():
        target = cache/name; shutil.copyfile(Path(e['public_cache_source'])/name, target)
        if common.sha(target) != digest: raise ValueError('Worker public cache copy differs')
    data = common.read(common.ROOT/p['dataset_path'])
    objects = common.resources(data, worker)
    objects[-2].cache_limit = 256
    if any(pool['num_threads'] != 1 for pool in threadpool_info()):
        raise ValueError('Worker native thread limit was not applied')
    _WORKER = (Path(out), p, e, data, objects)


def _run_job(job):
    out, p, e, data, objects = _WORKER
    family = next(f for f in data['families'] if f['family_id'] == job['family_id'])
    private = private_path(e['workdir'])/'private_state'/job['name'].removesuffix('.json.gz')
    private.mkdir(parents=True, mode=0o700, exist_ok=True); os.chmod(private, 0o700)
    objects[-2].cache.clear()
    began = time.perf_counter()
    value = common.run_family(family, data, job['draw'], tuple(p['configuration']['methods']), objects, private)
    common.compressed_save(out/'families'/job['name'], value)
    objects[-2].cache.clear()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(rss if sys.platform == 'darwin' else rss*1024)
    receipt = dict(name=job['name'], sha256=common.sha(out/'families'/job['name']),
        worker_pid=os.getpid(), elapsed_s=time.perf_counter()-began, worker_peak_rss_bytes=rss_bytes,
        native_threadpools=threadpool_info(),
        execution_protocol_sha256=common.sha(out/'execution_protocol.json'), origin='generated_parallel')
    common.save(out/'job_receipts'/(job['name']+'.json'), receipt)
    return receipt, objects[-1]


def execution_environment():
    names = ('numpy', 'scipy', 'scikit-learn', 'networkx', 'sumolib', 'eclipse-sumo', 'pyproj', 'shapely', 'threadpoolctl')
    versions = {}
    for name in names:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    return dict(versions, python=sys.version, platform=platform.platform(),
        argv=sys.argv, pid=os.getpid(), parent_native_threadpools=threadpool_info(),
        worker_thread_environment={n:os.environ.get(n) for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')},
        execution='three-or-fewer spawned worker processes; one native thread each')


def generate_parallel(out):
    out = Path(out); p, e = validate_execution(out)
    if (out/'generation_started.json').exists():
        raise FileExistsError('Write-once parallel generation already started; retain partials')
    common.save(out/'generation_started.json', dict(started_utc=now(), protocol_sha256=common.sha(out/'protocol.json'),
        execution_protocol_sha256=common.sha(out/'execution_protocol.json')))
    hashes = transfer_completed(out, e); (out/'job_receipts').mkdir()
    for name, digest in hashes.items():
        common.save(out/'job_receipts'/(name+'.json'), dict(name=name, sha256=digest,
            origin='imported_serial_byte_identical', execution_protocol_sha256=common.sha(out/'execution_protocol.json')))
    pending = [j for j in e['jobs'] if j['name'] not in hashes]
    for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = '1'
    common.save(out/'execution_environment.json', execution_environment())
    begin = time.perf_counter(); metadata = None; peak_by_worker = {}
    if not pending and e['transfer']:
        metadata = common.read(Path(e['transfer']['source_output'])/'resources.json')
    with ProcessPoolExecutor(max_workers=e['processes'], mp_context=multiprocessing.get_context('spawn'),
                             initializer=_initialize_worker, initargs=(str(out),)) as executor:
        futures = {executor.submit(_run_job, job): job for job in pending}
        for future in as_completed(futures):
            try: receipt, info = future.result()
            except Exception:
                for queued in futures: queued.cancel()
                raise
            hashes[receipt['name']] = receipt['sha256']
            stable = {k:v for k,v in info.items() if k != 'setup_elapsed_s'}
            if metadata is None: metadata = info
            elif stable != {k:v for k,v in metadata.items() if k != 'setup_elapsed_s'}:
                raise ValueError('Worker public resources differ')
            peak_by_worker[str(receipt['worker_pid'])] = receipt['worker_peak_rss_bytes']
            if sum(peak_by_worker.values()) > 4*1024**3:
                for queued in futures: queued.cancel()
                raise MemoryError('Conservative sum of worker RSS peaks exceeded the declared4GiB execution envelope')
            print('Parallel Q planner completed', receipt['name'], 'seconds', round(receipt['elapsed_s'],2), flush=True)
    ordered = {j['name']:hashes[j['name']] for j in e['jobs']}
    common.save(out/'resources.json', metadata)
    common.save(out/'generation.json', dict(family_files_sha256=ordered,
        family_count=len({j['family_id'] for j in e['jobs']}), draw_count=p['draw_count'],
        draws_by_split=p['draws_by_split'], elapsed_s=time.perf_counter()-begin,
        identical_private_transcript_asserted=True, private_keys_exported=False,
        execution_protocol_sha256=common.sha(out/'execution_protocol.json'),
        imported_completed_block_count=len(e['transfer']['family_files_sha256']) if e['transfer'] else 0,
        generated_block_count=len(pending), worker_peak_rss_bytes=peak_by_worker,
        conservative_sum_worker_peaks_bytes=sum(peak_by_worker.values()),
        processing_time_scope='imported serial and new parallel timings mixed; descriptive only'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, required=True)
    parser.add_argument('--public-cache', type=Path, required=True)
    parser.add_argument('--processes', type=int, default=3)
    parser.add_argument('--methods', nargs='+', choices=tuple(common.METHOD_CONFIGS))
    parser.add_argument('--draws', type=int)
    for split in ('train', 'selection', 'test'): parser.add_argument('--'+split+'-draws', type=int)
    parser.add_argument('--splits', nargs='+', choices=('train', 'selection', 'test'))
    parser.add_argument('--status')
    parser.add_argument('--import-output', type=Path); parser.add_argument('--import-private-root', type=Path)
    parser.add_argument('--stage', choices=('declare','generate','all'), default='all')
    args = parser.parse_args()
    if args.stage in ('declare','all'):
        prior = common.validate(args.output) if (args.output/'protocol.json').exists() else None
        dataset = args.dataset if args.dataset is not None else common.ROOT/prior['dataset_path'] if prior else common.DATA
        methods = args.methods if args.methods is not None else list(prior['configuration']['methods']) if prior else list(common.METHOD_CONFIGS)
        draws = args.draws if args.draws is not None else prior['draw_count'] if prior else 2
        splits = args.splits if args.splits is not None else prior['splits'] if prior else ['train','selection']
        status = args.status if args.status is not None else prior['status'] if prior else 'DEVELOPMENT: old native groups inspected; no fresh confirmation'
        schedule = {s:(getattr(args,s+'_draws') if getattr(args,s+'_draws') is not None else
                      prior['draws_by_split'][s] if prior and args.draws is None else draws) for s in splits}
        declare_execution(args.output,args.workdir,dataset=dataset,methods=methods,draws=draws,
            splits=splits,status=status,draws_by_split=schedule,processes=args.processes,
            public_cache=args.public_cache,import_output=args.import_output,import_private_root=args.import_private_root)
    if args.stage != 'declare':
        try: generate_parallel(args.output)
        except Exception as exc:
            if not (args.output/'failure.json').exists():
                common.save(args.output/'failure.json', dict(error=type(exc).__name__,message=str(exc),
                    traceback=traceback.format_exc(),stage='parallel_generation',created_utc=now()))
            raise


if __name__ == '__main__': main()
