"""Independent frozen-Q utility/ledger audit; no private RNG key or generation.

The runner/profile builder/optimizer are not imported. Existing directed local
ranking and public network APIs define the oracle. Source hashes certify exact
identity, not DP, attacker exhaustiveness or novelty. Validation is additive.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics

import numpy as np

from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from data.lane_states import build_lane_states
from evaluation.candidate_future_attack import CandidateFutureAttack, coordinates
from evaluation.lane_travel import LanePoiService, SparseTravel
from evaluation.ordered_endpoint_attacks import OrderedEndpointBank, ordered_endpoint_features
from benchmark.paper_comparators import PublicHistory
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'artifacts/benchmarks/qplanner_development_20261006_v1'
PURPOSES=tuple(p.value for p in QueryPurpose)
EXECUTOR='experiments/qplanner_parallel_generation_20261006.py'
PUBLIC_CACHE_FILES=('native.net.xml','reference5.npz','reply10.npz','reply20.npz',
                    'belief-0.01.npz','belief-0.00125.npz')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    path=Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_text())


def relative_path(root,name):
    path=Path(name)
    assert not path.is_absolute() and (Path(root)/path).resolve().is_relative_to(Path(root).resolve()),name
    return Path(root)/path


def expected_jobs(data,protocol):
    """Independently reconstruct the declared serial submission order."""
    families=sorted((f for f in data['families'] if f['split'] in protocol['splits']),
                    key=lambda f:(('selection','train','test').index(f['split']),f['family_id']))
    assert families and all(any(f['split']==s for f in families) for s in protocol['splits'])
    jobs=[]
    for family in families:
        count=protocol['draws_by_split'][family['split']]
        assert isinstance(count,int) and not isinstance(count,bool) and count>0
        for draw in range(1,count+1):
            jobs.append(dict(name=f'{family["family_id"]}--draw{draw}.json.gz',
                             family_id=family['family_id'],split=family['split'],draw=draw))
    assert len({j['name'] for j in jobs})==len(jobs)
    return jobs


def public_cache_check(execution,metadata,data):
    """Pin public cache metadata; local cache contents are optional for release.

    Rebuilt public service/geometry are independently checked elsewhere. A
    public clone need not retain the original /private/tmp execution cache.
    No RNG key, private transfer checkpoint, or ledger file is opened here.
    """
    manifest=execution['public_cache_files_sha256']
    assert set(manifest)==set(PUBLIC_CACHE_FILES)
    assert all(isinstance(d,str) and len(d)==64 and all(c in '0123456789abcdef' for c in d)
               for d in manifest.values())
    assert manifest['native.net.xml']==data['network']['native_sha256']
    cache=Path(execution['public_cache_source'])
    checks=0
    if cache.exists():
        for name,digest in manifest.items():
            path=cache/name
            assert path.is_file() and not path.is_symlink() and sha(path)==digest,name
            checks+=1
            if name.endswith('.npz'):
                with np.load(path,allow_pickle=False) as arrays:
                    digest=hashlib.sha256(str(arrays['metadata']).encode())
                    if name.startswith('belief-'):
                        digest.update(arrays['state_ids'].astype('<i8').tobytes())
                        digest.update(arrays['log_normalizers'].astype('<f8').tobytes())
                        expected=metadata['belief_sha256'][name[7:-4]]
                    else:
                        digest.update(arrays['signatures'].astype('<i4').tobytes())
                        digest.update(arrays['access'].astype('<i4').tobytes())
                        expected=metadata[{'reference5.npz':'reference_sha256','reply10.npz':'reply_sha256',
                                           'reply20.npz':'reply20_sha256'}[name]]
                    assert digest.hexdigest()==expected,name
    return dict(manifest_files=len(manifest),available_local_cache_files_verified=checks,
                original_public_cache_required=False)


def execution_check(out,protocol,generation,data,metadata,*,root=ROOT):
    """Audit parallel receipts independently without importing the executor."""
    out,root=Path(out),Path(root)
    if not (out/'execution_protocol.json').exists():return None
    e=read(out/'execution_protocol.json');digest=sha(out/'execution_protocol.json')
    assert digest==(out/'execution_protocol.sha256').read_text().strip()
    assert e['schema']=='qplanner-parallel-execution-v1'
    assert e['common_protocol_sha256']==sha(out/'protocol.json')
    assert generation['execution_protocol_sha256']==digest
    assert isinstance(e['processes'],int) and not isinstance(e['processes'],bool) and 1<=e['processes']<=3
    assert e['start_method']=='spawn' and e['native_threads_per_worker']==1
    assert set(e['source_sha256'])==set(protocol['source_sha256'])|{EXECUTOR}
    for name,pin in e['source_sha256'].items():
        assert sha(relative_path(root,name))==sha(relative_path(out/'execution_source_snapshot',name))==pin,name
        if name in protocol['source_sha256']:assert pin==protocol['source_sha256'][name]
    for name,pin in e.get('predeclared_files_sha256',{}).items():assert sha(relative_path(out,name))==pin
    jobs=expected_jobs(data,protocol);assert e['jobs']==jobs
    names=[j['name'] for j in jobs]
    assert list(generation['family_files_sha256'])==names
    assert generation['draws_by_split']==protocol['draws_by_split']
    assert generation['draw_count']==protocol['draw_count']
    assert generation['family_count']==len({j['family_id'] for j in jobs})
    assert generation['identical_private_transcript_asserted'] is True and generation['private_keys_exported'] is False
    assert not (out/'failure.json').exists()
    for field in ('workdir','public_cache_source'):
        path=Path(e[field]);assert path.is_absolute() and path.is_relative_to('/private/tmp')
        assert not path.resolve().is_relative_to(root.resolve())
    cache=public_cache_check(e,metadata,data)
    transfer=e['transfer'];imported={}
    if transfer is not None:
        imported=transfer['family_files_sha256'];assert list(imported)==names[:len(imported)]
        # Original output is published evidence. Relocate its artifacts suffix
        # when a release is verified from a different repository checkout.
        original=Path(transfer['source_output'])
        if not original.resolve().is_relative_to(root.resolve()):
            parts=original.parts;assert 'artifacts' in parts
            original=root/Path(*parts[parts.index('artifacts'):])
        assert original.is_dir() and original.resolve().is_relative_to(root.resolve())
        for name,pin in transfer['source_files_sha256'].items():assert sha(relative_path(original,name))==pin,name
        prior=read(original/'protocol.json')
        assert sha(original/'protocol.json')==(original/'protocol.sha256').read_text().strip()
        for field in ('configuration','dataset_path','dataset_sha256','source_sha256','draw_count',
                      'draws_by_split','splits','public_inputs_sha256','status'):
            assert prior[field]==protocol[field],field
        for name,pin in prior['source_sha256'].items():assert sha(relative_path(original/'source_snapshot',name))==pin,name
        assert {p.name for p in (original/'families').iterdir()}==set(imported)
        for name,pin in imported.items():
            before,after=original/'families'/name,out/'families'/name
            assert before.is_file() and after.is_file() and not before.is_symlink() and not after.is_symlink()
            assert sha(before)==sha(after)==generation['family_files_sha256'][name]==pin,name
            assert before.read_bytes()==after.read_bytes(),name
        if (original/'generation.json').exists():assert read(original/'generation.json')['family_files_sha256']==imported
        if (original/'interruption.json').exists():
            interrupted=read(original/'interruption.json')
            assert interrupted['completed_family_files_sha256']==imported
            assert interrupted['preserve_already_created_private_keys'] is True
            assert interrupted['no_performance_based_seed_or_family_replacement'] is True
    private=e['private_transfer']
    retained=[]
    if private is not None:
        assert set(private)=={'source_directory','retained_key_blocks','policy'}
        path=Path(private['source_directory']);assert path.is_absolute() and path.is_relative_to('/private/tmp')
        retained=private['retained_key_blocks']
        assert len(retained)==len(set(retained)) and set(imported)<=set(retained)<=set(names)
        assert retained==[n for n in names if n in set(retained)]
    else:assert not imported
    receipt=read(out/'transfer_receipt.json')
    assert receipt['execution_protocol_sha256']==digest and receipt['copied_family_files_sha256']==imported
    assert receipt['retained_private_key_block_count']==len(retained)
    assert receipt['private_key_bytes_or_hashes_exported'] is False
    started=read(out/'generation_started.json')
    assert started['protocol_sha256']==sha(out/'protocol.json') and started['execution_protocol_sha256']==digest
    receipt_names={p.name for p in (out/'job_receipts').iterdir()}
    assert receipt_names=={n+'.json' for n in names}
    peaks={};generated=0
    for job in jobs:
        item=read(out/'job_receipts'/(job['name']+'.json'))
        assert item['name']==job['name'] and item['sha256']==generation['family_files_sha256'][job['name']]
        assert item['execution_protocol_sha256']==digest
        if job['name'] in imported:
            assert item['origin']=='imported_serial_byte_identical'
        else:
            assert item['origin']=='generated_parallel';generated+=1
            assert isinstance(item['worker_pid'],int) and item['worker_pid']>0
            assert math.isfinite(item['elapsed_s']) and item['elapsed_s']>=0
            assert isinstance(item['worker_peak_rss_bytes'],int) and item['worker_peak_rss_bytes']>0
            assert all(pool['num_threads']==1 for pool in item['native_threadpools'])
            pid=str(item['worker_pid']);peaks[pid]=max(peaks.get(pid,0),item['worker_peak_rss_bytes'])
    assert generation['imported_completed_block_count']==len(imported)
    assert generation['generated_block_count']==generated==len(names)-len(imported)
    assert len(peaks)<=e['processes'] and generation['worker_peak_rss_bytes']==peaks
    assert generation['conservative_sum_worker_peaks_bytes']==sum(peaks.values())<=4*1024**3
    env=read(out/'execution_environment.json')
    assert all(env['worker_thread_environment'][name]=='1' for name in
               ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'))
    return dict(execution_protocol_sha256=digest,execution_sources=len(e['source_sha256']),
                declared_ordered_jobs=len(jobs),imported_byte_identical_prefix_blocks=len(imported),
                retained_private_key_block_count_metadata=len(retained),private_key_material_verified=False,
                generated_job_receipts=generated,worker_count=len(peaks),public_cache=cache)


def lower_tail(values, mass=.25):
    """Fractional lower utility tail, independently using Python arithmetic."""
    if not 0<mass<=1:raise ValueError('Positive tail mass <=1 required')
    if not values:return None
    ordered=sorted(values); remaining=len(values)*mass; total=0.
    for value in ordered:
        take=min(1.,remaining); total+=take*value;remaining-=take
        if remaining<=0:break
    return total/(len(values)*mass)


def summarize(rows):
    families=sorted({r['family_id'] for r in rows});result={}
    for purpose in PURPOSES:
        values={f:[r['purposes'][purpose]['recall5'] for r in rows if r['family_id']==f
                   and r['purposes'][purpose]['recall5'] is not None] for f in families}
        means={f:statistics.mean(v) if v else None for f,v in values.items()}; valid=[v for v in means.values() if v is not None]
        result[purpose]=dict(family_mean=statistics.mean(valid) if valid else None,family_values=means,
            family_lower_quartile_cvar=lower_tail(valid),minimum_family_mean=min(valid) if valid else None,
            event_lower_quartile_cvar_family_macro=statistics.mean(lower_tail(v) for v in values.values() if v) if valid else None,
            defined_windows=sum(len(v) for v in values.values()),total_windows=len(rows),
            defined_categories=sum(r['purposes'][purpose]['reference_category_count'] for r in rows),
            total_categories=sum(r['purposes'][purpose]['all_category_count'] for r in rows))
    overall={f:statistics.mean(result[p]['family_values'][f] for p in PURPOSES if result[p]['family_values'][f] is not None)
             for f in families if any(result[p]['family_values'][f] is not None for p in PURPOSES)}
    result['equal_purpose_macro']=dict(family_mean=statistics.mean(overall.values()) if overall else None,
        family_values=overall,family_lower_quartile_cvar=lower_tail(list(overall.values())))
    return result


def equal(expected,actual):
    if isinstance(expected,dict):
        for key,value in expected.items():assert key in actual,key;equal(value,actual[key])
    elif isinstance(expected,(list,tuple)):
        assert len(expected)==len(actual)
        for a,b in zip(expected,actual):equal(a,b)
    elif isinstance(expected,float):assert np.isclose(expected,actual,rtol=1e-10,atol=1e-10),(expected,actual)
    else:assert expected==actual,(expected,actual)


def ledger_check(record, times, *, unit, max_units, interval):
    """Recompute prospective read admission and realized integer branch charges."""
    assert len(record['ledger'])==len(record['states'])==len(record['anchors'])==len(times)
    assert record['admitted'] is True
    allocation=record['allocation']
    equal(dict(unit_epsilon_per_m=unit,max_units=max_units,effective_cap_per_m=unit*max_units,
               read_interval_s=interval),allocation)
    spent=0;reads=[];previous=None
    for time,entry,anchor in zip(times,record['ledger'],record['anchors']):
        first=previous is None;paced=first or time-reads[-1]>=interval
        should_read=paced and spent+(1 if first else 2)<=max_units
        assert entry['private_read']==should_read
        if should_read:
            assert entry['branch'] in ('fresh','reuse')
            assert not first or entry['branch']=='fresh'
            if entry['branch']=='reuse':assert anchor==previous
            cost=1 if first or entry['branch']=='reuse' else 2;reads.append(time)
        else:
            assert entry['branch']==('postprocess' if paced else 'public_clock_skip')
            assert anchor==previous;cost=0
        spent+=cost
        assert entry['cost_units']==cost and entry['spent_units']==spent<=max_units
        previous=anchor
    assert reads==record['supplier_times_s'] and len(reads)==record['private_reads']
    equal(spent*unit,record['spent_per_m'])
    return spent,len(reads)


def local_scores(local,state,destination,ids):
    mask=np.zeros(local.n,dtype=bool);mask[list(ids)]=True;all_ids=np.ones(local.n,dtype=bool);result={}
    for purpose in QueryPurpose:
        recalls=[];completion=[];hits=0;size=0
        for category in local.categories:
            q=QuerySpec(purpose,category,k=5,
                radius_m=1000. if purpose==QueryPurpose.WITHIN_RADIUS else None,
                destination_state=destination if purpose==QueryPurpose.MIN_DETOUR else None)
            ref=local.top(state,all_ids,q)
            if not ref:continue
            answer=local.top(state,mask,q);overlap=len(set(ref)&set(answer))
            recalls.append(overlap/len(ref));completion.append(len(answer)/len(ref));hits+=overlap;size+=len(ref)
        result[purpose.value]=dict(recall5=statistics.mean(recalls) if recalls else None,
            completion=statistics.mean(completion) if completion else None,
            reference_category_count=len(recalls),all_category_count=len(local.categories),
            overlap_total=hits,reference_poi_total=size)
    return result


def robust_rule(rows):
    """Independent Python statistics selection, including declared lexical ties."""
    families=sorted({r['family_id'] for r in rows});names=sorted(rows[0]['errors']);numbers={}
    assert len(families)>=2
    for name in names:
        grouped={f:[r['errors'][name] for r in rows if r['family_id']==f] for f in families}
        values={'mae':{f:statistics.mean(v) for f,v in grouped.items()}}
        values.update({'hit'+str(radius):{f:statistics.mean(float(x<=radius) for x in v) for f,v in grouped.items()}
                       for radius in (50,100,200,500)})
        numbers[name]={}
        for metric,by_family in values.items():
            v=list(by_family.values());mean=statistics.mean(v);se=statistics.stdev(v)/math.sqrt(len(v))
            numbers[name][metric]=dict(mean=mean,se=se,families=len(v),family_values=by_family,
                objective=mean+(se if metric=='mae' else -se))
    choices={'mae':min(names,key=lambda n:(round(numbers[n]['mae']['objective'],12),n))}
    choices.update({metric:min(names,key=lambda n:(-round(numbers[n][metric]['objective'],12),n))
                    for metric in ('hit50','hit100','hit200','hit500')})
    return choices,numbers


def forecast_stats(rows,probabilities):
    truth=np.asarray([r['choice_index'] for r in rows]);p=np.asarray(probabilities);prediction=p.argmax(axis=1)
    assert p.shape==(len(rows),2) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1.)
    correct=prediction==truth;errors=[];chance=[]
    for row,at in zip(rows,prediction):
        ends=np.array([c['destination_xy'] for c in row['public_context']['choices']])
        dist=np.linalg.norm(ends-row['destination_xy'],axis=1);errors.append(float(dist[at]));chance.append(float(dist.mean()))
    return dict(exact_candidate_edge_accuracy=float(correct.mean()),
        balanced_accuracy=float(np.mean([correct[truth==i].mean() for i in (0,1)])),
        brier=float(np.mean((p[:,1]-truth)**2)),log_loss=float(-np.log(np.maximum(1e-12,p[np.arange(len(rows)),truth])).mean()),
        destination_mae_m=statistics.mean(errors),destination_hit100=statistics.mean(float(x<=100.) for x in errors),
        uniform_destination_mae_m=statistics.mean(chance),
        routine_accuracy=statistics.mean(float(c) for row,c in zip(rows,correct) if row['destination_role']=='routine'),
        rare_accuracy=statistics.mean(float(c) for row,c in zip(rows,correct) if row['destination_role']=='rare'),
        n=len(rows),family_count=len({r['family_id'] for r in rows}))


def attack_checks(out,bundles,data,rn,methods):
    selection=read(out/'attack_selection.json');result=read(out/'attack_readout.json')
    assert selection['protocol_sha256']==result['protocol_sha256']==sha(out/'protocol.json')
    assert result['selection_sha256']==sha(out/'attack_selection.json')
    ids={b['evaluator_only']['family_id'] for b in bundles if b['evaluator_only']['split']=='train'}
    history=PublicHistory(rn,[[rn.nearest(p['lat'],p['lon'])[0] for p in data['traces'][s['session_id']][::20]]
        for f in data['families'] if f['family_id'] in ids for s in f['evaluator_only']['sessions']])
    future=read(out/'future_bank_predictions.json.gz')['rows'];endpoint=read(out/'endpoint_bank_errors.json.gz')
    lookup={(r['attack_key'],r['family_id'],r['draw'],r['slot']):r for r in future}
    assert len(lookup)==len(future);checks=dict(models=0,probability_banks=0,endpoint_errors=0,selectors=0)
    expected_keys={f'{m}--{stage}--{task}' for m in ('raw',*methods) for stage in ('shared_fork','turn_visible') for task in ('S5','S6')}
    expected_keys|={f'{m}--{scenario}' for m in ('raw',*methods) for scenario in ('S9','S10')}
    assert expected_keys==set(selection['selections'])==set(result['results'])
    for key,record in selection['selections'].items():
        path=out/'attack_models'/(key+'.pickle');assert sha(path)==record['model_sha256']
        model=pickle.loads(path.read_bytes()) # Trusted local SHA-verified artifact only.
        parts=key.split('--');method=parts[0];samples=[]
        if len(parts)==3:
            _,stage,task=parts
            for bundle in bundles:
                group,truth=bundle['public'],bundle['evaluator_only']
                for slot in (6,7):
                    target=truth['sessions'][slot]
                    samples.append(dict(family_id=truth['family_id'],split=truth['split'],draw=truth['draw'],query_slot=slot,
                        query={'events':[e for e in group['streams'][method][slot]['events'] if e['timestamp_s']<=group['public_clocks'][stage+'_t']]},
                        histories=group['streams'][method][:6],public_context=group['public_context'],choice_index=target['choice_index'],
                        destination_role=target['destination_role'],destination_xy=np.array(target['destination_xy'])))
            train=[r for r in samples if r['split']=='train'];assert all(r['family_id'] in ids for r in train)
            refit=CandidateFutureAttack(train,use_history=task=='S6');assert model.use_history==refit.use_history
            banks=[]
            for row in samples:
                if row['split']=='train':continue
                bank=model.predict(row['query'],row['public_context'],row['histories'])
                actual=lookup[key,row['family_id'],row['draw'],row['query_slot']]
                assert actual['truth_choice']==row['choice_index'] and actual['split']==row['split']
                equal({n:p.tolist() for n,p in bank.items()},actual['banks'])
                again=refit.predict(row['query'],row['public_context'],row['histories'])
                assert np.array_equal(again['candidate_trees'],bank['candidate_trees'])
                banks.append((row,bank));checks['probability_banks']+=len(bank)
            dev=[(r,b) for r,b in banks if r['split']=='selection']
            scores={n:forecast_stats([r for r,_ in dev],[b[n] for _,b in dev]) for n in dev[0][1]}
            equal(scores,record['selection_scores']);chosen=min(scores,key=lambda n:(-scores[n]['balanced_accuracy'],scores[n]['log_loss'],n))
            assert chosen==record['selected']
            for split in ('selection','test'):
                scoped=[(r,b) for r,b in banks if r['split']==split]
                if not scoped:continue
                expected=forecast_stats([r for r,_ in scoped],[b[chosen] for _,b in scoped])
                expected['family_values']={f:forecast_stats([r for r,_ in scoped if r['family_id']==f],
                    [b[chosen] for r,b in scoped if r['family_id']==f]) for f in sorted({r['family_id'] for r,_ in scoped})}
                equal(expected,result['results'][key][split])
        else:
            scenario=parts[1]
            for bundle in bundles:
                group,truth=bundle['public'],bundle['evaluator_only']
                for slot,session in enumerate(group['streams'][method]):
                    events=[{'timestamp_s':e['timestamp_s'],'coordinates':[[q['lat'],q['lon']] for q in e['candidates']]} for e in session['events']]
                    features,_=ordered_endpoint_features(events,scenario,rn,history,observable_close_s=600.)
                    target=truth['sessions'][slot]['origin_xy' if scenario=='S9' else 'destination_xy']
                    samples.append(dict(family_id=truth['family_id'],split=truth['split'],draw=truth['draw'],slot=slot,
                                        events=events,features=features,target=np.array(target)))
            train=[r for r in samples if r['split']=='train'];targets=np.array([r['target'] for r in train])
            features={n:np.array([r['features'][n][0] for r in train]) for n in train[0]['features']}
            refit=OrderedEndpointBank(features,targets)
            assert set(model.learners)==set(features)
            for n,x in features.items():
                learner=model.learners[n];mean=x.mean(axis=0);scale=x.std(axis=0);scale[scale<1e-9]=1.
                assert np.array_equal(learner.y,targets)
                assert np.allclose(learner.mean,mean,rtol=0.,atol=1e-12) and np.allclose(learner.scale,scale,rtol=0.,atol=1e-12)
                assert np.allclose(learner.x,(x-mean)/scale,rtol=0.,atol=1e-12)
            recorded={(r['family_id'],r['draw'],r['slot']):r for r in endpoint[key]};rows=[]
            assert len(recorded)==len(endpoint[key])
            for row in samples:
                if row['split']=='train':continue
                bank=model.predictions(row['events'],scenario,rn,history,observable_close_s=600.)
                again=refit.predictions(row['events'],scenario,rn,history,observable_close_s=600.)
                assert set(bank)==set(again)
                for n in bank:assert np.array_equal(bank[n],again[n]),n
                errors={n:float(np.linalg.norm(np.asarray(p)[0]-row['target'])) for n,p in bank.items()}
                expected=recorded[row['family_id'],row['draw'],row['slot']];equal(errors,expected['errors'])
                assert expected['split']==row['split'];rows.append(expected);checks['endpoint_errors']+=len(errors)
            assert len(rows)==len(recorded)
            chosen,numbers=robust_rule([r for r in rows if r['split']=='selection'])
            assert chosen==record['selected'];equal(numbers,record['selection_statistics'])
            for split in ('selection','test'):
                scoped=[r for r in rows if r['split']==split]
                if not scoped:continue
                families=sorted({r['family_id'] for r in scoped})
                values={f:dict(mae_m=statistics.mean(r['errors'][chosen['mae']] for r in scoped if r['family_id']==f),
                    **{h:statistics.mean(float(r['errors'][chosen[h]]<=int(h[3:])) for r in scoped if r['family_id']==f)
                       for h in ('hit100','hit500')}) for f in families}
                equal(dict(family_values=values,**{m:statistics.mean(v[m] for v in values.values()) for m in ('mae_m','hit100','hit500')}),result['results'][key][split])
        checks['models']+=1;checks['selectors']+=1
        print('Method-adapted attack verified',key,flush=True)
    return checks


def verify(out,*,validation_output=None):
    out=Path(out)
    protocol,generation,metadata=[read(out/name) for name in ('protocol.json','generation.json','resources.json')]
    assert sha(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()
    for name,digest in protocol['source_sha256'].items():assert sha(ROOT/name)==sha(out/'source_snapshot'/name)==digest,name
    data_path=ROOT/protocol['dataset_path'];assert sha(data_path)==protocol['dataset_sha256']
    for name,digest in protocol['public_inputs_sha256'].items():assert sha(ROOT/name)==digest,name
    data=read(data_path);network=ROOT/data['network']['compressed_path']
    assert sha(network)==data['network']['compressed_sha256']
    assert hashlib.sha256(gzip.decompress(network.read_bytes())).hexdigest()==data['network']['native_sha256']
    assert metadata['native_sha256']==data['network']['native_sha256']
    for name,digest in metadata['source_sha256'].items():assert protocol['public_inputs_sha256'][name]==digest
    execution=execution_check(out,protocol,generation,data,metadata)
    rn=build_lane_states(network,spacing_m=40.)
    assert str(network.relative_to(ROOT)) in protocol['public_inputs_sha256']
    assert 'artifacts/benchmarks/research_loop/resources.json' in protocol['public_inputs_sha256']
    pois=[{k:v for k,v in p.items() if k not in ('vertex','access_offset_m')} for p in
          read(ROOT/'artifacts/benchmarks/research_loop/resources.json')['pois_used']]
    reply=PublicPoiContext(LanePoiService(rn,pois,k=20))
    assert reply.sha256==metadata['reply20_sha256'] and len(reply.pois)==418
    # Rebuild latent cell representatives and prototype destinations directly
    # from public geometry, not from a family GPS/endpoint or builder output.
    cells,inverse=np.unique(np.floor(rn.xy/200.).astype(np.int64),axis=0,return_inverse=True)
    centres=(cells+.5)*200.;errors=np.linalg.norm(rn.xy-centres[inverse],axis=1)
    order=np.lexsort((np.arange(len(rn)),errors,inverse))
    first=np.r_[True,inverse[order[1:]]!=inverse[order[:-1]]];state_ids=order[first]
    profile_meta=metadata['profiles_metadata']
    assert state_ids.tolist()==profile_meta['state_ids']
    quantiles=protocol['configuration']['public_destination_grid_quantiles'];low,high=rn.xy.min(axis=0),rn.xy.max(axis=0)
    points=np.array([low+(high-low)*[a,b] for a in quantiles for b in quantiles])
    expected_destinations=sorted(set(map(int,state_ids[cKDTree(rn.xy[state_ids]).query(points)[1]])))
    assert expected_destinations==profile_meta['public_destination_states']
    assert profile_meta['reply_context_sha256']==reply.sha256 and profile_meta['reference_k']==5
    assert profile_meta['public_radius_m']==1000. and profile_meta['reply_l']==20
    local=MultiPurposeRoadRanking(reply,cache_limit=1024);travel=SparseTravel(rn,cache_limit=256)
    families={f['family_id']:f for f in data['families'] if f['split'] in protocol['splits']}
    methods=list(protocol['configuration']['methods']);budget=protocol['configuration']['budget']
    unit=budget['unit_epsilon_per_m'];max_units=2*budget['horizon']-1
    equal(budget['total_effective_epsilon_per_m']/(budget['session_slots']*max_units),unit)
    expected_names={f'{name}--draw{draw}.json.gz' for name,family in families.items()
                    for draw in range(1,protocol['draws_by_split'][family['split']]+1)}
    assert expected_names==set(generation['family_files_sha256'])
    counts=dict(bundles=0,events=0,ledger_steps=0,utility_windows=0,wire_rows=0,objective_records=0)
    utility_rows=[];wire_rows=[];timings=defaultdict(list);bundles=[]
    for name,digest in generation['family_files_sha256'].items():
        path=out/'families'/name;assert sha(path)==digest
        bundle=read(path);bundles.append(bundle);group,truth=bundle['public'],bundle['evaluator_only'];family=families[truth['family_id']]
        assert truth['split']==family['split'] and name==f'{truth["family_id"]}--draw{truth["draw"]}.json.gz'
        assert group['public_context']==family['public_context']
        assert group['public_clocks']=={k:family['clocks'][k] for k in ('shared_fork_t','turn_visible_t')}
        assert set(group['streams'])=={'raw',*methods}
        utilities={(r['slot'],r['method'],r['event_id'],r['cache']):r for r in bundle['utility']}
        wires={(r['slot'],r['method'],r['event_id']):r for r in bundle['wire']}
        assert len(utilities)==len(bundle['utility']) and len(wires)==len(bundle['wire'])
        cache={m:set() for m in ('raw',*methods)};epoch_spent={m:0. for m in methods}
        expected_events=0
        for slot,spec in enumerate(family['evaluator_only']['sessions']):
            trace={int(p['time_s']):p for p in data['traces'][spec['session_id']]};target=truth['sessions'][slot]
            assert target['slot']==slot and target['choice_index']==spec['choice_index'] and target['destination_role']==spec['destination_role']
            equal(list(rn.point_xy(trace[0]['lat'],trace[0]['lon'])),target['origin_xy'])
            end=trace[600];destination=rn.nearest(end['lat'],end['lon'])[0]
            equal(coordinates({'events':[{'event_id':'e','timestamp_s':0.,'candidates':[{'candidate_id':'q','lat':end['lat'],'lon':end['lon']}]}]})[0][0][0].tolist(),target['destination_xy'])
            clocks=sorted(set(range(0,601,20))|({family['clocks']['shared_fork_t'],family['clocks']['turn_visible_t']} if slot>=6 else set()))
            if slot>=6:
                for stage in ('shared_fork_t','turn_visible_t'):
                    cut=family['clocks'][stage]
                    label=next(p['edge_id'] for t,p in sorted(trace.items()) if t>cut and not p['edge_id'].startswith(':')
                               and (stage!='shared_fork_t' or p['edge_id']!=family['fork_edge']))
                    assert label==target['next_edge_labels'][stage]==family['public_context']['choices'][spec['choice_index']]['edge_id']
            tapes=[]
            for method in methods:
                record=target['ledger'][method];assert record['allocation']['slot']==slot
                ledger_check(record,clocks,unit=unit,max_units=max_units,interval=budget['read_interval_s'])
                tapes.append((record['anchors'],record['ledger'],record['supplier_times_s']))
                epoch_spent[method]+=record['spent_per_m'];timings[method].extend(record['step_ms'])
                counts['ledger_steps']+=len(clocks);counts['objective_records']+=len(record['planner_objectives'])
            assert all(tape==tapes[0] for tape in tapes)
            for method in ('raw',*methods):
                events=group['streams'][method][slot]['events'];assert [e['timestamp_s'] for e in events]==clocks
                coordinates({'events':events}) # Strict public-only feature schema.
                assert len(events)==len(clocks);expected_events+=len(events);previous=None;previous_t=None
                for event in events:
                    t=int(event['timestamp_s']);absolute=spec['depart_s']+t;positions=[(q['lat'],q['lon']) for q in event['candidates']]
                    assert len(positions)==(1 if method=='raw' else 5)
                    if method=='raw':assert positions==[(trace[t]['lat'],trace[t]['lon'])]
                    else:
                        at=clocks.index(t);states=target['ledger'][method]['states'][at]
                        assert positions==[rn.latlon(state) for state in states]
                        if previous is not None:
                            assert all(state in travel.reachable(old,t-previous_t) for old,state in zip(previous,states))
                        previous,previous_t=states,t
                    replies=[];reply_bytes=0;request_bytes=0
                    for lat,lon in positions:
                        state=rn.nearest(lat,lon)[0];ids=reply.query_indices(state).ravel();ids=list(map(int,ids[ids>=0]));replies.append(ids)
                        records=[{k:reply.pois[i][k] for k in ('id','category','lat','lon')} for i in ids]
                        reply_bytes+=len(json.dumps({'results':records},separators=(',',':'),ensure_ascii=False).encode())
                        request_bytes+=len(json.dumps({'timestamp_s':absolute,'lat':lat,'lon':lon,'categories':list(reply.categories),'L':20},separators=(',',':')).encode())
                    source=wires[slot,method,event['event_id']]
                    equal(dict(reply_poi_ids_by_Q=replies,requests=len(positions),request_bytes=request_bytes,reply_bytes=reply_bytes),source)
                    current={i for ids in replies for i in ids};cache[method].update(current)
                    point=trace[t];state=rn.nearest(point['lat'],point['lon'])[0]
                    for policy,ids in (('current',current),('static_epoch_cache',cache[method])):
                        row=utilities[slot,method,event['event_id'],policy]
                        equal(dict(family_id=family['family_id'],split=family['split'],draw=truth['draw'],slot=slot,
                                   method=method,t=t,available_count=len(ids),purposes=local_scores(local,state,destination,ids)),row)
                        counts['utility_windows']+=1
                    counts['wire_rows']+=1
            counts['events']+=expected_events;expected_events=0
        assert len(bundle['wire'])==sum(len(s['events']) for streams in group['streams'].values() for s in streams)
        assert len(bundle['utility'])==2*len(bundle['wire'])
        for method,spent in epoch_spent.items():
            equal(spent,truth['epoch_accounting'][method]['spent_per_m']);assert spent<=budget['total_effective_epsilon_per_m']+1e-12
            equal(budget['total_effective_epsilon_per_m'],truth['epoch_accounting'][method]['reserved_cap_per_m'])
        utility_rows.extend(bundle['utility']);wire_rows.extend(bundle['wire']);counts['bundles']+=1
        print('Q-planner utility/ledger verified',name,flush=True)
    if (out/'utility_readout.json').exists():
        result=read(out/'utility_readout.json')
        assert result['protocol_sha256']==sha(out/'protocol.json') and result['generation_sha256']==sha(out/'generation.json')
        for method,splits in result['summary'].items():
            for split,entries in splits.items():
                for cache in ('current','static_epoch_cache'):
                    rows=[r for r in utility_rows if r['method']==method and r['split']==split and r['cache']==cache]
                    scopes={'all':rows,'cold':[r for r in rows if r['slot']==0],
                            'early_0_180':[r for r in rows if r['t']<=180],
                            'temporal_tail_400_600':[r for r in rows if r['t']>=400]}
                    for scope,rows in scopes.items():equal(summarize(rows),entries[cache][scope])
                source=[r for r in wire_rows if r['method']==method and r['split']==split]
                equal({k:sum(r[k] for r in source) for k in ('requests','request_bytes','reply_bytes')},entries['cost'])
    attacks=attack_checks(out,bundles,data,rn,methods) if (out/'attack_readout.json').exists() else None
    record=dict(schema='qplanner-independent-utility-ledger-verification-v1',status='pass',
        protocol_sha256=sha(out/'protocol.json'),generation_sha256=sha(out/'generation.json'),verifier_sha256=sha(Path(__file__)),
        counts=counts,no_private_rng_key_required=True,identical_private_anchor_ledger_read_tapes=True,
        exact_received_service_and_causal_cache_replayed=True,independent_conditional_family_tail_arithmetic=True,
        exact_Q_coordinates_and_directed_track_reachability=True,attack_models_verified=attacks is not None,
        attack_checks=attacks,public_grid_prototype_destinations_independently_verified=True,
        parallel_execution_checks=execution,
        objective_optimizer_or_public_profile_semantics_verified=False,privacy_theorem_certified=False,
        scope='Frozen output arithmetic/causality audit; runtime measurements are descriptive, not replayed; development/generalization status unchanged')
    target=Path(validation_output) if validation_output else out/('validation.json' if attacks is not None else 'utility_ledger_validation.json')
    if target.exists():equal(record,read(target))
    else:target.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(record,flush=True);return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path,nargs='?',default=DEFAULT)
    parser.add_argument('--validation-output',type=Path);args=parser.parse_args();verify(args.out,validation_output=args.validation_output)
