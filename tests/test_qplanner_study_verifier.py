"""Independent verifier detects actual accounting/denominator mistakes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.verify_qplanner_study_20261006 import (
    EXECUTOR, PUBLIC_CACHE_FILES, PURPOSES, execution_check, forecast_stats, ledger_check,
    local_scores, lower_tail, public_cache_check, read, robust_rule, sha, summarize,
)
from tests.test_jisa_static_catalogue_control import fixture


def record():
    times=[0,20,60,80,120,140,180]
    branches=['fresh','public_clock_skip','fresh','public_clock_skip','fresh','public_clock_skip','postprocess']
    costs=[1,0,2,0,2,0,0];anchors=[[0.,0.],[0.,0.],[0.,.001],[0.,.001],[0.,.002],[0.,.002],[0.,.002]]
    spent=0;ledger=[]
    for branch,cost in zip(branches,costs):
        spent+=cost;ledger.append(dict(branch=branch,cost_units=cost,spent_units=spent,private_read=bool(cost)))
    return times,dict(admitted=True,allocation=dict(unit_epsilon_per_m=.01,max_units=5,effective_cap_per_m=.05,read_interval_s=60.),
        ledger=ledger,states=[[0]]*len(times),anchors=anchors,supplier_times_s=[0,60,120],private_reads=3,spent_per_m=.05)


def test_verifier_independently_checks_pacing_branch_cost_and_prospective_cap():
    times,state=record()
    assert ledger_check(state,times,unit=.01,max_units=5,interval=60.)==(5,3)
    for index,field,value in ((1,'private_read',True),(2,'cost_units',1),(6,'private_read',True)):
        broken=deepcopy(state);broken['ledger'][index][field]=value
        with pytest.raises(AssertionError):ledger_check(broken,times,unit=.01,max_units=5,interval=60.)
    broken=deepcopy(state);broken['anchors'][-1]=[1.,1.]
    with pytest.raises(AssertionError):ledger_check(broken,times,unit=.01,max_units=5,interval=60.)


def test_verifier_allows_valid_fresh_coordinate_collision_and_one_unit_reuse():
    times,state=record()
    # Fresh REM can return the same coordinate; infer branch from ledger, not Z equality.
    state['anchors']=[[0.,0.]]*len(times)
    assert ledger_check(state,times,unit=.01,max_units=5,interval=60.)==(5,3)
    times=[0,60,120,180,240]
    state.update(states=[[0]]*5,anchors=[[0.,0.]]*5,supplier_times_s=[0,60,120,180],private_reads=4,spent_per_m=.04)
    state['ledger']=[dict(branch='fresh' if i==0 else 'reuse' if i<4 else 'postprocess',
                         cost_units=1 if i<4 else 0,spent_units=min(i+1,4),private_read=i<4) for i in range(5)]
    # U5 admits first+three reuses (four reads), then reserve2 denies the fifth
    # read despite one remaining unit. H3 is a calibration, not three-read cap.
    assert ledger_check(state,times,unit=.01,max_units=5,interval=60.)==(4,4)


def test_conditional_four_purpose_family_tail_keeps_n_a_and_family_weight():
    def row(f,value):return dict(family_id=f,purposes={p:dict(recall5=value,
        reference_category_count=int(value is not None),all_category_count=2) for p in PURPOSES})
    result=summarize([row('a',0.),row('a',1.),row('b',1.),row('c',None)])
    for p in PURPOSES:
        assert result[p]['family_mean']==.75 and result[p]['family_values']['c'] is None
        assert result[p]['defined_windows']==3 and result[p]['total_windows']==4
        assert result[p]['family_lower_quartile_cvar']==.5
        assert result[p]['defined_categories']==3 and result[p]['total_categories']==8
    assert lower_tail([0.,.5,1.],.5)==pytest.approx(1/6)
    assert lower_tail([]) is None


def test_independent_local_scores_use_exact_candidates_and_empty_category_denominator():
    local=fixture()
    full=local_scores(local,0,3,{0,1,2});missing=local_scores(local,0,3,set())
    for purpose in PURPOSES:
        assert full[purpose]['recall5']==full[purpose]['completion']==1.
        assert missing[purpose]['recall5']==missing[purpose]['completion']==0.
        assert full[purpose]['reference_category_count']==1
        assert full[purpose]['all_category_count']==2
        assert full[purpose]['overlap_total']==full[purpose]['reference_poi_total']==2


def test_independent_robust_selection_penalizes_family_instability_and_lexical_hit_ties():
    rows=[dict(family_id=f,errors={'a':a,'b':b}) for f,a,b in (('one',0.,6.),('two',10.,6.))]
    choices,numbers=robust_rule(rows)
    assert numbers['a']['mae']['mean']==5. and numbers['a']['mae']['se']==5.
    assert numbers['a']['mae']['objective']==10. and numbers['b']['mae']['objective']==6.
    assert choices['mae']=='b' # Mean-only would incorrectly choose a.
    assert all(choices[key]=='a' for key in ('hit50','hit100','hit200','hit500'))


def test_independent_future_metrics_keep_raw_shared_fork_ambiguity():
    rows=[dict(family_id='x',choice_index=i,destination_role=role,destination_xy=[i*200.,0.],
        public_context={'choices':[{'destination_xy':[0.,0.]},{'destination_xy':[200.,0.]}]})
          for i,role in enumerate(('routine','rare'))]
    result=forecast_stats(rows,[[.5,.5],[.5,.5]])
    assert result['exact_candidate_edge_accuracy']==result['balanced_accuracy']==.5
    assert result['destination_mae_m']==100. and result['brier']==.25
    assert result['routine_accuracy']==1. and result['rare_accuracy']==0.
    assert result['log_loss']==pytest.approx(0.6931471805599453)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def execution_fixture(tmp_path):
    root=tmp_path;out=root/'artifacts/new';prior=root/'artifacts/prior'
    source='benchmark/example.py';(root/source).parent.mkdir(parents=True)
    (root/source).write_text('public=1\n')
    (root/EXECUTOR).parent.mkdir(parents=True);(root/EXECUTOR).write_text('executor=1\n')
    protocol=dict(source_sha256={source:sha(root/source)},configuration={'methods':['one']},
        dataset_path='dataset.json.gz',dataset_sha256='a'*64,draw_count=1,
        draws_by_split={'selection':2,'train':1},splits=['selection','train'],
        public_inputs_sha256={'public.json':'b'*64},status='DEVELOPMENT')
    for folder in (out,prior):
        save(folder/'protocol.json',protocol)
        (folder/'protocol.sha256').write_text(sha(folder/'protocol.json')+'\n')
    snapshot=prior/'source_snapshot'/source;snapshot.parent.mkdir(parents=True);snapshot.write_bytes((root/source).read_bytes())
    data={'families':[dict(family_id='a',split='train'),dict(family_id='b',split='selection')],
          'network':{'native_sha256':'1'*64}}
    jobs=[dict(name=f'{f}--draw{draw}.json.gz',family_id=f,split=split,draw=draw)
          for f,split,draw in (('b','selection',1),('b','selection',2),('a','train',1))]
    for index,job in enumerate(jobs):
        path=out/'families'/job['name'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(bytes([index]))
    before=prior/'families'/jobs[0]['name'];before.parent.mkdir(parents=True);before.write_bytes(b'\x00')
    imported={jobs[0]['name']:sha(before)}
    save(prior/'interruption.json',dict(completed_family_files_sha256=imported,
        preserve_already_created_private_keys=True,no_performance_based_seed_or_family_replacement=True))
    pins={p:sha(root/p) for p in (source,EXECUTOR)}
    for name,digest in pins.items():
        snapshot=out/'execution_source_snapshot'/name;snapshot.parent.mkdir(parents=True,exist_ok=True)
        snapshot.write_bytes((root/name).read_bytes())
    execution=dict(schema='qplanner-parallel-execution-v1',common_protocol_sha256=sha(out/'protocol.json'),
        source_sha256=pins,predeclared_files_sha256={},processes=2,start_method='spawn',native_threads_per_worker=1,
        workdir='/private/tmp/qplanner-verifier-fixture',
        public_cache_source='/private/tmp/qplanner-verifier-no-cache-'+tmp_path.name,
        public_cache_files_sha256={name:'1'*64 for name in PUBLIC_CACHE_FILES},jobs=jobs,
        transfer=dict(source_output=str(prior),source_files_sha256={name:sha(prior/name) for name in
                     ('protocol.json','protocol.sha256','interruption.json')},family_files_sha256=imported),
        private_transfer=dict(source_directory='/private/tmp/old-keys',retained_key_blocks=[j['name'] for j in jobs[:2]],policy='no keys exported'))
    save(out/'execution_protocol.json',execution)
    (out/'execution_protocol.sha256').write_text(sha(out/'execution_protocol.json')+'\n');digest=sha(out/'execution_protocol.json')
    generation=dict(execution_protocol_sha256=digest,family_files_sha256={j['name']:sha(out/'families'/j['name']) for j in jobs},
        draws_by_split=protocol['draws_by_split'],draw_count=1,family_count=2,
        identical_private_transcript_asserted=True,private_keys_exported=False,
        imported_completed_block_count=1,generated_block_count=2,worker_peak_rss_bytes={'55':3000},
        conservative_sum_worker_peaks_bytes=3000)
    save(out/'generation_started.json',dict(protocol_sha256=sha(out/'protocol.json'),execution_protocol_sha256=digest))
    save(out/'transfer_receipt.json',dict(execution_protocol_sha256=digest,copied_family_files_sha256=imported,
        retained_private_key_block_count=2,private_key_bytes_or_hashes_exported=False))
    save(out/'execution_environment.json',dict(worker_thread_environment={name:'1' for name in
        ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')}))
    for index,job in enumerate(jobs):
        item=dict(name=job['name'],sha256=generation['family_files_sha256'][job['name']],execution_protocol_sha256=digest,
            origin='imported_serial_byte_identical' if index==0 else 'generated_parallel')
        if index:item.update(worker_pid=55,elapsed_s=1.,worker_peak_rss_bytes=1000*(index+1),native_threadpools=[{'num_threads':1}])
        save(out/'job_receipts'/(job['name']+'.json'),item)
    return root,out,protocol,generation,data,execution


def test_parallel_verifier_checks_exact_import_and_receipts_without_private_keys_or_cache(tmp_path):
    root,out,protocol,generation,data,_=execution_fixture(tmp_path)
    result=execution_check(out,protocol,generation,data,{},root=root)
    assert result['declared_ordered_jobs']==3 and result['imported_byte_identical_prefix_blocks']==1
    assert result['retained_private_key_block_count_metadata']==2
    assert result['private_key_material_verified'] is False and result['generated_job_receipts']==2
    assert result['public_cache']['available_local_cache_files_verified']==0
    assert result['public_cache']['original_public_cache_required'] is False


@pytest.mark.parametrize('fault',('resampled_import','changed_snapshot','wrong_key_count','missing_receipt',
                                 'changed_receipt_hash','extra_threads','generation_order'))
def test_parallel_verifier_rejects_concrete_execution_and_import_faults(tmp_path,fault):
    root,out,protocol,generation,data,execution=execution_fixture(tmp_path)
    first,last=execution['jobs'][0]['name'],execution['jobs'][-1]['name']
    if fault=='resampled_import':
        (out/'families'/first).write_bytes(b'resampled')
        generation['family_files_sha256'][first]=sha(out/'families'/first)
    elif fault=='changed_snapshot':(out/'execution_source_snapshot'/EXECUTOR).write_text('changed=1')
    elif fault=='wrong_key_count':
        value=read(out/'transfer_receipt.json');value['retained_private_key_block_count']=1;save(out/'transfer_receipt.json',value)
    elif fault=='missing_receipt':(out/'job_receipts'/(last+'.json')).unlink()
    elif fault in ('changed_receipt_hash','extra_threads'):
        path=out/'job_receipts'/(last+'.json');value=read(path)
        if fault=='changed_receipt_hash':value['sha256']='0'*64
        else:value['native_threadpools'][0]['num_threads']=2
        save(path,value)
    elif fault=='generation_order':generation['family_files_sha256']=dict(reversed(list(generation['family_files_sha256'].items())))
    with pytest.raises(AssertionError):execution_check(out,protocol,generation,data,{},root=root)


def test_parallel_verifier_rejects_rehashed_wrong_declared_job_order(tmp_path):
    root,out,protocol,generation,data,execution=execution_fixture(tmp_path)
    execution['jobs']=list(reversed(execution['jobs']));save(out/'execution_protocol.json',execution)
    digest=sha(out/'execution_protocol.json');(out/'execution_protocol.sha256').write_text(digest+'\n')
    generation['execution_protocol_sha256']=digest
    # These hashes match: failure must come from independently rebuilt schedule.
    with pytest.raises(AssertionError):execution_check(out,protocol,generation,data,{},root=root)


def test_optional_public_cache_manifest_binds_semantic_context_and_belief_hashes(tmp_path):
    cache=tmp_path/'cache';cache.mkdir();native=cache/'native.net.xml';native.write_text('<net/>')
    metadata={'belief_sha256':{}}
    for name in PUBLIC_CACHE_FILES[1:]:
        text='{}';digest=hashlib.sha256(text.encode())
        if name.startswith('belief-'):
            states=np.array([1,2]);logs=np.array([0.,1.])
            np.savez_compressed(cache/name,metadata=text,state_ids=states,log_normalizers=logs)
            digest.update(states.astype('<i8').tobytes());digest.update(logs.astype('<f8').tobytes())
            metadata['belief_sha256'][name[7:-4]]=digest.hexdigest()
        else:
            ids=np.array([[-1,0]]);access=np.array([0])
            np.savez_compressed(cache/name,metadata=text,signatures=ids,access=access)
            digest.update(ids.astype('<i4').tobytes());digest.update(access.astype('<i4').tobytes())
            metadata[{'reference5.npz':'reference_sha256','reply10.npz':'reply_sha256','reply20.npz':'reply20_sha256'}[name]]=digest.hexdigest()
    execution={'public_cache_source':str(cache),'public_cache_files_sha256':{n:sha(cache/n) for n in PUBLIC_CACHE_FILES}}
    data={'network':{'native_sha256':sha(native)}}
    assert public_cache_check(execution,metadata,data)['available_local_cache_files_verified']==6
    metadata['reply20_sha256']='0'*64
    with pytest.raises(AssertionError):public_cache_check(execution,metadata,data)
