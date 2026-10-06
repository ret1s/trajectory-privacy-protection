"""Full sole-legacy base-Q audit for a separately frozen response-depth study.

No prior verifier/source is altered. The pre-test depth/source/configuration
contract is checked BEFORE the fresh dataset or base-Q metrics are read. This
file does not read fresh depth utility scores. Public service/graph/profile
reconstruction, realized budget, Q/clock, causal cache, and method-adapted
attacker checks reuse the independent frozen arithmetic APIs.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking
from data.lane_states import build_lane_states
from evaluation.candidate_future_attack import coordinates
from evaluation.lane_travel import LanePoiService, SparseTravel
from experiments.verify_qplanner_study_20261006 import (
    ROOT, PURPOSES, attack_checks, equal, expected_jobs, ledger_check, local_scores,
    public_cache_check, read, relative_path, sha, summarize,
)
from experiments.verify_qplanner_study_20261006_v2 import (
    public_anchor_model, rebuild_profiles,
)

DEFAULT=ROOT/'artifacts/benchmarks/qplanner_depth_base_q_generalization_20261006_v1'
EXECUTOR='experiments/qplanner_parallel_generation_20261006_v2.py'
DRAW_SCHEDULE={'train':1,'selection':1,'test':3}


def fresh_contract(out,*,root=ROOT):
    """Validate sole-legacy adoption before fresh dataset/base/depth scores.

    The depth helper authenticates the common/development source/configuration
    and selected-depth freeze. This adapter independently binds that validated
    contract to the base-Q output it will audit. Historical scores may be read
    for the declared development selection; fresh depth scores are never read.
    """
    out,root=Path(out),Path(root)
    freeze=read(out/'freeze.json')
    assert freeze['schema']=='qplanner-base-Q-for-selected-depth-freeze-v1'
    assert freeze['fresh_test_evaluated'] is False
    depth_output=relative_path(root,freeze['depth_output'])
    from experiments.qplanner_response_depth_generalization_20261006_v2 import depth_contract
    contract=depth_contract(depth_output,root=root)
    depth=read(depth_output/'protocol.json');protocol=read(out/'protocol.json')
    digest=sha(out/'protocol.json')
    assert digest==(out/'protocol.sha256').read_text().strip()==freeze['fresh_protocol_sha256']
    assert digest==depth['base_q_protocol_sha256']==contract['base_q_protocol_sha256']
    assert out.resolve()==relative_path(root,depth['base_q_output']).resolve()
    assert freeze['configuration']==protocol['configuration']==depth['base_q_configuration']
    assert list(protocol['configuration']['methods'])==['legacy_l10']
    assert protocol['configuration']['methods']['legacy_l10']['mode']=='legacy_l10'
    assert protocol['splits']==['train','selection','test'] and protocol['draws_by_split']==DRAW_SCHEDULE
    assert freeze['depth_protocol_sha256']==sha(depth_output/'protocol.json')==contract['depth_protocol_sha256']
    assert freeze['depth_freeze_sha256']==sha(depth_output/'depth_freeze.json')==contract['depth_freeze_sha256']
    assert contract['fresh_dataset_opened_for_contract'] is False and contract['fresh_metrics_opened_for_contract'] is False
    required={'experiments/verify_qplanner_study_20261006.py',
              'experiments/verify_qplanner_study_20261006_v2.py',
              'experiments/qplanner_response_depth_generalization_20261006_v2.py',
              'experiments/verify_qplanner_depth_base_q_20261006_v2.py'}
    assert required<=set(depth['source_sha256'])
    for name in required:assert sha(relative_path(root,name))==depth['source_sha256'][name],name
    return dict(contract,freeze_sha256=sha(out/'freeze.json'),
                sole_legacy_Q_method=True,no_fresh_depth_scores_opened=True,
                old_development_realization_reused=False)


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
    assert e.get('paired_development') is None
    assert generation['paired_development_private_realization'] is False
    assert receipt['paired_development_private_key_block_count']==0
    env=read(out/'execution_environment.json')
    assert all(env['worker_thread_environment'][name]=='1' for name in
               ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'))
    return dict(execution_protocol_sha256=digest,execution_sources=len(e['source_sha256']),
                declared_ordered_jobs=len(jobs),imported_byte_identical_prefix_blocks=len(imported),
                retained_private_key_block_count_metadata=len(retained),private_key_material_verified=False,
                generated_job_receipts=generated,worker_count=len(peaks),public_cache=cache,
                old_development_private_realization_reused=False)

def verify(out,*,validation_output=None):
    out=Path(out)
    contract=fresh_contract(out)
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
    reference=PublicPoiContext(LanePoiService(rn,pois,k=5))
    assert reference.sha256==metadata['reference_sha256']
    profiles=rebuild_profiles(rn,reference,reply,state_ids,expected_destinations,profile_meta,protocol)
    assert profiles.sha256==metadata['profiles_sha256']
    base=public_anchor_model(rn,reference)
    assert base.sha256==metadata['belief_sha256']['0.00125']
    assert np.array_equal(base.state_ids,profiles.state_ids)
    local=MultiPurposeRoadRanking(reply,cache_limit=1024);travel=SparseTravel(rn,cache_limit=256)
    families={f['family_id']:f for f in data['families'] if f['split'] in protocol['splits']}
    methods=list(protocol['configuration']['methods']);budget=protocol['configuration']['budget']
    unit=budget['unit_epsilon_per_m'];max_units=2*budget['horizon']-1
    assert unit==base.epsilon_release==base.epsilon_test and protocol['configuration']['theta_m']==base.theta_m
    equal(budget['total_effective_epsilon_per_m']/(budget['session_slots']*max_units),unit)
    expected_names={f'{name}--draw{draw}.json.gz' for name,family in families.items()
                    for draw in range(1,protocol['draws_by_split'][family['split']]+1)}
    assert expected_names==set(generation['family_files_sha256'])
    counts=dict(bundles=0,events=0,ledger_steps=0,utility_windows=0,wire_rows=0,objective_records=0)
    utility_rows=[];wire_rows=[];timings=defaultdict(list);bundles=[]
    objective_counts=dict(protected_belief_steps=0,normalized_records=0,action_scores=0)
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
    record=dict(schema='qplanner-independent-frozen-depth-base-Q-verification-v1',status='pass',
        protocol_sha256=sha(out/'protocol.json'),generation_sha256=sha(out/'generation.json'),verifier_sha256=sha(Path(__file__)),
        base_verifier_sha256=sha(ROOT/'experiments/verify_qplanner_study_20261006.py'),
        counts=counts,no_private_rng_key_required=True,identical_private_anchor_ledger_read_tapes=True,
        exact_received_service_and_causal_cache_replayed=True,independent_conditional_family_tail_arithmetic=True,
        exact_Q_coordinates_and_directed_track_reachability=True,attack_models_verified=attacks is not None,
        attack_checks=attacks,public_grid_prototype_destinations_independently_verified=True,
        parallel_execution_checks=execution,normalized_profile_checks=objective_counts,
        normalized_public_profile_construction_verified=True,
        normalized_profiles_sha256=profiles.sha256,fresh_freeze_contract=contract,
        old_development_private_realization_reused=False,
        normalized_profile_verifier_sha256=sha(ROOT/'experiments/verify_qplanner_study_20261006_v2.py'),
        bounded_optimizer_score_and_feasibility_verified=False,
        normalized_profile_objective_used_by_legacy_method=False,
        protected_tapes_verified=True,no_depth_scores_opened=True,no_fresh_depth_scores_opened=True,
        global_or_submodular_optimizer_guarantee_certified=False,privacy_theorem_certified=False,
        efficacy_criterion_or_confidence_intervals_verified=False,
        scope='Frozen single-legacy base-Q budget/public output/service/attacker audit before fresh depth utility; unused normalized public profile resource hash also rebuilt. Runtime measurements descriptive, not replayed; same-map synthetic only; no privacy theorem certification')
    target=Path(validation_output) if validation_output else out/('validation.json' if attacks is not None else 'utility_ledger_validation.json')
    if target.exists():equal(record,read(target))
    else:target.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(record,flush=True);return record

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path,nargs='?',default=DEFAULT)
    parser.add_argument('--validation-output',type=Path)
    args=parser.parse_args();verify(args.out,validation_output=args.validation_output)
