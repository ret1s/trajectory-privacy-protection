"""Independent normalized-profile study audit; sealed V1 verifier unchanged.

No normalized profile builder, planner, or generation runner is imported.
Public references are rebuilt with the existing directed local-ranking oracle.
The base audit helpers are imported from the immutable previous verifier.
This verifier is development-only: every job must pair with the complete
immutable earlier development cohort. Fresh validation requires a new scope.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from benchmark.anchor_belief import AnchorBelief, PublicAnchorModel
from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from data.lane_states import build_lane_states
from evaluation.candidate_future_attack import coordinates
from evaluation.lane_travel import LanePoiService, SparseTravel
from experiments.verify_qplanner_study_20261006 import (
    ROOT, PURPOSES, attack_checks, equal, expected_jobs, ledger_check,
    local_scores, public_cache_check, read, relative_path, sha, summarize,
)

DEFAULT=ROOT/'artifacts/benchmarks/qplanner_development_20261006_v3'
BASELINE=ROOT/'artifacts/benchmarks/qplanner_development_20261006_v2'
EXECUTOR='experiments/qplanner_parallel_generation_20261006_v2.py'
PROFILE_BUILDER='benchmark/public_service_profiles_v2.py'
PAIRED_CONFIGURATION_FIELDS=('budget','K','server_L','reference_k','public_radius_m',
    'public_destination_rule','public_destination_grid_quantiles','theta_m',
    'utility_purposes','private_utility_destination','cache')
PAIRED_PRIVACY_SOURCES=('core/mechanisms.py','core/session_budget.py',
    'benchmark/anchor_belief.py','benchmark/response_aware_belief.py','benchmark/public_poi_context.py',
    'benchmark/engines/public_service_planner.py','benchmark/engines/paced_slack.py',
    'benchmark/engines/paced_guard.py','benchmark/engines/matched_filter.py',
    'benchmark/engines/filtered_cover.py','benchmark/engines/slack_progress.py',
    'benchmark/engines/progress_cover.py','benchmark/engines/quotient_cover.py',
    'benchmark/engines/fair_cover.py','benchmark/engines/service_cover.py',
    'data/lane_states.py','evaluation/lane_travel.py')


def artifact_source(root,name):
    """Relocate published artifact provenance, never resolve a private key path."""
    path=Path(name);root=Path(root).resolve()
    if not path.resolve().is_relative_to(root):
        parts=path.parts;assert 'artifacts' in parts
        path=root/Path(*parts[parts.index('artifacts'):])
    assert path.is_dir() and path.resolve().is_relative_to(root)
    return path


def paired_development_check(out,execution,protocol,generation,data,*,root=ROOT):
    """Validate public pre-generation pairing contract; do not open its keys."""
    plan=execution['paired_development'];assert plan is not None
    assert set(plan)=={'schema','source_output','source_files_sha256','family_files_sha256',
        'privacy_control_source_sha256','planned_jobs','source_private_root','retained_key_blocks',
        'matched_controls','scope','key_policy'}
    assert plan['schema']=='qplanner-matched-development-private-realization-v1'
    assert execution['transfer'] is execution['private_transfer'] is None
    source=artifact_source(root,plan['source_output']);prior=read(source/'protocol.json')
    assert sha(source/'protocol.json')==(source/'protocol.sha256').read_text().strip()
    assert 'test' not in prior['splits'] and 'test' not in protocol['splits']
    for field in ('dataset_path','dataset_sha256','draws_by_split','splits','public_inputs_sha256'):
        assert prior[field]==protocol[field],field
    for field in PAIRED_CONFIGURATION_FIELDS:
        assert prior['configuration'][field]==protocol['configuration'][field],field
    controls=['legacy_l10','aligned_nearest'];assert plan['matched_controls']==controls
    for method in controls:
        old=prior['configuration']['methods'][method];new=protocol['configuration']['methods'][method]
        assert old['mode']==new['mode']==method
        assert old.get('utility_slack',prior['configuration']['utility_slack'])==new.get(
            'utility_slack',protocol['configuration']['utility_slack'])
    assert set(plan['privacy_control_source_sha256'])==set(PAIRED_PRIVACY_SOURCES)
    for name,pin in plan['privacy_control_source_sha256'].items():
        assert prior['source_sha256'][name]==protocol['source_sha256'][name]==sha(relative_path(root,name))==pin,name
    for name,pin in prior['source_sha256'].items():
        assert sha(relative_path(root,name))==sha(relative_path(source/'source_snapshot',name))==pin,name
    files={'protocol.json','protocol.sha256','generation.json','generation_started.json','resources.json',
           'execution_protocol.json','execution_protocol.sha256','execution_environment.json'}
    assert set(plan['source_files_sha256'])==files
    for name,pin in plan['source_files_sha256'].items():assert sha(source/name)==pin,name
    old_execution=read(source/'execution_protocol.json')
    assert sha(source/'execution_protocol.json')==(source/'execution_protocol.sha256').read_text().strip()
    assert old_execution['common_protocol_sha256']==sha(source/'protocol.json')
    for name,pin in old_execution['source_sha256'].items():
        assert sha(relative_path(root,name))==sha(relative_path(source/'execution_source_snapshot',name))==pin,name
    jobs=expected_jobs(data,protocol);assert plan['planned_jobs']==jobs==expected_jobs(data,prior)
    names=[job['name'] for job in jobs];old_generation=read(source/'generation.json')
    assert list(old_generation['family_files_sha256'])==names
    assert plan['family_files_sha256']==old_generation['family_files_sha256']
    assert {p.name for p in (source/'families').iterdir()}==set(names)
    for name,pin in plan['family_files_sha256'].items():
        path=source/'families'/name;assert path.is_file() and not path.is_symlink() and sha(path)==pin,name
    assert plan['retained_key_blocks']==names
    private=Path(plan['source_private_root'])
    assert private.is_absolute() and private.is_relative_to('/private/tmp')
    assert not private.resolve().is_relative_to(Path(root).resolve())
    receipt=read(Path(out)/'transfer_receipt.json')
    assert receipt['paired_development_private_key_block_count']==len(names)
    assert receipt['copied_family_files_sha256']=={} and receipt['retained_private_key_block_count']==0
    assert receipt['private_key_bytes_or_hashes_exported'] is False
    assert generation['imported_completed_block_count']==0 and generation['generated_block_count']==len(names)
    assert generation['paired_development_private_realization'] is True
    return dict(source_protocol_sha256=sha(source/'protocol.json'),source_generation_sha256=sha(source/'generation.json'),
        paired_blocks=len(names),source_files_verified=len(files),unchanged_privacy_control_sources=len(PAIRED_PRIVACY_SOURCES),
        retained_private_key_block_count_metadata=len(names),private_key_material_verified=False,
        old_completed_output_blocks_imported=False,fresh_test_pairing_permitted=False)


def profile_resource_digest(references,masses,case_counts,metadata,poi_count):
    """Reproduce the declared serialization digest, not the builder algorithm."""
    combined=sum(masses[p] for p in PURPOSES)/len(PURPOSES)
    rows=[];cols=[]
    for i,ref in enumerate(references):rows.extend([i]*len(ref));cols.extend(ref)
    incidence=csr_matrix((np.ones(len(rows),dtype=np.int16),(rows,cols)),
                         shape=(len(references),poi_count+1))
    digest=hashlib.sha256(json.dumps(metadata,sort_keys=True,separators=(',',':')).encode())
    for array in (combined.data.astype('<f8'),combined.indices.astype('<i8'),combined.indptr.astype('<i8'),
                  incidence.indices.astype('<i8'),incidence.indptr.astype('<i8')):
        digest.update(array.tobytes())
    outer=hashlib.sha256(digest.hexdigest().encode())
    for purpose in PURPOSES:
        m=masses[purpose]
        for array in (m.data.astype('<f8'),m.indices.astype('<i8'),m.indptr.astype('<i8'),
                      case_counts[purpose].astype('<i4')):
            outer.update(array.tobytes())
    return outer.hexdigest()


def rebuild_profiles(rn,reference,reply,state_ids,destinations,metadata,protocol):
    """Enumerate local references independently, then average valid categories.

    This intentionally uses the existing QuerySpec/local oracle, not the new
    builder's batched cost tables, profile objects, or normalization routine.
    The first-seen reference order is part of the pinned resource serialization.
    """
    assert reference.rn is rn and reply.rn is rn and reference.k==5 and reply.k==20
    assert reference.pois==reply.pois and reference.categories==reply.categories
    assert np.array_equal(reference.access,reply.access)
    assert np.array_equal(reference.signatures,reply.signatures[:,:,:5])
    state_ids=np.asarray(state_ids,dtype=int);destinations=sorted(destinations)
    local=MultiPurposeRoadRanking(reply,cache_limit=256);all_ids=np.ones(local.n,dtype=bool)
    references=[];indexes={};entries={};counts={};empty_categories=0
    for purpose in PURPOSES:
        cases=destinations if purpose==QueryPurpose.MIN_DETOUR.value else [None]
        counts[purpose]=np.zeros((len(state_ids),len(cases)),dtype=np.int16);entries[purpose]=[]
        for column,destination in enumerate(cases):
            for latent,state in enumerate(state_ids):
                accessed=int(reference.access[state]);valid=[]
                for category in local.categories:
                    query=QuerySpec(QueryPurpose(purpose),category,k=5,
                        radius_m=metadata['public_radius_m'] if purpose==QueryPurpose.WITHIN_RADIUS.value else None,
                        destination_state=destination)
                    ref=tuple(sorted(local.top(accessed,all_ids,query)))
                    if ref:valid.append(ref)
                counts[purpose][latent,column]=len(valid)
                empty_categories+=len(local.categories)-len(valid)
                for ref in valid or [()]:
                    if ref not in indexes:indexes[ref]=len(references);references.append(ref)
                    probability=1./(len(cases)*len(valid)) if valid else 1./len(cases)
                    entries[purpose].append((latent,indexes[ref],probability))
    masses={}
    for purpose in PURPOSES:
        rows,columns,values=zip(*entries[purpose])
        masses[purpose]=csr_matrix((values,(rows,columns)),shape=(len(state_ids),len(references)))
        assert np.allclose(np.asarray(masses[purpose].sum(axis=1)).ravel(),1.,rtol=0.,atol=1e-12)
    expected=dict(schema='public-service-profiles-purpose-normalized-v2',reference_k=5,reply_l=20,
        reference_context_sha256=reference.sha256,reply_context_sha256=reply.sha256,
        catalogue_sha256=rn.catalogue_sha256,state_ids=state_ids.tolist(),
        public_destination_states=destinations,public_radius_m=float(metadata['public_radius_m']),
        purpose_prior='equal defined purposes after within-purpose protected valid-case conditioning',
        destination_prior='uniform fixed public graph states BEFORE within-purpose N/A conditioning',
        category_prior='equal VALID categories separately within each latent/purpose/destination case',
        normalization_denominator='D_p(b)=sum_x,d b_x*pi_p(d)*1[n_valid_categories(x,p,d)>0]',
        empty_reference='category N/A excluded inside case; whole empty case retained; entirely undefined purpose N/A',
        input_profile_count=len(state_ids)*(3+len(destinations))*len(local.categories),
        empty_input_category_profile_count=empty_categories,
        all_empty_case_count_by_purpose={p:int(np.sum(v==0)) for p,v in counts.items()},
        coalesced_profile_count=len(references),server_service='nearest-distance top-L per ALL public categories',
        access_rule='public coordinate-to-state mapping, including duplicate-coordinate ambiguity',
        builder_sha256=protocol['source_sha256'][PROFILE_BUILDER])
    equal(expected,metadata);assert set(expected)==set(metadata)
    digest=profile_resource_digest(references,masses,counts,expected,len(reply.pois))
    return SimpleNamespace(reference_profiles=tuple(references),purpose_masses=masses,
        case_valid_category_counts=counts,state_ids=state_ids,metadata=expected,context=reply,
        lengths=np.asarray(list(map(len,references))),sha256=digest)


def normalized_mass_oracle(profiles,belief):
    """Direct conditional expectation, independent of the V2 mass adapter."""
    belief=np.asarray(belief,dtype=float)
    assert belief.shape==(len(profiles.state_ids),) and np.isfinite(belief).all()
    assert np.all(belief>=0) and belief.sum()>0
    belief=belief/belief.sum();nonempty=profiles.lengths>0
    conditional={};valid_mass={};undefined={}
    for purpose in PURPOSES:
        values=np.asarray(belief@profiles.purpose_masses[purpose]).ravel()
        valid_mass[purpose]=float(sum(values[i] for i in np.flatnonzero(nonempty)))
        undefined[purpose]=float(sum(values[i] for i in np.flatnonzero(~nonempty)))
        if valid_mass[purpose]>0:
            conditional[purpose]=np.where(nonempty,values/valid_mass[purpose],0.)
    mass=np.zeros(len(profiles.reference_profiles))
    for values in conditional.values():mass+=values/len(conditional)
    diagnostics=dict(normalization='valid_categories_per_case_then_valid_cases_per_purpose_then_equal_defined_purposes',
        purpose_valid_case_mass=valid_mass,purpose_undefined_case_mass=undefined,
        defined_purposes=list(conditional),undefined_purposes=[p for p in PURPOSES if p not in conditional],
        effective_purpose_weights={p:1./len(conditional) if p in conditional else 0. for p in PURPOSES},
        empty_profile_mass=math.fsum(undefined.values())/len(undefined),
        empty_mass_definition='equal-prior probability of ALL categories undefined in a latent/destination case; partial category N/A normalized inside case')
    if conditional:assert np.isclose(mass.sum(),1.,rtol=0,atol=1e-12)
    else:assert np.all(mass==0.)
    return mass,diagnostics


def direct_score(profiles,mass,selected,*,tail_mass,risk_weight):
    """Exact set intersections and a sorted fractional weighted lower tail."""
    assert 0<tail_mass<=1 and 0<=risk_weight<=1
    if not np.any(mass>0):return None
    available={int(i) for state in selected for i in profiles.context.query_indices(state).ravel() if i>=0}
    rows=sorted((len(set(ref)&available)/len(ref),float(weight))
                for ref,weight in zip(profiles.reference_profiles,mass) if weight>0 and ref)
    mean=math.fsum(value*weight for value,weight in rows);remaining=tail_mass;terms=[]
    for value,weight in rows:
        taken=min(weight,max(0.,remaining));terms.append(taken*value);remaining-=taken
        if remaining<=0:break
    assert abs(remaining)<=1e-12
    tail=math.fsum(terms)/tail_mass
    return dict(mean=mean,lower_tail_cvar=tail,objective=(1-risk_weight)*mean+risk_weight*tail)


def public_anchor_model(rn,reference):
    # The frozen native helper declares uniform public120m cell occupancy and
    # 200m latent cells. No session/family GPS is accepted by this reconstruction.
    _,inverse,counts=np.unique(np.floor(rn.xy/120.).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    prior=1./counts[inverse];prior/=prior.sum()
    return PublicAnchorModel(rn,reference,prior,spacing_m=200.,epsilon_release=.00125,
                             epsilon_test=.00125,theta_m=200.)


def normalized_objective_checks(bundle,family,base,profiles,protocol,travel,counts):
    methods=protocol['configuration']['methods'];targets=bundle['evaluator_only']['sessions']
    for slot,target in enumerate(targets):
        events=bundle['public']['streams']['raw'][slot]['events']
        baseline=target['ledger']['legacy_l10'];belief=AnchorBelief(base);previous={m:None for m in methods}
        previous_t=None
        for index,(event,anchor,entry) in enumerate(zip(events,baseline['anchors'],baseline['ledger'])):
            t=event['timestamp_s'];absolute=family['evaluator_only']['sessions'][slot]['depart_s']+t
            weights=belief.update(anchor,absolute,observed=entry['private_read'])
            mass,diagnostics=normalized_mass_oracle(profiles,weights);counts['protected_belief_steps']+=1
            for method,config in methods.items():
                states=target['ledger'][method]['states'][index]
                if not config['mode'].startswith('normalized_'):
                    previous[method]=states;continue
                record=target['ledger'][method]['planner_objectives'][index]
                assert record['normalization_mode']==config['mode']
                equal(diagnostics,record['normalization_diagnostics'])
                equal(diagnostics['empty_profile_mass'],record['empty_profile_mass'])
                assert record['active_profile_count']==int(np.sum(mass>0))
                score=direct_score(profiles,mass,states,tail_mass=.25,risk_weight=config['risk_weight'])
                if score is None:
                    assert record['value'] is record['mean_value'] is record['lower_tail_cvar'] is None
                    assert record['risk_history']==[] and record['risk_termination']=='undefined_public_reference'
                else:
                    equal(dict(value=score['objective'],mean_value=score['mean'],lower_tail_cvar=score['lower_tail_cvar']),record)
                    assert states==record['risk_states'] and record['risk_history'][0]['states']==record['mean_baseline_states']
                    initial=direct_score(profiles,mass,record['mean_baseline_states'],tail_mass=.25,risk_weight=config['risk_weight'])
                    floor=initial['mean']-.01;equal(floor,record['risk_mean_floor'])
                    prior=None
                    assert 1<=len(record['risk_history'])<=4
                    for step in record['risk_history']:
                        exact=direct_score(profiles,mass,step['states'],tail_mass=.25,risk_weight=config['risk_weight'])
                        equal(exact,step);assert exact['mean']>=floor-1e-12
                        if prior is not None:
                            assert step['objective']>prior['objective']+1e-12
                            assert sum(a!=b for a,b in zip(step['states'],prior['states']))==1
                        if previous[method] is not None:
                            assert all(state in travel.reachable(old,t-previous_t) for old,state in zip(previous[method],step['states']))
                        prior=step;counts['action_scores']+=1
                    assert record['risk_history'][-1]['states']==states
                    if config['risk_weight']==0.:
                        assert len(record['risk_history'])==1 and record['risk_termination']=='mean_baseline'
                previous[method]=states;counts['normalized_records']+=1
            previous_t=t


def control_pairing_inputs(out,protocol,generation,baseline):
    """Pin immutable earlier controls; private sampler keys are never opened."""
    baseline=Path(baseline);prior=read(baseline/'protocol.json');manifest=read(baseline/'generation.json')
    assert sha(baseline/'protocol.json')==(baseline/'protocol.sha256').read_text().strip()
    plan=read(Path(out)/'execution_protocol.json')['paired_development'];assert plan is not None
    assert sha(baseline/'protocol.json')==plan['source_files_sha256']['protocol.json']
    assert sha(baseline/'generation.json')==plan['source_files_sha256']['generation.json']
    assert manifest['family_files_sha256']==plan['family_files_sha256']
    assert prior['dataset_path']==protocol['dataset_path'] and prior['dataset_sha256']==protocol['dataset_sha256']
    assert prior['public_inputs_sha256']==protocol['public_inputs_sha256']
    assert list(manifest['family_files_sha256'])==list(generation['family_files_sha256'])
    for name,pin in manifest['family_files_sha256'].items():assert sha(baseline/'families'/name)==pin,name
    return dict(path=baseline,protocol_sha256=sha(baseline/'protocol.json'),generation_sha256=sha(baseline/'generation.json'),
                family_files_sha256=manifest['family_files_sha256'],blocks=0,controls=['legacy_l10','aligned_nearest'],
                private_keys_verified=False)


def control_pairing_bundle(bundle,name,receipt,methods):
    old=read(receipt['path']/'families'/name)
    for method in receipt['controls']:
        assert method in methods
        assert bundle['public']['streams'][method]==old['public']['streams'][method]
        for field in ('utility','wire'):
            assert [r for r in bundle[field] if r['method']==method]==[r for r in old[field] if r['method']==method]
        assert bundle['evaluator_only']['epoch_accounting'][method]==old['evaluator_only']['epoch_accounting'][method]
    for at,target in enumerate(bundle['evaluator_only']['sessions']):
        previous=old['evaluator_only']['sessions'][at]
        for field in ('slot','choice_index','destination_role','origin_xy','destination_xy','next_edge_labels'):
            assert target[field]==previous[field]
        for method in methods:
            source=previous['ledger']['legacy_l10'];current=target['ledger'][method]
            for field in ('anchors','ledger','supplier_times_s','private_reads','spent_per_m','allocation'):
                assert current[field]==source[field],(name,method,field)
        for method in receipt['controls']:
            assert target['ledger'][method]['states']==previous['ledger'][method]['states']
    receipt['blocks']+=1


def public_pairing_receipt(receipt):return {k:v for k,v in receipt.items() if k!='path'}



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
    pairing=paired_development_check(out,e,protocol,generation,data,root=root)
    env=read(out/'execution_environment.json')
    assert all(env['worker_thread_environment'][name]=='1' for name in
               ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'))
    return dict(execution_protocol_sha256=digest,execution_sources=len(e['source_sha256']),
                declared_ordered_jobs=len(jobs),imported_byte_identical_prefix_blocks=len(imported),
                retained_private_key_block_count_metadata=len(retained),private_key_material_verified=False,
                generated_job_receipts=generated,worker_count=len(peaks),public_cache=cache,
                paired_development=pairing)

def verify(out,*,validation_output=None,baseline=BASELINE):
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
    baseline_receipt=control_pairing_inputs(out,protocol,generation,baseline)
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
        normalized_objective_checks(bundle,family,base,profiles,protocol,travel,objective_counts)
        control_pairing_bundle(bundle,name,baseline_receipt,methods)
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
    record=dict(schema='qplanner-independent-normalized-profile-verification-v2',status='pass',
        protocol_sha256=sha(out/'protocol.json'),generation_sha256=sha(out/'generation.json'),verifier_sha256=sha(Path(__file__)),
        base_verifier_sha256=sha(ROOT/'experiments/verify_qplanner_study_20261006.py'),
        counts=counts,no_private_rng_key_required=True,identical_private_anchor_ledger_read_tapes=True,
        exact_received_service_and_causal_cache_replayed=True,independent_conditional_family_tail_arithmetic=True,
        exact_Q_coordinates_and_directed_track_reachability=True,attack_models_verified=attacks is not None,
        attack_checks=attacks,public_grid_prototype_destinations_independently_verified=True,
        parallel_execution_checks=execution,normalized_profile_checks=objective_counts,
        normalized_public_profile_construction_verified=True,
        normalized_profiles_sha256=profiles.sha256,immutable_control_pairing=public_pairing_receipt(baseline_receipt),
        bounded_optimizer_score_and_feasibility_verified=True,
        global_or_submodular_optimizer_guarantee_certified=False,privacy_theorem_certified=False,
        scope='Frozen normalized public profile/score and output arithmetic/causality audit; runtime measurements are descriptive, not replayed; development status unchanged')
    target=Path(validation_output) if validation_output else out/('validation.json' if attacks is not None else 'utility_ledger_validation.json')
    if target.exists():equal(record,read(target))
    else:target.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(record,flush=True);return record

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path,nargs='?',default=DEFAULT)
    parser.add_argument('--baseline',type=Path,default=BASELINE)
    parser.add_argument('--validation-output',type=Path)
    args=parser.parse_args();verify(args.out,validation_output=args.validation_output,baseline=args.baseline)
