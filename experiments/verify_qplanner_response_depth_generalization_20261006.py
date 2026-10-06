"""Independent frozen two-depth utility/cost and family-cluster arithmetic.

The shared adoption contract is authenticated before fresh data are opened.
No sampler key or new location draw is needed. The static public service/local
answers are replayed by the independent development oracle. Family, draw,
N/A, bootstrap and criterion calculations below do not import the production
paired-readout scorer. This verifies a paid-bandwidth utility study, not DP.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking
from data.lane_states import build_lane_states, catalogue_summary
from evaluation.lane_travel import LanePoiService
from experiments.verify_qplanner_study_20261006 import ROOT, PURPOSES, expected_jobs, read, relative_path, sha
from experiments.verify_qplanner_response_depth_20261006 import canonical_sha, replay_expected, same

REPLICATES=10000
SEED=2026100617
PHASES=('all','cold','early_0_180','temporal_tail_400_600')
THIS='experiments/verify_qplanner_response_depth_generalization_20261006.py'
TEST='tests/test_qplanner_response_depth_generalization_verifier.py'
STATISTICAL_TEST_COUNT=20


def verification_sources(protocol):
    return sorted(set(protocol['source_sha256'])|{THIS,TEST,
        'experiments/verify_qplanner_response_depth_20261006.py',
        'experiments/verify_qplanner_study_20261006.py'})


def declare_verification(out,*,root=ROOT):
    """Seal this independent audit source BEFORE fresh depth replay/scoring."""
    out,root=Path(out),Path(root)
    from experiments.qplanner_response_depth_generalization_20261006_v2 import depth_contract
    contract=depth_contract(out,root=root);p=read(out/'protocol.json')
    assert contract['fresh_dataset_opened_for_contract'] is False and contract['fresh_metrics_opened_for_contract'] is False
    assert not any((out/name).exists() for name in ('replay_started.json','readout.json','paired_protocol.json','paired_readout.json','families'))
    if (out/'verification_protocol.json').exists():raise FileExistsError('Keep existing independent verification declaration')
    sources={name:sha(relative_path(root,name)) for name in verification_sources(p)}
    assert sources[THIS]==sha(Path(__file__))
    value=dict(schema='qplanner-response-depth-independent-verification-protocol-v1',
        created_utc=datetime.now(timezone.utc).isoformat(),depth_protocol_sha256=sha(out/'protocol.json'),
        depth_freeze_sha256=sha(out/'depth_freeze.json'),source_sha256=sources,
        selected_depth=p['selected_depth'],depths=p['depths'],criterion=p['criterion'],
        family_bootstrap_replicates=REPLICATES,public_analysis_seed=SEED,fractional_lower_tail_mass=.25,
        synthetic_fixture_test_count=STATISTICAL_TEST_COUNT,synthetic_test_source=TEST,
        declaration_before_depth_replay=True,fresh_dataset_or_depth_scores_opened_for_declaration=False,
        verification_schema='qplanner-independent-response-depth-generalization-verification-v1',
        scope='Independent fixed-Q static replies/local utility/cost and whole-family nested-draw arithmetic; no sampler key, privacy theorem or real-data certificate')
    with (out/'verification_protocol.json').open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
    with (out/'verification_protocol.sha256').open('x') as stream:stream.write(sha(out/'verification_protocol.json')+'\n')
    for name in sources:
        target=relative_path(out/'verification_source_snapshot',name);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(relative_path(root,name).read_bytes())
    return value


def verification_contract(out,*,root=ROOT):
    out,root=Path(out),Path(root);p=read(out/'protocol.json');v=read(out/'verification_protocol.json')
    assert sha(out/'verification_protocol.json')==(out/'verification_protocol.sha256').read_text().strip()
    assert v['schema']=='qplanner-response-depth-independent-verification-protocol-v1'
    assert v['depth_protocol_sha256']==sha(out/'protocol.json') and v['depth_freeze_sha256']==sha(out/'depth_freeze.json')
    assert v['depths']==p['depths'] and v['selected_depth']==p['selected_depth'] and v['criterion']==p['criterion']
    assert (v['family_bootstrap_replicates'],v['public_analysis_seed'],v['fractional_lower_tail_mass'])==(REPLICATES,SEED,.25)
    assert v['synthetic_fixture_test_count']==STATISTICAL_TEST_COUNT and v['synthetic_test_source']==TEST
    assert v['declaration_before_depth_replay'] is True and v['fresh_dataset_or_depth_scores_opened_for_declaration'] is False
    assert v['verification_schema']=='qplanner-independent-response-depth-generalization-verification-v1'
    assert set(v['source_sha256'])==set(verification_sources(p))
    for name,pin in v['source_sha256'].items():
        assert sha(relative_path(root,name))==sha(relative_path(out/'verification_source_snapshot',name))==pin,name
        if name in p['source_sha256']:assert pin==p['source_sha256'][name]
    assert v['source_sha256'][THIS]==sha(Path(__file__))
    return dict(verification_protocol_sha256=sha(out/'verification_protocol.json'),
        verification_sources=len(v['source_sha256']),synthetic_fixture_test_count=STATISTICAL_TEST_COUNT)


def phases(row):
    return [p for p,yes in (('all',True),('cold',row['slot']==0),('early_0_180',row['t']<=180),
            ('temporal_tail_400_600',row['t']>=400)) if yes]


def family_cells(rows,wires,schedule):
    """Conditional event means nested inside draw/family; missing stays None."""
    groups=defaultdict(list);costs=defaultdict(lambda:dict(requests=0,request_bytes=0,reply_bytes=0))
    family_splits={};seen=set();wire_seen=set()
    for row in rows:
        f,s,d=row['family_id'],row['split'],row['draw'];assert family_splits.setdefault(f,s)==s
        key=f,d,row['method'],row['cache'],row['slot'],row['t'];assert key not in seen;seen.add(key)
        assert tuple(row['purposes'])==PURPOSES
        for phase in phases(row):groups[row['method'],s,row['cache'],phase].append(row)
    for row in wires:
        key=row['family_id'],row['draw'],row['method'],row['slot'],row['t'];assert key not in wire_seen;wire_seen.add(key)
        cost=costs[row['method'],row['split'],row['family_id'],row['draw']]
        for field in cost:cost[field]+=row[field]
    output={}
    for key,subset in sorted(groups.items()):
        method,split,cache,phase=key;families=sorted(f for f,s in family_splits.items() if s==split)
        draws=list(range(1,schedule[split]+1));cell={}
        for purpose in PURPOSES:
            values={};draw_values={};coverage=dict(defined_windows=0,total_windows=0,defined_categories=0,total_categories=0)
            for f in families:
                draw_values[f]={};all_valid=[];reference_counts=[]
                for d in draws:
                    items=[r['purposes'][purpose] for r in subset if r['family_id']==f and r['draw']==d]
                    assert items,(key,f,d)
                    valid=[]
                    for v in items:
                        assert (v['recall5'] is None)==(v['reference_category_count']==0)
                        if v['recall5'] is not None:
                            assert np.isfinite(v['recall5']) and 0<=v['recall5']<=1;valid.append(v['recall5'])
                        coverage['defined_windows']+=v['recall5'] is not None;coverage['total_windows']+=1
                        coverage['defined_categories']+=v['reference_category_count'];coverage['total_categories']+=v['all_category_count']
                    reference_counts.append((len(valid),len(items),sum(v['reference_category_count'] for v in items),sum(v['all_category_count'] for v in items)))
                    draw_values[f][str(d)]=statistics.mean(valid) if valid else None;all_valid.extend(valid)
                assert len(set(reference_counts))==1,'Nested draws must share reference coverage'
                values[f]=statistics.mean(all_valid) if all_valid else None
            cell[purpose]=dict(family_values=values,family_draw_values=draw_values,coverage=coverage)
        macro={};macro_draws={};complete=[]
        for f in families:
            defined=[cell[p]['family_values'][f] for p in PURPOSES if cell[p]['family_values'][f] is not None]
            macro[f]=statistics.mean(defined) if defined else None
            if len(defined)==len(PURPOSES):complete.append(f)
            macro_draws[f]={}
            for d in draws:
                defined=[cell[p]['family_draw_values'][f][str(d)] for p in PURPOSES if cell[p]['family_draw_values'][f][str(d)] is not None]
                macro_draws[f][str(d)]=statistics.mean(defined) if defined else None
        cell['equal_purpose_macro']=dict(family_values=macro,family_draw_values=macro_draws,complete_all_purpose_families=complete,
            partial_or_undefined_purpose_families=[f for f in families if f not in complete],
            coverage_rule='average defined purposes locally; separately retain complete-purpose paired sensitivity')
        output[key]=cell
    return output,dict(costs)


def tail_mean(values,mass=.25):
    """Explicit fractional rank weights, including a partial boundary rank."""
    v=np.sort(np.asarray(values,float),axis=-1);n=v.shape[-1]
    assert 0<mass<=1
    if not n:return None
    rank=np.arange(n);weights=np.clip(n*mass-rank,0.,1.)/(n*mass)
    result=v@weights
    return float(result) if result.ndim==0 else result


def paired(left,right,*,replicates=REPLICATES,seed=SEED):
    all_ids=sorted(set(left)|set(right));ids=[f for f in all_ids if left.get(f) is not None and right.get(f) is not None]
    result=dict(family_ids=ids,independent_family_clusters=len(ids),excluded_missing_or_undefined_family_pairs=[f for f in all_ids if f not in ids],
        replicates=replicates,statistical_seed=seed,sign='left minus right; utility positive favors left')
    if not ids:return dict(result,status='N/A: no common defined family pairs',mean_difference=None,percentile95_family_bootstrap=None,
        lower_tail_mean_difference=None,percentile95_lower_tail_difference=None)
    a,b=np.array([left[f] for f in ids]),np.array([right[f] for f in ids]);assert np.isfinite(a).all() and np.isfinite(b).all()
    delta=a-b;resample=np.random.default_rng(seed).integers(len(ids),size=(replicates,len(ids)))
    means=np.sum(delta[resample],axis=1)/len(ids)
    tails=tail_mean(a[resample])-tail_mean(b[resample]);at,bt=tail_mean(a),tail_mean(b)
    return dict(result,status='defined',family_differences=dict(zip(ids,delta.tolist())),mean_difference=float(statistics.mean(delta)),
        percentile95_family_bootstrap=np.quantile(means,[.025,.975]).tolist(),lower_tail_mass=.25,
        left_lower_tail_mean=at,right_lower_tail_mean=bt,lower_tail_mean_difference=at-bt,
        percentile95_lower_tail_difference=np.quantile(tails,[.025,.975]).tolist())


def draw_diagnostic(left,right,draws):
    ids=sorted(f for f in set(left)&set(right) if all(left[f].get(str(d)) is not None and right[f].get(str(d)) is not None for d in draws))
    values={str(d):dict(family_clusters=len(ids),paired_mean_difference=statistics.mean(left[f][str(d)]-right[f][str(d)] for f in ids) if ids else None) for d in draws}
    means=[v['paired_mean_difference'] for v in values.values() if v['paired_mean_difference'] is not None]
    return dict(scope='descriptive within-draw means; draws remain nested in family, not independent sample count',
        complete_family_pairs=ids,draws=values,sign_consistency=dict(positive_draws=sum(x>0 for x in means),zero_draws=sum(x==0 for x in means),
        negative_draws=sum(x<0 for x in means),defined_draws=len(means)),draw_mean_min=min(means) if means else None,draw_mean_max=max(means) if means else None)


def contrasts(cells,schedule,depths,*,fresh=True):
    result={};baseline='service_l20';selected=f'service_l{depths[-1]}'
    for (method,split,cache,phase),cell in sorted(cells.items()):
        if method==baseline:continue
        right=cells[baseline,split,cache,phase]
        for purpose,left in cell.items():
            other=right[purpose];value=paired(left['family_values'],other['family_values'])
            value['within_draw']=draw_diagnostic(left['family_draw_values'],other['family_draw_values'],range(1,schedule[split]+1))
            value['primary_contrast']=fresh and (method,split,cache,phase,purpose)==(selected,'test','current','all','equal_purpose_macro')
            if purpose=='equal_purpose_macro':
                common=set(left['complete_all_purpose_families'])&set(other['complete_all_purpose_families'])
                value['complete_all_purpose_sensitivity']=paired({f:v for f,v in left['family_values'].items() if f in common},
                    {f:v for f,v in other['family_values'].items() if f in common})
                value['partial_or_undefined_purpose_families']=sorted(set(left['partial_or_undefined_purpose_families'])|set(other['partial_or_undefined_purpose_families']))
            else:value.update(left_reference_coverage=left['coverage'],right_reference_coverage=other['coverage'])
            result[f'{method}--minus--{baseline}--{split}--{cache}--{phase}--{purpose}']=value
    return result


def decision(values,depth,criterion):
    key=f'service_l{depth}--minus--service_l20--test--current--all--equal_purpose_macro';value=values.get(key)
    if value is None or value['mean_difference'] is None:return dict(status='N/A: primary has no defined paired family score',passes=False)
    draws=[r['paired_mean_difference'] for r in value['within_draw']['draws'].values()]
    gates=dict(mean_gain=value['mean_difference']>=criterion['minimum_absolute_mean_gain'],
        paired95_lower_bound=value['percentile95_family_bootstrap'][0]>criterion['paired95_lower_bound_gt'],
        every_private_draw_gain=bool(draws) and all(d is not None and d>criterion['every_private_draw_gain_gt'] for d in draws))
    return dict(status='defined',passes=all(gates.values()),gates=gates,primary_key=key,
        independent_family_clusters=value['independent_family_clusters'],
        scope='predeclared conditional static-utility criterion only; not a privacy equivalence or real-data confirmation claim')


def verify(out,*,validation_output=None):
    out=Path(out)
    from experiments.qplanner_response_depth_generalization_20261006_v2 import depth_contract,base_q_receipt
    contract=depth_contract(out);audit_contract=verification_contract(out)
    base,generation,base_certificate=base_q_receipt(out)
    p=read(out/'protocol.json');basep=read(base/'protocol.json');saved=read(out/'readout.json');metadata=read(out/'resources.json')
    assert saved['schema']=='qplanner-response-depth-fresh-replay-v1'
    assert saved['protocol_sha256']==sha(out/'protocol.json') and saved['resources_sha256']==sha(out/'resources.json')
    assert saved['base_generation_sha256']==sha(base/'generation.json') and saved['base_validation_sha256']==sha(base/'validation.json')
    assert saved['only_selected_and_baseline_scored'] is True and saved['no_private_generation'] is True
    assert saved['L20_source_rows_exactly_reproduced'] is True and saved['static_event_recall_monotonicity_asserted'] is True
    started=read(out/'replay_started.json')
    assert started['depth_protocol_sha256']==sha(out/'protocol.json') and started['base_generation_sha256']==sha(base/'generation.json')
    assert started['base_validation_sha256']==sha(base/'validation.json')
    data_path=relative_path(ROOT,p['dataset_path']);assert sha(data_path)==p['dataset_sha256'];data=read(data_path)
    network=relative_path(ROOT,data['network']['compressed_path']);assert sha(network)==data['network']['compressed_sha256']
    assert hashlib.sha256(gzip.decompress(network.read_bytes())).hexdigest()==data['network']['native_sha256']
    jobs=expected_jobs(data,basep);assert list(generation['family_files_sha256'])==[j['name'] for j in jobs]
    assert saved['source_family_files_sha256']==generation['family_files_sha256']
    assert list(saved['family_files_sha256'])==list(generation['family_files_sha256'])
    assert set(x.name for x in (out/'families').iterdir())==set(generation['family_files_sha256'])
    families={f['family_id']:f for f in data['families']};assert sum(f['split']=='test' for f in families.values())>=24
    rn=build_lane_states(network,spacing_m=40.)
    pois=[{k:v for k,v in poi.items() if k not in ('vertex','access_offset_m')} for poi in read(ROOT/'artifacts/benchmarks/research_loop/resources.json')['pois_used']]
    full=PublicPoiContext(LanePoiService(rn,pois,k=60));reply=PublicPoiContext(LanePoiService(rn,pois,k=20));reference=PublicPoiContext(LanePoiService(rn,pois,k=5))
    assert np.array_equal(full.signatures[:,:,:20],reply.signatures) and np.array_equal(full.access,reply.access)
    assert full.pois==reply.pois==reference.pois and metadata['reference_sha256']==reference.sha256
    assert metadata['native_sha256']==data['network']['native_sha256'] and metadata['catalogue']==catalogue_summary(rn)
    expected_kernel=dict(full_reply60_sha256=full.sha256,frozen_reply20_sha256=reply.sha256,
        depth_context_sha256={str(d):canonical_sha({'parent_sha256':full.sha256,'depth':d}) for d in (20,30,40,60)},exact_l20_prefix_asserted=True)
    same(expected_kernel,p['service_kernel'])
    for key,value in expected_kernel.items():same(value,metadata[key])
    assert reply.sha256==read(base/'resources.json')['reply20_sha256']
    for name,pin in metadata['source_sha256'].items():assert p['public_inputs_sha256'][name]==pin
    if metadata['public_cache_copies_sha256']:same(metadata['public_cache_copies_sha256'],read(base/'execution_protocol.json')['public_cache_files_sha256'])
    local=MultiPurposeRoadRanking(reference,cache_limit=1024);rows=[];wires=[]
    for name,pin in generation['family_files_sha256'].items():
        assert sha(base/'families'/name)==pin and sha(out/'families'/name)==saved['family_files_sha256'][name]
        original=read(base/'families'/name);expected=replay_expected(original,families[original['evaluator_only']['family_id']],rn,full,local,depths=p['depths'])
        expected['source_bundle_sha256']=pin;same(expected,read(out/'families'/name))
        rows.extend(expected['utility']);wires.extend(expected['wire']);print('Fresh fixed-Q response-depth verified',name,flush=True)
    cells,costs=family_cells(rows,wires,p['draws_by_split']);values=contrasts(cells,p['draws_by_split'],p['depths'])
    pairedp=read(out/'paired_protocol.json');pairedr=read(out/'paired_readout.json')
    assert pairedp['scope']=='independent-synthetic-generalization' and pairedp['depths']==p['depths'] and pairedp['criterion']==p['criterion']
    assert pairedp['source_protocol_sha256']==sha(out/'protocol.json') and pairedp['source_readout_sha256']==sha(out/'readout.json')
    assert pairedp['analysis_source_sha256']==p['source_sha256']['experiments/qplanner_response_depth_generalization_20261006_v2.py']
    assert pairedp['family_bootstrap_replicates']==REPLICATES and pairedp['public_analysis_seed']==SEED and pairedp['primary']==p['primary']
    assert pairedr['schema']=='qplanner-response-depth-paired-readout-v1' and pairedr['paired_protocol_sha256']==sha(out/'paired_protocol.json')
    assert pairedr['source_readout_sha256']==sha(out/'readout.json') and pairedr['defense_selected_by_this_readout'] is False
    assert pairedr['exact_Q_clock_ledger_certificate_checked'] is True and pairedr['no_private_generation'] is True
    same({f'{m}--{s}--{c}--{ph}':cell for (m,s,c,ph),cell in cells.items()},pairedr['conditional_family_cells'])
    same([dict(method=m,split=s,family_id=f,draw=d,**cost) for (m,s,f,d),cost in sorted(costs.items())],pairedr['family_draw_costs'])
    same(values,pairedr['contrasts']);primary=decision(values,p['selected_depth'],p['criterion']);same(primary,pairedr['primary_criterion_result'])
    certificate=dict(schema='qplanner-independent-response-depth-generalization-verification-v1',status='pass',
        protocol_sha256=sha(out/'protocol.json'),readout_sha256=sha(out/'readout.json'),paired_readout_sha256=sha(out/'paired_readout.json'),
        verifier_sha256=sha(Path(__file__)),depth_oracle_sha256=sha(ROOT/'experiments/verify_qplanner_response_depth_20261006.py'),
        independent_verification_contract=audit_contract,
        verification_protocol_sha256=audit_contract['verification_protocol_sha256'],
        base_Q_validation_sha256=sha(base/'validation.json'),frozen_adoption_contract=contract,
        bundles=len(generation['family_files_sha256']),utility_windows=len(rows),wire_rows=len(wires),selected_depth=p['selected_depth'],
        exact_L20_scores_replies_cost_reproduced=True,current_and_causal_cache_monotonicity_verified=True,
        independent_whole_family_nested_draw_bootstrap_and_NA_arithmetic=True,primary_criterion_result=primary,
        no_private_rng_key_required=True,no_new_Q_GPS_anchors_or_ledger_reads=True,
        privacy_theorem_certified=False,scope='Frozen same-map synthetic fixed-Q static depth utility/cost; paid bandwidth; no timing, new location method, real-data or privacy-superiority certificate')
    target=Path(validation_output) if validation_output else out/'validation.json'
    if target.exists():same(certificate,read(target))
    else:target.write_text(json.dumps(certificate,indent=2,allow_nan=False)+'\n')
    print(certificate,flush=True);return certificate


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    parser.add_argument('--validation-output',type=Path);parser.add_argument('--declare',action='store_true');args=parser.parse_args()
    if args.declare:print(declare_verification(args.out),flush=True)
    else:verify(args.out,validation_output=args.validation_output)
