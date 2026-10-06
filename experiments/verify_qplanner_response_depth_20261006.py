"""Independent fixed-Q static response-depth audit, with no private RNG key.

Rebuilds ordered service replies and local purpose answers rather than importing
the depth runner, its PrefixContext, scorer, summaries or selector. Public JSON
costs are application payload estimates, not timing/HTTP/TLS measurements.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking
from data.lane_states import build_lane_states, catalogue_summary
from evaluation.lane_travel import LanePoiService
from experiments.verify_qplanner_study_20261006 import (
    ROOT, PURPOSES, expected_jobs, local_scores, read, relative_path, sha, summarize,
)

DEPTHS=(20,30,40,60)
METHOD='legacy_l10'
REPLAY='experiments/qplanner_response_depth_20261006.py'
BASE_VERIFIER='experiments/verify_qplanner_study_20261006_v2.py'
CRITERION={'split':'selection','cache':'current','phase':'all',
    'candidate_depths_in_order':[30,40,60],'baseline_depth':20,
    'minimum_equal_family_equal_purpose_recall_gain':.02,
    'minimum_nearest_distance_recall_gain':0.,'maximum_total_reply_json_byte_ratio':2.5,
    'choice':'smallest eligible depth; NONE if no candidate passes all gates',
    'undefined':'N/A is never zero; require identical complete family pairs and reference coverage',
    'status':'exploratory development selection; fresh confirmation requires a separate freeze'}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def same(expected,actual):
    """Exact inventories and discrete values, a small arithmetic tolerance only."""
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and set(expected)==set(actual),(set(expected),set(actual))
        for key,value in expected.items():same(value,actual[key])
    elif isinstance(expected,(tuple,list)):
        assert isinstance(actual,(tuple,list)) and len(expected)==len(actual)
        for a,b in zip(expected,actual):same(a,b)
    elif isinstance(expected,float):
        assert np.isfinite(expected) and np.isclose(expected,actual,rtol=0.,atol=1e-12),(expected,actual)
    else:assert expected==actual,(expected,actual)


def source_path(value,root):
    path=Path(value)
    if not path.is_absolute():return relative_path(root,path)
    if path.resolve().is_relative_to(root.resolve()):return path
    assert 'artifacts' in path.parts
    return root/Path(*path.parts[path.parts.index('artifacts'):])


def development_contract(out,*,root=ROOT):
    """Authenticate development policy and complete original Q before scoring."""
    out,root=Path(out),Path(root);p=read(out/'protocol.json')
    assert p['schema']=='qplanner-response-depth-development-v1'
    assert sha(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()
    assert p['depths']==list(DEPTHS) and p['selection_criterion']==CRITERION
    assert p['frozen_coordinate_method']==METHOD and 'test' not in p['splits']
    for name,pin in p['source_sha256'].items():
        assert sha(relative_path(root,name))==sha(relative_path(out/'source_snapshot',name))==pin,name
    assert REPLAY in p['source_sha256']
    source=source_path(p['source_output'],root)
    for name,pin in p['source_files_sha256'].items():assert sha(source/name)==pin,name
    old=read(source/'protocol.json');g=read(source/'generation.json')
    assert sha(source/'protocol.json')==(source/'protocol.sha256').read_text().strip()
    assert old['configuration']==p['fixed_configuration'] and old['splits']==p['splits']
    assert old['draws_by_split']==p['draws_by_split'] and old['dataset_path']==p['dataset_path']
    assert old['dataset_sha256']==p['dataset_sha256']
    assert g['family_files_sha256']==p['family_files_sha256']
    assert set(x.name for x in (source/'families').iterdir())==set(g['family_files_sha256'])
    for name,pin in g['family_files_sha256'].items():assert sha(source/'families'/name)==pin,name
    for name,pin in old['source_sha256'].items():
        assert sha(relative_path(root,name))==sha(relative_path(source/'source_snapshot',name))==pin,name
    for name,pin in old['public_inputs_sha256'].items():assert sha(relative_path(root,name))==pin,name
    c=old['configuration'];assert (c['K'],c['server_L'],c['reference_k'],c['public_radius_m'])==(5,20,5,1000.)
    assert c['methods'][METHOD]['mode']==METHOD
    certificate=read(source/'validation.json')
    assert certificate['status']=='pass' and certificate['protocol_sha256']==sha(source/'protocol.json')
    assert certificate['generation_sha256']==sha(source/'generation.json')
    assert certificate['verifier_sha256']==sha(relative_path(root,BASE_VERIFIER))
    assert certificate['identical_private_anchor_ledger_read_tapes'] is True
    assert certificate['exact_received_service_and_causal_cache_replayed'] is True
    return p,source,old,g


def monotonic(previous,current):
    for purpose in PURPOSES:
        a,b=previous[purpose],current[purpose]
        for key in ('reference_category_count','all_category_count','reference_poi_total'):assert a[key]==b[key]
        for key in ('recall5','completion'):
            assert (a[key] is None)==(b[key] is None)
            if a[key] is not None:assert b[key]+1e-12>=a[key],(purpose,key,a[key],b[key])
        assert b['overlap_total']>=a['overlap_total']


def reply_at(context,state,depth):
    ids=context.query_indices(state)[:,:depth].ravel()
    ids=list(map(int,ids[ids>=0]))
    records=[{k:context.pois[i][k] for k in ('id','category','lat','lon')} for i in ids]
    return ids,len(json.dumps({'results':records},separators=(',',':'),ensure_ascii=False).encode())


def replay_expected(bundle,family,rn,full,local,*,depths=DEPTHS):
    """Pure fixed-stream oracle; private GPS/endpoint used only for local scoring."""
    depths=tuple(depths);assert depths[0]==20 and sorted(set(depths))==list(depths)
    truth,group=bundle['evaluator_only'],bundle['public']
    original_util={(r['slot'],r['t'],r['cache']):r for r in bundle['utility'] if r['method']==METHOD}
    original_wire={(r['slot'],r['t']):r for r in bundle['wire'] if r['method']==METHOD}
    assert len(original_util)==sum(r['method']==METHOD for r in bundle['utility'])
    assert len(original_wire)==sum(r['method']==METHOD for r in bundle['wire'])
    inventory={(s,e['timestamp_s']) for s,session in enumerate(group['streams'][METHOD]) for e in session['events']}
    assert set(original_wire)==inventory and set(original_util)=={(s,t,c) for s,t in inventory for c in ('current','static_epoch_cache')}
    assert len(group['streams'][METHOD])==len(family['evaluator_only']['sessions'])==8
    rows,wires=[],[];cache={d:set() for d in depths};response_cache={}
    for slot,session in enumerate(group['streams'][METHOD]):
        raw=group['streams']['raw'][slot]['events']
        assert [e['timestamp_s'] for e in raw]==[e['timestamp_s'] for e in session['events']]
        assert raw[-1]['timestamp_s']==600 and all(len(e['candidates'])==1 for e in raw)
        end=raw[-1]['candidates'][0];destination=rn.nearest(end['lat'],end['lon'])[0]
        departure=family['evaluator_only']['sessions'][slot]['depart_s']
        for event,gps in zip(session['events'],raw):
            assert event['event_id']==gps['event_id']
            t=original_wire[slot,event['timestamp_s']]['t'];assert t==event['timestamp_s']
            point=gps['candidates'][0];state=rn.nearest(point['lat'],point['lon'])[0]
            positions=[(q['lat'],q['lon']) for q in event['candidates']];assert len(positions)==5
            before_pool=set();before_cache=set();before_scores={}
            for depth in depths:
                responses=[]
                for lat,lon in positions:
                    qstate=rn.nearest(lat,lon)[0];key=qstate,depth
                    if key not in response_cache:response_cache[key]=reply_at(full,qstate,depth)
                    responses.append(response_cache[key])
                current={i for ids,_ in responses for i in ids};cache[depth].update(current)
                assert before_pool<=current and before_cache<=cache[depth]
                before_pool,before_cache=current,cache[depth].copy()
                fields={k:truth[k] for k in ('family_id','split','draw')}
                fields.update(slot=slot,method=f'service_l{depth}',event_id=event['event_id'],t=t)
                requests=[dict(timestamp_s=departure+t,lat=lat,lon=lon,categories=list(full.categories),L=depth) for lat,lon in positions]
                wire=dict(fields,requests=5,request_bytes=sum(len(json.dumps(v,separators=(',',':')).encode()) for v in requests),
                    reply_bytes=sum(size for _,size in responses),reply_poi_ids_by_Q=[ids for ids,_ in responses])
                wires.append(wire)
                for policy,pool in (('current',current),('static_epoch_cache',cache[depth])):
                    scores=local_scores(local,state,destination,pool)
                    if policy in before_scores:monotonic(before_scores[policy],scores)
                    before_scores[policy]=scores
                    row=dict(fields,cache=policy,purposes=scores,available_count=len(pool));rows.append(row)
                    if depth==20:same(original_util[slot,t,policy],dict(row,method=METHOD))
                if depth==20:same(original_wire[slot,t],dict(wire,method=METHOD))
    tapes=[{k:v for k,v in s['ledger'][METHOD].items() if k!='step_ms'} for s in truth['sessions']]
    return dict(schema='qplanner-service-depth-replay-family-v1',evaluator_only={k:truth[k] for k in ('family_id','split','draw')},
        frozen_controls=dict(Q_stream_sha256=canonical_sha(group['streams'][METHOD]),ledger_and_anchors_sha256=canonical_sha(tapes),
                             Q_not_regenerated=True,private_reads_not_performed=True),utility=rows,wire=wires)


def independent_summary(rows,wires,splits,*,depths=DEPTHS):
    result={}
    for depth in depths:
        result[str(depth)]={}
        for split in splits:
            cells={};selected=[r for r in rows if r['method']==f'service_l{depth}' and r['split']==split]
            for policy in ('current','static_epoch_cache'):
                subset=[r for r in selected if r['cache']==policy]
                cells[policy]={'all':summarize(subset),'cold':summarize([r for r in subset if r['slot']==0]),
                    'temporal_tail_400_600':summarize([r for r in subset if r['t']>=400])}
            cost=[r for r in wires if r['method']==f'service_l{depth}' and r['split']==split]
            cells['cost']={k:sum(r[k] for r in cost) for k in ('requests','request_bytes','reply_bytes')}
            result[str(depth)][split]=cells
    return result


def independent_selection(summary):
    base=summary['20']['selection'];candidates=[];chosen=None
    for depth in (30,40,60):
        row=summary[str(depth)]['selection'];delta={}
        for p in ('equal_purpose_macro','nearest_distance'):
            a,b=base['current']['all'][p]['family_values'],row['current']['all'][p]['family_values']
            assert a and set(a)==set(b) and all(a[f] is not None and b[f] is not None for f in a)
            delta[p]=sum(b[f]-a[f] for f in sorted(a))/len(a)
        for p in PURPOSES:
            for key in ('defined_windows','total_windows','defined_categories','total_categories'):
                assert base['current']['all'][p][key]==row['current']['all'][p][key]
        assert base['cost']['reply_bytes']>0 and row['cost']['requests']==base['cost']['requests']
        ratio=row['cost']['reply_bytes']/base['cost']['reply_bytes']
        gates=dict(macro_gain_at_least_2pp=delta['equal_purpose_macro']>=.02-1e-12,
                   nearest_no_loss=delta['nearest_distance']>=-1e-12,reply_byte_ratio_at_most_2_5=ratio<=2.5+1e-12)
        eligible=all(gates.values())
        candidates.append(dict(depth=depth,differences=delta,reply_json_byte_ratio=ratio,gates=gates,eligible=eligible))
        if chosen is None and eligible:chosen=depth
    return dict(schema='qplanner-response-depth-selection-v1',selected_depth=chosen,criterion=CRITERION,candidates=candidates,
                fresh_confirmation='PENDING; not scored by this development replay')


def verify(out,*,validation_output=None):
    out=Path(out);p,source,old,generation=development_contract(out)
    assert sha(relative_path(ROOT,p['dataset_path']))==p['dataset_sha256']
    data=read(relative_path(ROOT,p['dataset_path']));network=relative_path(ROOT,data['network']['compressed_path'])
    assert sha(network)==data['network']['compressed_sha256']
    assert hashlib.sha256(gzip.decompress(network.read_bytes())).hexdigest()==data['network']['native_sha256']
    assert list(p['family_files_sha256'])==[j['name'] for j in expected_jobs(data,old)]
    rn=build_lane_states(network,spacing_m=40.)
    pois=[{k:v for k,v in poi.items() if k not in ('vertex','access_offset_m')} for poi in read(ROOT/'artifacts/benchmarks/research_loop/resources.json')['pois_used']]
    full=PublicPoiContext(LanePoiService(rn,pois,k=60));reply=PublicPoiContext(LanePoiService(rn,pois,k=20))
    reference=PublicPoiContext(LanePoiService(rn,pois,k=5));metadata=read(out/'resources.json');source_meta=read(source/'resources.json')
    assert full.pois==reply.pois==reference.pois and np.array_equal(full.access,reply.access)
    assert np.array_equal(full.signatures[:,:,:20],reply.signatures)
    assert metadata['full_reply60_sha256']==full.sha256 and metadata['frozen_reply20_sha256']==reply.sha256==source_meta['reply20_sha256']
    assert metadata['reference_sha256']==reference.sha256 and metadata['exact_l20_prefix_asserted'] is True
    assert metadata['native_sha256']==data['network']['native_sha256'] and metadata['catalogue']==catalogue_summary(rn)
    same({str(d):canonical_sha({'parent_sha256':full.sha256,'depth':d}) for d in DEPTHS},metadata['depth_context_sha256'])
    for name,pin in metadata['source_sha256'].items():assert old['public_inputs_sha256'][name]==pin
    copied=metadata['public_cache_copies_sha256']
    if copied:same(copied,read(source/'execution_protocol.json')['public_cache_files_sha256'])
    local=MultiPurposeRoadRanking(reference,cache_limit=1024)
    families={f['family_id']:f for f in data['families']};readout=read(out/'readout.json')
    assert readout['schema']=='qplanner-response-depth-readout-v1' and readout['protocol_sha256']==sha(out/'protocol.json')
    assert readout['resources_sha256']==sha(out/'resources.json') and readout['L20_source_rows_exactly_reproduced'] is True
    assert readout['static_event_recall_monotonicity_asserted'] is True and readout['no_private_generation'] is True
    assert read(out/'replay_started.json')['protocol_sha256']==sha(out/'protocol.json')
    assert list(readout['family_files_sha256'])==list(p['family_files_sha256'])
    assert set(x.name for x in (out/'families').iterdir())==set(p['family_files_sha256'])
    rows,wires=[],[]
    for name,pin in p['family_files_sha256'].items():
        original=read(source/'families'/name);record=read(out/'families'/name)
        assert sha(out/'families'/name)==readout['family_files_sha256'][name]
        expected=replay_expected(original,families[original['evaluator_only']['family_id']],rn,full,local)
        expected['source_bundle_sha256']=pin;same(expected,record)
        rows.extend(expected['utility']);wires.extend(expected['wire'])
        print('Independent response prefixes/cost/utility verified',name,flush=True)
    summary=independent_summary(rows,wires,p['splits']);same(summary,readout['summary'])
    selected=None
    if (out/'depth_selection.json').exists():
        value=independent_selection(summary);selected=value['selected_depth']
        value.update(protocol_sha256=sha(out/'protocol.json'),readout_sha256=sha(out/'readout.json'))
        same(value,read(out/'depth_selection.json'))
    record=dict(schema='qplanner-independent-response-depth-verification-v1',status='pass',
        protocol_sha256=sha(out/'protocol.json'),readout_sha256=sha(out/'readout.json'),verifier_sha256=sha(Path(__file__)),
        base_verifier_sha256=sha(ROOT/'experiments/verify_qplanner_study_20261006.py'),
        source_q_validation_sha256=sha(source/'validation.json'),bundles=len(p['family_files_sha256']),
        utility_windows=len(rows),wire_rows=len(wires),selected_depth=selected,
        exact_L20_utility_reply_cost_reproduced=True,all_category_ordered_prefixes_verified=True,
        current_and_causal_cache_monotonic_recall_completion_verified=True,
        fixed_Q_clock_anchors_ledger_inputs_unchanged=True,no_private_rng_key_required=True,
        independent_compact_JSON_cost_and_conditional_family_arithmetic=True,
        privacy_theorem_certified=False,timing_resource_savings_measured=False,
        scope='DEVELOPMENT-only fixed public response-depth/static-service utility-cost ablation; source Q ledger certificate reused; no new location mechanism or privacy superiority')
    target=Path(validation_output) if validation_output else out/'validation.json'
    if target.exists():same(record,read(target))
    else:target.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(record,flush=True);return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    parser.add_argument('--validation-output',type=Path);args=parser.parse_args()
    verify(args.out,validation_output=args.validation_output)
