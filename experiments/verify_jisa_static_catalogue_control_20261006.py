"""Read-only source, causal-pool, ranking and arithmetic verification.

Does not import the new application runner/helper. Existing local.top supplies
an independent path from the runner's score/sort implementation. Evidence is
only added as validation.json; original pilot files are never written.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from data.lane_states import build_lane_states
from evaluation.lane_travel import LanePoiService

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'artifacts/benchmarks/jisa_static_catalogue_control_20261006_v1'
DATA=ROOT/'artifacts/datasets/future_controlled_20261005_v2/dataset.json.gz'
REPLY=ROOT/'artifacts/benchmarks/future_native_depth_20261005_v1/public_reply40.npz'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    path=Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_text())
def encoded(value):return json.dumps(value,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def summary(rows,answer):
    groups=defaultdict(list)
    for row in rows:groups[row['family_id'],row['slot']].append(row['recall'][answer])
    sessions={key:statistics.mean(v for v in values if v is not None)
              if any(v is not None for v in values) else None for key,values in groups.items()}
    families={f:statistics.mean(v for (family,_),v in sessions.items() if family==f and v is not None)
              if any(family==f and v is not None for (family,_),v in sessions.items()) else None
              for f in sorted({r['family_id'] for r in rows})}
    defined=[v for v in families.values() if v is not None]; usable=[v for v in sessions.values() if v is not None]
    return dict(family_macro_recall5=statistics.mean(defined) if defined else None,family_values=families,
        represented_family_count=len(families),defined_family_count=len(defined),represented_session_count=len(sessions),
        undefined_session_count=sum(v is None for v in sessions.values()),
        minimum_defined_session_recall5=min(usable) if usable else None,
        reference_defined_queries=sum(bool(r['reference']) for r in rows),
        empty_reference_queries=sum(not r['reference'] for r in rows),total_queries=len(rows))


def equal(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:equal(a[key],b[key])
    elif isinstance(a,float):assert np.isclose(a,b,rtol=0.,atol=1e-12),(a,b)
    else:assert a==b,(a,b)


def verify(out):
    protocol,result=read(out/'protocol.json'),read(out/'results.json')
    assert sha(out/'protocol.json')==(out/'protocol.sha256').read_text().strip()==result['protocol_sha256']
    pilot=Path(protocol['pilot']);pilot=pilot if pilot.is_absolute() else ROOT/pilot
    for name,digest in protocol['pilot_sha256'].items():assert sha(pilot/name)==digest,name
    for name,digest in protocol['source_sha256'].items():
        assert sha(ROOT/name)==sha(out/'source_snapshot'/name)==digest,name
    for name,digest in result['file_sha256'].items():assert sha(out/name)==digest,name
    assert sha(DATA)==protocol['dataset_sha256'] and sha(REPLY)==protocol['reply40_file_sha256']
    data,public=read(DATA),read(pilot/'public_transcripts.json.gz')
    wire=read(pilot/'wire_rows.json.gz')['rows']; rows=read(out/'utility_rows.json.gz')['rows']
    costs=read(out/'cost_rows.json');bulk_request=read(out/'bulk_request.json');bulk_response=read(out/'bulk_response.json')
    rn=build_lane_states(ROOT/data['network']['compressed_path'],spacing_m=40.)
    pois=[{k:v for k,v in p.items() if k not in ('vertex','access_offset_m')} for p in
          read(ROOT/'artifacts/benchmarks/research_loop/resources.json')['pois_used']]
    context=PublicPoiContext(LanePoiService(rn,pois,k=40),REPLY)
    local=MultiPurposeRoadRanking(context,cache_limit=256)
    records=[{k:p[k] for k in ('id','category','lat','lon')} for p in local.pois]
    version=hashlib.sha256(encoded(records)).hexdigest()
    assert bulk_response=={'catalogue_version':version,'results':records}
    assert bulk_request=={'schema':'static_catalogue_prefetch_v1','catalogue_version':version,
                          'epoch_id':protocol['public_epoch']['epoch_id'],'timestamp_s':0.}
    assert len(records)==result['poi_count']==418 and version==result['catalogue_version']
    wire_lookup={(r['family_id'],r['slot'],r['method'],r['event_id']):r for r in wire}
    row_lookup={(r['family_id'],r['slot'],r['event_id'],r['purpose'],r['category']):r for r in rows}
    assert len(wire_lookup)==len(wire) and len(row_lookup)==len(rows)
    expected_costs=[];answer_checks=0; query_count=0
    for group,family in zip(public['groups'],data['families']):
        expected_costs.append(dict(family_id=family['family_id'],split=family['split'],method='full_catalogue',
            requests=1,request_bytes=len(encoded(bulk_request)),reply_bytes=len(encoded(bulk_response))))
        for method in protocol['methods']:
            source=[r for r in wire if r['family_id']==family['family_id'] and r['method']==method]
            expected_costs.append(dict(family_id=family['family_id'],split=family['split'],method=method,
                requests=sum(r['requests'] for r in source),request_bytes=sum(r['request_bytes'] for r in source),
                reply_bytes=sum(r['reply_bytes'] for r in source)))
        accumulated={m:set() for m in protocol['methods']}
        for slot,spec in enumerate(family['evaluator_only']['sessions']):
            trace={int(p['time_s']):p for p in data['traces'][spec['session_id']]}
            end=trace[600];destination=rn.nearest(end['lat'],end['lon'])[0]
            for event in group['streams']['raw'][slot]['events']:
                t=int(event['timestamp_s']);absolute=spec['depart_s']+t
                assert 0<=absolute<12000.
                pools={'full_catalogue':set(range(local.n))}
                for method in protocol['methods']:
                    reply=wire_lookup[family['family_id'],slot,method,event['event_id']]
                    assert reply['absolute_public_time_s']==absolute
                    current={i for ids in reply['reply_poi_indices_by_Q'] for i in ids}
                    assert all(0<=i<local.n for i in current)
                    accumulated[method].update(current)
                    pools[method+'_current_only'],pools[method+'_epoch_cache']=current,accumulated[method]
                point=trace[t];state=rn.nearest(point['lat'],point['lon'])[0]
                for purpose in QueryPurpose:
                    for category in local.categories:
                        row=row_lookup[family['family_id'],slot,event['event_id'],purpose.value,category]
                        assert row['local_state_evaluator_only']==state and row['local_destination_evaluator_only']==destination
                        assert row['split']==family['split'] and row['session_id']==spec['session_id']
                        assert row['timestamp_s']==t and row['absolute_public_time_s']==absolute
                        q=QuerySpec(purpose,category,k=5,
                            radius_m=1000. if purpose==QueryPurpose.WITHIN_RADIUS else None,
                            destination_state=destination if purpose==QueryPurpose.MIN_DETOUR else None)
                        reference=local.top(state,np.ones(local.n,dtype=bool),q)
                        assert reference==row['reference']
                        for name,ids in pools.items():
                            mask=np.zeros(local.n,dtype=bool);mask[list(ids)]=True
                            answer=local.top(state,mask,q)
                            assert answer==row['returned'][name]
                            assert row['recall'][name]==(len(set(reference)&set(answer))/len(reference) if reference else None)
                            if name=='full_catalogue':assert answer==reference
                            answer_checks+=1
                        query_count+=1
        print('Independent static control family verified',family['family_id'],flush=True)
    assert query_count==len(rows)==result['local_query_count'] and expected_costs==costs
    for split,answers in result['utility'].items():
        subset=[r for r in rows if r['split']==split]
        for name,purposes in answers.items():
            for purpose,entry in purposes.items():equal(summary([r for r in subset if r['purpose']==purpose],name),entry)
        for method,entry in result['cost'][split].items():
            scoped=[r for r in costs if r['split']==split and r['method']==method]
            expected={key:sum(r[key] for r in scoped) for key in ('requests','request_bytes','reply_bytes')}
            expected['total_bytes']=expected['request_bytes']+expected['reply_bytes'];equal(expected,entry)
    record={'schema':'jisa-static-catalogue-independent-verification-v1','status':'pass',
        'verifier_sha256':sha(Path(__file__)),'results_sha256':sha(out/'results.json'),
        'queries_checked':query_count,'exact_answer_rankings_checked':answer_checks,
        'source_pilot_unchanged':True,'received_only_current_and_causal_epoch_union':True,
        'full_catalogue_request_coordinate_free':True,'bulk_payload_bytes_recomputed':True,
        'raw_destination_only_in_local_utility':True,'no_new_protection_draws':True,
        'timing_measured':False,'real_provider_bulk_access_verified':False,'dynamic_freshness_verified':False}
    target=out/'validation.json'
    if target.exists():assert read(target)==record
    else:target.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(record,flush=True)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=DEFAULT)
    verify(parser.parse_args().out)
