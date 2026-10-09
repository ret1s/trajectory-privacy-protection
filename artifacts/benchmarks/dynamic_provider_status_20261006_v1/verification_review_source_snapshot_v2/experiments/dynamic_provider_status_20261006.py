"""Controlled dynamic-status sensitivity on immutable Geo-I public Q.

Top-L STATIC nearest-category records carry current public-epoch availability
bits. They are not top-L available results. Status worlds are synthetic, pinned
before scoring, and never enter client cache/ranking through an oracle API.
This already-inspected same-map cohort is a secondary diagnostic, not a fresh
confirmation or a new privacy mechanism. A permitted current full-catalogue
bulk API is an explicit dominating control, not hidden from the comparison.
"""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np

from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from data.lane_states import build_lane_states, catalogue_summary
from evaluation.lane_travel import LanePoiService

ROOT=Path(__file__).resolve().parents[1]
THIS='experiments/dynamic_provider_status_20261006.py'
TEST='tests/test_dynamic_provider_status.py'
SOURCE='artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1'
OUT='artifacts/benchmarks/dynamic_provider_status_20261006_v1'
PUBLIC='artifacts/benchmarks/research_loop/resources.json'
DEFAULT_CACHE=Path('/private/tmp/qplanner-response-depth-generalization-20261006-v1/public_resources/reply60.npz')
SEED=2026100623
EPOCH_SECONDS=60
DEPARTURES=tuple(float(1500*s) for s in range(8))
PURPOSES=tuple(p.value for p in QueryPurpose)
ARMS=('geoi_l20_current','geoi_l20_epoch60','geoi_l30_current','geoi_l30_epoch60',
      'raw_l20_current','raw_l30_current','static_catalogue_stale_safe',
      'current_catalogue_bulk_oracle','static_catalogue_stale_unsafe')


def read(path):
    path=Path(path);raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def size(value):return len(json.dumps(value,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':raw=gzip.compress(raw,mtime=0)
    with path.open('xb') as stream:stream.write(raw)


def epoch_at(timestamp):
    if not math.isfinite(timestamp) or timestamp<0:raise ValueError('Nonnegative finite absolute public clock required')
    return int(timestamp//EPOCH_SECONDS)


def status_world(ids,epochs,seed=SEED):
    """Evaluator/provider-only public-ID hash; no coordinates or targets input."""
    if len(set(ids))!=len(ids) or not ids:raise ValueError('Distinct public POI IDs required')
    snapshots={}
    for epoch in sorted(set(epochs)):
        if isinstance(epoch,bool) or int(epoch)!=epoch or epoch<0:raise ValueError('Integer public epochs required')
        snapshots[str(epoch)]=[int.from_bytes(hashlib.sha256(json.dumps(
            ['synthetic-availability-v1',int(seed),str(pid),int(epoch)],separators=(',',':')).encode()).digest()[:8],'big')<2**63 for pid in ids]
    return dict(schema='public-id-hash-availability-world-v1',seed=seed,probability=.5,
        epoch_seconds=EPOCH_SECONDS,poi_ids=list(ids),snapshots=snapshots,
        client_world_or_seed_access=False,status_generated_without_GPS_targets_or_scores=True)


class CurrentStatusCache:
    """Received static IDs persist; dynamic bits expire at PUBLIC epoch boundary."""
    def __init__(self,count):
        self.count=count;self.static=np.zeros(count,bool);self.known=np.zeros(count,bool)
        self.available=np.zeros(count,bool);self.epoch=None;self.last_t=None

    def receive(self,timestamp,ids,bits):
        if self.last_t is not None and timestamp<=self.last_t:raise ValueError('Strictly increasing absolute public clock required')
        epoch=epoch_at(timestamp)
        ids=list(ids);bits=list(bits)
        if len(ids)!=len(bits) or any(type(v) is not bool for v in bits):raise ValueError('Every received ID needs a current boolean bit')
        incoming={}
        for index,bit in zip(ids,bits):
            if isinstance(index,bool) or int(index)!=index or not 0<=index<self.count:raise ValueError('Public POI index out of range')
            if index in incoming and incoming[index]!=bit:raise ValueError('Conflicting duplicate status bits')
            if epoch==self.epoch and self.known[index] and self.available[index]!=bit:raise ValueError('Conflicting statuses within fixed public epoch')
            incoming[index]=bit
        # Validate the entire reply before an invalid payload can clear a
        # previous epoch or partially add records.
        if epoch!=self.epoch:self.known[:]=False;self.available[:]=False;self.epoch=epoch
        current=np.zeros(self.count,bool);current_available=np.zeros(self.count,bool)
        for index,bit in incoming.items():
            self.static[index]=self.known[index]=current[index]=True
            self.available[index]=current_available[index]=bit
        self.last_t=timestamp
        return (current,current.copy(),current_available),(self.static.copy(),self.known.copy(),self.available.copy())


def requests(timestamp,positions,categories,depth):
    """No private purpose, radius, category choice or destination parameter."""
    return [dict(timestamp_s=timestamp,lat=float(lat),lon=float(lon),categories=list(categories),
        L=depth,status_epoch=epoch_at(timestamp),include_availability=True) for lat,lon in positions]


def score_cases(cases,truth,static,known,available,*,unsafe_candidates=None):
    """Reference=top5 currently available; unknown never means unavailable."""
    truth,static,known,available=[np.asarray(x,dtype=bool) for x in (truth,static,known,available)]
    if any(x.ndim!=1 for x in (truth,static,known,available)) or len({x.shape for x in (truth,static,known,available)})!=1:raise ValueError('Full masks must share public catalogue shape')
    safe=unsafe_candidates is None
    candidate=static&known&available if safe else np.asarray(unsafe_candidates,bool)
    if candidate.shape!=truth.shape:raise ValueError('Unsafe-control mask must retain public catalogue shape')
    if safe and np.any(available&known&~truth):raise ValueError('Current known-available bit contradicts provider truth')
    result={}
    for purpose,entries in cases.items():
        recalls=[];completions=[];counts=dict(reference_poi_total=0,overlap_total=0,empty_reference_categories=0,
            retrieval_miss_reference_pois=0,current_status_unknown_reference_pois=0,known_available_reference_pois=0,
            returned_items=0,returned_unavailable_items=0,returned_current_status_unknown_items=0,
            reference_exists_zero_answer_categories=0,current_known_unavailable_candidates=int(np.sum(static&known&~available)),
            current_status_unknown_candidates=int(np.sum(static&~known)))
        for ordered in entries:
            ordered=np.asarray(ordered,int);ref=ordered[truth[ordered]][:5]
            answer=ordered[candidate[ordered]][:5]
            overlap=len(set(map(int,answer))&set(map(int,ref)))
            counts['returned_items']+=len(answer);counts['returned_unavailable_items']+=int(np.sum(~truth[answer]))
            counts['returned_current_status_unknown_items']+=int(np.sum(~known[answer]))
            if not len(ref):counts['empty_reference_categories']+=1;continue
            recalls.append(overlap/len(ref));completions.append(min(int(np.sum(truth[answer])),len(ref))/len(ref))
            counts['reference_poi_total']+=len(ref);counts['overlap_total']+=overlap
            counts['retrieval_miss_reference_pois']+=int(np.sum(~static[ref]))
            counts['current_status_unknown_reference_pois']+=int(np.sum(static[ref]&~known[ref]))
            counts['known_available_reference_pois']+=int(np.sum(static[ref]&known[ref]&available[ref]))
            counts['reference_exists_zero_answer_categories']+=not len(answer)
        if safe:
            assert not counts['returned_unavailable_items'] and not counts['returned_current_status_unknown_items']
            assert counts['reference_poi_total']==sum(counts[k] for k in ('retrieval_miss_reference_pois','current_status_unknown_reference_pois','known_available_reference_pois'))
        result[purpose]=dict(recall5=float(np.mean(recalls)) if recalls else None,
            completion=float(np.mean(completions)) if completions else None,
            reference_category_count=len(recalls),all_category_count=len(entries),**counts)
    return result


class DynamicLocalEvaluator:
    def __init__(self,ranking):self.ranking=ranking;self.refs={}

    def references(self,state,destination):
        key=int(state),int(destination)
        if key not in self.refs:
            result={}
            for purpose in QueryPurpose:
                entries=[]
                for category in self.ranking.categories:
                    query=QuerySpec(purpose,category,k=5,radius_m=1000. if purpose==QueryPurpose.WITHIN_RADIUS else None,
                        destination_state=destination if purpose==QueryPurpose.MIN_DETOUR else None)
                    costs=self.ranking.scores(state,query)
                    ids=[i for i,p in enumerate(self.ranking.pois) if p['category']==category and np.isfinite(costs[i])]
                    ids.sort(key=lambda i:(float(costs[i]),self.ranking.pois[i]['id']))
                    entries.append(np.asarray(ids,int))
                result[purpose.value]=entries
            self.refs[key]=result
        return self.refs[key]


def source_closure():
    pending=[THIS];found=set()
    def add(module):
        for path in (module.replace('.','/')+'.py',module.replace('.','/')+'/__init__.py'):
            if (ROOT/path).is_file() and path not in found:pending.append(path)
    while pending:
        name=pending.pop()
        if name in found:continue
        found.add(name);package=name.removesuffix('.py').split('/')[:-1]
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            if isinstance(node,ast.Import):
                for alias in node.names:add(alias.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=package[:len(package)-node.level+1] if node.level else []
                module='.'.join(prefix+([node.module] if node.module else []))
                if module:add(module)
                for alias in node.names:
                    if alias.name!='*':add('.'.join(filter(None,(module,alias.name))))
    return sorted(found|{TEST,'requirements.txt','requirements-sumo.txt'})


def cache_context_contract(cache,expected_context,expected_catalogue):
    """Authenticate the public NPZ semantics BEFORE IDs define the workload."""
    with np.load(cache,allow_pickle=False) as arrays:
        encoded=str(arrays['metadata']);metadata=json.loads(encoded)
        signatures,access=arrays['signatures'],arrays['access']
        assert metadata['schema']=='public-poi-context-v1' and metadata['k']==60
        assert metadata['catalogue_sha256']==expected_catalogue
        assert signatures.shape==(len(access),len(metadata['categories']),60) and access.ndim==1
        assert np.all((access>=0)&(access<len(access)))
        assert np.all((signatures>=-1)&(signatures<len(metadata['pois'])))
        digest=hashlib.sha256(encoded.encode())
        digest.update(signatures.astype('<i4').tobytes());digest.update(access.astype('<i4').tobytes())
        assert digest.hexdigest()==expected_context
    return metadata


def declare(output,cache=DEFAULT_CACHE):
    output=Path(output).resolve();source=ROOT/SOURCE
    if not output.is_relative_to(ROOT):raise ValueError('Artifact must stay inside repository')
    if output.exists() and any(output.iterdir()):raise FileExistsError('New write-once output required')
    p=read(source/'protocol.json');cert=read(source/'validation.json');saved=read(source/'readout.json')
    if cert['status']!='pass' or cert['protocol_sha256']!=sha(source/'protocol.json') or cert['readout_sha256']!=sha(source/'readout.json'):
        raise ValueError('Complete frozen source validation required')
    assert p['depths']==[20,30] and p['selected_depth']==30 and saved['no_private_generation'] is True
    base=ROOT/p['base_q_output'];generation=read(base/'generation.json')
    assert saved['base_generation_sha256']==sha(base/'generation.json') and saved['base_validation_sha256']==sha(base/'validation.json')
    assert sha(ROOT/p['dataset_path'])==p['dataset_sha256']
    data=read(ROOT/p['dataset_path']);test=sorted(f['family_id'] for f in data['families'] if f['split']=='test')
    assert len(test)==24
    assert all(tuple(f['evaluator_only']['sessions'][s]['depart_s'] for s in range(8))==DEPARTURES for f in data['families'] if f['family_id'] in test)
    names=[f'{f}--draw{d}.json.gz' for f in test for d in (1,2,3)]
    epochs={0};controls={};inventory={}
    for name in names:
        assert sha(source/'families'/name)==saved['family_files_sha256'][name]
        assert sha(base/'families'/name)==generation['family_files_sha256'][name]
        b=read(base/'families'/name);depth=read(source/'families'/name)
        assert b['evaluator_only']['split']=='test' and depth['source_bundle_sha256']==generation['family_files_sha256'][name]
        controls[name]=depth['frozen_controls'];inventory[name]=[]
        for slot,session in enumerate(b['public']['streams']['legacy_l10']):
            for event in session['events']:
                absolute=DEPARTURES[slot]+event['timestamp_s'];epochs.add(epoch_at(absolute))
                inventory[name].append([slot,event['timestamp_s'],event['event_id'],absolute])
    source_resource=read(source/'resources.json')
    metadata=cache_context_contract(cache,p['service_kernel']['full_reply60_sha256'],source_resource['catalogue']['sha256'])
    ids=[poi['id'] for poi in metadata['pois']];assert len(ids)==418 and ids==sorted(ids)
    world=status_world(ids,epochs)
    plan=dict(schema='frozen-Q-dynamic-provider-status-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source_output=SOURCE,base_Q_output=p['base_q_output'],dataset_path=p['dataset_path'],dataset_sha256=p['dataset_sha256'],
        source_files_sha256={n:sha(source/n) for n in ('protocol.json','readout.json','resources.json','validation.json')},
        base_files_sha256={n:sha(base/n) for n in ('protocol.json','generation.json','resources.json','validation.json')},
        source_family_files_sha256={n:saved['family_files_sha256'][n] for n in names},
        base_family_files_sha256={n:generation['family_files_sha256'][n] for n in names},
        frozen_controls=controls,public_event_inventory=inventory,source_sha256={n:sha(ROOT/n) for n in source_closure()},
        public_resource_sha256=sha(ROOT/PUBLIC),public_reply60_sha256=p['service_kernel']['full_reply60_sha256'],
        reply_cache_sha256=sha(cache),public_catalogue=metadata['pois'],status_world_sha256=canonical(world),
        workload_seed=SEED,availability_probability=.5,status_epoch_seconds=60,public_departures_s=DEPARTURES,
        included_blocks=names,families=test,draws=[1,2,3],split='test',depths=[20,30],arms=ARMS,
        provider_contract='Top-L STATIC nearest records per public category + CURRENT epoch availability bit; no available-first filtering; all-purpose fixed requests',
        stale_snapshot_epoch=0,cache_policy='static IDs persist within linked eight-session scope; received status bits expire at absolute public60s boundary, no rolling TTL',
        utility='top5 currently available reference; all4 purposes/6 categories; conditional nonempty categories; explicit all-event/category totals; equal-family/equal-nested-draw',
        cost='Compact synthetic JSON application payload counts; static metadata and current-status upload/download separately; no HTTP/TLS/real bandwidth/latency',
        status_only_bulk='One static catalogue prefetch plus one all-ID status download per observed public epoch, no coordinate upload; explicit stronger API control',
        scope='SECONDARY already-inspected same-map synthetic workload sensitivity; no retuningL, new Q, Geo-I theorem, independent confirmation or SOTA/privacy victory',
        local_GPS_destination='Evaluator-only reuse of saved raw truth; no additional GPS/protection queries sampled',
        unsafe_control='static_catalogue_stale_unsafe deliberately uses expired epoch0 bits; INVALID operational baseline, measured false availability/unknown separately')
    plan['workload_interface_assumption']='Client API receives current bits only for requested records; evaluator seed/fullworld are excluded from that interface, but public in the research artifact for replay. A client given the fixture can compute all synthetic statuses locally. This tests freshness/dataflow under assumed provider APIs, not remote necessity or unpredictable real status.'
    output.mkdir(parents=True);save(output/'protocol.json',plan);save(output/'status_world.json.gz',world)
    (output/'protocol.sha256').write_text(sha(output/'protocol.json')+'\n')
    shutil.copyfile(cache,output/'public_reply60.npz')
    for name in plan['source_sha256']:
        target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    return plan


def validate(output):
    output=Path(output);p=read(output/'protocol.json')
    assert sha(output/'protocol.json')==(output/'protocol.sha256').read_text().strip()
    assert p['workload_seed']==SEED and p['availability_probability']==.5 and p['depths']==[20,30] and tuple(p['arms'])==ARMS
    for name,pin in p['source_sha256'].items():assert sha(ROOT/name)==sha(output/'source_snapshot'/name)==pin
    for folder,field in ((ROOT/p['source_output'],'source_files_sha256'),(ROOT/p['base_Q_output'],'base_files_sha256')):
        for name,pin in p[field].items():assert sha(folder/name)==pin
    assert sha(ROOT/p['dataset_path'])==p['dataset_sha256'] and sha(ROOT/PUBLIC)==p['public_resource_sha256']
    assert sha(output/'public_reply60.npz')==p['reply_cache_sha256']
    world=read(output/'status_world.json.gz');assert canonical(world)==p['status_world_sha256']
    assert world==status_world(world['poi_ids'],map(int,world['snapshots']),SEED)
    return p,world


def response_cost(ids,context,epoch,bits):
    records=[{k:context.pois[i][k] for k in ('id','category','lat','lon')} for i in ids]
    status=[dict(id=context.pois[i]['id'],available=bool(bits[i])) for i in ids]
    return size({'results':records}),size({'epoch':epoch,'status':status})


def replay_block(bundle,depth_bundle,context,evaluator,world):
    truth=bundle['evaluator_only'];streams=bundle['public']['streams'];n=len(context.pois)
    caches={d:CurrentStatusCache(n) for d in (20,30)};wire_index={(r['method'],r['slot'],r['t']):r for r in depth_bundle['wire']}
    full=np.ones(n,bool);empty=np.zeros(n,bool);stale=np.asarray(world['snapshots']['0'],bool)
    rows=[];cost={arm:dict(requests=0,static_request_bytes=0,status_upload_bytes=0,static_reply_bytes=0,status_download_bytes=0) for arm in ARMS}
    bulk_seen=set();cache_max={d:0 for d in (20,30)}
    for slot,session in enumerate(streams['legacy_l10']):
        raw=streams['raw'][slot]['events'];assert [e['timestamp_s'] for e in raw]==[e['timestamp_s'] for e in session['events']]
        endpoint=raw[-1]['candidates'][0];destination=context.rn.nearest(endpoint['lat'],endpoint['lon'])[0]
        for event,gps in zip(session['events'],raw):
            t=wire_index['service_l20',slot,event['timestamp_s']]['t'];absolute=DEPARTURES[slot]+t;epoch=epoch_at(absolute)
            live=np.asarray(world['snapshots'][str(epoch)],bool);point=gps['candidates'][0];state=context.rn.nearest(point['lat'],point['lon'])[0]
            cases=evaluator.references(state,destination);configs={}
            positions=[(q['lat'],q['lon']) for q in event['candidates']];assert len(positions)==5
            for depth in (20,30):
                old=wire_index[f'service_l{depth}',slot,t];replies=[]
                for lat,lon in positions:
                    ids=context.query_indices(context.rn.nearest(lat,lon)[0])[:,:depth].ravel();replies.append(list(map(int,ids[ids>=0])))
                assert replies==old['reply_poi_ids_by_Q']
                ids=[i for reply in replies for i in reply];current,cached=caches[depth].receive(absolute,ids,[bool(live[i]) for i in ids])
                cache_max[depth]=max(cache_max[depth],int(np.sum(cached[0])))
                for policy,masks in (('current',current),('epoch60',cached)):
                    arm=f'geoi_l{depth}_{policy}';configs[arm]=(*masks,None)
                    payloads=requests(absolute,positions,context.categories,depth)
                    original=[{k:v for k,v in r.items() if k not in ('status_epoch','include_availability')} for r in payloads]
                    assert sum(size(p) for p in original)==old['request_bytes']
                    static_bytes=sum(response_cost(reply,context,epoch,live)[0] for reply in replies);assert static_bytes==old['reply_bytes']
                    cost[arm]['requests']+=5;cost[arm]['static_request_bytes']+=old['request_bytes']
                    cost[arm]['status_upload_bytes']+=sum(size(p) for p in payloads)-old['request_bytes']
                    cost[arm]['static_reply_bytes']+=static_bytes
                    cost[arm]['status_download_bytes']+=sum(response_cost(reply,context,epoch,live)[1] for reply in replies)
                raw_ids=context.query_indices(state)[:,:depth].ravel();raw_ids=list(map(int,raw_ids[raw_ids>=0]));mask=np.zeros(n,bool);mask[raw_ids]=True
                arm=f'raw_l{depth}_current';configs[arm]=(mask,mask.copy(),mask&live,None)
                payload=requests(absolute,[(point['lat'],point['lon'])],context.categories,depth)[0]
                original={k:v for k,v in payload.items() if k not in ('status_epoch','include_availability')}
                static_bytes,status_bytes=response_cost(raw_ids,context,epoch,live)
                cost[arm]['requests']+=1;cost[arm]['static_request_bytes']+=size(original);cost[arm]['status_upload_bytes']+=size(payload)-size(original)
                cost[arm]['static_reply_bytes']+=static_bytes;cost[arm]['status_download_bytes']+=status_bytes
            configs['static_catalogue_stale_safe']=(full,full if epoch==0 else empty,stale if epoch==0 else empty,None)
            configs['current_catalogue_bulk_oracle']=(full,full,live,None)
            configs['static_catalogue_stale_unsafe']=(full,full if epoch==0 else empty,stale,stale)
            if not bulk_seen:
                for arm in ('static_catalogue_stale_safe','static_catalogue_stale_unsafe','current_catalogue_bulk_oracle'):
                    static_bytes,status_bytes=response_cost(range(n),context,0,stale)
                    cost[arm]['requests']+=1;cost[arm]['static_request_bytes']+=size({'schema':'full_public_catalogue_v1'})
                    cost[arm]['static_reply_bytes']+=static_bytes
                    # Oracle metadata prefetch carries no separate epoch0
                    # status download; its once-per-epoch loop supplies that.
                    if arm!='current_catalogue_bulk_oracle':cost[arm]['status_download_bytes']+=status_bytes
            if epoch not in bulk_seen:
                arm='current_catalogue_bulk_oracle';cost[arm]['requests']+=1
                cost[arm]['status_upload_bytes']+=size({'schema':'full_catalogue_status_v1','epoch':epoch})
                cost[arm]['status_download_bytes']+=response_cost(range(n),context,epoch,live)[1]
                bulk_seen.add(epoch)
            refs=None
            for arm,(static,known,available,unsafe) in configs.items():
                scores=score_cases(cases,live,static,known,available,unsafe_candidates=unsafe)
                coverage={p:(v['reference_category_count'],v['reference_poi_total']) for p,v in scores.items()}
                if refs is None:refs=coverage
                else:assert coverage==refs
                if arm=='current_catalogue_bulk_oracle':assert all(v['recall5'] is None or v['recall5']==1. for v in scores.values())
                rows.append(dict(family_id=truth['family_id'],draw=truth['draw'],split='test',arm=arm,slot=slot,t=t,absolute_t=absolute,
                    epoch=epoch,operationally_valid=arm!='static_catalogue_stale_unsafe',purposes=scores))
    return dict(schema='dynamic-status-frozen-Q-block-v1',family_id=truth['family_id'],draw=truth['draw'],split='test',
        rows=rows,cost=cost,static_cache_max_records={str(k):v for k,v in cache_max.items()},
        frozen_controls=depth_bundle['frozen_controls'],private_reads_or_Q_regeneration=False)


def summarize(blocks):
    cells=defaultdict(list);costs=defaultdict(lambda:defaultdict(int));max_cache=[]
    for block in blocks:
        for row in block['rows']:
            phases=['all']+(['cold'] if row['slot']==0 else [])+(['temporal_tail_400_600'] if row['t']>=400 else [])
            for phase in phases:
                for purpose,value in row['purposes'].items():cells[row['arm'],phase,purpose,row['family_id'],row['draw']].append(value)
        for arm,value in block['cost'].items():
            for field,count in value.items():costs[arm][field]+=count
        max_cache.append(block['static_cache_max_records'])
    summary={};local=[]
    for arm in ARMS:
        summary[arm]={}
        for phase in ('all','cold','temporal_tail_400_600'):
            result={}
            for purpose in PURPOSES:
                groups={(f,d):v for (a,p,q,f,d),v in cells.items() if (a,p,q)==(arm,phase,purpose)}
                family=defaultdict(list);per_draw=defaultdict(list);denominators=defaultdict(int)
                for (f,d),values in sorted(groups.items()):
                    valid=[v['recall5'] for v in values if v['recall5'] is not None]
                    mean=float(np.mean(valid)) if valid else None;family[f].append(mean);per_draw[d].append(mean)
                    local.append(dict(arm=arm,phase=phase,purpose=purpose,family_id=f,draw=d,recall5=mean,
                        total_events=len(values),defined_events=len(valid)))
                    denominators['total_events']+=len(values);denominators['defined_events']+=len(valid)
                    for v in values:
                        for field,count in v.items():
                            if field not in ('recall5','completion'):denominators[field]+=count
                means={f:float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else None for f,values in family.items()}
                defined=[v for v in means.values() if v is not None]
                result[purpose]=dict(family_mean=float(np.mean(defined)) if defined else None,family_values=means,
                    within_draw_family_mean={str(d):float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else None for d,values in per_draw.items()},
                    explicit_denominators=dict(denominators))
            family_ids=result[PURPOSES[0]]['family_values'];macro={f:float(np.mean([result[p]['family_values'][f] for p in PURPOSES if result[p]['family_values'][f] is not None])) for f in family_ids if any(result[p]['family_values'][f] is not None for p in PURPOSES)}
            result['equal_purpose_macro']=dict(family_mean=float(np.mean(list(macro.values()))) if macro else None,family_values=macro,
                minimum_family_mean=min(macro.values()) if macro else None,
                within_draw_family_mean={str(d):float(np.mean([np.mean([v['recall5'] for v in local if (v['arm'],v['phase'],v['family_id'],v['draw'])==(arm,phase,f,d) and v['recall5'] is not None]) for f in macro])) for d in (1,2,3)})
            summary[arm][phase]=result
    return dict(summary=summary,family_draw_purpose_values=local,cost={a:dict(v) for a,v in costs.items()},
        max_static_cache_records={str(d):max(int(v[str(d)]) for v in max_cache) for d in (20,30)})


def replay(output):
    output=Path(output).resolve();p,world=validate(output)
    if (output/'replay_started.json').exists():raise FileExistsError('Preserve existing replay attempt')
    save(output/'replay_started.json',dict(protocol_sha256=sha(output/'protocol.json'),started_utc=datetime.now(timezone.utc).isoformat()))
    data=read(ROOT/p['dataset_path']);network=ROOT/data['network']['compressed_path']
    assert sha(network)==data['network']['compressed_sha256'] and hashlib.sha256(gzip.decompress(network.read_bytes())).hexdigest()==data['network']['native_sha256']
    rn=build_lane_states(network,spacing_m=40.)
    public_pois=[{k:v for k,v in poi.items() if k not in ('vertex','access_offset_m')} for poi in read(ROOT/PUBLIC)['pois_used']]
    context=PublicPoiContext(LanePoiService(rn,public_pois,k=60),output/'public_reply60.npz')
    assert context.sha256==p['public_reply60_sha256'] and list(context.pois)==p['public_catalogue'] and world['poi_ids']==[p['id'] for p in context.pois]
    evaluator=DynamicLocalEvaluator(MultiPurposeRoadRanking(context,cache_limit=256));blocks=[];pins={}
    for name in p['included_blocks']:
        source=ROOT/p['source_output']/'families'/name;base=ROOT/p['base_Q_output']/'families'/name
        assert sha(source)==p['source_family_files_sha256'][name] and sha(base)==p['base_family_files_sha256'][name]
        depth=read(source);assert depth['frozen_controls']==p['frozen_controls'][name]
        b=read(base);assert canonical(b['public']['streams']['legacy_l10'])==depth['frozen_controls']['Q_stream_sha256']
        value=replay_block(b,depth,context,evaluator,world)
        value['source_bundle_sha256']=sha(base);value['source_depth_reply_bundle_sha256']=sha(source)
        save(output/'blocks'/name,value);pins[name]=sha(output/'blocks'/name);blocks.append(value)
        print('Dynamic frozen-Q block',name,flush=True)
    result=summarize(blocks);save(output/'readout.json',dict(schema='dynamic-status-workload-readout-v1',
        protocol_sha256=sha(output/'protocol.json'),status_world_file_sha256=sha(output/'status_world.json.gz'),
        block_files_sha256=pins,protected_Q_clocks_ledger_unchanged=True,no_private_key_GPS_queries_or_sampler_calls=True,
        source_already_inspected_secondary_only=True,workload_interface_assumption=p['workload_interface_assumption'],**result))
    save(output/'resources.json',dict(native_sha256=data['network']['native_sha256'],catalogue=catalogue_summary(rn),
        public_reply60_sha256=context.sha256,poi_count=len(context.pois),status_epochs=len(world['snapshots'])))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=('declare','replay','contract'))
    parser.add_argument('--output',type=Path,default=ROOT/OUT);parser.add_argument('--public-cache',type=Path,default=DEFAULT_CACHE)
    args=parser.parse_args()
    if args.operation=='declare':declare(args.output,args.public_cache);print('Dynamic public workload/source frozen',args.output)
    elif args.operation=='contract':validate(args.output);print('Dynamic pinned contract PASS',args.output)
    else:replay(args.output)
