"""Predeclared paired retrieval test on existing frozen Geo-I Q, not fresh data.

No location model, sampler, dataset or sealed result is changed. Candidate
selection uses SELECTION only and is sealed before the existing TEST replay.
All candidates remain reported, including failures. Destination/radius on the
wire are public fixed prototypes; private endpoint is evaluator-local only.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import gzip,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.sparse.csgraph import dijkstra
from benchmark.multi_purpose_retrieval import PublicRetrievalPlan,PublicPurposePoiService
from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking
from evaluation.lane_travel import LanePoiService,matrix
from experiments import qplanner_study_20261006_v2 as common
from experiments.qplanner_paired_readout_20261006 import paired

ROOT=common.ROOT
BASE=ROOT/'artifacts/benchmarks/qplanner_depth_base_q_generalization_20261006_v1'
DEPTH=ROOT/'artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1'
OUT=ROOT/'artifacts/benchmarks/multi_purpose_retrieval_20261009_v2'
WORK=Path('/private/tmp/multi-purpose-retrieval-20261009-v1')
PUBLIC_CACHE=Path('/private/tmp/qplanner-response-depth-generalization-20261006-v1')
CHANNELS={
    **{f'nearest_l{l}':(('nearest_distance',l),) for l in (10,20,30,40)},
    'nearest15_fastest15':(('nearest_distance',15),('fastest_travel',15)),
    'four_types_l10':(('nearest_distance',10),('fastest_travel',10),('within_radius',10),('public_detour_bank',10)),
    'nearest20_fastest10_detour10':(('nearest_distance',20),('fastest_travel',10),('public_detour_bank',10)),
}
CANDIDATES=('nearest15_fastest15','four_types_l10','nearest20_fastest10_detour10')
MATCHED={'nearest15_fastest15':'nearest_l30','four_types_l10':'nearest_l40','nearest20_fastest10_detour10':'nearest_l40'}
GATES=dict(min_macro_gain=.01,min_each_purpose_gain=-.005,max_total_reply_ratio_to_l30=1.25,
    min_matched_ceiling_macro_gain=0.,test_paired95_lower_gt=0.,all_test_draw_gains_gt=0.)


def read(path):return common.read(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def json_size(value):return len(json.dumps(value,separators=(',',':'),ensure_ascii=False).encode())


def declare():
    if OUT.exists():raise FileExistsError('Evidence is write-once; use another version after a source/config change')
    base=read(BASE/'protocol.json');generation=read(BASE/'generation.json')
    for name,pin in base['source_sha256'].items():assert sha(ROOT/name)==pin,name
    for name,pin in generation['family_files_sha256'].items():assert sha(BASE/'families'/name)==pin,name
    paths=['benchmark/multi_purpose_retrieval.py','experiments/multi_purpose_retrieval_20261009.py',
           'tests/test_multi_purpose_retrieval.py']
    source_pins=dict(base['source_sha256'])|{n:sha(ROOT/n) for n in paths}
    p=dict(schema='fixed-multipurpose-retrieval-hypothesis-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source_output=str(BASE.relative_to(ROOT)),dataset_path=base['dataset_path'],dataset_sha256=base['dataset_sha256'],
        source_protocol_sha256=sha(BASE/'protocol.json'),source_generation_sha256=sha(BASE/'generation.json'),
        source_sha256=source_pins,source_family_files_sha256=generation['family_files_sha256'],
        existing_depth_protocol_sha256=sha(DEPTH/'protocol.json'),channels=CHANNELS,matched_ceiling=MATCHED,
        gates=GATES,splits=base['splits'],draws_by_split=base['draws_by_split'],
        public_radius_m=1000.,public_destination_grid_quantiles=[.15,.5,.85],
        public_destination_rule='nearest BELIEF grid states to nine fixed bounding-box points; stable sorted IDs; round-robin equal rank, unique top-L per category',
        selection_rule='SELECTION only; pass all fixed gates then minimum actual reply bytes; NONE allowed; freeze before TEST',
        primary='current-only; equal-family, equal-purpose conditional category Recall@5; nested draws; empty references N/A',
        cost='compactJSON requests/replies, identical POI record fields; baseline historical schema preserved; mixed templates separate requests per Q, repeated records counted; no measured HTTP latency/energy',
        privacy='same Q/anchors/ledger/reads. Public fixed templates/radius/destination bank independent of private QuerySpec. No new privacy theorem; coordinate protection inherited under existing ideal assumptions.',
        scope='Already-used same-map SUMO cohort; predeclared service ablation, NOT independent fresh generalization or evidence of endpoint improvement',
        adoption='Only opt-in service candidate if selection gates and TEST confirmation pass. No automatic overwrite of frozen L30 artifacts or Geo-I backbone.')
    save(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    for name in source_pins:
        dest=OUT/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    return read(OUT/'protocol.json')


def resources(p):
    data=read(ROOT/p['dataset_path']);assert sha(ROOT/p['dataset_path'])==p['dataset_sha256']
    # The published native_resources validates catalogue and public caches.
    rn,ref,_,beliefs,metadata=common.native_resources(data,PUBLIC_CACHE)
    full=PublicPoiContext(LanePoiService(rn,list(ref.pois),k=60),PUBLIC_CACHE/'public_resources/reply60.npz')
    WORK.mkdir(exist_ok=True)
    vertices=np.array([poi['vertex'] for poi in full.pois],int)
    costs=[]
    for name,is_time in [('distance',False),('time',True)]:
        path=WORK/(name+'.npy')
        if not path.exists():
            print('Building public',name,'costs',len(vertices),'POIs ×',len(rn),'states',flush=True)
            value=dijkstra(matrix(rn,time=is_time).transpose().tocsr(),directed=True,indices=vertices)
            np.save(path,value);del value
        costs.append(np.load(path,mmap_mode='r'))
    base=beliefs[.00125].base;low,high=rn.xy.min(axis=0),rn.xy.max(axis=0)
    points=np.array([low+(high-low)*[a,b] for a in [.15,.5,.85] for b in [.15,.5,.85]])
    destinations=tuple(sorted(set(map(int,base.state_ids[base.tree.query(points)[1]]))))
    service=PublicPurposePoiService(rn,full,*costs,radius_m=1000.,destination_states=destinations)
    ranking=MultiPurposeRoadRanking(ref,cache_limit=128)
    evaluator=common.UtilityEvaluator(rn,full,ranking)
    plans={m:PublicRetrievalPlan(channels,1000.,destinations) for m,channels in CHANNELS.items()}
    metadata.update(full_reply_sha256=full.sha256,public_destination_states=destinations,
        public_cost_file_sha256={name:sha(WORK/(name+'.npy')) for name in ('distance','time')},
        network_cost_assumptions='Directed free-flow capped8m/s native graph, no congestion')
    save(OUT/'resources.json',metadata)
    return rn,service,evaluator,plans


def summarize(rows,wires):
    result={}
    for method in CHANNELS:
        utility=[r for r in rows if r['method']==method]
        wire=[r for r in wires if r['method']==method]
        result[method]=dict(utility=common.summarize_utility(utility),
            cost={k:sum(r[k] for r in wire) for k in ['requests','request_bytes','reply_bytes','reply_records']},
            mean_unique_pool=float(np.mean([r['available_count'] for r in utility])))
    return result


def select(summary):
    decisions=[];base=summary['nearest_l30'];metric='equal_purpose_macro'
    for method in CANDIDATES:
        other=summary[method];delta={purpose:other['utility'][purpose]['family_mean']-base['utility'][purpose]['family_mean']
                                  for purpose in (*common.PURPOSES,metric)}
        matched=other['utility'][metric]['family_mean']-summary[MATCHED[method]]['utility'][metric]['family_mean']
        ratio=other['cost']['reply_bytes']/base['cost']['reply_bytes']
        gates=dict(macro_gain_at_least_1pp=delta[metric]>=GATES['min_macro_gain'],
            all_purposes_loss_at_most_half_pp=all(delta[v]>=GATES['min_each_purpose_gain'] for v in common.PURPOSES),
            reply_ratio_at_most_1_25=ratio<=GATES['max_total_reply_ratio_to_l30'],
            matched_ceiling_macro_not_worse=matched>=GATES['min_matched_ceiling_macro_gain'])
        decisions.append(dict(method=method,differences=delta,matched_ceiling_gain=matched,reply_byte_ratio=ratio,
                              gates=gates,eligible=all(gates.values())))
    eligible=[r['method'] for r in decisions if r['eligible']]
    chosen=min(eligible,key=lambda m:(summary[m]['cost']['reply_bytes'],m)) if eligible else None
    return dict(selected_method=chosen,candidates=decisions,gates=GATES,test_scores_viewed=False)


def replay(p):
    started=time.perf_counter();rn,service,evaluator,plans=resources(p)
    data=read(ROOT/p['dataset_path']);families={f['family_id']:f for f in data['families']}
    inventory={split:[] for split in p['splits']}
    for name,pin in p['source_family_files_sha256'].items():
        family_id=name.split('--draw')[0];inventory[families[family_id]['split']].append((name,pin))
    summaries={};file_pins={};all_rows={};all_wires={};examples=[]
    for split in ['train','selection','test']:
        rows=[];wires=[]
        for name,pin in sorted(inventory[split]):
            assert sha(BASE/'families'/name)==pin
            bundle=read(BASE/'families'/name);truth=bundle['evaluator_only'];streams=bundle['public']['streams']
            original={(r['slot'],r['t']):r for r in bundle['utility'] if r['method']=='legacy_l10' and r['cache']=='current'}
            original_wire={(r['slot'],r['t']):r for r in bundle['wire'] if r['method']=='legacy_l10'}
            local_rows=[];local_wire=[];checked=0
            for slot,session in enumerate(streams['legacy_l10']):
                raw=streams['raw'][slot]['events'];end=raw[-1]['candidates'][0]
                destination=rn.nearest(end['lat'],end['lon'])[0]
                departure=families[truth['family_id']]['evaluator_only']['sessions'][slot]['depart_s']
                for event,raw_event in zip(session['events'],raw):
                    assert event['timestamp_s']==raw_event['timestamp_s'] and len(event['candidates'])==5
                    t=original_wire[slot,event['timestamp_s']]['t'];gps=raw_event['candidates'][0]
                    state=rn.nearest(gps['lat'],gps['lon'])[0]
                    coords=[(q['lat'],q['lon']) for q in event['candidates']]
                    qstates=[rn.nearest(*q)[0] for q in coords]
                    for method,plan in plans.items():
                        responses=[];payloads=[]
                        for coordinate,qs in zip(coords,qstates):
                            for template,l in plan.channels:
                                responses.append(service.query(qs,template,l))
                            if method.startswith('nearest_l'):
                                payloads.append(dict(timestamp_s=departure+t,lat=coordinate[0],lon=coordinate[1],categories=list(service.categories),L=plan.channels[0][1]))
                            else:payloads.extend(plan.requests(departure+t,coordinate,service.categories,int((departure+t)//60)))
                        ids=[[v for cat in reply for v in cat] for reply in responses]
                        pool={v for reply in ids for v in reply}
                        score=evaluator.score(state,destination,pool)
                        fields={k:truth[k] for k in ['family_id','split','draw']};fields.update(slot=slot,t=t,method=method)
                        row=dict(fields,purposes=score,available_count=len(pool));local_rows.append(row)
                        records=[[{k:service.pois[i][k] for k in ('id','category','lat','lon')} for i in reply] for reply in ids]
                        wire=dict(fields,requests=len(payloads),request_bytes=sum(map(json_size,payloads)),
                            reply_bytes=sum(json_size(dict(results=r)) for r in records),reply_records=sum(map(len,ids)),pool_sha256=canonical(sorted(pool)))
                        local_wire.append(wire)
                        if method=='nearest_l20':
                            assert score==original[slot,t]['purposes'],'Frozen L20 scores differ'
                            for key in ('requests','request_bytes','reply_bytes'):assert wire[key]==original_wire[slot,t][key],key
                            checked+=1
                        if len(examples)<len(CHANNELS) and slot==0 and t==0 and split=='train':
                            examples.append(dict(method=method,requests=payloads,responses=records,
                                                 private_query_on_wire=False,public_destination_bank=list(service.destination_states)))
            frozen=dict(Q_stream_sha256=canonical(streams['legacy_l10']),
                ledger_and_anchors_sha256=canonical([{k:v for k,v in s['ledger']['legacy_l10'].items() if k!='step_ms'} for s in truth['sessions']]),
                Q_not_regenerated=True,private_reads_not_performed=True,exact_L20_events_verified=checked)
            payload=dict(evaluator_only={k:truth[k] for k in ['family_id','split','draw']},source_bundle_sha256=pin,
                frozen_controls=frozen,utility=local_rows,wire=local_wire)
            path=OUT/'families'/name;path.parent.mkdir(exist_ok=True)
            path.write_bytes(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0))
            file_pins[name]=sha(path);rows.extend(local_rows);wires.extend(local_wire)
            print(split,name,'exact baseline windows',checked,'elapsed',round(time.perf_counter()-started,1),flush=True)
        summaries[split]=summarize(rows,wires);all_rows[split]=rows;all_wires[split]=wires
        if split=='selection':
            freeze=select(summaries[split]);freeze.update(protocol_sha256=sha(OUT/'protocol.json'),
                selection_family_files_sha256={n:file_pins[n] for n,_ in inventory[split]})
            save(OUT/'selection_freeze.json',freeze)
    comparisons={}
    for method in CANDIDATES:
        comparisons[method]={}
        for metric in (*common.PURPOSES,'equal_purpose_macro'):
            a=summaries['test'][method]['utility'][metric]['family_values'];b=summaries['test']['nearest_l30']['utility'][metric]['family_values']
            comparisons[method][metric]=paired(a,b)
        comparisons[method]['within_draw_macro_gain']={}
        for draw in (1,2,3):
            values=common.summarize_utility([r for r in all_rows['test'] if r['method']==method and r['draw']==draw])
            base=common.summarize_utility([r for r in all_rows['test'] if r['method']=='nearest_l30' and r['draw']==draw])
            comparisons[method]['within_draw_macro_gain'][str(draw)]=values['equal_purpose_macro']['family_mean']-base['equal_purpose_macro']['family_mean']
    chosen=freeze['selected_method'];adopt=False
    if chosen:
        stats=comparisons[chosen]['equal_purpose_macro'];gains=comparisons[chosen]['within_draw_macro_gain']
        adopt=all(select(summaries['test'])['candidates'][list(CANDIDATES).index(chosen)]['gates'].values()) and stats['percentile95_family_bootstrap'][0]>0 and all(v>0 for v in gains.values())
    save(OUT/'wire_examples.json',examples)
    save(OUT/'readout.json',dict(schema='fixed-multipurpose-paired-readout-v1',protocol_sha256=sha(OUT/'protocol.json'),
        selection_freeze_sha256=sha(OUT/'selection_freeze.json'),family_files_sha256=file_pins,summary=summaries,
        test_comparisons=comparisons,adoption=dict(selected_on_selection=chosen,adopt=adopt,
            active_configuration=chosen if adopt else 'nearest_l30',reason='Fixed selection gates and held-out replay confirmation' if adopt else 'No candidate passed all fixed selection/confirmation gates'),
        elapsed_s=time.perf_counter()-started,scope=p['scope']))
    print('ADOPTION',chosen,adopt,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['declare','replay','all'],default='all');args=parser.parse_args()
    if args.stage in ('declare','all'):p=declare()
    else:p=read(OUT/'protocol.json')
    if args.stage in ('replay','all'):
        assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
        for name,pin in p['source_sha256'].items():assert sha(ROOT/name)==pin,name
        replay(p)

if __name__=='__main__':main()
