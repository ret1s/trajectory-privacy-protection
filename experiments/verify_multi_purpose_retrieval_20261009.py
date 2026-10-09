"""Independent arithmetic/inventory and exact frozen-L30 reconciliation.

Does not use the study's summarize/select functions. Native template checks
use the original forward shortest-path local ranking, not reverse matrices.
"""
from collections import defaultdict
from pathlib import Path
import gzip,hashlib,json
import numpy as np
from benchmark.query_purpose import QuerySpec,QueryPurpose,MultiPurposeRoadRanking
from benchmark.public_poi_context import PublicPoiContext
from evaluation.lane_travel import LanePoiService
from experiments.future_sumo_eval import native_resources

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/benchmarks/multi_purpose_retrieval_20261009_v2'
DEPTH=ROOT/'artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1'
PUBLIC_CACHE=Path('/private/tmp/qplanner-response-depth-generalization-20261006-v1')
PURPOSES=('nearest_distance','fastest_travel','within_radius','minimum_detour')
def read(path):
    raw=Path(path).read_bytes();return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    p=read(OUT/'protocol.json');r=read(OUT/'readout.json');freeze=read(OUT/'selection_freeze.json')
    assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()==r['protocol_sha256']
    assert sha(OUT/'selection_freeze.json')==r['selection_freeze_sha256']
    for name,pin in p['source_sha256'].items():
        assert sha(ROOT/name)==sha(OUT/'source_snapshot'/name)==pin,name
    assert sha(ROOT/p['dataset_path'])==p['dataset_sha256']
    base=ROOT/p['source_output'];assert sha(base/'protocol.json')==p['source_protocol_sha256']
    assert sha(base/'generation.json')==p['source_generation_sha256']
    assert set(r['family_files_sha256'])==set(p['source_family_files_sha256'])
    means=defaultdict(list);costs=defaultdict(lambda:defaultdict(int));counts=defaultdict(int)
    event_count=0;exact_l30=0;draws=defaultdict(set)
    for name,pin in sorted(r['family_files_sha256'].items()):
        path=OUT/'families'/name;assert sha(path)==pin
        assert sha(base/'families'/name)==p['source_family_files_sha256'][name]
        block=read(path);original=read(base/'families'/name);depth=read(DEPTH/'families'/name)
        truth=block['evaluator_only'];split=truth['split'];fid=truth['family_id'];draw=truth['draw'];draws[split,fid].add(draw)
        old={(u['slot'],u['t']):u for u in depth['utility'] if u['method']=='service_l30' and u['cache']=='current'}
        old_wire={(w['slot'],w['t']):w for w in depth['wire'] if w['method']=='service_l30'}
        assert block['source_bundle_sha256']==p['source_family_files_sha256'][name]
        canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert block['frozen_controls']['Q_stream_sha256']==canonical(original['public']['streams']['legacy_l10'])
        ledgers=[{k:v for k,v in s['ledger']['legacy_l10'].items() if k!='step_ms'} for s in original['evaluator_only']['sessions']]
        assert block['frozen_controls']['ledger_and_anchors_sha256']==canonical(ledgers)
        expected={(w['slot'],w['t'],m) for w in old_wire.values() for m in p['channels']}
        assert {(u['slot'],u['t'],u['method']) for u in block['utility']}==expected
        assert {(w['slot'],w['t'],w['method']) for w in block['wire']}==expected
        assert len(block['utility'])==len(block['wire'])==len(expected)
        for u in block['utility']:
            method=u['method'];counts[split,method]+=1
            for purpose in PURPOSES:
                value=u['purposes'][purpose]['recall5']
                assert u['purposes'][purpose]['reference_category_count']==old[u['slot'],u['t']]['purposes'][purpose]['reference_category_count']
                if value is not None:
                    assert 0<=value<=1
                    means[split,method,fid,purpose].append(value)
            if method=='nearest_l30':
                assert u['purposes']==old[u['slot'],u['t']]['purposes']
                exact_l30+=1
        for w in block['wire']:
            key=split,w['method'];channels=p['channels'][w['method']]
            assert w['requests']==5*len(channels)
            for metric in ('requests','request_bytes','reply_bytes','reply_records'):costs[key][metric]+=w[metric]
            if w['method']=='nearest_l30':
                for metric in ('requests','request_bytes','reply_bytes'):assert w[metric]==old_wire[w['slot'],w['t']][metric]
        event_count+=len(old)
    for (split,fid),ds in draws.items():assert ds==set(range(1,p['draws_by_split'][split]+1))
    for split,summary in r['summary'].items():
        families=sorted({f for s,f in draws if s==split})
        for method,entry in summary.items():
            macro=[]
            for fid in families:
                values=[]
                for purpose in PURPOSES:
                    data=means[split,method,fid,purpose];v=float(np.mean(data)) if data else None
                    actual=entry['utility'][purpose]['family_values'][fid]
                    assert (v is None)==(actual is None)
                    if v is not None:assert abs(v-actual)<1e-12;values.append(v)
                macro.append(float(np.mean(values)))
            assert abs(np.mean(macro)-entry['utility']['equal_purpose_macro']['family_mean'])<1e-12
            assert dict(costs[split,method])==entry['cost']
    chosen=[];selection=r['summary']['selection'];base30=selection['nearest_l30']
    for candidate in freeze['candidates']:
        m=candidate['method'];a=selection[m];g=p['gates']
        delta={purpose:a['utility'][purpose]['family_mean']-base30['utility'][purpose]['family_mean'] for purpose in (*PURPOSES,'equal_purpose_macro')}
        gates=dict(macro_gain_at_least_1pp=delta['equal_purpose_macro']>=g['min_macro_gain'],
            all_purposes_loss_at_most_half_pp=all(delta[v]>=g['min_each_purpose_gain'] for v in PURPOSES),
            reply_ratio_at_most_1_25=a['cost']['reply_bytes']/base30['cost']['reply_bytes']<=g['max_total_reply_ratio_to_l30'],
            matched_ceiling_macro_not_worse=a['utility']['equal_purpose_macro']['family_mean']>=selection[p['matched_ceiling'][m]]['utility']['equal_purpose_macro']['family_mean'])
        assert gates==candidate['gates'] and all(gates.values())==candidate['eligible']
        if all(gates.values()):chosen.append(m)
    selected=min(chosen,key=lambda m:(selection[m]['cost']['reply_bytes'],m)) if chosen else None
    assert freeze['selected_method']==selected and freeze['test_scores_viewed'] is False
    # Verify each paired bootstrap independently with the declared seed/clusters.
    for method,metrics in r['test_comparisons'].items():
        for purpose in (*PURPOSES,'equal_purpose_macro'):
            stat=metrics[purpose];fids=stat['family_ids']
            left=r['summary']['test'][method]['utility'][purpose]['family_values']
            right=r['summary']['test']['nearest_l30']['utility'][purpose]['family_values']
            d=np.array([left[f]-right[f] for f in fids]);assert len(fids)==24
            indices=np.random.default_rng(stat['statistical_seed']).integers(0,len(fids),size=(stat['replicates'],len(fids)))
            ci=np.quantile(d[indices].mean(axis=1),[.025,.975])
            assert abs(float(d.mean())-stat['mean_difference'])<1e-12
            assert np.allclose(ci,stat['percentile95_family_bootstrap'],rtol=0,atol=1e-12)
    # Independent forward-cost oracle for every channel in the actual first wire example.
    data=read(ROOT/p['dataset_path']);rn,ref,_,_,_=native_resources(data,PUBLIC_CACHE)
    rank=MultiPurposeRoadRanking(ref);mask=np.ones(rank.n,bool);examples=read(OUT/'wire_examples.json');checked_native=0
    for ex in examples:
        if ex['method'].startswith('nearest_l'):continue
        for request,records in zip(ex['requests'],ex['responses']):
            state=rn.nearest(*request['coordinate'])[0]
            state=int(rn.tree.query(rn.xy[state])[1]);purpose=request['retrieval_type'];l=request['response_l']
            ids=[]
            for category in rank.categories:
                if purpose=='public_detour_bank':
                    rows=[rank.top(state,mask,QuerySpec(QueryPurpose.MIN_DETOUR,category,k=l,destination_state=s)) for s in request['public_destination_states']]
                    union=[]
                    for i in range(max(map(len,rows),default=0)):
                        for row in rows:
                            if i<len(row) and row[i] not in union:union.append(row[i])
                    ids.extend(union[:l])
                else:
                    spec=QuerySpec(QueryPurpose(purpose),category,k=l,radius_m=request['public_radius_m'] if purpose=='within_radius' else None)
                    ids.extend(rank.top(state,mask,spec))
            assert [rank.pois[i]['id'] for i in ids]==[item['id'] for item in records],purpose
            checked_native+=1
    result=dict(status='pass',verifier_sha256=sha(Path(__file__)),protocol_sha256=sha(OUT/'protocol.json'),
        readout_sha256=sha(OUT/'readout.json'),family_blocks=len(r['family_files_sha256']),source_windows=event_count,
        exact_frozen_l30_windows=exact_l30,all_family_aggregates_costs_and_bootstraps_recomputed=True,
        forward_oracle_native_response_checks=checked_native,forward_oracle_scope='All 45 requests of three candidate first-event examples, not all native states',
        selected_method=selected,scope=p['scope'])
    with (OUT/'validation.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
