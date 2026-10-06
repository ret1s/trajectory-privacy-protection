"""Independent arithmetic/service replay for the sealed public-purpose study.

Does not select a new defense, rerun Geo-I, or write to original evidence. POI
scores, availability, wire bytes and public geometric attacks are reconstructed
from frozen Q streams and dataset truth; learned attack summaries are recomputed
using the saved error table; fitted models were not saved for independent replay.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'artifacts/benchmarks/geoi_purpose_refinement_20261005'
CACHE = Path('/private/tmp/trajectory-research-20261005-public-map')
DATA = ROOT/'artifacts/datasets/research_loop_expanded_v1/dataset.json'
PURPOSES = ('nearest_distance','fastest_travel','within_radius','minimum_detour')
AUDIT_ONLY_DEPENDENCIES = ['experiments/rng_util.py', 'experiments/endpoint_noise_loop.py',
    'benchmark/paper_comparators.py','benchmark/public_poi_context.py',
    'benchmark/response_aware_belief.py','evaluation/live_comparison_attacks.py',
    'evaluation/live_comparison_endpoint_attacks.py','data/lane_states.py',
    'benchmark/engines/quotient_cover.py','benchmark/engines/fair_cover.py']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    raw=Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if str(path).endswith('.gz') else raw)


def close(left,right):
    if left is None or right is None:
        assert left is None and right is None, (left,right)
    else:
        assert math.isfinite(left) and math.isfinite(right)
        assert math.isclose(left,right,rel_tol=1e-10,abs_tol=1e-10), (left,right)


def balanced_mean(rows,getter):
    families=defaultdict(list)
    for row in rows:
        value=getter(row)
        if value is not None:families[row['family_id']].append(float(value))
    assert families, 'No eligible family observations'
    return sum(sum(v)/len(v) for v in families.values())/len(families)


def utility_summary(rows,method,split):
    subset=[r for r in rows if r['method']==method and r['split']==split]
    by_purpose={p:balanced_mean([r for r in subset if r['purpose']==p],lambda r:r['recall']) for p in PURPOSES}
    return dict(by_purpose=by_purpose, mean_recall=sum(by_purpose.values())/4,
        worst_purpose_recall=min(by_purpose.values()), eligible_rows=sum(r['recall'] is not None for r in subset),
        empty_reference_rows=sum(r['recall'] is None for r in subset),
        invalid_returned_items=sum(r['invalid_returned_items'] for r in subset))


def attack_choices(rows):
    assert rows and all(r['status']=='ok' and r['errors'] for r in rows), 'Failed attacks cannot silently become scores'
    names=sorted(set.intersection(*(set(r['errors']) for r in rows)))
    assert names
    assert all(all(math.isfinite(v) and v>=0 for v in r['errors'].values()) for r in rows)
    maes={n:balanced_mean(rows,lambda r,n=n:r['errors'][n]) for n in names}
    chosen={'mae':min(names,key=lambda n:(maes[n],n))}
    for radius in (50,100,200,500):
        hits={n:balanced_mean(rows,lambda r,n=n:float(r['errors'][n]<=radius)) for n in names}
        chosen[f'hit{radius}']=min(names,key=lambda n:(-hits[n],maes[n],n))
    return chosen


def attack_summary(rows,choices):
    return dict(mae_m=balanced_mean(rows,lambda r:r['errors'][choices['mae']]),
        **{f'hit{radius}':balanced_mean(rows,lambda r:float(r['errors'][choices[f'hit{radius}']]<=radius))
           for radius in (50,100,200,500)}, selected_attackers=choices,
        families=len({r['family_id'] for r in rows}),observations=len(rows))


def compare_dict(left,right):
    assert set(left)==set(right), (set(left),set(right))
    for key,value in left.items():
        other=right[key]
        if isinstance(value,dict):compare_dict(value,other)
        elif isinstance(value,(float,np.floating)):close(float(value),other)
        else:assert value==other,(key,value,other)


def selection_from_summaries(configs,baseline,summaries):
    base=summaries[baseline];eligible=[];details={}
    for config in configs:
        name=config['id'];row=summaries[name]
        gain=row['utility']['worst_purpose_recall']-base['utility']['worst_purpose_recall']
        gates=dict(all_purposes90=row['utility']['worst_purpose_recall']>=.90,
            minimum_gain1pp=gain>=.01,
            selection_endpoint_guard=all(row['attacks'][s]['mae_m']>=.9*base['attacks'][s]['mae_m'] and
                row['attacks'][s]['hit100']<=base['attacks'][s]['hit100']+.05 for s in ('S9','S10')),
            public_cost_guard=row['bytes_per_input']<=1.5*base['bytes_per_input'] or gain>=.05)
        admitted=all(gates.values())
        details[name]=dict(gain_worst_purpose=gain,eligible=admitted,gates=gates)
        if admitted:eligible.append(name)
    selected=min(eligible,key=lambda n:(summaries[n]['bytes_per_input'],
                 -summaries[n]['utility']['worst_purpose_recall'],n)) if eligible else baseline
    return selected,details


def allowed_test_methods(selected,baseline,configs):
    chosen=next(c for c in configs if c['id']==selected)
    return {selected,baseline,f'alpha000_L{chosen["L"]}','raw'}


def planned_sessions(data,protocol):
    result={}
    for family in data['families']:
        if family['family_id'] not in protocol['split_by_family']:continue
        sid=min(s['session_id'] for s in family['sessions'] if s['session_id'] in data['traces'])
        trace=data['traces'][sid];start=trace[0]['time_s'];clock=[trace[0]];next_t=60.
        for point in trace[1:]:
            if point['time_s']-start>=next_t:
                clock.append(point);next_t=point['time_s']-start+60.
        if clock[-1] is not trace[-1]:clock.append(trace[-1])
        result[sid]=dict(family_id=family['family_id'],split=protocol['split_by_family'][family['family_id']],
            points={float(p['time_s']-start):p for p in clock},last=trace[-1],close=float(trace[-1]['time_s']-start))
    assert len(result)==len(protocol['split_by_family'])
    return result


def service_replay(runs,utility_rows,sessions,cache_dir):
    """Rebuild road objectives/availability/replies/bytes without production ranker.

    Uses SciPy directed shortest paths over independently constructed matrices.
    Distinct-source forward costs preserve the original floating tie behavior.
    """
    from experiments.public_research_resources import load_public_research_resources
    rn,_,_,_,_,ranking,metadata=load_public_research_resources(cache_dir)
    edges=list(rn.graph.edges(data=True));pairs=([int(a) for a,b,d in edges],[int(b) for a,b,d in edges])
    road=csr_matrix(([float(d['length']) for a,b,d in edges],pairs),shape=(len(rn),len(rn)))
    timed=csr_matrix(([float(d['length'])/float(d['speed']) for a,b,d in edges],pairs),shape=road.shape)
    pois=ranking.pois;vertices=np.array([p['vertex'] for p in pois]);categories=tuple(sorted(ranking.categories))
    by_category={c:np.array([i for i,p in enumerate(pois) if p['category']==c],dtype=int) for c in categories}
    all_server=dijkstra(road.transpose().tocsr(),directed=True,indices=vertices).T
    grouped=defaultdict(list)
    for row in utility_rows:
        grouped[row['method'],row['session_id'],row['rep'],row['timestamp_s']].append(row)
    costs,reverse_costs,availability={},{},{}
    byte_checks=0;list_checks=0;geometry_checks=0
    for run in runs:
        source=sessions[run['session_id']]
        destination=rn.nearest(source['last']['lat'],source['last']['lon'])[0]
        if destination not in reverse_costs:
            reverse_costs[destination]=dijkstra(road.transpose().tocsr(),directed=True,indices=destination)
        to_dest=reverse_costs[destination]
        previous_epoch=None;known=np.zeros(ranking.n,dtype=bool);total_bytes=0
        for event in run['events']:
            t=event['timestamp_s'];epoch=int(t//60.)
            if epoch not in availability:
                availability[epoch]=np.random.Generator(np.random.PCG64(
                    np.random.SeedSequence([2026100597,epoch]))).random(ranking.n)<.8
            available=availability[epoch]
            if epoch!=previous_epoch:known[:]=False
            previous_epoch=epoch
            requests=[];replies=[]
            for q in event['coordinates']:
                state=rn.nearest(*q)[0];reply=[]
                for category in categories:
                    ids=by_category[category]
                    valid=[int(i) for i in ids if available[i] and np.isfinite(all_server[state,i])]
                    valid.sort(key=lambda i:(float(all_server[state,i]),pois[i]['id']))
                    reply.append(valid[:run['config']['L']])
                replies.append(reply)
                for group in reply:known[group]=True
                requests.append(dict(schema='all_category_cover_v1',timestamp_s=t,coordinate=q,
                    categories=list(categories),response_l=run['config']['L'],epoch=epoch))
            payload=dict(requests=requests,replies=[[[pois[i]['id'] for i in cat] for cat in reply] for reply in replies])
            total_bytes+=len(json.dumps(payload,separators=(',',':'),sort_keys=True).encode())
            point=source['points'][t];origin=rn.nearest(point['lat'],point['lon'])[0]
            if origin not in costs:
                costs[origin]=(dijkstra(road,directed=True,indices=origin),dijkstra(timed,directed=True,indices=origin))
            distance,travel=costs[origin]
            local_rows=grouped[run['method'],run['session_id'],run['rep'],t]
            assert len(local_rows)==4*len(categories)
            assert {(r['purpose'],r['category']) for r in local_rows}=={(p,c) for p in PURPOSES for c in categories}
            for row in local_rows:
                scores=(travel if row['purpose']=='fastest_travel' else distance)[vertices].copy()
                if row['purpose']=='within_radius':scores[scores>1000.]=np.inf
                elif row['purpose']=='minimum_detour':
                    direct=distance[destination]
                    if not math.isfinite(direct):scores[:]=np.inf
                    else:
                        # Keep the displayed d(x,p)+d(p,d)-d(x,d) operation
                        # order: reassociation changes exact float ties at zero.
                        scores=scores+to_dest[vertices]-direct
                        finite=np.isfinite(scores);scores[finite]=np.maximum(0.,scores[finite])
                def top(mask):
                    ids=[int(i) for i in by_category[row['category']] if mask[i] and np.isfinite(scores[i])]
                    ids.sort(key=lambda i:(float(scores[i]),pois[i]['id']))
                    return ids[:5]
                assert row['reference']==top(available),(run['method'],t,row['purpose'],'reference')
                assert row['returned']==top(known),(run['method'],t,row['purpose'],'returned')
                invalid=sum(not available[i] or not np.isfinite(scores[i]) for i in row['returned'])
                assert invalid==row['invalid_returned_items']==0
                assert all(known[i] for i in row['returned'])
                list_checks+=2
        close(run['bytes_per_input'],total_bytes/len(run['events']));byte_checks+=1
        for scenario,at in (('S9',0),('S10',-1)):
            true=source['points'][run['events'][at]['timestamp_s']]
            expected=rn.point_xy(true['lat'],true['lon'])
            assert np.allclose(expected,run['target_xy_evaluator_only'][scenario],rtol=0,atol=1e-8)
            geometry_checks+=1
    speeds=sorted({float(d['speed']) for a,b,d in edges})
    return dict(rn=rn, list_checks=list_checks, byte_checks=byte_checks,target_checks=geometry_checks,
        graph_speed_values_m_s=speeds, graph_sha256=rn.catalogue_sha256,
        constant_speed_nearest_equals_fastest=len(speeds)==1)


def verify(out,cache_dir=CACHE):
    protocol=read(out/'protocol.json');selection=read(out/'selection.json');result=read(out/'readout.json')
    for file,digest in protocol['source_sha256'].items():assert sha(ROOT/file)==digest,file
    assert sha(DATA)==protocol['dataset_sha256']
    families=protocol['split_by_family'];assert set(families.values())=={'fit','selection','test'}
    data=read(DATA);sessions=planned_sessions(data,protocol)
    configs=protocol['configs'];baseline=protocol['baseline'];methods={c['id'] for c in configs}|{'raw'}
    assert len(configs)==9 and len(methods)==10
    assert {(c['alpha'],c['L']) for c in configs}=={(a,L) for a in (0.,.5,1.) for L in (10,20,40)}
    runs=[];development={}
    for method in methods:
        parts=read(out/f'development-{method}.json.gz')
        assert set(parts)=={'fit','selection'}
        development[method]=parts
        for split,group in parts.items():
            assert len(group)==2*sum(s==split for s in families.values())
            assert all(r['method']==method and r['split']==split for r in group)
            runs.extend(group)
    test=read(out/'test_runs.json.gz');allowed=allowed_test_methods(selection['selected'],baseline,configs)
    assert {r['method'] for r in test}==allowed
    assert set(result['test'])==allowed
    assert all(r['split']=='test' for r in test)
    assert len(test)==len(allowed)*2*sum(s=='test' for s in families.values())
    runs.extend(test)
    run_by_key={}
    for run in runs:
        sid,rep=run['session_id'],run['rep'];source=sessions[sid]
        assert run['family_id']==source['family_id'] and run['split']==source['split']
        assert rep in protocol['repetitions'] and run['close_s']==source['close']
        assert [e['timestamp_s'] for e in run['events']]==list(source['points'])
        words=np.frombuffer(hashlib.sha256(f'geoi-purpose-refinement-v1|{sid}/{rep}'.encode()).digest()[:16],dtype='<u4')
        expected_seed=int(np.random.default_rng(np.random.SeedSequence(words)).integers(0,2**63))
        assert run['seed']==expected_seed
        key=(run['method'],sid,rep);assert key not in run_by_key;run_by_key[key]=run
        for event in run['events']:
            assert set(event)=={'timestamp_s','coordinates'}
            assert len(event['coordinates'])==(1 if run['method']=='raw' else 5)
            assert all(len(q)==2 and math.isfinite(q[0]) and math.isfinite(q[1]) and -90<=q[0]<=90 and -180<=q[1]<=180 for q in event['coordinates'])
        if run['method']=='raw':continue
        assert len(run['ledger_evaluator_only'])==len(run['protected_anchors_evaluator_only'])==len(run['events'])
        units=0
        for index,row in enumerate(run['ledger_evaluator_only']):
            charge={'fresh':1 if index==0 else 2,'reuse':1,'postprocess':0,'public_clock_skip':0}[row['branch']]
            assert row['cost_units']==charge and row['private_read']==(charge>0)
            if charge:assert units+(1 if index==0 else 2)<=23
            units+=charge;assert units==row['spent_units']<=23
        close(run['spent_bound_per_m'],units*.0025);assert run['spent_bound_per_m']<=.0575+1e-12
    anchor_pairs=0
    for run in runs:
        if run['method']=='raw':continue
        base=run_by_key[baseline,run['session_id'],run['rep']]
        assert run['protected_anchors_evaluator_only']==base['protected_anchors_evaluator_only']
        assert run['ledger_evaluator_only']==base['ledger_evaluator_only']
        anchor_pairs+=1
    utility=read(out/'utility_rows.json.gz');utility_keys=set()
    for row in utility:
        assert row['method'] in methods and row['purpose'] in PURPOSES
        assert row['split']==families[row['family_id']]
        assert row['split']!='test' or row['method'] in allowed
        assert (row['method'],row['session_id'],row['rep']) in run_by_key
        assert len(row['reference'])==len(set(row['reference']))<=5
        assert len(row['returned'])==len(set(row['returned']))<=5
        expected=len(set(row['reference'])&set(row['returned']))/len(row['reference']) if row['reference'] else None
        close(row['recall'],expected)
        assert row['invalid_returned_items']==0
        key=(row['method'],row['session_id'],row['rep'],row['timestamp_s'],row['purpose'],row['category'])
        assert key not in utility_keys;utility_keys.add(key)
    selection_rows=read(out/'selection_attack_rows.json.gz');test_rows=read(out/'test_attack_rows.json.gz')
    assert all(families[r['family_id']]=='selection' and r['method'] in methods for r in selection_rows)
    assert all(families[r['family_id']]=='test' and r['method'] in allowed for r in test_rows)
    computed={};choices={}
    for method in methods:
        attacks={};choices[method]={}
        for scenario in ('S9','S10'):
            rows=[r for r in selection_rows if r['method']==method and r['scenario']==scenario]
            assert len(rows)==2*sum(s=='selection' for s in families.values())
            choices[method][scenario]=attack_choices(rows)
            attacks[scenario]=attack_summary(rows,choices[method][scenario])
        computed[method]=dict(utility=utility_summary(utility,method,'selection'),
            bytes_per_input=balanced_mean(development[method]['selection'],lambda r:r['bytes_per_input']),attacks=attacks)
    selected,gates=selection_from_summaries(configs,baseline,computed)
    assert selected==selection['selected']==result['selection']
    assert selection['test_used_for_selection'] is False
    assert selection['promoted']==result['promoted_in_development']==(selected!=baseline)
    for method,summary in computed.items():
        saved=selection['all_selection_summaries'][method]
        expected=dict(summary,**gates.get(method,{}))
        compare_dict(expected,saved)
    for method in allowed:
        expected=dict(utility=utility_summary(utility,method,'test'),
            bytes_per_input=balanced_mean([r for r in test if r['method']==method],lambda r:r['bytes_per_input']),attacks={})
        for scenario in ('S9','S10'):
            rows=[r for r in test_rows if r['method']==method and r['scenario']==scenario]
            assert len(rows)==2*sum(s=='test' for s in families.values())
            expected['attacks'][scenario]=attack_summary(rows,choices[method][scenario])
        compare_dict(expected,result['test'][method])
    weights_metadata=read(out/'public_purpose_weights.json');weights=np.load(out/'public_purpose_weights.npz')['weights']
    metadata={k:v for k,v in weights_metadata.items() if k!='sha256'}
    digest=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode());digest.update(weights.astype('<f8').tobytes())
    assert digest.hexdigest()==weights_metadata['sha256']
    assert weights.shape[0]==len(metadata['state_ids']) and np.isfinite(weights).all() and np.all(weights>=0)
    mass=weights.sum(axis=1)
    assert np.all(weights[:,-1]==0) and np.all(np.isclose(mass,0.,atol=1e-12,rtol=0)|np.isclose(mass,1.,atol=1e-12,rtol=0))
    replay=service_replay(runs,utility,sessions,cache_dir);rn=replay.pop('rn')
    assert weights.shape[1]==419
    assert replay['graph_sha256']==read(out/'resources.json')['catalogue']['sha256']
    geometry_errors=0
    for row in selection_rows+test_rows:
        matched=[r for r in runs if r['method']==row['method'] and r['session_id']==row['session_id'] and r['seed']==row['seed']]
        assert len(matched)==1;run=matched[0];scenario=row['scenario'];first=scenario=='S9';at=0 if first else -1
        sets=[np.unique(np.array([rn.point_xy(*q) for q in e['coordinates']]),axis=0) for e in run['events']]
        truth=np.array(run['target_xy_evaluator_only'][scenario]);times=np.array([e['timestamp_s'] for e in run['events']]);times-=times[0]
        for name,stream in [('centroid',np.array([p.mean(axis=0) for p in sets])),('median',np.array([np.median(p,axis=0) for p in sets]))]:
            close(row['errors'][name],float(np.linalg.norm(stream[at]-truth)));geometry_errors+=1
            for count in (2,3,6):
                ids=np.arange(min(count,len(times))) if first else np.arange(max(0,len(times)-count),len(times))
                design=np.column_stack([np.ones(len(ids)),times[ids]-times[at]])
                fit=np.linalg.lstsq(design,stream[ids],rcond=None)[0]
                for seconds in (30,60,120):
                    predicted=np.array([1.,(-1 if first else 1)*seconds])@fit
                    close(row['errors'][f'{name}_ols{count}_{seconds}s'],float(np.linalg.norm(predicted-truth)));geometry_errors+=1
                design=np.column_stack([np.ones(len(ids)),times[ids]])
                fit=np.linalg.lstsq(design,stream[ids],rcond=None)[0]
                target_time=0. if first else run['close_s']
                predicted=np.array([1.,target_time])@fit
                close(row['errors'][f'public_boundary_{name}_ols{count}'],float(np.linalg.norm(predicted-truth)));geometry_errors+=1
    return dict(status='passed', selected=selected,promoted=selected!=baseline,
        development_configurations_preserved=9,development_runs=len(runs)-len(test),heldout_runs=len(test),
        heldout_methods=sorted(allowed),no_unselected_candidate_heldout_scores=True,
        utility_rows_recomputed=len(utility),empty_references_preserved=sum(r['recall'] is None for r in utility),
        empty_public_objective_profiles=int(np.sum(np.isclose(mass,0.,atol=1e-12,rtol=0))),
        selected_attack_summaries_recomputed=True,geometric_attack_errors_recomputed=geometry_errors,
        private_anchor_and_ledger_equal_pairs=anchor_pairs,all_presealed_source_hashes_unchanged=True,
        service_replay=replay, invalid_returned_items=0,
        audit_only_current_dependency_sha256={f:sha(ROOT/f) for f in AUDIT_ONLY_DEPENDENCIES},
        limitations=['Some transitive dependencies/availability seed are pinned only through main source, not separately in original protocol.',
            'Learned/Viterbi errors are archived without fitted models; aggregate metrics are verified but their estimates are not independently refitted.',
            'Clock includes observable true final sample; fixed public length/timing remains an assumption.',
            'Only previously inspected three test families and two RNG draws; no independent confirmation.',
            'Finite-precision Geo-I/PRNG limitations remain; no identity/network anonymity claim.'],
        output_sha256={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='verification.json'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--cache-dir',type=Path,default=CACHE);args=parser.parse_args()
    result=verify(args.out,args.cache_dir)
    print(json.dumps({k:v for k,v in result.items() if k not in ('output_sha256','audit_only_current_dependency_sha256')},indent=2))


if __name__=='__main__':main()
