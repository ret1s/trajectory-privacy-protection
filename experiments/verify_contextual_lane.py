"""Independent arithmetic and motion audit, optional full public-context replay."""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

from benchmark.public_poi_context import PublicPoiContext
from core.demo_protocol import TrajectoryPoint
from evaluation.research_protocol import utility_metrics
from experiments.run_contextual_lane import ROOT, DATA, PREVIOUS, CASES, METHODS, prepare, generate, estimators, sha
from experiments.verify_lane_comparison import audit_native_graph
from experiments.rng_util import rng_from_key


def near(a,b):
    assert np.allclose(a,b,atol=1e-8,rtol=1e-10), (a,b)


def verify(path, replay=False):
    path=Path(path)
    p=json.loads(path.read_text())
    assert sha(path)==path.with_suffix('.sha256').read_text().strip()
    assert p['schema']=='contextual-lane-v1'
    assert p['dataset_sha256']==sha(DATA) and p['previous_results_sha256']==sha(PREVIOUS)
    assert p['method_grid']=={k:list(v) for k,v in METHODS.items()}
    for name,digest in p['source_sha256'].items():
        assert sha(ROOT/name)==digest, name
    assert sha(ROOT/p['network_path'])==p['network_sha256']
    assert sha(ROOT/p['service']['osm_path'])==p['service']['osm_sha256']
    data, previous, rn, service, context, _, prior, inputs = prepare()
    assert p['records']==inputs
    assert p['context_sha256']==context.sha256
    assert p['service']['pois_used']==service.pois
    assert p['context_bytes']==context.signatures.nbytes+context.access.nbytes
    if replay:
        print('Rebuilding public POI context without cache',flush=True)
        rebuilt=PublicPoiContext(service)
        assert rebuilt.sha256==context.sha256
    native=audit_native_graph(rn,ROOT/p['network_path'])
    records={r['record_id']:r for r in inputs}
    old={(r['record_id'],r['k'],r['replicate']):r for r in previous['rows'] if r['method']=='lane_br_raw'}
    assert len(records)==10 and len(p['rows'])==160
    expected={(rid,name,k,rep) for rid in records for name in METHODS for k in (3,5) for rep in (1,2)}
    actual=[(r['record_id'],r['method'],r['k'],r['replicate']) for r in p['rows']]
    assert len(set(actual))==len(actual) and set(actual)==expected
    anchors={}
    checked_states=set(np.linspace(0,len(rn)-1,32,dtype=int))
    events=queries=transitions=baseline_replays=prefixes=0
    for n,row in enumerate(p['rows']):
        r=records[row['record_id']]
        assert all(row[k]==r[k] for k in ('case_id','split','family_id'))
        seed=int(rng_from_key(r['record_id'],row['k'],row['replicate'],'paired_br',schema='lane-study-row-v1').integers(0,2**31))
        assert row['rng_seed']==seed
        public=row['public']
        assert set(public)=={'mechanism','output_kind','public_parameters','events'}
        assert public['output_kind']=='dummy_only'
        assert len(public['events'])==len(r['points'])==len(row['evaluator_states'])==len(row['evaluator_anchors'])==len(row['step_ms'])
        assert public['public_parameters']['route_weight']==METHODS[row['method']][0]
        assert public['public_parameters']['coverage_weight']==METHODS[row['method']][1]
        for event,x,states in zip(public['events'],r['points'],row['evaluator_states']):
            assert set(event)=={'event_id','timestamp_s','candidates'}
            assert event['timestamp_s']==x['timestamp_s'] and len(event['candidates'])==row['k']
            assert all(set(c)=={'candidate_id','lat','lon'} for c in event['candidates'])
            assert [rn.latlon(i) for i in states]==[(c['lat'],c['lon']) for c in event['candidates']]
            checked_states.update(states)
        assert row['distinct_coordinates']==[len({(c['lat'],c['lon']) for c in e['candidates']}) for e in public['events']]
        key=row['record_id'],row['k'],row['replicate']
        if key in anchors:
            assert anchors[key]==row['evaluator_anchors']
        anchors[key]=row['evaluator_anchors']
        near(row['spent_bound'],.01+.02*(min(12,len(public['events']))-1))
        predictions=estimators(public,rn,prior,r['scenario'])
        truth=np.array([rn.point_xy(x['lat'],x['lon']) for x in r['points']])
        assert predictions.keys()==row['errors_by_attack'].keys()
        for name,xy in predictions.items():
            near(row['errors_by_attack'][name],np.linalg.norm(xy-truth,axis=1))
        for t in range(1,len(public['events'])):
            dt=public['events'][t]['timestamp_s']-public['events'][t-1]['timestamp_s']
            for u,v in zip(row['evaluator_states'][t-1],row['evaluator_states'][t]):
                reached=nx.single_source_dijkstra_path_length(rn.graph,u,cutoff=dt+1e-8,weight=lambda a,b,d:d['length']/d['speed'])
                assert v in reached
                transitions+=1
        u=row['utility']
        recalls=[len(set(q['reference'])&set(q['returned']))/len(q['reference']) for q in u['poi_rows'] if q['reference']]
        near(u['poi_recall_at_5'],np.mean(recalls))
        assert u['poi_evaluable_n']==len(recalls)
        assert len(u['poi_rows'])==len(service.categories)*len(public['events'])
        if row['method']=='baseline' and key in old:
            for field in ('public','evaluator_states','spent_bound','utility'):
                lhs,rhs=row[field],old[key][field]
                assert (lhs['events']==rhs['events']) if field=='public' else lhs==rhs
            baseline_replays+=1
        if replay:
            points=tuple(TrajectoryPoint(**x) for x in r['points'])
            rerun=generate(row['method'],points,rn,context,row['k'],seed)
            for field in ('public','evaluator_states','evaluator_anchors','spent_bound'):
                assert rerun[field]==row[field], (key,field)
            short=generate(row['method'],points[:3],rn,context,row['k'],seed)
            assert short['public']['events']==public['events'][:3]
            assert utility_metrics(service,public,[(x.lat,x.lon) for x in points])==u
            prefixes+=1
        events+=len(public['events'])
        queries+=len(u['poi_rows'])
        if (n+1)%16==0:
            print(f'{n+1}/160 rows verified',flush=True)
    # Shared context is a score proxy; independently check actual POI membership,
    # not just the cached builder. Count disagreements instead of hiding them.
    index_checks=index_mismatches=0
    for state in sorted(checked_states):
        assert context.access[state]==rn.nearest(*rn.latlon(state))[0]
        for j,category in enumerate(context.categories):
            expected=set(service.query(rn.latlon(state),category))
            observed={context.pois[i]['id'] for i in context.query_indices(state)[j] if i>=0}
            index_checks+=1
            index_mismatches+=expected!=observed
    # Independent summary and validation-only mechanism selection arithmetic.
    assert len(p['summaries'])==80
    for s in p['summaries']:
        group=[r for r in p['rows'] if all(r[k]==s[k] for k in ('split','method','k','case_id'))]
        assert len(group)==s['rng_replicates']==2 and s['families']==1
        for a in group[0]['errors_by_attack']:
            near(s['mae_by_attack'][a],np.mean([np.mean(r['errors_by_attack'][a]) for r in group]))
            near(s['hit_by_attack'][a],np.mean([np.mean(np.asarray(r['errors_by_attack'][a])<=100) for r in group]))
        near(s['audit_min_mae_m'],min(s['mae_by_attack'].values()))
        near(s['audit_max_hit100'],max(s['hit_by_attack'].values()))
        near(s['poi_recall'],np.mean([r['utility']['poi_recall_at_5'] for r in group]))
        near(s['poi_complete'],np.mean([r['utility']['poi_complete_rate'] for r in group]))
        near(s['generation_mean_ms'],np.mean([np.mean(r['step_ms']) for r in group]))
        near(s['generation_p95_ms'],np.mean([np.percentile(r['step_ms'],95) for r in group]))
        if s['split']=='development_test':
            val=next(v for v in p['summaries'] if (v['split'],v['method'],v['k'],v['case_id'])==('development_validation',s['method'],s['k'],s['case_id']))
            ma=min(val['mae_by_attack'],key=lambda a:(val['mae_by_attack'][a],a))
            hi=max(val['hit_by_attack'],key=lambda a:(val['hit_by_attack'][a],a))
            assert s['selected_mae_attack']==ma and s['selected_hit_attack']==hi
            assert s['selected_mae_m']==s['mae_by_attack'][ma] and s['selected_hit100']==s['hit_by_attack'][hi]
    for k in (3,5):
        candidates=[]
        for name in METHODS:
            group=[s for s in p['summaries'] if (s['split'],s['method'],s['k'])==('development_validation',name,k)]
            assert {s['case_id'] for s in group}==set(CASES)
            candidates.append({'method':name,'min_case_recall':min(s['poi_recall'] for s in group),
                               'macro_hit100':float(np.mean([s['audit_max_hit100'] for s in group]))})
        valid=[c for c in candidates if round(c['min_case_recall'],12)>=.90]
        best=min(valid,key=lambda c:(round(c['macro_hit100'],12),c['method'])) if valid else min(candidates,key=lambda c:(-round(c['min_case_recall'],12),round(c['macro_hit100'],12),c['method']))
        assert p['method_selection'][str(k)]=={'chosen':best,'utility_feasible':bool(valid),'candidates':candidates}
    result={'verified':True,'rows':len(p['rows']),'events':events,'POI_queries':queries,
            'native_connections':native,'motion_transitions':transitions,'previous_baseline_replays':baseline_replays,
            'paired_anchor_groups':len(anchors),'prefix_checks':prefixes,'replay':replay,
            'context_rebuilt':replay,'context_membership_checks':index_checks,'context_membership_mismatches':index_mismatches,
            'results_sha256':sha(path)}
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'artifacts/benchmarks/contextual_lane/results.json')
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    verify(args.results,args.replay)
