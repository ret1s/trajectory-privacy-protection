"""Recheck both frozen stages, native paths, attack/POI arithmetic and replay."""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np

from core.demo_protocol import TrajectoryPoint
from evaluation.research_protocol import utility_metrics
from experiments.run_belief_suite import ROOT, DATA, OUTPUT, CASES, METHODS, prepare, records_for, generate
from experiments.run_contextual_lane import sha, estimators
from experiments.verify_lane_comparison import audit_native_graph
from experiments.rng_util import rng_from_key


def near(a,b):
    assert np.allclose(a,b,atol=1e-8,rtol=1e-10),(a,b)


def verify(output=OUTPUT,replay=False):
    output=Path(output)
    data,previous,rn,service,context,prior,belief,_=prepare()
    selection_path=output/'selection.json'; selection=json.loads(selection_path.read_text())
    assert sha(selection_path)==selection_path.with_suffix('.sha256').read_text().strip()
    assert selection['validation_sha256']==sha(output/'validation.json')
    native=audit_native_graph(rn,ROOT/previous['network_path'])
    counts=dict(rows=0,events=0,POI_queries=0,motion_transitions=0,paired_anchor_groups=0,
                replay_rows=0,prefix_checks=0,independent_POI_origins=0,native_connections=native)
    summaries={}
    for phase in ('validation','confirmation'):
        path=output/f'{phase}.json'; p=json.loads(path.read_text())
        assert sha(path)==path.with_suffix('.sha256').read_text().strip()
        assert p['schema']=='belief-suite-v1' and p['phase']==phase
        assert p['dataset_sha256']==sha(DATA) and p['context_sha256']==context.sha256
        assert p['belief_model_sha256']==belief.sha256==selection['belief_model_sha256']
        assert p['method_grid']=={k:list(v) for k,v in METHODS.items()}
        for name,h in p['source_sha256'].items(): assert sha(ROOT/name)==h,name
        assert p['source_sha256']==selection['source_sha256']
        assert p['selection_sha256']==(sha(selection_path) if phase=='confirmation' else None)
        inputs=records_for(data,phase)
        assert inputs==p['records']; records={r['record_id']:r for r in inputs}
        expected={(rid,name,k,1) for rid in records for name in METHODS for k in (3,5)}
        keys=[(r['record_id'],r['method'],r['k'],r['replicate']) for r in p['rows']]
        assert len(keys)==len(set(keys))==180 and set(keys)==expected
        anchors={}
        # Independent directed-distance service oracle, across both families.
        for r in inputs[::4]:
            x=r['points'][0]; point=x['lat'],x['lon']; state,_=rn.nearest(*point)
            distances=nx.single_source_dijkstra_path_length(rn.graph,state,weight='length')
            expected_dist={poi['id']:distances[poi['vertex']] for poi in service.pois if poi['vertex'] in distances}
            actual=service.distances(point)
            assert actual.keys()==expected_dist.keys()
            near(list(actual.values()),[expected_dist[k] for k in actual])
            counts['independent_POI_origins']+=1
        for n,row in enumerate(p['rows']):
            r=records[row['record_id']]
            assert all(row[k]==r[k] for k in ('case_id','family_id','split'))
            seed=int(rng_from_key(r['record_id'],row['k'],1,schema='belief-suite-row-v1').integers(0,2**31))
            assert seed==row['rng_seed']
            public=row['public']
            assert set(public)=={'mechanism','output_kind','public_parameters','events'}
            assert public['output_kind']=='dummy_only'
            assert len(public['events'])==len(r['points'])==len(row['evaluator_states'])==len(row['evaluator_anchors'])
            assert len(row['step_ms'])==len(public['events']) and min(row['step_ms'])>=0
            assert public['public_parameters']['coverage_weight']==METHODS[row['method']][1]
            if METHODS[row['method']][0]:
                assert public['public_parameters']['center_mode']==METHODS[row['method']][2]
                assert len(row['evaluator_belief'])==len(public['events'])
                assert all(np.isfinite(b['entropy_nats']) and b['effective_states']>=1 for b in row['evaluator_belief'])
            else: assert row['evaluator_belief'] is None
            for e,x,states in zip(public['events'],r['points'],row['evaluator_states']):
                assert set(e)=={'event_id','timestamp_s','candidates'} and e['timestamp_s']==x['timestamp_s']
                assert len(e['candidates'])==row['k']
                assert all(set(c)=={'candidate_id','lat','lon'} for c in e['candidates'])
                assert [rn.latlon(i) for i in states]==[(c['lat'],c['lon']) for c in e['candidates']]
            key=row['record_id'],row['k']
            if key in anchors: assert anchors[key]==row['evaluator_anchors']
            anchors[key]=row['evaluator_anchors']
            near(row['spent_bound'],.01+.02*(len(r['points'])-1))
            predicted=estimators(public,rn,prior,r['scenario'])
            truth=np.array([rn.point_xy(x['lat'],x['lon']) for x in r['points']])
            assert predicted.keys()==row['errors_by_attack'].keys()
            for a,xy in predicted.items(): near(row['errors_by_attack'][a],np.linalg.norm(xy-truth,axis=1))
            for t in range(1,len(public['events'])):
                dt=r['points'][t]['timestamp_s']-r['points'][t-1]['timestamp_s']
                for u,v in zip(row['evaluator_states'][t-1],row['evaluator_states'][t]):
                    reachable=nx.single_source_dijkstra_path_length(rn.graph,u,cutoff=dt+1e-8,weight=lambda a,b,d:d['length']/d['speed'])
                    assert v in reachable
                    counts['motion_transitions']+=1
            u=row['utility']; recalls=[]; complete=[]
            assert len(u['poi_rows'])==6*len(public['events'])
            for q in u['poi_rows']:
                if q['reference']:
                    val=len(set(q['reference'])&set(q['returned']))/len(q['reference'])
                    near(q['recall'],val); recalls.append(val)
                    flag=len(q['reference'])==len(q['returned'])
                    assert q['complete']==flag; complete.append(flag)
            near(u['poi_recall_at_5'],np.mean(recalls)); near(u['poi_complete_rate'],np.mean(complete))
            assert u['poi_evaluable_n']==len(recalls)
            if replay:
                points=tuple(TrajectoryPoint(**x) for x in r['points'])
                again=generate(row['method'],points,rn,context,belief,row['k'],seed)
                for field in ('public','evaluator_states','evaluator_anchors','evaluator_belief','spent_bound'):
                    assert again[field]==row[field],(key,row['method'],field)
                short=generate(row['method'],points[:3],rn,context,belief,row['k'],seed)
                assert short['public']['events']==public['events'][:3]
                assert utility_metrics(service,public,[(x.lat,x.lon) for x in points])==u
                counts['replay_rows']+=1; counts['prefix_checks']+=1
            counts['rows']+=1; counts['events']+=len(public['events']); counts['POI_queries']+=len(u['poi_rows'])
            if (n+1)%30==0: print(f'{phase}: {n+1}/180 verified',flush=True)
        counts['paired_anchor_groups']+=len(anchors)
        assert len(p['summaries'])==90
        for s in p['summaries']:
            g=[r for r in p['rows'] if all(r[k]==s[k] for k in ('method','k','case_id'))]
            assert len(g)==2 and len({r['family_id'] for r in g})==s['families']==2
            assert s['rng_replicates_per_family']==1
            for a in g[0]['errors_by_attack']:
                near(s['mae_by_attack'][a],np.mean([np.mean(r['errors_by_attack'][a]) for r in g]))
                near(s['hit_by_attack'][a],np.mean([np.mean(np.asarray(r['errors_by_attack'][a])<=100) for r in g]))
            near(s['audit_min_mae_m'],min(s['mae_by_attack'].values()))
            near(s['audit_max_hit100'],max(s['hit_by_attack'].values()))
            near(s['poi_recall'],np.mean([r['utility']['poi_recall_at_5'] for r in g]))
            near(s['poi_complete'],np.mean([r['utility']['poi_complete_rate'] for r in g]))
            near(s['generation_mean_ms'],np.mean([np.mean(r['step_ms']) for r in g]))
            key=f'{s["method"]}/{s["k"]}/{s["case_id"]}'
            if phase=='validation':
                expected_a={'mae':min(s['mae_by_attack'],key=lambda a:(s['mae_by_attack'][a],a)),
                            'hit':min(s['hit_by_attack'],key=lambda a:(-s['hit_by_attack'][a],a))}
                assert selection['attackers'][key]==expected_a
            else:
                a=selection['attackers'][key]
                assert s['selected_mae_attack']==a['mae'] and s['selected_hit_attack']==a['hit']
                assert s['selected_mae_m']==s['mae_by_attack'][a['mae']]
                assert s['selected_hit100']==s['hit_by_attack'][a['hit']]
        summaries[phase]=p['summaries']
    for k in (3,5):
        candidates=[]
        for name in METHODS:
            g=[s for s in summaries['validation'] if (s['method'],s['k'])==(name,k)]
            candidates.append({'method':name,'min_case_recall':min(s['poi_recall'] for s in g),
                               'macro_hit100':float(np.mean([s['audit_max_hit100'] for s in g]))})
        valid=[c for c in candidates if round(c['min_case_recall'],12)>=.90]
        best=min(valid,key=lambda c:(round(c['macro_hit100'],12),c['method'])) if valid else min(candidates,key=lambda c:(-round(c['min_case_recall'],12),round(c['macro_hit100'],12),c['method']))
        assert selection['method_selection'][str(k)]=={'chosen':best,'utility_feasible':bool(valid),'candidates':candidates}
    return {'verified':True,**counts,'replay':replay,'dataset_sha256':sha(DATA),
            'validation_sha256':sha(output/'validation.json'),'confirmation_sha256':sha(output/'confirmation.json'),
            'selection_sha256':sha(selection_path),
            'verifier_source_sha256':{p:sha(ROOT/p) for p in ('experiments/verify_belief_suite.py','experiments/verify_scenario_suite_v2.py','experiments/scenario_v2_checks.py')}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--replay',action='store_true')
    a=p.parse_args(); result=verify(a.output,a.replay)
    (a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
