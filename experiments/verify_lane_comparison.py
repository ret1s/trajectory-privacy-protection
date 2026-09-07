"""Audit lane-study provenance, arithmetic, public surface and native motion.

NetworkX independently checks SciPy paths; --replay regenerates every release
and POI response. These are implementation checks, not an optimal-adversary or
pure-DP certification. No frozen earlier benchmark is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np

from core.demo_protocol import TrajectoryPoint
from data.lane_states import build_lane_states, catalogue_summary, coordinate_catalogue
from data.scenario_suite.records import device_view
from data.sumo_demo import _load_sumolib, load_sumo_road_network
from evaluation.lane_travel import LanePoiService
from evaluation.research_protocol import utility_metrics
from evaluation.scenario_metrics import read_osm_pois
from experiments.run_lane_comparison import DATA, ROOT, METHODS, CASES, attacks, release, spatial_prior
from experiments.rng_util import rng_from_key


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def close(a,b):
    assert np.allclose(a,b,rtol=1e-10,atol=1e-7), (a,b)


def audit_native_graph(rn,path):
    net=_load_sumolib().net.readNet(str(path),withInternal=True)
    lanes={l.getID():l for e in net.getEdges(withInternal=True) for l in e.getLanes() if l.allows('passenger')}
    assert set(lanes)==set(rn.lane_indices)
    native_connections={(lid,c.getViaLaneID() or c.getToLane().getID())
                        for lid,l in lanes.items() for c in l.getOutgoing()
                        if c.allows('passenger') and (c.getViaLaneID() or c.getToLane().getID()) in lanes}
    seen=set()
    for a,b,d in rn.graph.edges(data=True):
        x,y=rn.graph.nodes[a],rn.graph.nodes[b]
        assert 0<d['speed']<=8 and d['length']>=0
        if d['kind']=='lane_progress':
            assert x['lane_id']==y['lane_id'] and y['lane_pos_m']>x['lane_pos_m']
            assert d['length']<=20+1e-8
        else:
            pair=x['lane_id'],y['lane_id']
            assert pair in native_connections
            assert a==rn.lane_indices[pair[0]][-1] and b==rn.lane_indices[pair[1]][0]
            seen.add(pair)
    assert seen==native_connections
    return len(seen)


def verify(path,replay=False):
    path=Path(path)
    p=json.loads(path.read_text())
    assert sha(path)==path.with_suffix('.sha256').read_text().strip()
    assert p['schema']=='lane-comparison-v1'
    assert p['prior']=={'source_family':91,'bandwidth_m':250,'floor':1e-9,'sampling_stride':20}
    assert p['dataset_sha256']==sha(DATA)
    for name,digest in p['source_sha256'].items():
        assert sha(ROOT/name)==digest, f'Stale scientific source: {name}'
    assert sha(ROOT/p['network_path'])==p['network_sha256']
    assert sha(ROOT/p['service']['osm_path'])==p['service']['osm_sha256']
    d=json.loads(DATA.read_text())
    rn=build_lane_states(ROOT/p['network_path'])
    sites=coordinate_catalogue(rn)
    assert p['catalogue']=={**catalogue_summary(rn),'distinct_coordinates':len(sites)}
    connections=audit_native_graph(rn,ROOT/p['network_path'])
    old=load_sumo_road_network(ROOT/p['network_path'])
    train_ids={s['session_id'] for f in d['families'] if f['seed']==91 for s in f['sessions']}
    prior=spatial_prior(rn,[d['traces'][sid] for sid in sorted(train_ids)])
    service=LanePoiService(rn,read_osm_pois(ROOT/p['service']['osm_path'],tuple(d['network']['bbox_lon_lat'])))
    assert service.pois==p['service']['pois_used'] and service.excluded==p['service']['pois_excluded']
    source_records={r['record_id']:r for r in d['records']}
    records={r['record_id']:r for r in p['records']}
    assert len(records)==6 and len(p['rows'])==144 and len(p['summaries'])==36
    for r in records.values():
        source=source_records[r['record_id']]
        assert r['case_id'] in CASES
        assert all(r[key]==source[key] for key in ('case_id','scenario','family_id','split'))
        expected=[{'timestamp_s':x['time_s'],'lat':x['lat'],'lon':x['lon']}
                  for x in list(device_view(source,d['traces']))[:12]]
        assert r['points']==expected and r['retained_events']==len(expected)
        assert r['available_events']==len(source['observed_indices'][0])
        close(r['nearest_lane_error_m'],[rn.nearest(x['lat'],x['lon'])[1] for x in expected])
        close(r['nearest_junction_error_m'],[old.nearest(x['lat'],x['lon'])[1] for x in expected])
    # Independent NetworkX distance oracle at up to eight fixture points, all POIs.
    oracle_points=[x for r in records.values() for x in r['points']][::6][:8]
    for x in oracle_points:
        point=x['lat'],x['lon']
        v,_=rn.nearest(*point)
        distances=nx.single_source_dijkstra_path_length(rn.graph,v,weight='length')
        expected={poi['id']:distances[poi['vertex']] for poi in service.pois if poi['vertex'] in distances}
        actual=service.distances(point)
        assert expected.keys()==actual.keys()
        close(list(expected.values()),[actual[i] for i in expected])
    keys=[]
    transitions=queries=events=prefix_checks=0
    for n,row in enumerate(p['rows']):
        r=records[row['record_id']]
        assert row['case_id']==r['case_id'] and row['split']==r['split']
        key=row['record_id'],row['method'],row['k'],row['replicate']
        keys.append(key)
        method_key='paired_br' if row['method'].startswith('lane_br') else row['method']
        expected_seed=int(rng_from_key(row['record_id'],row['k'],row['replicate'],method_key,
                                      schema='lane-study-row-v1').integers(0,2**31))
        assert row['rng_seed']==expected_seed
        public=row['public']
        assert set(public)=={'mechanism','output_kind','events','public_parameters'}
        assert len(public['events'])==len(r['points'])
        for e,x in zip(public['events'],r['points']):
            assert set(e)=={'event_id','timestamp_s','candidates'} and e['timestamp_s']==x['timestamp_s']
            assert len(e['candidates'])==(1 if row['method']=='unprotected' else row['k'])
            assert all(set(c)=={'candidate_id','lat','lon'} for c in e['candidates'])
        estimates=attacks(public,rn,prior,r['scenario'])
        truth=np.array([rn.point_xy(x['lat'],x['lon']) for x in r['points']])
        assert estimates.keys()==row['errors_by_attack'].keys()
        for name,value in estimates.items():
            close(row['errors_by_attack'][name],np.linalg.norm(value-truth,axis=1))
        distinct=[len({(c['lat'],c['lon']) for c in e['candidates']}) for e in public['events']]
        assert distinct==row['distinct_coordinates_per_event']
        if row['method'] in ('dls','enhanced_dls','uniform_sets'):
            assert all(k==row['k'] for k in distinct)
        assert row['request_json_bytes']==len(json.dumps(public['events'],separators=(',',':')).encode())
        u=row['utility']
        assert len(u['poi_rows'])==6*len(r['points'])
        recalls=[]
        complete=[]
        extra=[]
        for q in u['poi_rows']:
            if q['reference']:
                value=len(set(q['reference'])&set(q['returned']))/len(q['reference'])
                close(q['recall'],value)
                recalls.append(value)
                flag=len(q['returned'])==len(q['reference'])
                assert flag==q['complete']
                complete.append(flag)
                if flag:
                    assert q['extra_distance_m']>=0
                    extra.append(q['extra_distance_m'])
        close(u['poi_recall_at_5'],np.mean(recalls))
        close(u['poi_complete_rate'],np.mean(complete))
        close(u['poi_extra_distance_m'],np.mean(extra))
        assert u['poi_evaluable_n']==len(recalls) and u['poi_extra_distance_n']==len(extra)
        if row['method'].startswith('lane_br'):
            states=row['evaluator_states']
            assert len(states)==len(public['events'])
            for ids,e in zip(states,public['events']):
                assert len(ids)==row['k']
                assert [rn.latlon(i) for i in ids]==[(c['lat'],c['lon']) for c in e['candidates']]
            for t in range(1,len(states)):
                dt=public['events'][t]['timestamp_s']-public['events'][t-1]['timestamp_s']
                for a,b in zip(states[t-1],states[t]):
                    reached=nx.single_source_dijkstra_path_length(rn.graph,a,cutoff=dt+1e-8,
                                                                  weight=lambda i,j,z:z['length']/z['speed'])
                    assert b in reached
                    transitions+=1
            close(row['spent_bound'],.01+.02*(min(12,len(states))-1))
        else:
            assert row['evaluator_states'] is None and row['spent_bound'] is None
        if replay:
            points=tuple(TrajectoryPoint(**x) for x in r['points'])
            regenerated=release(row['method'],points,rn,prior,row['k'],row['rng_seed'],sites)
            for field in ('public','evaluator_states','spent_bound'):
                assert regenerated[field]==row[field], (key,field)
            assert utility_metrics(service,public,[(x.lat,x.lon) for x in points])==u
            short=release(row['method'],points[:3],rn,prior,row['k'],row['rng_seed'],sites)
            assert short['public']['events']==public['events'][:3]
            prefix_checks+=1
        events+=len(public['events'])
        queries+=len(u['poi_rows'])
        if (n+1)%24==0:
            print(f'{n+1}/144 rows verified',flush=True)
    expected={(rid,m,k,rep) for rid in records for m in METHODS for k in (3,5) for rep in (1,2)}
    assert len(set(keys))==144 and set(keys)==expected
    for s in p['summaries']:
        group=[r for r in p['rows'] if (r['method'],r['k'],r['case_id'])==(s['method'],s['k'],s['case_id'])]
        val=[r for r in group if r['split']=='development_validation']
        test=[r for r in group if r['split']=='development_test']
        selection=p['attacker_selection'][f"{s['method']}/{s['k']}/{s['case_id']}"]
        maes={a:float(np.mean([np.mean(r['errors_by_attack'][a]) for r in val])) for a in val[0]['errors_by_attack']}
        hits={a:float(np.mean([np.mean(np.array(r['errors_by_attack'][a])<=100) for r in val])) for a in maes}
        assert selection['validation_mae']==maes and selection['validation_hit100']==hits
        assert selection['mae_attack']==min(maes,key=lambda a:(maes[a],a))
        assert selection['hit_attack']==max(hits,key=lambda a:(hits[a],a))
        close(s['mae_m'],np.mean([np.mean(r['errors_by_attack'][selection['mae_attack']]) for r in test]))
        close(s['hit100'],np.mean([np.mean(np.array(r['errors_by_attack'][selection['hit_attack']])<=100) for r in test]))
        close(s['poi_recall'],np.mean([r['utility']['poi_recall_at_5'] for r in test]))
        close(s['poi_complete'],np.mean([r['utility']['poi_complete_rate'] for r in test]))
        close(s['generation_ms_per_event'],np.mean([r['generation_ms']/len(r['public']['events']) for r in test]))
        close(s['mean_distinct_coordinates'],np.mean([np.mean(r['distinct_coordinates_per_event']) for r in test]))
        assert s['trips']==1 and s['rng_replicates']==2
    audit_path=path.parent/'audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        assert audit['source_sha256']==sha(path)
        assert audit['summarizer_sha256']==sha(ROOT/'experiments/summarize_lane_comparison.py')
        assert audit['interpretation']=='exploratory_report_family_attack_envelope_NOT_held_out_selection'
        assert len(audit['rows'])==36
        for a,s in zip(audit['rows'],p['summaries']):
            assert all(a[key]==value for key,value in s.items())
            group=[r for r in p['rows'] if (r['method'],r['k'],r['case_id'],r['split'])==
                   (s['method'],s['k'],s['case_id'],'development_test')]
            maes={name:np.mean([np.mean(r['errors_by_attack'][name]) for r in group]) for name in group[0]['errors_by_attack']}
            hits={name:np.mean([np.mean(np.array(r['errors_by_attack'][name])<=100) for r in group]) for name in maes}
            close(a['audit_min_mae_m'],min(maes.values()))
            close(a['audit_max_hit100'],max(hits.values()))
            assert a['audit_mae_attack']==min(maes,key=lambda n:(maes[n],n))
            assert a['audit_hit_attack']==max(hits,key=lambda n:(hits[n],n))
        for field in ('nearest_lane_error_m','nearest_junction_error_m'):
            values=[v for r in records.values() for v in r[field]]
            expected={'n':len(values),'mean_m':np.mean(values),'p95_m':np.percentile(values,95),'max_m':max(values)}
            for key,value in expected.items():
                close(audit['projection'][field][key],value)
    print(json.dumps({'verified':True,'rows':len(keys),'events':events,'POI_queries':queries,
                      'native_connections':connections,'BR_transitions':transitions,
                      'prefix_checks':prefix_checks,'replay':replay,'independent_POI_sources':len(oracle_points)},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'artifacts/benchmarks/lane_comparison/results.json')
    parser.add_argument('--replay',action='store_true')
    args=parser.parse_args()
    verify(args.results,args.replay)
