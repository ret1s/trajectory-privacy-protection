"""Development comparison: richer road states, raw inputs and diverse controls.

NOT an all-scenario leaderboard or a paper-equivalent DL reproduction. This
cycle uses frozen S1.A/S2.B/S3.A fixtures, H=12, K=3/5, two RNG replicates;
seed91 trains the public spatial prior, seed92 selects attacks, seed93 reports.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from benchmark.engines.dls import DLSGraph
from benchmark.engines.enhanced_dls import EnhancedDLSGraph
from benchmark.engines.lane_budgeted import LaneBudgetedDummy
from core.demo_protocol import TrajectoryPoint, PublicCandidate, PublicEvent, PublicTranscript, ProtectedRun, EvaluationTruth, OutputKind, make_replacement_run
from data.lane_states import build_lane_states, catalogue_summary, coordinate_catalogue
from data.scenario_suite.records import device_view
from data.sumo_demo import _existing_default_osm, load_sumo_road_network
from evaluation.lane_travel import LanePoiService
from evaluation.scenario_metrics import read_osm_pois, public_features, continuity_estimates
from evaluation.research_protocol import utility_metrics
from experiments.rng_util import rng_from_key

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'artifacts/datasets/urban_scenarios_v1/dataset.json'
METHODS=('unprotected','uniform_sets','dls','enhanced_dls','lane_br_projected','lane_br_raw')
CASES=('S1.A','S2.B','S3.A')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spatial_prior(rn,training,bandwidth=250.):
    """Public Gaussian-mixture prior using only training-family 20 s samples.

    Equal sample weights, bandwidth fixed before comparisons, positive floor.
    This is a spatial occupancy approximation, not paper-native query counts.
    """
    xy=np.array([rn.point_xy(p['lat'],p['lon']) for trace in training for p in trace[::20]])
    q=np.empty(len(rn))
    for start in range(0,len(rn),256):
        dist2=((rn.xy[start:start+256,None,:]-xy[None,:,:])**2).sum(axis=2)
        q[start:start+256]=np.exp(logsumexp(-dist2/(2*bandwidth**2),axis=1)-np.log(len(xy)))+1e-9
    return q/q.sum()


def instantiate(name,rn,prior,k,seed):
    rng=rng_from_key(seed,schema='lane-comparison-v1')
    if name=='dls':
        return DLSGraph(rn,prior,k=k,rng=rng)
    if name=='enhanced_dls':
        return EnhancedDLSGraph(rn,prior,k=k,rng=rng)
    if name.startswith('lane_br'):
        return LaneBudgetedDummy(rn,budget=.24,horizon=12,k=k,theta_m=200,offset_m=80,rng=rng)
    return None


def release(name,points,rn,prior,k,seed,sites=None):
    # Paired BR variants use the SAME random stream; their only difference
    # is preprocessing raw inputs versus snapping to the new lane catalogue.
    start=time.perf_counter()
    sites=coordinate_catalogue(rn) if sites is None else sites
    selection_rn=sites if name in ('dls','enhanced_dls') else rn
    selection_prior=prior[sites.state_indices] if name in ('dls','enhanced_dls') else prior
    model=instantiate(name,selection_rn,selection_prior,k,seed)
    init_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter()
    inputs=tuple(TrajectoryPoint(p.timestamp_s,*rn.latlon(rn.nearest(p.lat,p.lon)[0])) for p in points) if name=='lane_br_projected' else points
    if model:
        run=model.protect_run(inputs)
    elif name=='unprotected':
        run=make_replacement_run(name,points,points)
    else:
        rng=rng_from_key(seed,schema='lane-comparison-uniform-v1')
        events,real_ids,representations=[],[],[]
        for j,p in enumerate(points):
            true=sites.nearest(p.lat,p.lon)[0]
            others=rng.choice(np.delete(np.arange(len(sites)),true),k-1,replace=False)
            ids=rng.permutation(np.r_[true,others])
            cs=tuple(PublicCandidate(f'e{j}_c{i}',*sites.latlon(int(v))) for i,v in enumerate(ids))
            events.append(PublicEvent(f'e{j}',p.timestamp_s,cs))
            real_ids.append(cs[list(ids).index(true)].candidate_id)
            representations.append(TrajectoryPoint(p.timestamp_s,*sites.latlon(true)))
        run=ProtectedRun(PublicTranscript(name,OutputKind.REAL_PLUS_DUMMIES,tuple(events),{'k':k,'linkage':'event_local'}),
                         EvaluationTruth(points,tuple(real_ids),tuple(representations)))
    params=dict(run.transcript.public_parameters)
    if name in ('uniform_sets','dls','enhanced_dls'):
        params.update(selection_unit='unique_exact_public_coordinates',selection_catalogue_size=len(sites))
    if name=='lane_br_projected':
        params['input_representation']='nearest_lane_state_ablation'
    run=replace(run,transcript=replace(run.transcript,public_parameters=params))
    elapsed=(time.perf_counter()-start)*1000
    return {'public':run.to_attacker_dict(),
            'evaluator_states':model.evaluator_states if isinstance(model,LaneBudgetedDummy) else None,
            'spent_bound':model.spent_bound if isinstance(model,LaneBudgetedDummy) else None,
            'init_ms':init_ms,'generation_ms':elapsed}


def attacks(public,rn,prior,scenario):
    center=public_features(public,rn,'S1')
    choices={'centroid':center,'continuity':continuity_estimates(public,rn),
             'population_mean':np.tile(np.average(rn.xy,axis=0,weights=prior),(len(center),1))}
    if scenario!='S1':
        choices['full_mean']=np.tile(center.mean(axis=0),(len(center),1))
        choices['running_mean']=np.cumsum(center,axis=0)/np.arange(1,len(center)+1)[:,None]
    if public['output_kind']=='real_plus_dummies':
        sets=[{rn.nearest(c['lat'],c['lon'])[0] for c in e['candidates']} for e in public['events']]
        choices['candidate_prior']=np.array([rn.xy[max(sorted(ids),key=lambda i:prior[i])] for ids in sets])
        if scenario=='S2':
            common=set.intersection(*sets)
            choices['stationary_intersection']=np.tile(rn.xy[max(sorted(common),key=lambda i:prior[i])],(len(center),1)) if common else center.copy()
    return choices


def run(output,replace=False):
    output=Path(output)
    if (output/'results.json').exists() and not replace:
        raise FileExistsError('Use a fresh output directory or explicit --replace')
    output.mkdir(parents=True,exist_ok=True)
    dataset=json.loads(DATA.read_text())
    assert sha(DATA)==DATA.with_suffix('.sha256').read_text().strip()
    network=ROOT/dataset['network']['path']
    print('Building shared lane-state catalogue',flush=True)
    rn=build_lane_states(network)
    sites=coordinate_catalogue(rn)
    old=load_sumo_road_network(network)
    train_ids={s['session_id'] for f in dataset['families'] if f['seed']==91 for s in f['sessions']}
    prior=spatial_prior(rn,[dataset['traces'][sid] for sid in sorted(train_ids)])
    osm=_existing_default_osm()
    service=LanePoiService(rn,read_osm_pois(osm,tuple(dataset['network']['bbox_lon_lat'])))
    records=[]
    for r in dataset['records']:
        if r['case_id'] in CASES and r['split']!='development_train':
            stream=list(device_view(r,dataset['traces']))[:12]
            points=tuple(TrajectoryPoint(p['time_s'],p['lat'],p['lon']) for p in stream)
            records.append({'record_id':r['record_id'],'case_id':r['case_id'],'scenario':r['scenario'],
                            'family_id':r['family_id'],'split':r['split'],'points':[p.to_dict() for p in points],
                            'retained_events':len(points),'available_events':len(r['observed_indices'][0]),
                            'nearest_lane_error_m':[rn.nearest(p.lat,p.lon)[1] for p in points],
                            'nearest_junction_error_m':[old.nearest(p.lat,p.lon)[1] for p in points]})
    rows=[]
    for r in records:
        points=tuple(TrajectoryPoint(**p) for p in r['points'])
        truth_xy=np.array([rn.point_xy(p.lat,p.lon) for p in points])
        for k in (3,5):
            for replicate in (1,2):
                for name in METHODS:
                    seed=int(rng_from_key(r['record_id'],k,replicate,name if not name.startswith('lane_br') else 'paired_br',schema='lane-study-row-v1').integers(0,2**31))
                    generated=release(name,points,rn,prior,k,seed,sites)
                    estimates=attacks(generated['public'],rn,prior,r['scenario'])
                    errors={n:np.linalg.norm(v-truth_xy,axis=1).tolist() for n,v in estimates.items()}
                    utility=utility_metrics(service,generated['public'],[(p.lat,p.lon) for p in points])
                    unique=[len({(c['lat'],c['lon']) for c in e['candidates']}) for e in generated['public']['events']]
                    row={'record_id':r['record_id'],'case_id':r['case_id'],'split':r['split'],'method':name,'k':k,
                         'replicate':replicate,'rng_seed':seed,**generated,'errors_by_attack':errors,
                         'utility':utility,'distinct_coordinates_per_event':unique,
                         'request_json_bytes':len(json.dumps(generated['public']['events'],separators=(',',':')).encode())}
                    rows.append(row)
        print(r['record_id'],r['case_id'],r['split'],'completed',flush=True)
    selected={}
    for name in METHODS:
        for k in (3,5):
            for case in CASES:
                val=[r for r in rows if r['method']==name and r['k']==k and r['case_id']==case and r['split']=='development_validation']
                names=val[0]['errors_by_attack']
                maes={n:float(np.mean([np.mean(v['errors_by_attack'][n]) for v in val])) for n in names}
                hits={n:float(np.mean([np.mean(np.array(v['errors_by_attack'][n])<=100) for v in val])) for n in names}
                selected[f'{name}/{k}/{case}']={'mae_attack':min(maes,key=lambda n:(maes[n],n)),
                    'hit_attack':max(hits,key=lambda n:(hits[n],n)),'validation_mae':maes,'validation_hit100':hits}
    summaries=[]
    for name in METHODS:
        for k in (3,5):
            for case in CASES:
                test=[r for r in rows if r['method']==name and r['k']==k and r['case_id']==case and r['split']=='development_test']
                selection=selected[f'{name}/{k}/{case}']
                summaries.append({'method':name,'k':k,'case_id':case,'trips':1,'rng_replicates':2,
                    'mae_m':float(np.mean([np.mean(r['errors_by_attack'][selection['mae_attack']]) for r in test])),
                    'hit100':float(np.mean([np.mean(np.array(r['errors_by_attack'][selection['hit_attack']])<=100) for r in test])),
                    'poi_recall':float(np.mean([r['utility']['poi_recall_at_5'] for r in test])),
                    'poi_complete':float(np.mean([r['utility']['poi_complete_rate'] for r in test])),
                    'generation_ms_per_event':float(np.mean([r['generation_ms']/len(r['public']['events']) for r in test])),
                    'mean_distinct_coordinates':float(np.mean([np.mean(r['distinct_coordinates_per_event']) for r in test]))})
    files=[Path(__file__),ROOT/'data/lane_states.py',ROOT/'data/sumo_demo.py',ROOT/'core/road_network.py',ROOT/'core/mechanisms.py',
           ROOT/'core/demo_protocol.py',ROOT/'benchmark/engines/budgeted.py',ROOT/'benchmark/engines/lane_budgeted.py',
           ROOT/'benchmark/engines/dls.py',ROOT/'benchmark/engines/enhanced_dls.py',ROOT/'evaluation/lane_travel.py',
           ROOT/'evaluation/research_protocol.py',ROOT/'evaluation/scenario_metrics.py',ROOT/'experiments/rng_util.py',
           ROOT/'data/scenario_suite/records.py']
    payload={'schema':'lane-comparison-v1','scope':'development_S1A_S2B_S3A_not_final_benchmark',
             'dataset_sha256':sha(DATA),'network_path':str(network.relative_to(ROOT)),'network_sha256':sha(network),
             'catalogue':{**catalogue_summary(rn),'distinct_coordinates':len(sites)},
             'prior':{'source_family':91,'bandwidth_m':250,'floor':1e-9,'sampling_stride':20},
             'service':{'osm_path':str(Path(osm).relative_to(ROOT)),'osm_sha256':sha(osm),
                        'categories':service.categories,'pois_used':service.pois,'pois_excluded':service.excluded,
                        'contract':'directed_nearest_lane_state_access_POI_top5_all_six_categories'},
             'records':records,'rows':rows,'attacker_selection':selected,'summaries':summaries,
             'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files}}
    target=output/'results.json'
    target.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
    (output/'results.sha256').write_text(sha(target)+'\n')
    print(f'{len(rows)} rows, {len(summaries)} summaries: {target}',flush=True)
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/benchmarks/lane_comparison')
    p.add_argument('--replace',action='store_true')
    args=p.parse_args()
    run(args.output,args.replace)
