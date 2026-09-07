"""Two-stage, source-pinned belief-lane diagnostic on SUMO suite v2."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.belief_lane import BeliefLaneDummy
from benchmark.engines.contextual_lane import ContextualLaneDummy
from benchmark.public_poi_context import PublicPoiContext
from core.demo_protocol import TrajectoryPoint
from data.lane_states import build_lane_states, catalogue_summary
from data.scenario_suite.records import device_view
from evaluation.lane_travel import LanePoiService
from evaluation.research_protocol import utility_metrics
from evaluation.scenario_metrics import read_osm_pois
from experiments.run_contextual_lane import sha, choose_configuration, estimators
from experiments.run_lane_comparison import ROOT, spatial_prior
from experiments.rng_util import rng_from_key

DATA=ROOT/'artifacts/datasets/urban_scenarios_v2/dataset.json'
PREVIOUS=ROOT/'artifacts/benchmarks/contextual_lane/results.json'
OUTPUT=ROOT/'artifacts/benchmarks/belief_suite'
CASES=tuple(f'S{s}.{c}' for s in (1,2,3) for c in 'ABC')
METHODS={'baseline':(False,0.,'anchor'),'point6':(False,6.,'anchor'),
         'belief6':(True,6.,'anchor'),'belief24':(True,24.,'anchor'),
         'belief24_center':(True,24.,'belief_mean')}


def sources(previous,data):
    paths=set(previous['source_sha256']) | set(data['source_sha256']) | {
        'benchmark/anchor_belief.py','benchmark/engines/belief_lane.py',
        'experiments/run_belief_suite.py','thesis/notes/belief_evaluation_protocol.md'}
    return {p:sha(ROOT/p) for p in sorted(paths)}


def prepare():
    d=json.loads(DATA.read_text()); previous=json.loads(PREVIOUS.read_text())
    assert sha(DATA)==DATA.with_suffix('.sha256').read_text().strip()
    for p,h in {**d['source_sha256'],**previous['source_sha256']}.items():
        assert sha(ROOT/p)==h,p
    verified=json.loads((DATA.parent/'verification.json').read_text())
    assert verified['dataset_sha256']==sha(DATA)
    assert verified['raw_fcd_points_compared']==d['summary']['raw_fcd_samples']
    assert sha(ROOT/previous['network_path'])==previous['network_sha256']
    assert sha(ROOT/previous['service']['osm_path'])==previous['service']['osm_sha256']
    start=time.perf_counter(); rn=build_lane_states(ROOT/previous['network_path'])
    graph_ms=(time.perf_counter()-start)*1000
    service=LanePoiService(rn,read_osm_pois(ROOT/previous['service']['osm_path'],tuple(d['network']['bbox_lon_lat'])))
    start=time.perf_counter()
    context=PublicPoiContext(service,ROOT/'cache/contextual_lane_v1/public_poi.npz')
    context_ms=(time.perf_counter()-start)*1000
    ids=sorted(s['session_id'] for f in d['families'] if f['split']=='development_train' for s in f['sessions'])
    start=time.perf_counter(); prior=spatial_prior(rn,[d['traces'][s] for s in ids])
    prior_ms=(time.perf_counter()-start)*1000
    cache=ROOT/'cache/belief_suite_v1/model.npz'; warm=cache.exists()
    start=time.perf_counter()
    belief=PublicAnchorModel(rn,context,prior,cache_path=cache)
    belief_ms=(time.perf_counter()-start)*1000
    return d,previous,rn,service,context,prior,belief,{
        'graph_ms':graph_ms,'context_ms':context_ms,'prior_ms':prior_ms,
        'belief_ms':belief_ms,'belief_cache_warm':warm,'latent_states':len(belief.xy)}


def records_for(data,phase):
    split='development_validation' if phase=='validation' else 'confirmation'
    result=[]
    for r in data['records']:
        if r['split']!=split or r['case_id'] not in CASES: continue
        stream=list(device_view(r,data['traces'])); retained=stream[:12]
        item={k:r[k] for k in ('record_id','case_id','scenario','split','family_id')}
        item.update(available_events=len(stream),retained_events=len(retained),
                    points=[{'timestamp_s':x['time_s'],'lat':x['lat'],'lon':x['lon']} for x in retained])
        if r['case_id']=='S2.C':
            idx=r['observed_indices'][0][:12]
            item['retained_visits']=[sum(a<=i<=b for i in idx) for a,b in r['labels']['stop_intervals']]
            assert item['retained_visits']==[9,3]
        result.append(item)
    assert len(result)==18 and all(sum(r['case_id']==c for r in result)==2 for c in CASES)
    return result


def generate(name,points,rn,context,belief,k,seed):
    use_belief,weight,center=METHODS[name]
    kwargs=dict(coverage_weight=weight,budget=.24,horizon=12,k=k,theta_m=200,offset_m=80,
                rng=rng_from_key(seed,schema='lane-comparison-v1'))
    start=time.perf_counter()
    model=(BeliefLaneDummy(rn,belief_model=belief,center_mode=center,**kwargs) if use_belief else
           ContextualLaneDummy(rn,context=context,**kwargs))
    init_ms=(time.perf_counter()-start)*1000
    start=time.perf_counter(); run=model.protect_run(points)
    elapsed=(time.perf_counter()-start)*1000
    return {'public':run.to_attacker_dict(),'evaluator_states':model.evaluator_states,
            'evaluator_anchors':model.evaluator_anchors,'spent_bound':model.spent_bound,
            'evaluator_belief':getattr(model,'evaluator_belief',None),
            'init_ms':init_ms,'generation_ms':elapsed,'step_ms':model.step_ms}


def summarize(rows):
    result=[]
    for name in METHODS:
        for k in (3,5):
            for case in CASES:
                group=[r for r in rows if (r['method'],r['k'],r['case_id'])==(name,k,case)]
                assert len(group)==2 and len({r['family_id'] for r in group})==2
                errors={a:float(np.mean([np.mean(r['errors_by_attack'][a]) for r in group])) for a in group[0]['errors_by_attack']}
                hits={a:float(np.mean([np.mean(np.asarray(r['errors_by_attack'][a])<=100) for r in group])) for a in errors}
                result.append({'method':name,'k':k,'case_id':case,'families':2,'rng_replicates_per_family':1,
                    'mae_by_attack':errors,'hit_by_attack':hits,'audit_min_mae_m':min(errors.values()),
                    'audit_max_hit100':max(hits.values()),
                    'poi_recall':float(np.mean([r['utility']['poi_recall_at_5'] for r in group])),
                    'poi_complete':float(np.mean([r['utility']['poi_complete_rate'] for r in group])),
                    'generation_mean_ms':float(np.mean([np.mean(r['step_ms']) for r in group]))})
    return result


def select(summaries):
    result={}
    for k in (3,5):
        candidates=[]
        for name in METHODS:
            s=[r for r in summaries if (r['method'],r['k'])==(name,k)]
            candidates.append({'method':name,'min_case_recall':min(r['poi_recall'] for r in s),
                               'macro_hit100':float(np.mean([r['audit_max_hit100'] for r in s]))})
        result[str(k)]=choose_configuration(candidates)
    return result


def run(phase,output=OUTPUT):
    output=Path(output); target=output/f'{phase}.json'
    if target.exists(): raise FileExistsError('Preserve frozen results; use a new output directory')
    print(f'Preparing public context for {phase}',flush=True)
    data,previous,rn,service,context,prior,belief,timings=prepare()
    hashes=sources(previous,data)
    selection=None; selection_path=output/'selection.json'
    if phase=='confirmation':
        selection=json.loads(selection_path.read_text())
        assert sha(selection_path)==selection_path.with_suffix('.sha256').read_text().strip()
        assert selection['source_sha256']==hashes and selection['dataset_sha256']==sha(DATA)
        assert selection['validation_sha256']==sha(output/'validation.json')
        assert selection['belief_model_sha256']==belief.sha256
    records=records_for(data,phase); rows=[]
    print(f'Public belief: {len(belief.xy)} states; {timings}',flush=True)
    for record in records:
        points=tuple(TrajectoryPoint(**p) for p in record['points'])
        truth=np.array([rn.point_xy(p.lat,p.lon) for p in points])
        for k in (3,5):
            seed=int(rng_from_key(record['record_id'],k,1,schema='belief-suite-row-v1').integers(0,2**31))
            paired=[]
            for name in METHODS:
                generated=generate(name,points,rn,context,belief,k,seed)
                predictions=estimators(generated['public'],rn,prior,record['scenario'])
                errors={a:np.linalg.norm(v-truth,axis=1).tolist() for a,v in predictions.items()}
                utility=utility_metrics(service,generated['public'],[(p.lat,p.lon) for p in points])
                rows.append({**{key:record[key] for key in ('record_id','case_id','split','family_id')},
                    'method':name,'k':k,'replicate':1,'rng_seed':seed,**generated,
                    'errors_by_attack':errors,'utility':utility})
                paired.append(generated['evaluator_anchors'])
            assert all(a==paired[0] for a in paired)
        print(f'Completed {record["record_id"]} {record["case_id"]}',flush=True)
    summaries=summarize(rows)
    if selection:
        for s in summaries:
            key=f'{s["method"]}/{s["k"]}/{s["case_id"]}'
            a=selection['attackers'][key]
            s.update(selected_mae_attack=a['mae'],selected_hit_attack=a['hit'],
                     selected_mae_m=s['mae_by_attack'][a['mae']],selected_hit100=s['hit_by_attack'][a['hit']])
    payload={'schema':'belief-suite-v1','phase':phase,'scope':'nine_case_within_city_diagnostic',
        'dataset_path':str(DATA.relative_to(ROOT)),'dataset_sha256':sha(DATA),
        'previous_results_sha256':sha(PREVIOUS),'network_path':previous['network_path'],
        'network_sha256':previous['network_sha256'],'catalogue':catalogue_summary(rn),
        'service':previous['service'],'context_sha256':context.sha256,'belief_model_sha256':belief.sha256,
        'preparation':timings,'method_grid':METHODS,'records':records,'rows':rows,
        'summaries':summaries,'source_sha256':hashes,
        'selection_sha256':sha(selection_path) if selection else None}
    output.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
    target.with_suffix('.sha256').write_text(sha(target)+'\n')
    if phase=='validation':
        attackers={f'{s["method"]}/{s["k"]}/{s["case_id"]}':{
            'mae':min(s['mae_by_attack'],key=lambda a:(s['mae_by_attack'][a],a)),
            'hit':min(s['hit_by_attack'],key=lambda a:(-s['hit_by_attack'][a],a))} for s in summaries}
        selection={'schema':'belief-selection-v1','dataset_sha256':sha(DATA),
            'validation_sha256':sha(target),'source_sha256':hashes,'belief_model_sha256':belief.sha256,
            'method_selection':select(summaries),'attackers':attackers}
        if selection_path.exists(): raise FileExistsError('Selection already exists')
        selection_path.write_text(json.dumps(selection,indent=2)+'\n')
        selection_path.with_suffix('.sha256').write_text(sha(selection_path)+'\n')
    print(f'{len(rows)} rows saved to {target}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('validation','confirmation'),required=True)
    p.add_argument('--output',type=Path,default=OUTPUT)
    a=p.parse_args(); run(a.phase,a.output)
