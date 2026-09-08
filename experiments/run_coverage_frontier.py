"""Prespecified development frontier; no historical artifact or defender overwritten."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import sklearn

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.contextual_lane import ContextualLaneDummy
from benchmark.engines.service_cover import ServiceCoverLaneDummy
from benchmark.engines.fair_cover import FairCoverLaneDummy
from benchmark.public_poi_context import PublicPoiContext
from core.demo_protocol import TrajectoryPoint
from evaluation.expanded_shadow import fit_trees, predict as expanded_predict, TREE_PARAMS
from evaluation.loss_aware_shadow import predict as loss_predict
from evaluation.service_shadow import features, fit, predict as knn_predict
from evaluation.retrieval_frontier import evaluate_retrieval
from evaluation.lane_travel import LanePoiService
from experiments.run_contextual_lane import estimators
from experiments.run_expanded_shadow import setup as auxiliary_setup
from experiments.run_prior_factors import OUTPUT as PARENT
from experiments.run_service_cover import read, write, sha, ROOT
from experiments.rng_util import rng_from_key

OUTPUT = ROOT/'artifacts/benchmarks/coverage_frontier'
METHODS = ('geometric','mean_greedy','mean_exchange','capped_exchange')
BUDGETS = (.12,.24,.48)
DEPTHS = (5,10)
PATHS = ('benchmark/engines/fair_cover.py','evaluation/retrieval_frontier.py',
         'experiments/run_coverage_frontier.py','thesis/notes/coverage_frontier_protocol.md',
         'thesis/notes/coverage_frontier_execution_amendment.md',
         'thesis/notes/coverage_frontier_timing_amendment.md')


def recover_unsealed(path):
    """Preserve an interrupted write that never received its commit hash."""
    if path.exists() and not path.with_suffix('.sha256').exists():
        destination=path.with_name(path.stem+f'.interrupted_{time.time_ns()}.json')
        path.rename(destination)
        print(f'Preserved incomplete write at {destination}',flush=True)


def prepare(budget):
    if budget not in BUDGETS:
        raise ValueError('Budget is not in the declared grid')
    auxiliary,rn,prior,models,provenance=auxiliary_setup('training')
    original=models['uniform_initial_learned_motion'].initial
    context=original.context
    flat=np.zeros(len(rn))
    # Distribute equal mass over public cells, not lane sampling density.
    from experiments.diagnose_service_cover import uniform_cell_prior
    flat=uniform_cell_prior(rn,original.spacing_m)
    belief=PublicAnchorModel(rn,context,flat,epsilon_release=budget/24,
        epsilon_test=budget/24,theta_m=200.,
        cache_path=ROOT/f'cache/coverage_frontier/b{budget:.2f}_belief.npz')
    if budget==.24:
        assert belief.sha256==original.sha256
    service=LanePoiService(rn,list(context.pois),k=5)
    deeper=PublicPoiContext(LanePoiService(rn,list(context.pois),k=10),
        ROOT/'cache/coverage_frontier/poi10.npz')
    assert np.array_equal(deeper.signatures[:,:,:5],context.signatures)
    provenance.update(source_sha256={p:sha(ROOT/p) for p in sorted(set(provenance['source_sha256'])|set(PATHS))},
        frontier_belief_sha256=belief.sha256,reply_context_sha256=deeper.sha256,
        budget=budget,methods=METHODS,reply_depths=DEPTHS,k=5,horizon=12,
        core_phase_sha256={p:sha(PARENT/f'{p}.json') for p in ('training','validation','development')})
    assert sklearn.__version__=='1.7.1'
    for r in auxiliary:
        r['scenario']='S3'  # not a fitting feature; AUX only enters training
    return auxiliary,rn,prior,belief,service,deeper,provenance


def make_model(method,rn,belief,budget,seed):
    kwargs=dict(budget=budget,horizon=12,k=5,theta_m=200,
                rng=rng_from_key(seed,schema='lane-comparison-v1'))
    if method=='geometric':
        model=ContextualLaneDummy(rn,context=belief.context,**kwargs)
    elif method=='mean_greedy':
        model=ServiceCoverLaneDummy(rn,belief_model=belief,**kwargs)
    elif method in ('mean_exchange','capped_exchange'):
        model=FairCoverLaneDummy(rn,belief_model=belief,
            category_cap=.9 if method=='capped_exchange' else None,max_exchanges=3,**kwargs)
    else:
        raise ValueError(method)
    return model


def generate(method,points,rn,belief,budget,seed,pool=None):
    started=time.perf_counter()
    if pool is None:
        model=make_model(method,rn,belief,budget,seed)
    else:
        model=pool[method]
        # Same two streams as BudgetedReachableDummy.__init__. Only immutable
        # map/SCC/travel/filter-model resources are reused; protect_run resets
        # every session-dependent state before observing the new trajectory.
        seeds=rng_from_key(seed,schema='lane-comparison-v1').integers(0,2**63,size=2,dtype=np.int64)
        model.anchor_rng,model.dummy_rng=(np.random.default_rng(int(s)) for s in seeds)
    init_ms=(time.perf_counter()-started)*1000
    started=time.perf_counter(); run=model.protect_run(points)
    return {'public':run.to_attacker_dict(),'evaluator_states':model.evaluator_states,
        'evaluator_anchors':model.evaluator_anchors,
        'evaluator_objective':getattr(model,'evaluator_objective',[]),
        'spent_bound':model.spent_bound,'step_ms':model.step_ms,'init_ms':init_ms,
        'generation_ms':(time.perf_counter()-started)*1000}


def compact_utility(result):
    # Public outputs + pinned service suffice for independent exact reconstruction.
    return {depth:{**value,'poi_rows':[{k:v for k,v in r.items()
        if k not in ('reference','returned','replies')} for r in value['poi_rows']]}
        for depth,value in result.items()}


def summarize(rows,selection=None):
    groups=defaultdict(list)
    for r in rows: groups[r['method'],r['case_id']].append(r)
    summaries=[]
    for (method,case),group in sorted(groups.items()):
        attacks=sorted(group[0]['errors_by_attack'])
        assert all(sorted(r['errors_by_attack'])==attacks for r in group)
        mae={a:float(np.mean([np.mean(r['errors_by_attack'][a]) for r in group])) for a in attacks}
        hit={a:float(np.mean([np.mean(np.asarray(r['errors_by_attack'][a])<=100) for r in group])) for a in attacks}
        s={'method':method,'case_id':case,'rows':len(group),
           'families':len({r['family_id'] for r in group}), 'mae_by_attack':mae,'hit_by_attack':hit,
           'envelope_mae_m':min(mae.values()),'envelope_hit100':max(hit.values()),
           'step_mean_ms':float(np.mean([np.mean(r['step_ms']) for r in group])),
           'step_p95_ms':float(np.percentile(np.concatenate([r['step_ms'] for r in group]),95)),
           'utility':{}}
        for depth in map(str,DEPTHS):
            values=[r['utility'][depth] for r in group]
            s['utility'][depth]={k:float(np.mean([v[k] for v in values])) for k in (
                'poi_recall_at_5','poi_complete_rate','requests_per_event','reply_items_per_event','response_id_bytes_per_event')}
            categories=sorted({q['category'] for v in values for q in v['poi_rows']})
            s['utility'][depth]['by_category']={c:float(np.mean([
                np.mean([q['recall'] for q in v['poi_rows'] if q['category']==c and q['recall'] is not None])
                for v in values if any(q['category']==c and q['recall'] is not None for q in v['poi_rows'])]))
                for c in categories if any(q['category']==c and q['recall'] is not None for v in values for q in v['poi_rows'])}
        if selection:
            choice=selection['attackers'][method+'/'+case]
            s.update(selected_mae_attack=choice['mae'],selected_hit_attack=choice['hit'],
                     selected_mae_m=mae[choice['mae']],selected_hit100=hit[choice['hit']])
        summaries.append(s)
    return summaries


def select(summaries):
    attackers={s['method']+'/'+s['case_id']:{
        'mae':min(s['mae_by_attack'],key=lambda a:(s['mae_by_attack'][a],a)),
        'hit':min(s['hit_by_attack'],key=lambda a:(-s['hit_by_attack'][a],a))} for s in summaries}
    choices={}
    for depth in map(str,DEPTHS):
        candidates=[]
        for name in METHODS:
            group=[s for s in summaries if s['method']==name]
            candidates.append({'method':name,'min_case_recall':min(s['utility'][depth]['poi_recall_at_5'] for s in group),
                'macro_hit100':float(np.mean([s['envelope_hit100'] for s in group]))})
        feasible=[c for c in candidates if round(c['min_case_recall'],12)>=.90]
        choices[depth]={'candidates':candidates,'utility_feasible':bool(feasible),
            'chosen':min(feasible,key=lambda c:(round(c['macro_hit100'],12),c['method'])) if feasible else None}
    return {'attackers':attackers,'method_selection_by_depth':choices}


def run(budget,output=OUTPUT,resume=False):
    output=Path(output)/f'b{budget:.2f}'
    if output.exists() and not resume: raise FileExistsError('Use --resume for exact pinned continuation or a fresh destination')
    if resume:
        for phase in ('training','validation','selection','development'):
            recover_unsealed(output/f'{phase}.json')
    if (output/'development.json').exists(): raise FileExistsError('Completed evidence stays immutable')
    auxiliary,rn,prior,belief,service,deeper,provenance=prepare(budget)
    print(f'Prepared B={budget:.2f}: {len(auxiliary)} auxiliary windows',flush=True)
    output.mkdir(parents=True,exist_ok=True)
    runtime_start=time.perf_counter()
    pool={m:make_model(m,rn,belief,budget,0) for m in METHODS}
    provenance['public_runtime_build_ms']=(time.perf_counter()-runtime_start)*1000
    provenance['initialization_mode']='public_resources_reused_per_worker; per_run_init_ms_is_rng_reseed_only'
    fingerprint=hashlib.sha256(json.dumps({k:v for k,v in provenance.items() if not k.endswith('_ms')},
                                         sort_keys=True).encode()).hexdigest()
    checkpoints=ROOT/'cache/coverage_frontier_checkpoints'/fingerprint
    checkpoints.mkdir(parents=True,exist_ok=True)
    shadow_models={}; forests={}; selection=None
    for phase in ('training','validation','development'):
        parent=read(PARENT/f'{phase}.json')
        records=parent['records']+(auxiliary if phase=='training' else [])
        if (output/f'{phase}.json').exists():
            saved=read(output/f'{phase}.json')
            for k,v in provenance.items():
                if not k.endswith('_ms'): assert saved[k]==json.loads(json.dumps(v)),k
            assert saved['records']==records
            if phase=='training':
                shadow_models=saved['shadow_models']
                for method,info in saved['forests'].items():
                    path=output/info['path'];assert sha(path)==info['sha256']
                    with np.load(path,allow_pickle=False) as arrays:
                        forests[method]={k:arrays[k] for k in arrays.files}
            if phase=='validation':
                if not (output/'selection.json').exists():
                    write(output/'selection.json',{**provenance,**select(saved['summaries']),
                        'training_sha256':sha(output/'training.json'),'validation_sha256':sha(output/'validation.json')})
                selection=read(output/'selection.json')
            print(f'Resumed sealed {phase}',flush=True)
            continue
        rows=[]
        for index,record in enumerate(records):
            checkpoint=checkpoints/f'{phase}_{index:04d}.json'
            recover_unsealed(checkpoint)
            if checkpoint.exists():
                cached=read(checkpoint)
                assert cached['record']==record and cached['fingerprint']==fingerprint
                rows.extend(cached['rows'])
                if index%25==0:print(f'Resumed {phase} record {index+1}',flush=True)
                continue
            first_row=len(rows)
            points=tuple(TrajectoryPoint(**p) for p in record['points'])
            truth=np.array([rn.point_xy(p.lat,p.lon) for p in points])
            aux=record['case_id'].startswith('AUX')
            for rep in ((1,) if aux else (1,2,3)):
                seed=int((rng_from_key(record['record_id'],schema='expanded-shadow-v1') if aux else
                          rng_from_key(record['record_id'],5,rep,schema='service-cover-row-v1')).integers(0,2**31))
                anchors=None
                for method in METHODS:
                    generated=generate(method,points,rn,belief,budget,seed,pool=pool)
                    assert anchors is None or anchors==generated['evaluator_anchors']
                    anchors=generated['evaluator_anchors']
                    row={**{k:record[k] for k in ('record_id','family_id','case_id','split')},
                         'method':method,'k':5,'replicate':rep,'rng_seed':seed,**generated}
                    if phase!='training':
                        x=features(row['public'],rn)
                        predictions=estimators(row['public'],rn,prior,record['scenario'])
                        small=shadow_models[method]['core']
                        predictions.update({f'shadow_knn_{k}':knn_predict(small,x,k) for k in (1,5,15)})
                        predictions.update(loss_predict(small,x)[0])
                        predictions.update(expanded_predict(shadow_models[method]['expanded'],forests[method],x))
                        row['predictions']={a:v.tolist() for a,v in predictions.items()}
                        row['errors_by_attack']={a:np.linalg.norm(v-truth,axis=1).tolist() for a,v in predictions.items()}
                        row['utility']=compact_utility(evaluate_retrieval(service,deeper,row['public'],
                                                       [(p.lat,p.lon) for p in points]))
                    rows.append(row)
            write(checkpoint,{'fingerprint':fingerprint,'record':record,'rows':rows[first_row:]})
            if index%10==0 or index==len(records)-1:
                print(f'B={budget:.2f} {phase}: {index+1}/{len(records)} records',flush=True)
        payload={'schema':'coverage-frontier-v1','scope':'reused_development_not_confirmation',
            'phase':phase,**provenance,'records':records,'rows':rows,
            'training_sha256':sha(output/'training.json') if phase!='training' else None,
            'selection_sha256':sha(output/'selection.json') if phase=='development' else None}
        if phase=='training':
            lookup={r['record_id']:r for r in records}
            forest_info={}
            for method in METHODS:
                shadow_models[method]={}
                for kind in ('core','expanded'):
                    group=[r for r in rows if r['method']==method and
                           (kind=='expanded' or not r['case_id'].startswith('AUX'))]
                    x=np.concatenate([features(r['public'],rn) for r in group])
                    y=np.concatenate([np.array([rn.point_xy(p['lat'],p['lon'])
                        for p in lookup[r['record_id']]['points']]) for r in group])
                    shadow_models[method][kind]=fit(x,y,{'families':sorted({r['family_id'] for r in group}),
                        'row_keys':[[r['record_id'],r['replicate']] for r in group],'holdout_used':False})
                    assert len(x)==(360 if kind=='core' else 3233)
                forests[method]=fit_trees(shadow_models[method]['expanded'])
                path=output/f'{method}_forests.npz'
                if path.exists():
                    with np.load(path,allow_pickle=False) as saved:
                        assert set(saved.files)==set(forests[method])
                        assert all(np.array_equal(saved[k],v) for k,v in forests[method].items())
                else:
                    with path.open('xb') as stream: np.savez_compressed(stream,**forests[method])
                forest_info[method]={'path':path.name,'sha256':sha(path)}
            payload.update(shadow_models=shadow_models,forests=forest_info)
        else:
            payload['summaries']=summarize(rows,selection)
        write(output/f'{phase}.json',payload)
        if phase=='validation':
            selection={**provenance,**select(payload['summaries']),
                'training_sha256':sha(output/'training.json'),'validation_sha256':sha(output/'validation.json')}
            write(output/'selection.json',selection)
            print(json.dumps(selection['method_selection_by_depth']),flush=True)
        print(f'Saved B={budget:.2f} {phase}: {len(rows)} runs',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--budget',type=float,choices=BUDGETS,required=True)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();run(args.budget,args.output,args.resume)
