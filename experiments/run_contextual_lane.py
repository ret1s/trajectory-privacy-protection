"""Fixed 2x2 development ablation; preserves the preceding lane-study evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from benchmark.engines.contextual_lane import ContextualLaneDummy
from benchmark.public_poi_context import PublicPoiContext
from core.demo_protocol import TrajectoryPoint
from data.lane_states import build_lane_states, catalogue_summary
from data.scenario_suite.records import device_view
from evaluation.lane_travel import LanePoiService
from evaluation.research_protocol import utility_metrics
from evaluation.scenario_metrics import read_osm_pois
from experiments.run_lane_comparison import ROOT, DATA, attacks, spatial_prior
from experiments.rng_util import rng_from_key

PREVIOUS = ROOT/'artifacts/benchmarks/lane_comparison/results.json'
CONTEXT_CACHE = ROOT/'cache/contextual_lane_v1/public_poi.npz'
CASES = ('S1.A', 'S2.B', 'S3.A', 'S3.B', 'S3.C')
METHODS = {'baseline': (0., 0.), 'route': (.5, 0.), 'coverage': (0., 6.), 'combined': (.5, 6.)}


def choose_configuration(candidates):
    """Fix decimal decision precision; 1 ulp is not scientific evidence."""
    eligible = [c for c in candidates if round(c['min_case_recall'],12) >= .90]
    quality = lambda c: (round(c['macro_hit100'],12), c['method'])
    fallback = lambda c: (-round(c['min_case_recall'],12), *quality(c))
    choice = min(eligible,key=quality) if eligible else min(candidates,key=fallback)
    return {'chosen':choice,'utility_feasible':bool(eligible),'candidates':candidates}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    data = json.loads(DATA.read_text())
    previous = json.loads(PREVIOUS.read_text())
    assert sha(DATA) == previous['dataset_sha256']
    rn = build_lane_states(ROOT/previous['network_path'])
    service = LanePoiService(rn, read_osm_pois(ROOT/previous['service']['osm_path'], tuple(data['network']['bbox_lon_lat'])))
    started = time.perf_counter()
    context = PublicPoiContext(service, CONTEXT_CACHE)
    context_ms = (time.perf_counter() - started)*1000
    train_ids = sorted(s['session_id'] for f in data['families'] if f['seed']==91 for s in f['sessions'])
    prior = spatial_prior(rn, [data['traces'][sid] for sid in train_ids])
    records = []
    for r in data['records']:
        if r['case_id'] in CASES and r['split'] != 'development_train':
            stream = list(device_view(r, data['traces']))
            records.append({**{k:r[k] for k in ('record_id','case_id','scenario','split','family_id')},
                            'available_events':len(stream), 'retained_events':min(12,len(stream)),
                            'points':[{'timestamp_s':x['time_s'],'lat':x['lat'],'lon':x['lon']} for x in stream[:12]]})
    return data, previous, rn, service, context, context_ms, prior, records


def generate(name, points, rn, context, k, seed):
    started = time.perf_counter()
    route, cover = METHODS[name]
    model = ContextualLaneDummy(rn, context=context, route_weight=route, coverage_weight=cover,
                                budget=.24, horizon=12, k=k, theta_m=200, offset_m=80,
                                rng=rng_from_key(seed, schema='lane-comparison-v1'))
    init_ms = (time.perf_counter()-started)*1000
    started = time.perf_counter()
    run = model.protect_run(points)
    elapsed = (time.perf_counter()-started)*1000
    return {'public':run.to_attacker_dict(), 'evaluator_states':model.evaluator_states,
            'evaluator_anchors':model.evaluator_anchors, 'spent_bound':model.spent_bound,
            'init_ms':init_ms, 'generation_ms':elapsed, 'step_ms':model.step_ms}


def estimators(public, rn, prior, scenario):
    result = attacks(public, rn, prior, scenario)
    for j in range(len(public['events'][0]['candidates'])):
        trajectory = np.array([rn.point_xy(e['candidates'][j]['lat'],e['candidates'][j]['lon']) for e in public['events']])
        result[f'track_{j}'] = trajectory
        if scenario != 'S1':
            result[f'track_mean_{j}'] = np.tile(trajectory.mean(axis=0), (len(trajectory),1))
    return result


def summarize(rows):
    summaries = []
    for split in ('development_validation', 'development_test'):
        for name in METHODS:
            for k in (3,5):
                for case in CASES:
                    group = [r for r in rows if (r['split'],r['method'],r['k'],r['case_id'])==(split,name,k,case)]
                    if not group:
                        raise ValueError(f'Missing prespecified group {split}/{name}/{k}/{case}')
                    errors = {a:float(np.mean([np.mean(r['errors_by_attack'][a]) for r in group])) for a in group[0]['errors_by_attack']}
                    hits = {a:float(np.mean([np.mean(np.asarray(r['errors_by_attack'][a])<=100) for r in group])) for a in errors}
                    summaries.append({'split':split,'method':name,'k':k,'case_id':case,
                                      'audit_min_mae_m':min(errors.values()),'audit_max_hit100':max(hits.values()),
                                      'mae_by_attack':errors,'hit_by_attack':hits,
                                      'poi_recall':float(np.mean([r['utility']['poi_recall_at_5'] for r in group])),
                                      'poi_complete':float(np.mean([r['utility']['poi_complete_rate'] for r in group])),
                                      'generation_mean_ms':float(np.mean([np.mean(r['step_ms']) for r in group])),
                                      'generation_p95_ms':float(np.mean([np.percentile(r['step_ms'],95) for r in group])),
                                      'rng_replicates':len(group),'families':len({r['family_id'] for r in group})})
    selected = {}
    for k in (3,5):
        candidates = []
        for name in METHODS:
            s = [r for r in summaries if (r['split'],r['method'],r['k'])==('development_validation',name,k)]
            candidates.append({'method':name,'min_case_recall':min(r['poi_recall'] for r in s),
                               'macro_hit100':float(np.mean([r['audit_max_hit100'] for r in s]))})
        selected[str(k)] = choose_configuration(candidates)
    # Preserve the distinction between validation-selected attacks and exploratory
    # reporting envelopes. Never describe the latter as held-out selection.
    for s in summaries:
        if s['split'] == 'development_test':
            val = next(v for v in summaries if (v['split'],v['method'],v['k'],v['case_id'])==('development_validation',s['method'],s['k'],s['case_id']))
            ma = min(val['mae_by_attack'],key=lambda a:(val['mae_by_attack'][a],a))
            hi = max(val['hit_by_attack'],key=lambda a:(val['hit_by_attack'][a],a))
            s.update(selected_mae_attack=ma,selected_hit_attack=hi,
                     selected_mae_m=s['mae_by_attack'][ma],selected_hit100=s['hit_by_attack'][hi])
    return summaries, selected


def run(output):
    output = Path(output)
    if (output/'results.json').exists():
        raise FileExistsError('Preserve results: use a fresh output directory')
    print('Preparing fixed public lane and POI context',flush=True)
    data, previous, rn, service, context, context_ms, prior, records = prepare()
    print(f'Public context ready: {context.sha256}; {len(records)} input records',flush=True)
    rows = []
    for record in records:
        points = tuple(TrajectoryPoint(**x) for x in record['points'])
        truth = np.array([rn.point_xy(p.lat,p.lon) for p in points])
        for k in (3,5):
            for replicate in (1,2):
                seed = int(rng_from_key(record['record_id'],k,replicate,'paired_br',schema='lane-study-row-v1').integers(0,2**31))
                for name in METHODS:
                    generated = generate(name,points,rn,context,k,seed)
                    predictions = estimators(generated['public'],rn,prior,record['scenario'])
                    errors = {a:np.linalg.norm(v-truth,axis=1).tolist() for a,v in predictions.items()}
                    utility = utility_metrics(service,generated['public'],[(x.lat,x.lon) for x in points])
                    rows.append({**{key:record[key] for key in ('record_id','case_id','split','family_id')},
                                 'method':name,'k':k,'replicate':replicate,'rng_seed':seed,**generated,
                                 'errors_by_attack':errors,'utility':utility,
                                 'distinct_coordinates':[len({(c['lat'],c['lon']) for c in e['candidates']}) for e in generated['public']['events']]})
        print(f"Completed {record['record_id']} {record['case_id']} {record['split']}",flush=True)
    summaries, selected = summarize(rows)
    sources = sorted(set(previous['source_sha256']) | {'experiments/run_contextual_lane.py',
                     'experiments/run_lane_comparison.py','benchmark/engines/contextual_lane.py',
                     'benchmark/public_poi_context.py','thesis/notes/contextual_lane_protocol.md'})
    payload = {'schema':'contextual-lane-v1','scope':'five_case_development_ablation_not_final_test',
               'dataset_sha256':sha(DATA),'previous_results_sha256':sha(PREVIOUS),
               'network_path':previous['network_path'],'network_sha256':previous['network_sha256'],
               'catalogue':catalogue_summary(rn),'service':previous['service'],
               'context_sha256':context.sha256,'context_prepare_ms':context_ms,
               'context_bytes':context.signatures.nbytes+context.access.nbytes,
               'method_grid':METHODS,'records':records,'rows':rows,
               'summaries':summaries,'method_selection':selected,
               'source_sha256':{name:sha(ROOT/name) for name in sources}}
    output.mkdir(parents=True,exist_ok=True)
    target = output/'results.json'
    target.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')))
    (output/'results.sha256').write_text(sha(target)+'\n')
    print(f'{len(rows)} rows saved to {target}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/benchmarks/contextual_lane')
    run(p.parse_args().output)
