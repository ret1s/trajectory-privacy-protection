"""Paired city benchmark of an exact selector reduction, development GPS only."""
import json,time
from pathlib import Path
import numpy as np
from benchmark.engines.fair_cover import FairCoverLaneDummy
from benchmark.engines.quotient_cover import QuotientCoverLaneDummy
from core.demo_protocol import TrajectoryPoint
from experiments.research_loop_resources import load,ROOT,sha
from experiments.rng_util import rng_from_key
OUT=ROOT/'artifacts/benchmarks/research_loop/iteration01_quotient.json'

def main():
    if OUT.exists():raise FileExistsError('Use a new iteration, preserve completed evidence')
    rn,service,context,belief,metadata=load()
    data=json.loads((ROOT/'artifacts/datasets/urban_fresh_v2/dataset.json').read_text())
    cases={f'S{s}.{c}' for s in (1,2,3,9,10) for c in 'ABC'}
    records=[r for r in data['records'] if r['case_id'] in cases and r['family_id'] in ('family-301','family-302')]
    models={};init={}
    for name,cls in [('exchange',FairCoverLaneDummy),('quotient',QuotientCoverLaneDummy)]:
        t=time.perf_counter();models[name]=cls(rn,belief_model=belief,k=5,budget=.24,horizon=12,
            category_cap=None,max_exchanges=3,rng=np.random.default_rng(0));init[name]=(time.perf_counter()-t)*1000
    rows=[]
    for ri,r in enumerate(records):
        # This measures computation and exactness, not scenario endpoint protection.
        sid=r['session_ids'][0];indices=r['observed_indices'][0][:12]
        points=tuple(TrajectoryPoint(data['traces'][sid][i]['time_s'],data['traces'][sid][i]['lat'],data['traces'][sid][i]['lon']) for i in indices)
        for rep in range(2):
            generated={}
            for name in (('exchange','quotient') if (ri+rep)%2==0 else ('quotient','exchange')):
                model=models[name]
                # Paired independent streams; resetting always starts a new source record.
                seeds=rng_from_key(r['record_id'],rep,schema='quotient-city-v1').integers(0,2**63,size=2,dtype=np.int64)
                model.anchor_rng,model.dummy_rng=(np.random.default_rng(int(s)) for s in seeds)
                started=time.perf_counter();run=model.protect_run(points)
                generated[name]={'events':run.to_attacker_dict()['events'],'states':model.evaluator_states,
                    'anchors':model.evaluator_anchors,'spent_bound':model.spent_bound,
                    'step_ms':model.step_ms,'generation_ms':(time.perf_counter()-started)*1000,
                    'counts':[{'full':x['reachable_counts'],'reduced':x.get('quotient_counts',x['reachable_counts'])} for x in model.evaluator_objective]}
            for key in ('events','states','anchors','spent_bound'):
                assert generated['exchange'][key]==generated['quotient'][key],(r['record_id'],rep,key)
            rows.append({'record_id':r['record_id'],'family_id':r['family_id'],'case_id':r['case_id'],'replicate':rep,'events':len(points),
                'exact_public_match':True,'methods':{m:{k:v for k,v in v.items() if k not in ('events','states','anchors')} for m,v in generated.items()}})
        print('Verified',r['record_id'],r['case_id'],len(rows),flush=True)
    summaries={}
    for name in models:
        timings=[t for r in rows for t in r['methods'][name]['step_ms']]
        summaries[name]={'steps':len(timings),'step_mean_ms':float(np.mean(timings)),
            'step_median_ms':float(np.median(timings)),'step_p95_ms':float(np.percentile(timings,95)),
            'initialization_ms':init[name],
            'total_generation_ms':sum(r['methods'][name]['generation_ms'] for r in rows)}
    result={'schema':'quotient-city-v1','scope':'computation_only; historical_development_GPS_on_new_OSM; not endpoint privacy confirmation',
        'resources':metadata,'source_sha256':{p:sha(ROOT/p) for p in ('benchmark/engines/quotient_cover.py','benchmark/engines/fair_cover.py','experiments/research_loop_quotient.py','artifacts/datasets/urban_fresh_v2/dataset.json')},
        'rows':rows,'summaries':summaries,'all_transcripts_identical':True}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
