"""Sequential local-host timings, separate from concurrent experiment jobs."""
import platform

import numpy as np

from core.demo_protocol import TrajectoryPoint
from experiments.run_coverage_frontier import OUTPUT,METHODS,BUDGETS,prepare,generate
from experiments.run_service_cover import ROOT,read,write,sha


def run():
    # Requiring every verifier first prevents profiling against our own workers.
    for b in BUDGETS: assert read(OUTPUT/f'b{b:.2f}'/'verification.json')['verified']
    rows=[]
    for budget in BUDGETS:
        _,rn,_,belief,_,_,_=prepare(budget)
        stored=read(OUTPUT/f'b{budget:.2f}'/'validation.json')
        records=[r for r in stored['records'] if r['family_id']=='family-103' and r['case_id'] in ('S1.A','S3.A','S3.C')]
        assert len(records)==3
        for record in records:
            points=tuple(TrajectoryPoint(**p) for p in record['points'])
            for method in METHODS:
                parent=next(r for r in stored['rows'] if r['record_id']==record['record_id'] and r['method']==method and r['replicate']==1)
                actual=generate(method,points,rn,belief,budget,parent['rng_seed'])
                assert actual['public']==parent['public']
                rows.append({'budget':budget,'method':method,'record_id':record['record_id'],
                    'case_id':record['case_id'],'events':len(points),'init_ms':actual['init_ms'],
                    'step_ms':actual['step_ms'],'public_matches':True})
                print(f'Profile {budget} {method} {record["case_id"]}',flush=True)
    summaries=[]
    for budget in BUDGETS:
        for method in METHODS:
            group=[r for r in rows if (r['budget'],r['method'])==(budget,method)]
            steps=np.concatenate([r['step_ms'] for r in group])
            summaries.append(dict(budget=budget,method=method,events=len(steps),
                step_mean_ms=float(np.mean(steps)),step_p95_ms=float(np.percentile(steps,95)),
                init_mean_ms=float(np.mean([r['init_ms'] for r in group]))))
    write(OUTPUT/'timing.json',dict(schema='coverage-frontier-sequential-timing-v1',
        platform=platform.platform(),machine=platform.machine(),python=platform.python_version(),
        scope='36 sequential runs; 3 fixed family-103 records; local host not mobile benchmark',
        source_sha256=sha(__file__),amendment_sha256=sha(ROOT/'thesis/notes/coverage_frontier_timing_amendment.md'),
        rows=rows,summaries=summaries))


if __name__=='__main__':run()
