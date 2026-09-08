"""Independent raw-row aggregation and finite-grid dominance audit."""
from collections import defaultdict
import argparse
import math

import numpy as np

from experiments.run_coverage_frontier import OUTPUT,BUDGETS,METHODS,DEPTHS
from experiments.run_service_cover import ROOT,read,write,sha


def average(values):
    values=list(values)
    assert values and all(math.isfinite(v) for v in values)
    return math.fsum(values)/len(values)


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)


def verify(check_only=False):
    result=read(OUTPUT/'readout.json');timing=read(OUTPUT/'timing.json')
    assert result['timing_sha256']==sha(OUTPUT/'timing.json')
    assert timing['source_sha256']==sha(ROOT/'experiments/profile_coverage_frontier.py')
    assert len(result['rows'])==24 and len(result['family_deltas'])==72 and len(result['cases'])==216
    counts=defaultdict(int)
    for budget in BUDGETS:
        directory=OUTPUT/f'b{budget:.2f}'
        data=read(directory/'development.json');validation=read(directory/'validation.json')
        selection=read(directory/'selection.json');receipt=read(directory/'verification.json')
        assert receipt['verified'] and receipt['verifier_sha256']==sha(ROOT/'experiments/verify_coverage_frontier.py')
        for phase,h in receipt['phase_sha256'].items():assert h==sha(directory/f'{phase}.json')
        assert result['sources'][str(budget)]=={
            'development_sha256':sha(directory/'development.json'),
            'verification_sha256':sha(directory/'verification.json')}
        assert result['selection'][str(budget)]==selection['method_selection_by_depth']
        for method in METHODS:
            rows=[r for r in data['rows'] if r['method']==method]
            by_case=defaultdict(list)
            for r in rows:by_case[r['case_id']].append(r)
            assert len(by_case)==9 and all(len(g)==12 for g in by_case.values())
            selected_hit={};selected_mae={};envelope_hit={};envelope_mae={}
            for case,g in by_case.items():
                attacks=g[0]['errors_by_attack']
                hits={a:average(average(int(e<=100) for e in r['errors_by_attack'][a]) for r in g) for a in attacks}
                maes={a:average(average(r['errors_by_attack'][a]) for r in g) for a in attacks}
                choice=selection['attackers'][method+'/'+case]
                selected_hit[case]=hits[choice['hit']];selected_mae[case]=maes[choice['mae']]
                envelope_hit[case]=max(hits.values());envelope_mae[case]=min(maes.values())
            for depth in DEPTHS:
                key=str(depth)
                row=next(r for r in result['rows'] if (r['budget'],r['method'],r['depth'])==(budget,method,depth))
                recall={case:average(r['utility'][key]['poi_recall_at_5'] for r in g) for case,g in by_case.items()}
                close(row['recall'],average(recall.values()));close(row['min_case_recall'],min(recall.values()))
                for name,values in [('selected_hit100',selected_hit),('selected_mae_m',selected_mae),
                                    ('envelope_hit100',envelope_hit),('envelope_mae_m',envelope_mae)]:
                    close(row[name],average(values.values()))
                for name,metric in [('complete','poi_complete_rate'),('requests_per_event','requests_per_event'),
                    ('reply_items','reply_items_per_event'),('response_id_bytes','response_id_bytes_per_event')]:
                    close(row[name],average(average(r['utility'][key][metric] for r in g) for g in by_case.values()))
                assert row['requests_per_event']==30 and row['k']==5 and row['families']==4 and row['cases']==9
                vc=defaultdict(list)
                for r in validation['rows']:
                    if r['method']==method:vc[r['case_id']].append(r['utility'][key]['poi_recall_at_5'])
                val_min=min(average(v) for v in vc.values())
                close(row['validation_min_recall'],val_min)
                assert row['validation_feasible']==(round(val_min,12)>=.9)
                assert row['development_feasible']==(round(min(recall.values()),12)>=.9)
                for case,g in by_case.items():
                    case_row=next(c for c in result['cases'] if
                        (c['budget'],c['method'],c['depth'],c['case_id'])==(budget,method,depth,case))
                    close(case_row['recall'],recall[case]);close(case_row['selected_hit100'],selected_hit[case])
                    close(case_row['envelope_hit100'],envelope_hit[case])
                    for category,score in case_row['categories'].items():
                        close(score,average(average(q['recall'] for q in r['utility'][key]['poi_rows']
                            if q['category']==category and q['recall'] is not None) for r in g
                            if any(q['category']==category and q['recall'] is not None for q in r['utility'][key]['poi_rows'])))
                    counts['case_rows']+=1
                for category,score in row['categories'].items():
                    close(score,average(c['categories'][category] for c in result['cases']
                        if (c['budget'],c['method'],c['depth'])==(budget,method,depth) and category in c['categories']))
                profile=[p for p in timing['rows'] if (p['budget'],p['method'])==(budget,method)]
                assert len(profile)==3 and all(p['public_matches'] for p in profile)
                steps=[s for p in profile for s in p['step_ms']]
                close(row['profile_step_mean_ms'],average(steps));close(row['profile_step_p95_ms'],float(np.percentile(steps,95)))
                counts['readout_rows']+=1
        for delta in result['family_deltas']:
            if delta['budget']!=budget:continue
            means={}
            for method in ('mean_greedy',delta['method']):
                g=[r for r in data['rows'] if (r['method'],r['family_id'])==(method,delta['family'])]
                assert len(g)==27
                means[method]=(average(r['utility'][str(delta['depth'])]['poi_recall_at_5'] for r in g),
                    average(average(int(e<=100) for e in r['errors_by_attack'][selection['attackers'][method+'/'+r['case_id']]['hit']]) for r in g))
            close(delta['recall_delta'],means[delta['method']][0]-means['mean_greedy'][0])
            close(delta['selected_hit_delta'],means[delta['method']][1]-means['mean_greedy'][1])
            counts['family_deltas']+=1
    # A separately expressed pairwise relation, not the exporting helper.
    for field,metrics in [('frontier_ids',('response_id_bytes',)),
                          ('timing_frontier_ids',('response_id_bytes','profile_step_mean_ms'))]:
        survivors=[]
        for candidate in result['rows']:
            def values(r):return [-round(r['recall'],12),round(r['selected_hit100'],12)]+[round(r[k],12) for k in metrics]
            a=values(candidate);dominated=False
            for rival in result['rows']:
                if rival['budget']!=candidate['budget']:continue
                b=values(rival)
                if all(x<=y for x,y in zip(b,a)) and any(x<y for x,y in zip(b,a)):
                    dominated=True;break
            if not dominated:survivors.append(candidate['id'])
        assert set(survivors)==set(result[field]);counts['frontiers']+=1
    receipt={'verified':True,'counts':dict(counts),'readout_sha256':sha(OUTPUT/'readout.json'),
        'tables_sha256':sha(OUTPUT/'tables.tex'),'timing_sha256':sha(OUTPUT/'timing.json'),
        'verifier_sha256':sha(__file__),'exporter_sha256':sha(ROOT/'experiments/export_coverage_frontier.py')}
    if not check_only:write(OUTPUT/'readout_verification.json',receipt)
    print(receipt,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    verify(parser.parse_args().check_only)
