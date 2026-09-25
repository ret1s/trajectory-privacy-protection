"""Cost of calendar cover versus an equally cached on-demand public plan.

Accounting-only companion after evaluation: no model/attacker selection or
defense-parameter change. Both clients use the already frozen plans/epochs.
"""
import json
from pathlib import Path
import numpy as np

from benchmark.category_client import PublicCategoryClient
from evaluation.live_poi import RankedRoadPois,LivePointService,AvailabilityWorld,score_returned
from experiments.endpoint_calendar_study import OUT,DATA,plan_for,protocol,scheduled_service
from experiments.research_loop_resources import ROOT,CACHE,load,sha
from experiments.research_loop_category_confirmation import sources
from experiments.run_live_paper_comparison import write_json,encoded


def main():
    p=protocol();rn,service,*_=load();ranking=RankedRoadPois(service,CACHE/'live_poi_full_rank_v1.npy')
    data=json.loads((DATA/'dataset.json').read_text());ss,_=sources(data,rn)
    prior=json.loads((OUT/'holdout_service.json').read_text());phases=prior['phases_evaluator_only'];rows=[];checked=0
    for method in ('calendar30','calendar67'):
        plan=plan_for(method);states={tuple(q['coordinate']):rn.nearest(*q['coordinate'])[0] for q in plan['queries']}
        for seed in p['world_seeds']:
            server=LivePointService(ranking,AvailabilityWorld(ranking.n,seed,.8,60),10)
            for source in ss:
                sid=source['session_id'];phase=phases[sid];client=PublicCategoryClient(plan,ranking.n,refresh_once_per_epoch=True)
                request_bytes=response_bytes=emissions=0
                expected,calendar=scheduled_service(method,source,server,ranking,rn,phase)
                for i,t in enumerate(source['timestamps_s']):
                    now=phase+t;reply=client.step(now,lambda q:server.query(states[q['coordinate']],q['epoch'])[q['category_index']])
                    if reply['requests']:
                        queries=[[q['category_index'],*q['coordinate']] for q in reply['requests']]
                        request_bytes+=encoded({'time_s':now,'epoch':reply['epoch'],'queries':queries})
                        response_bytes+=encoded({'epoch':reply['epoch'],'results':reply['replies']});emissions+=1
                    state=source['reference_states'][i];available=server.world.at_epoch(reply['epoch'])
                    actual=score_returned(ranking.top(state,available,5),ranking.top(state,reply['known'],5),available)['recall']
                    target=expected[source['clock_indices'][i]]['recall']
                    assert actual is None and target is None or np.isclose(actual,target)
                    checked+=1
                rows.append({'method':method,'world_seed':seed,'session_id':sid,'family_id':source['family_id'],
                    'service_events':len(source['timestamps_s']),'active_epoch':{'request_bytes':request_bytes,'response_bytes':response_bytes,'emissions':emissions},
                    'public_calendar':{k:calendar[k] for k in ('request_bytes','response_bytes','emissions')}})
        print('Ablation verified',method,flush=True)
    write_json(OUT/'calendar_ablation.json',{'rows':rows,'service_events_equal_utility_checked':checked,
        'protocol_sha256':sha(OUT/'protocol.json'),'service_sha256':sha(OUT/'holdout_service.json'),
        'code_sha256':sha(Path(__file__)),'public_client_sha256':sha(ROOT/'benchmark/category_client.py'),
        'scope':'post-evaluation accounting check, unchanged frozen plan; on-demand epoch refresh still discloses activity bounds'})


if __name__=='__main__':main()
