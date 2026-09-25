"""Counterfactual full request/response checks on the actual public POI service."""
import hashlib
import json
from pathlib import Path

from benchmark.scheduled_category_client import ScheduledCategoryClient
from evaluation.live_poi import RankedRoadPois,LivePointService,AvailabilityWorld
from experiments.endpoint_calendar_study import OUT,plan_for,protocol
from experiments.research_loop_resources import CACHE,load,sha
from experiments.run_live_paper_comparison import write_json


def main():
    p=protocol();rn,service,*_=load();ranking=RankedRoadPois(service,CACHE/'live_poi_full_rank_v1.npy');checks=[]
    for m in ('calendar30','calendar67'):
        plan=plan_for(m);states={tuple(q['coordinate']):rn.nearest(*q['coordinate'])[0] for q in plan['queries']}
        for seed in p['world_seeds']:
            server=LivePointService(ranking,AvailabilityWorld(ranking.n,seed,.8,60),10)
            bodies=[]
            for reads in ([310,350,470,590],[915,1010,1559,2301],[]):
                client=ScheduledCategoryClient(plan,ranking.n);transcript=[]
                for t in range(0,3600,60):
                    transcript.append(client.tick(t,lambda q:server.query(states[q['coordinate']],q['epoch'])[q['category_index']]))
                    for local_time in reads:
                        if t<=local_time<t+60:client.local_snapshot(local_time)
                body=json.dumps(transcript,sort_keys=True,separators=(',',':')).encode();bodies.append(body)
            assert bodies[0]==bodies[1]==bodies[2]
            checks.append({'method':m,'world_seed':seed,'private_activity_variants':3,
                           'all_request_response_bytes_equal':True,'sha256':hashlib.sha256(bodies[0]).hexdigest()})
    write_json(OUT/'calendar_flow_checks.json',{'status':'passed','protocol_sha256':sha(OUT/'protocol.json'),
        'source_sha256':{'benchmark/scheduled_category_client.py':sha(Path('benchmark/scheduled_category_client.py')),
                        str(Path(__file__).resolve().relative_to(Path.cwd())):sha(Path(__file__))},
        'checks':checks,'scope':'Application payloads within a pre-subscribed public hour, same exogenous server state; not IP, account, activation or packet timing under faults'})
    print('Six actual-service checks: early trip, late trip and no trip have identical full payloads',flush=True)


if __name__=='__main__':main()
