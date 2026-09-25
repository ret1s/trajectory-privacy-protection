"""Run six named paper adaptations and the frozen proposed plans on one service.

CLI stages preserve source-locked evidence. Model/attack fit: 501/502; attack
selection: 601/602; evaluation: 701--712 and 901--904. Previously seen data is
not relabelled as a new independent confirmation. AnotherMe's full-trip adapter
is reported in an offline stratum and never ranked as an online mechanism.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import networkx as nx

from benchmark.paper_comparators import PublicHistory, shared_models, generate
from core.demo_protocol import TrajectoryPoint
from evaluation.live_poi import RankedRoadPois, LivePointService, AvailabilityWorld, EpochResponseCache, score_returned
from experiments.research_loop_category_confirmation import sources
from experiments.research_loop_cases import SCENARIOS, mean_optional
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.rng_util import rng_from_key

OUT=ROOT/'artifacts/benchmarks/live_paper_comparison_v1'
DATASETS={'auxiliary':'research_loop_development_v1','development':'research_loop_expanded_v1',
          'new_groups':'research_loop_confirmation_v1'}
METHODS=('raw','dls','rdg','transprotect_markov','semantic_poi','fake_queries','anotherme_offline','ours30','ours67')
LABELS={'raw':'Vị trí thật','dls':'DLS · road adaptation','rdg':'RDG · road adaptation',
 'transprotect_markov':'TransProtect · Markov adaptation','semantic_poi':'Semantic correlation · POI adaptation',
 'fake_queries':'Fake-query insertion · road adaptation','anotherme_offline':'AnotherMe · VTGA offline adaptation',
 'ours30':'Đề xuất: 30 query','ours67':'Đề xuất: 67 query'}
CODE=['benchmark/paper_comparators.py','experiments/run_live_paper_comparison.py','evaluation/live_poi.py',
 'benchmark/engines/dls.py','benchmark/engines/transprotect.py','benchmark/engines/anotherme.py',
 'benchmark/engines/semantic_correlation.py','benchmark/methods/transprotect.py','benchmark/methods/anotherme.py',
 'benchmark/methods/semantic_correlation.py','evaluation/live_comparison_attacks.py']


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def data_path(split):return ROOT/f'artifacts/datasets/{DATASETS[split]}/dataset.json'


def protocol():
    p=json.loads((OUT/'protocol.json').read_text())
    for path,digest in p['source_sha256'].items():assert sha(ROOT/path)==digest,path
    return p


def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Protocol already frozen')
    files=CODE+[str(data_path(s).relative_to(ROOT)) for s in DATASETS]+[
        'artifacts/benchmarks/research_loop/iteration29_plans.json',
        'cache/research_loop_20260924/resources.json']
    p={'schema':'common-live-paper-comparison-v1','date':'2026-09-25',
       'source_sha256':{x:sha(ROOT/x) for x in files},'methods':list(METHODS),'labels':LABELS,
       'datasets':DATASETS,'fit_families':['family-501','family-502'],
       'attack_selection_families':['family-601','family-602'],
       'evaluation_families':{'development':[f'family-{i}' for i in range(701,713)],'new_groups':[f'family-{i}' for i in range(901,905)]},
       'mechanism_repetitions':[0,1],'world_seeds':[3101,3102,3103],'probabilities':[.5,.8,.95],
       'response_L':10,'client_k':5,'epoch_seconds':60,'categories':'all six for every paper coordinate',
       'request_policy':'every service event; all methods may union valid responses in the same epoch; all fake traffic counted',
       'cost_policy':'actual compact UTF-8 JSON request and response bodies; public time/epoch included; no HTTP/TLS',
       'clock':'union of background 20-second cadence and exact required sample indices; public conditional clock',
       'input_domain':'all methods receive same nearest-lane-state coordinates; evaluator targets raw FCD',
       'dls':{'k':5,'subset_trials':50,'grid_prior_m':100,'prior':'background visits, Laplace cell smoothing, per-state area adjustment'},
       'rdg':{'k':5,'dummy_pool':20,'algorithm':'arXiv:1805.06104 Algorithm 6','transition_smoothing':.1},
       'transprotect':{'predictor':'disjoint SUMO empirical Markov; NOT published Transformer','candidate_k':10,'target_count':8,'alpha':10000,'epsilon_m_inv':.005},
       'semantic':{'predictor':'empirical previous POI type; NOT published LSTM weights','labels':'nearest public POI category','k':5,'candidate_pool':128},
       'fake':{'k':5,'max_fake':2,'timer_wait_seconds':[5,15],'candidate_N':24,'sigma':.75,
               'history':'disjoint population transitions instead of unavailable personal logs','geometry':'directed lane travel; length/direction similarity; DLS fallback',
               'failed_cover_policy':'suppress fake and retain failure count'},
       'anotherme':{'execution':'offline full-trip VTGA reference, excluded from online ranking','routing':'SUMO passenger graph for this vehicle dataset','minimum_raw_samples':20},
       'privacy':'finite common geometry/Viterbi/prior/learned attack bank; selection only on 601/602; not formal universal efficacy',
       'not_independent_confirmation':True,'scope':'Named local adaptations on one common service; no reproduced-paper superiority claim',
       'primary_sources':{'RDG':'https://arxiv.org/abs/1805.06104','fake_queries':'https://link.springer.com/article/10.1007/s44443-025-00438-z'}}
    write_json(OUT/'protocol.json',p)
    (OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    print('Frozen',OUT/'protocol.json',flush=True)


def resources():
    rn,service,*_=load()
    aux=json.loads(data_path('auxiliary').read_text())
    train_ids={s['session_id'] for f in aux['families'] if f['family_id'] in ('family-501','family-502') for s in f['sessions']}
    training=[]
    for sid in sorted(train_ids):
        trace=aux['traces'][sid]
        training.append([rn.nearest(p['lat'],p['lon'])[0] for p in trace[::20]])
    history=PublicHistory(rn,training)
    started=time.perf_counter()
    shared=shared_models(rn,history,training,service.pois)
    setup={'setup_ms':(time.perf_counter()-started)*1000,'training_sessions':sorted(train_ids),
           'training_points':sum(map(len,training)),'platform':platform.platform(),'python':platform.python_version()}
    return rn,service,history,shared,setup


def shard_path(split,method,sid,rep):return OUT/'transcripts'/split/method/f'{sid}_r{rep}.json.gz'


def generate_all(split,only=None,pilot=False):
    p=protocol();rn,service,history,shared,setup=resources()
    data=json.loads(data_path(split).read_text());ss,records=sources(data,rn)
    plans=json.loads((ROOT/'artifacts/benchmarks/research_loop/iteration29_plans.json').read_text())['plans']
    methods=[only] if only else METHODS
    if pilot:ss=ss[:1]
    for method in methods:
        for source in ss:
            sid=source['session_id'];trace=data['traces'][sid]
            ids=source['clock_indices']
            points=tuple(TrajectoryPoint(t,*rn.latlon(state)) for t,state in zip(source['timestamps_s'],source['reference_states']))
            for rep in ([0] if pilot else p['mechanism_repetitions']):
                path=shard_path(split,method,sid,rep)
                if path.exists() and not pilot:
                    old=json.loads(gzip.decompress(path.read_bytes()))
                    assert old['protocol_sha256']==sha(OUT/'protocol.json')
                    continue
                seed=int(rng_from_key(method,sid,rep,schema='live-paper-comparison-v1').integers(0,2**31))
                start=time.perf_counter()
                try:
                    if method.startswith('ours'):
                        plan=plans['public_category_budget30' if method=='ours30' else 'public_category_full_cover']
                        queries=[[q['category_index'],*q['coordinate']] for q in plan['queries']]
                        result={'events':[{'timestamp_s':t,'queries':queries} for t in source['timestamps_s']],
                                'service_event_positions':{i:i for i in range(len(ids))},'generation_ms':1000*(time.perf_counter()-start),'diagnostics':{}}
                    else:result=generate(method,points,rn,history,shared,seed)
                    # Cache server snap results outside mechanism timing; public coordinates only.
                    for e in result['events']:
                        coords=e.get('coordinates') or [q[1:] for q in e['queries']]
                        e['server_states']=[rn.nearest(*xy)[0] for xy in coords]
                    result.update(status='ok')
                except (ValueError,RuntimeError,IndexError,nx.NetworkXException) as exc:
                    result={'status':'failed','error':f'{type(exc).__name__}: {exc}',
                            'generation_ms':1000*(time.perf_counter()-start)}
                result.update(method=method,session_id=sid,family_id=source['family_id'],rep=rep,clock_indices=ids,
                              protocol_sha256=sha(OUT/'protocol.json'),seed=seed,setup=setup)
                if not pilot:
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(gzip.compress(json.dumps(result,separators=(',',':'),allow_nan=False).encode(),mtime=0))
                print(split,method,sid,rep,result['status'],round(result['generation_ms']), 'ms',result.get('diagnostics',result.get('error')),flush=True)


def encoded(value):return len(json.dumps(value,separators=(',',':'),ensure_ascii=False).encode())


def score_execution(ex,source,server,ranking):
    """Count every emission; private service-event map never enters attack input."""
    if ex['status']!='ok':return None
    cache=EpochResponseCache(ranking.n);utility={};counts=defaultdict(float)
    positions={int(v):int(k) for k,v in ex['service_event_positions'].items()}
    for j,e in enumerate(ex['events']):
        epoch=server.world.epoch(e['timestamp_s'])
        if 'queries' in e:
            partial=[server.query(s,epoch)[q[0]] for s,q in zip(e['server_states'],e['queries'])]
            replies=[[ids] for ids in partial]
            request={'time_s':e['timestamp_s'],'epoch':epoch,'queries':e['queries']}
            response={'epoch':epoch,'results':partial}
            query_count=len(partial)
        else:
            replies=[server.query(s,epoch) for s in e['server_states']]
            request={'time_s':e['timestamp_s'],'epoch':epoch,'points':e['coordinates']}
            response={'epoch':epoch,'topL':replies}
            query_count=len(replies)*len(ranking.categories)
        _,known=cache.receive(epoch,replies)
        counts['request_bytes']+=encoded(request);counts['response_bytes']+=encoded(response)
        counts['category_queries']+=query_count;counts['emissions']+=1
        coords=e.get('coordinates') or [q[1:] for q in e['queries']]
        counts['coordinates']+=len(coords);counts['distinct_coordinates']+=len(set(map(tuple,coords)))
        if j in positions:
            i=positions[j];state=source['reference_states'][i]
            available=server.world.at_epoch(epoch)
            score=score_returned(ranking.top(state,available,5),ranking.top(state,known,5),available)
            assert score['unavailable_returned_items']==0
            utility[source['clock_indices'][i]]=score
    assert set(utility)==set(source['clock_indices'])
    counts['service_events']=len(utility)
    counts['generation_ms']=ex['generation_ms']
    return utility,dict(counts)


def evaluate(split):
    p=protocol();dst=OUT/f'service_{split}.json'
    if dst.exists():raise FileExistsError(dst)
    rn,service,*_=load();ranking=RankedRoadPois(service,CACHE/'live_poi_full_rank_v1.npy')
    data=json.loads(data_path(split).read_text());ss,records=sources(data,rn)
    rows=[];costs=[];failures=[];inputs={}
    for method in METHODS:
        executions={}
        for source in ss:
            for rep in p['mechanism_repetitions']:
                path=shard_path(split,method,source['session_id'],rep)
                inputs[str(path.relative_to(ROOT))]=sha(path)
                ex=json.loads(gzip.decompress(path.read_bytes()))
                assert ex['protocol_sha256']==sha(OUT/'protocol.json')
                executions[source['session_id'],rep]=ex
                if ex['status']!='ok':failures.append({k:ex[k] for k in ('method','session_id','rep','error')})
        for probability in p['probabilities']:
            for seed in p['world_seeds']:
                server=LivePointService(ranking,AvailabilityWorld(ranking.n,seed,probability,60),10)
                lookup={}
                for source in ss:
                    for rep in p['mechanism_repetitions']:
                        ex=executions[source['session_id'],rep]
                        result=score_execution(ex,source,server,ranking)
                        if result:
                            utilities,count=result;lookup[source['session_id'],rep]=utilities
                            if probability==.8:costs.append(dict(method=method,rep=rep,world_seed=seed,session_id=source['session_id'],family_id=source['family_id'],**count))
                for r in records:
                    for rep in p['mechanism_repetitions']:
                        complete=all((sid,rep) in lookup for sid in r['session_ids'])
                        refs=[ranking.top(rn.nearest(data['traces'][sid][i]['lat'],data['traces'][sid][i]['lon'])[0],server.world.at_epoch(server.world.epoch(data['traces'][sid][i]['time_s']-data['traces'][sid][0]['time_s'])),5)
                              for sid,ids in zip(r['session_ids'],r['observed_indices']) for i in ids] if not complete else None
                        values=[lookup[sid,rep][i]['recall'] for sid,ids in zip(r['session_ids'],r['observed_indices']) for i in ids] if complete else [0. if any(x) else None for x in refs]
                        rows.append({'method':method,'case_id':r['case_id'],'record_id':r['record_id'],'family_id':r['family_id'],
                            'rep':rep,'world_seed':seed,'probability':probability,'status':'ok' if complete else 'failed',
                            'recall':mean_optional(values),'eligible_events':sum(v is not None for v in values),'events':len(values)})
        print('Service',split,method,'done',flush=True)
    write_json(dst,{'schema':'common-live-paper-service-v1','protocol_sha256':sha(OUT/'protocol.json'),'rows':rows,'costs':costs,'failures':failures,'transcript_sha256':inputs})


if __name__=='__main__':
    import networkx as nx
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','generate','evaluate'])
    parser.add_argument('--split',choices=list(DATASETS),default='auxiliary');parser.add_argument('--method',choices=METHODS);parser.add_argument('--pilot',action='store_true')
    a=parser.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='generate':generate_all(a.split,a.method,a.pilot)
    else:evaluate(a.split)
