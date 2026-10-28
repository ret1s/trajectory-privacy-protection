"""Matched paper-adaptation study on the sealed, previously inspected cohort.

GeoI-BQ Q and REM Z are replayed, NEVER regenerated. New comparator randomness
is private. Freeze source/config before generation; fit TRAIN, select SELECTION,
then score TEST. Failures and all declared arms are retained.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import ast
import copy
import gzip
import hashlib
import json
import os
import pickle
import platform
import secrets
import tempfile
import time
import traceback

import numpy as np

from benchmark.paper_comparators import PublicHistory, shared_models, generate as paper_generate
from benchmark.session_poi_cache import SessionCoverClient
from core.demo_protocol import TrajectoryPoint
from core.research_release import load_release
from evaluation.live_comparison_attacks import LearnedAttack, features_and_geometry
from evaluation.ordered_endpoint_attacks import OrderedEndpointBank, ordered_endpoint_features
from evaluation.robust_endpoint_selection import robust_select
from experiments import main_depth_study_20261014 as sealed

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/benchmarks/paper_comparison_20261028_v1'
WORK=Path('/private/tmp/trajectory-paper-comparison-20261028-v1')
METHODS=('raw','geo_i_Z_k1','transprotect_markov','semantic_poi','fake_queries','geoi_bq')
NEW_METHODS=('raw','transprotect_markov','semantic_poi','fake_queries')
LABELS={'raw':'Raw GPS','geo_i_Z_k1':'Geo-I/REM Z (K1)',
    'transprotect_markov':'TransProtect (Markov adaptation)',
    'semantic_poi':'Semantic correlation (OSM/empirical adaptation)',
    'fake_queries':'Fake-query insertion (road/history adaptation)',
    'geoi_bq':'GeoI-BQ v1 (proposed)'}
SCENARIOS=('S1','S2','S3','S9','S10')
CACHES=('current','known','session')
CONFIG=dict(L=60,reference_k=5,radius_m=1000,window_s=[0,600],truth_step_s=20,
    late_window_s=[540,600],splits={'train':20,'selection':8,'test':20},
    draws_by_split={'train':1,'selection':1,'test':3},bootstrap_replicates=10000,
    bootstrap_seed=20261028,primary='geoi_bq',reused_geometry=True,
    new_defense_samples=False,selection='family mean+SE MAE / mean-SE Hit, lexical ties',
    original_GeoIBQ_request_label=30,actual_original_provider_L=60,
    normalized_request_label=60,request_label_change='public constant only; same Q/Z, POIs and utility checked',
    TP=dict(candidate_K=10,targets=8,alpha=10000,epsilon_per_m=.005,predictor='TRAIN Markov'),
    Semantic=dict(K=5,pool=128,semantics='nearest public POI',predictor='empirical'),
    Fake=dict(K=5,max_fake=2,wait_seconds=[5,15],candidate_limit=24,similarity=.75))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    p=Path(p);b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if p.suffix=='.gz':payload=gzip.compress(payload,mtime=0)
    fd,tmp=tempfile.mkstemp(prefix='.pending-',dir=p.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(payload)
        os.link(tmp,p)  # Atomic, fail closed if this public result already exists.
    finally:os.unlink(tmp)
def packed(mask):return np.packbits(mask).tobytes().hex()
def unpacked(value,size):return np.unpackbits(np.frombuffer(bytes.fromhex(value),dtype=np.uint8))[:size].astype(bool)


def public_events(events,lo=0.,hi=600.):
    """Time-window view; never consumes a real/fake flag or evaluator ledger."""
    out=[]
    for e in events:
        t=float(e['timestamp_s'])
        if lo<=t<=hi:
            coords=[[float(x),float(y)] for x,y in e['coordinates']]
            if not coords or not np.isfinite(coords).all():raise ValueError('Finite public coordinates required')
            out.append(dict(timestamp_s=t,coordinates=coords))
    if not out or any(b['timestamp_s']<=a['timestamp_s'] for a,b in zip(out,out[1:])):
        raise ValueError('Nonempty strict public clock required')
    return out


def source_closure():
    found=set();pending=[str(Path(__file__).relative_to(ROOT)),
        'experiments/verify_paper_comparison_20261028.py','tests/test_paper_comparison_20261028.py']
    def add(module):
        for name in (module.replace('.','/')+'.py',module.replace('.','/')+'/__init__.py'):
            if (ROOT/name).is_file() and name not in found:pending.append(name)
    while pending:
        name=pending.pop()
        if name in found:continue
        found.add(name);tree=ast.parse((ROOT/name).read_text());package=name[:-3].split('/')[:-1]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for a in node.names:add(a.name)
            elif isinstance(node,ast.ImportFrom):
                parent=package[:len(package)-node.level+1] if node.level else []
                module='.'.join(parent+([node.module] if node.module else []))
                add(module)
                for a in node.names:
                    if a.name!='*':add('.'.join(filter(None,(module,a.name))))
    found.update('docs/reproduction/'+n+'.md' for n in ('transprotect','semantic_correlation','fake_queries'))
    found.update(('requirements-continual-geoi.txt','requirements-paper-comparison.txt'))
    found.add('docs/research/2026-10-28_paper_comparison_protocol.md')
    return sorted(found)


def freeze():
    load_release();sealed.validate_sources(sealed.OUT)
    if OUT.exists():raise FileExistsError('Use a new version; never replace frozen evidence')
    files=source_closure()
    inputs=[sealed.DATA,sealed.OUT/'protocol.json',sealed.OUT/'generation.json',
            sealed.OUT/'resources.json',sealed.OUT/'results.json',ROOT/sealed.public.NETWORK,ROOT/sealed.public.RESOURCES]
    protocol=dict(schema='matched-paper-comparison-v1',date='2026-10-28',config=CONFIG,
        methods=list(METHODS),labels=LABELS,adaptations_not_official_reproductions=True,
        source_sha256={n:sha(ROOT/n) for n in files},
        inputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        original_families=read(sealed.OUT/'generation.json')['family_files_sha256'],
        proposed_output='exact saved main-depth Q coordinate/time stream; no model rerun',
        source_request_label_discrepancy='old request field30 but provider returned60; normalize field60 for ALL arms; replay checks same POIs/utility',
        data_scope='all48 existing families; inspected same-map synthetic data, NOT new geometry confirmation',
        observation='only public coordinates/time; all fake events in scenario time window; clock/close public',
        scenarios={'S1':'single event at each declared public service tick',
            'S2':'all wire in540..600; target mean of four GPS ticks, not semantic stop',
            'S3':'all wire0..600; reconstruct each of31 GPS ticks',
            'S9':'all wire0..600; first GPS','S10':'all wire0..600; GPS at public close600'},
        failure='retain every trip/arm; report coverage; unavailable privacy is N/A, delivered utility0 for failed arm',
        privacy_rng='fresh private keys outside repository; outputs archived; seeds not attacker inputs',
        runtime=dict(python=platform.python_version(),platform=platform.platform(),
            thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')}))
    save(OUT/'protocol.json',protocol)
    for name in files:
        p=OUT/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes())
    print('Frozen matched protocol; no TEST output generated',flush=True)


def validate_sources():
    p=read(OUT/'protocol.json');load_release()
    assert p['config']==CONFIG and p['methods']==list(METHODS)
    for n,d in p['source_sha256'].items():
        assert sha(ROOT/n)==sha(OUT/'source_snapshot'/n)==d,('Source changed',n)
    for n,d in p['inputs_sha256'].items():assert sha(ROOT/n)==d,n
    for n,d in p['original_families'].items():assert sha(sealed.OUT/'families'/n)==d,n
    return p


def resources():return sealed.public.resources(WORK/'public_resources')


def model_setup(families,rn,reply,dataset):
    train=[t for f in families if f['split']=='train' for t in f['trips']]
    sequences=[[x['state'] for x in t['truth']] for t in train]
    history=PublicHistory(rn,sequences)
    training=[points_for(t,dataset) for t in train]
    start=time.perf_counter();shared=shared_models(rn,history,training,reply.pois)
    return history,shared,time.perf_counter()-start


def points_for(trip,dataset):
    by_time={int(p['time_s']):p for p in dataset['traces'][trip['native_session_id']]}
    return [TrajectoryPoint(float(x['t']),by_time[x['t']]['lat'],by_time[x['t']]['lon']) for x in trip['truth']]


def smoke():
    load_release();families=sealed.load_families(sealed.OUT);dataset=read(sealed.DATA)
    rn,ref,reply,_=resources();history,shared,seconds=model_setup(families,rn,reply,dataset)
    trip=next(f for f in families if f['split']=='train')['trips'][0]
    points=points_for(trip,dataset)[:4]
    changed=points[:3]+[TrajectoryPoint(points[3].timestamp_s,points[0].lat,points[0].lon)]
    for m in NEW_METHODS:
        a=paper_generate(m,points,rn,history,shared,20261028)
        b=paper_generate(m,changed,rn,history,shared,20261028)
        assert public_events(a['events'],0,40)==public_events(b['events'],0,40),(m,'future dependence')
        print('TRAIN smoke / causal prefix:',m,len(a['events']),'events',flush=True)
    print('Setup seconds:',round(seconds,3),flush=True)


def replay(events,trip,rn,reply,utility):
    events=public_events(events);ks={len(e['coordinates']) for e in events}
    if len(ks)!=1:raise ValueError('Declared adapters require fixed within-trip K')
    client=SessionCoverClient(reply.categories,len(reply.pois),k=next(iter(ks)),response_l=60,
        session_token=trip['session_id'],static_catalogue_id=reply.sha256)
    cost=dict(requests=0,packets=0,request_bytes=0,reply_bytes=0)
    score=[];masks=[];cursor=0;last=None
    def server(request):
        assert request['response_l']==60
        state=int(rn.nearest(*request['coordinate'])[0])
        return [list(map(int,row[row>=0])) for row in reply.query_indices(state)]
    destination=trip['truth'][-1]['state']
    start=time.perf_counter()
    for truth in trip['truth']:
        t=truth['t']
        while cursor<len(events) and events[cursor]['timestamp_s']<=t:
            event=events[cursor];last=client.step(event['timestamp_s'],event['coordinates'],server)
            cost['requests']+=len(last['requests']);cost['packets']+=1
            for name,field in [('request_bytes','requests'),('reply_bytes','replies')]:
                cost[name]+=sum(len(json.dumps(x,separators=(',',':')).encode()) for x in last[field])
            cursor+=1
        assert last is not None
        score.append(dict(t=t,**{c:utility.score(truth['state'],destination,last[c]) for c in CACHES}))
        masks.append({c:packed(last[c]) for c in CACHES})
    assert cursor==len(events),'Queries after public close excluded before replay'
    return dict(utility=score,masks=masks,cost=cost,service_replay_ms=(time.perf_counter()-start)*1000)


def native_metrics(events,trip,rn,target_utility,generation_ms):
    singleton=all(len(e['coordinates'])==1 for e in events)
    d=dict(generation_ms=generation_ms,Eq13=None,displacement_m=None,
        ASR=None,DER=None,path_count=None,
        unavailable={'ASR':'No paper-calibrated real-member posterior; Q-only has no real member',
            'DER':'Source Sim/effectiveness rule and AMap model unavailable',
            'path_count':'Source-compatible full arrival/direction similarity counts unavailable'})
    if singleton:
        public={e['timestamp_s']:e for e in events};released=[];real=[];dist=[]
        for truth in trip['truth']:
            e=public[float(truth['t'])];lat,lon=e['coordinates'][0]
            released.append(int(rn.nearest(lat,lon)[0]));real.append(truth['state'])
            dist.append(float(np.linalg.norm(np.array(rn.point_xy(lat,lon))-truth['xy'])))
        costs=target_utility.cost_to_targets
        d['Eq13']=float(np.mean(np.abs(costs[real]-costs[released])@target_utility.target_prior))
        d['displacement_m']=float(np.mean(dist))
    return d


def generate_all():
    validate_sources()
    if (OUT/'generation.json').exists():raise FileExistsError('Generation already complete')
    families=sealed.load_families(sealed.OUT);dataset=read(sealed.DATA)
    rn,ref,reply,_=resources();history,shared,setup=model_setup(families,rn,reply,dataset)
    utility=sealed.Utility(ref,reply);private=WORK/'private';private.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(private,0o700)
    metadata=dict(native_states=len(rn),pois=len(reply.pois),catalogue_sha256=rn.catalogue_sha256,
        reference_sha256=ref.sha256,reply_sha256=reply.sha256,shared_setup_seconds=setup,
        TRAIN_sequences=history.training_sequences,
        Eq13_target_indices=shared['transprotect_markov'].utility_provider.target_indices.tolist(),
        Eq13_target_prior=shared['transprotect_markov'].utility_provider.target_prior.tolist(),
        Eq13_unreachable_penalty=shared['transprotect_markov'].utility_provider.disconnection_penalty,
        Eq13_scope='same TRAIN-population target proxy, public coordinate snapping; not original Rome/SF metric parity')
    if not (OUT/'resources.json').exists():save(OUT/'resources.json',metadata)
    for family in families:
        dest=OUT/'families'/(family['family_id']+'.json.gz')
        if dest.exists():
            assert read(dest)['protocol_sha256']==sha(OUT/'protocol.json')
            print('Retained completed family',family['family_id'],flush=True);continue
        trips=[]
        for source in family['trips']:
            trip={k:copy.deepcopy(source[k]) for k in ('session_id','native_session_id','draw','truth')}
            trip.update(events={},utility={},masks={},cost={},native={},diagnostics={},status={},runtime={})
            points=points_for(source,dataset)
            for m in METHODS:
                generation_ms=None
                try:
                    if m=='geoi_bq':events=source['events']['cache_target_g1'];diagnostics={'saved_Q_replayed':True}
                    elif m=='geo_i_Z_k1':
                        events=[dict(timestamp_s=float(x['t']),coordinates=[z]) for x,z in zip(source['truth'],source['GeoI_anchors'])]
                        diagnostics={'saved_REM_Z_replayed':True}
                    else:
                        keypath=private/(source['session_id']+'--'+m+'.key')
                        if keypath.exists():key=keypath.read_bytes()
                        else:
                            key=secrets.token_bytes(32);fd=os.open(keypath,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
                            with os.fdopen(fd,'wb') as stream:stream.write(key)
                        seed=int.from_bytes(key[:16],'little')
                        result=paper_generate(m,points,rn,history,shared,seed)
                        events=result['events'];generation_ms=result['generation_ms'];diagnostics=result['diagnostics']
                    events=public_events(events)
                    result=replay(events,source,rn,reply,utility)
                    if m=='geoi_bq':
                        assert events==source['events']['cache_target_g1']
                        assert result['utility']==source['utility']['cache_target_g1'],'Proposed utility replay differs'
                        assert all(result['cost'][k]==v for k,v in source['cost']['cache_target_g1'].items())
                        for masks,old in zip(result['masks'],source['retrieved_POI_indices']['cache_target_g1']):
                            for c in CACHES:assert np.flatnonzero(unpacked(masks[c],len(reply.pois))).tolist()==old[c]
                    for key in ('events','utility','masks','cost'):
                        trip[key][m]=events if key=='events' else result[key]
                    trip['runtime'][m]=dict(generation_ms=generation_ms,service_replay_ms=result['service_replay_ms'])
                    trip['native'][m]=native_metrics(events,source,rn,shared['transprotect_markov'].utility_provider,generation_ms)
                    trip['diagnostics'][m]=diagnostics;trip['status'][m]='ok'
                except Exception as error:
                    trip['status'][m]='failed'
                    trip['diagnostics'][m]=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
                    print('RETAINED FAILURE',family['family_id'],source['session_id'],m,type(error).__name__,str(error),flush=True)
            trips.append(trip)
        save(dest,dict(family_id=family['family_id'],split=family['split'],trips=trips,protocol_sha256=sha(OUT/'protocol.json')))
        print('Completed',family['family_id'],family['split'],len(trips),'trip-draws',flush=True)
    save(OUT/'generation.json',dict(protocol_sha256=sha(OUT/'protocol.json'),
        family_files_sha256={p.name:sha(p) for p in sorted((OUT/'families').glob('*.gz'))},private_keys_exported=False))


def families():
    receipt=read(OUT/'generation.json');out=[]
    for name,digest in receipt['family_files_sha256'].items():
        p=OUT/'families'/name;assert sha(p)==digest;out.append(read(p))
    return out


def attack_sample(trip,scenario,method,rn,history):
    truth=trip['truth'];xy=np.array([x['xy'] for x in truth]);events=public_events(trip['events'][method])
    grid={float(x['t']) for x in truth}
    if scenario in ('S9','S10'):
        features,geometry=ordered_endpoint_features(events,scenario,rn,history,observable_close_s=600.)
        return features,geometry,xy[[0 if scenario=='S9' else -1]]
    if scenario=='S2':
        window=public_events(events,540,600);features,geometry=features_and_geometry(window,'S2',rn,history)
        target=xy[[i for i,x in enumerate(truth) if 540<=x['t']<=600]].mean(axis=0,keepdims=True)
    elif scenario=='S1':
        observed=[e for e in events if e['timestamp_s'] in grid]
        assert [e['timestamp_s'] for e in observed]==[float(x['t']) for x in truth]
        parts=[features_and_geometry([e],'S1',rn,history) for e in observed]
        features=np.concatenate([a for a,b in parts]);geometry={n:np.concatenate([b[n] for a,b in parts]) for n in parts[0][1]};target=xy
    elif scenario=='S3':
        features,geometry=features_and_geometry(events,'S3',rn,history)
        positions=[i for i,e in enumerate(events) if e['timestamp_s'] in grid]
        assert len(positions)==len(truth)
        features=features[positions];geometry={n:p[positions] for n,p in geometry.items()};target=xy
    else:raise ValueError('Unknown scenario')
    return {'shadow':features},geometry,target


def attacks(all_families,rn):
    train=[f for f in all_families if f['split']=='train'];selection=[f for f in all_families if f['split']=='selection'];test=[f for f in all_families if f['split']=='test']
    history=PublicHistory(rn,[[p['state'] for p in t['truth']] for f in train for t in f['trips']])
    rows=[]
    for m in METHODS:
        if any(t['status'][m]!='ok' for f in all_families for t in f['trips']):
            rows.extend(dict(method=m,scenario=s,status='incomplete_coverage',Hit100=None,MAE_m=None) for s in SCENARIOS);continue
        for s in SCENARIOS:
            path=OUT/'attack_metrics'/(m+'--'+s+'.json')
            if path.exists():rows.append(read(path));continue
            features=defaultdict(list);labels=[]
            for f in train:
                for t in f['trips']:
                    x,_,y=attack_sample(t,s,m,rn,history)
                    for name,a in x.items():features[name].extend(a)
                    labels.extend(y)
            bank=OrderedEndpointBank(features,labels) if s in ('S9','S10') else LearnedAttack(features['shadow'],labels)
            def errors(t):
                x,g,y=attack_sample(t,s,m,rn,history)
                predictions=(bank.predictions(public_events(t['events'][m]),s,rn,history,observable_close_s=600.)
                    if s in ('S9','S10') else g|{'shadow_'+n:p for n,p in bank.predict(x['shadow']).items()})
                return {n:np.linalg.norm(p-y,axis=1) for n,p in predictions.items()}
            selection_rows=[]
            for f in selection:
                for t in f['trips']:
                    es=errors(t)
                    for i in range(len(next(iter(es.values())))):
                        selection_rows.append(dict(family_id=f['family_id'],session_id=t['session_id']+f'-e{i}',seed=t['draw'],errors={n:float(e[i]) for n,e in es.items()}))
            chosen,stats=robust_select(selection_rows,[f['family_id'] for f in selection])
            model_path=OUT/'attacker_models'/(m+'--'+s+'.pkl.gz');model_path.parent.mkdir(parents=True,exist_ok=True)
            if model_path.exists():bank=pickle.loads(gzip.decompress(model_path.read_bytes()))
            else:
                payload=gzip.compress(pickle.dumps(bank,protocol=5),mtime=0)
                with model_path.open('xb') as stream:stream.write(payload)
            selpath=OUT/'attacker_selection'/(m+'--'+s+'.json')
            selection_value=dict(selected=chosen,statistics=stats,fit_split='train',selection_split='selection',
                model_sha256=sha(model_path),before_test_predictions=True)
            if selpath.exists():assert read(selpath)==selection_value
            else:save(selpath,selection_value)
            predictions=[];fm=[]
            for f in test:
                trip_rows=[]
                for t in f['trips']:
                    es=errors(t);trip_rows.append(dict(MAE_m=float(np.mean(es[chosen['mae']])),Hit100=float(np.mean(es[chosen['hit100']]<=100))))
                    predictions.append(dict(family_id=f['family_id'],session_id=t['session_id'],errors={n:e.tolist() for n,e in es.items()}))
                fm.append(dict(family_id=f['family_id'],**{k:float(np.mean([x[k] for x in trip_rows])) for k in ('MAE_m','Hit100')}))
            predpath=OUT/'attacker_predictions'/(m+'--'+s+'.json.gz')
            if predpath.exists():assert read(predpath)==predictions
            else:save(predpath,predictions)
            indices=np.random.default_rng(CONFIG['bootstrap_seed']).integers(0,len(fm),(10000,len(fm)))
            result=dict(method=m,scenario=s,status='ok',selected_attacker=chosen,family_metrics=fm)
            for key in ('MAE_m','Hit100'):
                values=np.array([x[key] for x in fm]);result[key]=float(values.mean())
                result[key+'_CI95']=np.quantile(values[indices].mean(axis=1),[.025,.975]).tolist()
            save(path,result);rows.append(result)
            print('Attacker selected/scored',m,s,flush=True)
    return rows


def summarize(all_families):
    test=[f for f in all_families if f['split']=='test'];indices=np.random.default_rng(CONFIG['bootstrap_seed']).integers(0,len(test),(10000,len(test)))
    purposes=[p.value for p in sealed.QueryPurpose];utility=[];family_values={};cost={};native={};coverage={}
    for m in METHODS:
        test_trips=[t for f in test for t in f['trips']];successful=[t for t in test_trips if t['status'][m]=='ok']
        coverage[m]=dict(attempted=len(test_trips),completed=len(successful),failed=len(test_trips)-len(successful))
        for cache in CACHES:
            fm=[]
            for f in test:
                values={}
                for p in purposes:
                    valid=[]
                    for t in f['trips']:
                        if t['status'][m]!='ok':valid.extend([0.]*len(t['truth']))
                        else:valid.extend(x[cache][p]['recall5'] for x in t['utility'][m] if x[cache][p]['recall5'] is not None)
                    values[p]=float(np.mean(valid)) if valid else None
                values['macro']=float(np.mean([v for v in values.values() if v is not None]))
                fm.append(dict(family_id=f['family_id'],**values))
            family_values[m,cache]=fm
            metrics={}
            for p in purposes+['macro']:
                values=np.array([x[p] for x in fm]);metrics[p]=dict(mean=float(values.mean()),absolute95=np.quantile(values[indices].mean(axis=1),[.025,.975]).tolist())
            utility.append(dict(method=m,cache=cache,metrics=metrics,family_metrics=fm))
        fields=('requests','packets','request_bytes','reply_bytes')
        cost[m]={k:float(np.mean([t['cost'][m][k] for t in successful])) if successful else None for k in fields}
        for field in ('generation_ms','service_replay_ms'):
            v=[t['runtime'][m][field] for t in successful if t['runtime'][m][field] is not None]
            cost[m][field]=float(np.mean(v)) if v else None
        native[m]={}
        for field in ('Eq13','displacement_m','generation_ms'):
            v=[t['native'][m][field] for t in successful if t['native'][m][field] is not None]
            native[m][field]=dict(mean=float(np.mean(v)) if v else None,eligible=len(v),status='computed' if v else 'not_applicable_or_not_measured')
        for field in ('ASR','DER','path_count'):
            native[m][field]=dict(mean=None,status='not_available',reason=successful[0]['native'][m]['unavailable'][field] if successful else 'No valid execution')
    comparisons=[]
    for m in METHODS[:-1]:
        metrics={}
        for p in purposes+['macro']:
            delta=np.array([a[p]-b[p] for a,b in zip(family_values['geoi_bq','session'],family_values[m,'session'])])
            metrics[p]=dict(delta=float(delta.mean()),paired95=np.quantile(delta[indices].mean(axis=1),[.025,.975]).tolist())
        comparisons.append(dict(control=m,metrics=metrics))
    return dict(utility=utility,cost=cost,native_metrics=native,coverage=coverage,comparisons=comparisons)


def score():
    validate_sources();all_families=families();rn,_,_,_=resources()
    result=summarize(all_families);result['attackers']=attacks(all_families,rn)
    result.update(schema='matched-paper-comparison-results-v1',protocol_sha256=sha(OUT/'protocol.json'),
        generation_sha256=sha(OUT/'generation.json'),scope='controlled comparison on reused native cohort',
        proposed_model_changed=False,proposed_randomness_regenerated=False,
        limitations=['paper adaptations, not official reproductions','same inspected map/cohort',
            'static418-POI service, local destination known','finite loss-selected attacker bank',
            'reply-index JSON bytes, no metadata/HTTP/TLS or field latency','source public request label normalized30to60; received POIs unchanged'])
    save(OUT/'results.json',result)
    print('Matched comparison complete',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('smoke','freeze','generate','score'))
    args=p.parse_args();{'smoke':smoke,'freeze':freeze,'generate':generate_all,'score':score}[args.stage]()


if __name__=='__main__':main()
