"""Authentic native S5 next-edge and S6 fork/history forecast experiment.

Predeclared fresh family splits, public candidate context and public clocks.
Geo-I is unchanged: current REM/noisy-test + belief/road/POI postprocessing.
The epoch8 branch wraps that engine with eight prospectively reserved caps.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import secrets
import time
import numpy as np
from data.lane_states import build_lane_states,catalogue_summary
from evaluation.lane_travel import LanePoiService
from benchmark.public_poi_context import PublicPoiContext
from benchmark.anchor_belief import PublicAnchorModel
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from core.session_budget import FixedEpochPolicy,PersistentEpochBudget,FixedEpochProtectedSessions
from evaluation.candidate_future_attack import CandidateFutureAttack,forecast_metrics,coordinates
from experiments.build_future_sumo_cohort import ROOT,sha,save

DATA=ROOT/'artifacts/datasets/future_controlled_20261005_v2/dataset.json.gz'
OUT=ROOT/'artifacts/benchmarks/future_native_20261005_v1'
WORK=Path('/private/tmp/trajectory-native-future-evaluation-v1')
METHODS=('raw','geoi_session_reset','geoi_epoch8')


def compressed_save(path,payload):
    if path.exists():raise FileExistsError(f'Preserve completed evidence:{path}')
    path.write_bytes(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0))


def declare():
    return {'schema':'native-future-evaluation-protocol-v1','dataset_sha256':sha(DATA),
        'splits':{'train':[f'native-{i:02d}' for i in range(1,13)],
                  'selection':[f'native-{i:02d}' for i in range(13,19)],
                  'test':[f'native-{i:02d}' for i in range(19,25)]},
        'methods':METHODS,'GeoI_backbone':'unchanged paced REM/noisy-reuse, matched epsilon belief, K5,theta200,H12,slack.03',
        'public_observations':'eight fixed windows0..600 every20seconds; querywindows additionally include BOTH predeclared calibrationforkclocks',
        'forecast_cut':'only queryevents<=shared_fork_t or <=turn_visible_t; six FULL public histories days1..6',
        'auxiliary_context':'both outgoing native edge IDs/via geometries and both public candidate destination positions, '
            'fixed before hidden choice; neither selected future route nor private target GPS enters features',
        'targets':'S5:first FUTURE noninternal edge after forecastcut; S6:native parked destinationGPS at600s',
        'history_prior':'five routine/one rare historicaltrips; QUERY targets deliberately balanced and private order randomized',
        'S4_sideinfo':'S6 attacker assumed already able to link six historical windows to this driver; no identity privacy claim',
        'budget':{'geoi_session_reset':{'epoch_effective_cap':1.84,'slot_effective_cap':.23,'unit_epsilon':.01},
                  'geoi_epoch8':{'epoch_effective_cap':.23,'slot_effective_cap':.02875,'unit_epsilon':.00125}},
        'attacker_bank':'96candidate-scoring ExtraTrees depth12 leaf2; public curve mean/min/recent likelihood '
            'at20/100/300m; heading; S6 publichistoryprior and history+query; uniform control',
        'selection':'permethod/task/stage maximum selectionbalancedaccuracy, then minimumlogloss, stable lexicalname; no fit on test',
        'permutation':'one balanced within-family choice flip in TRAIN only; no test tuning',
        'evaluation':'exact native candidate-edge accuracy, Brier/logloss; destinationMAE/Hit100; routine/rare separately',
        'public_native_map':'same converted SUMOnetwork for FCD, road catalogue, service and decoder',
        'RNG':'OSprivatekey, HMAC-independent sessions; keys/ledgers local0600 outside artifact; frozenpublicoutput replayable',
        'utility':'static public POI union Recall@5, six categories,L10,K5,all available; not dynamicavailabilitybenchmark',
        'privacy_scope':'coordinate transcript within declared public windows and one eight-session epoch, ideal kernel caveat; '
            'publicknowncandidategeometry narrows attack target; not full openworldrouteforecast or faithfulpaperSOTA'}


def native_resources(data,work):
    cache=work/'public_resources';cache.mkdir(parents=True,exist_ok=True)
    compressed=ROOT/data['network']['compressed_path'];assert sha(compressed)==data['network']['compressed_sha256']
    net=cache/'native.net.xml'
    if not net.exists():net.write_bytes(gzip.decompress(compressed.read_bytes()))
    assert sha(net)==data['network']['native_sha256']
    rn=build_lane_states(net,spacing_m=40.)
    pois=[{k:v for k,v in p.items() if k not in ('vertex','access_offset_m')}
          for p in json.loads((ROOT/'artifacts/benchmarks/research_loop/resources.json').read_text())['pois_used']]
    service=LanePoiService(rn,pois,k=5)
    reference=PublicPoiContext(service,cache/'reference5.npz')
    reply=PublicPoiContext(LanePoiService(rn,pois,k=10),cache/'reply10.npz')
    _,inv,counts=np.unique(np.floor(rn.xy/120.).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    prior=1./counts[inv];prior/=prior.sum()
    beliefs={}
    for epsilon in (.01,.00125):
        base=PublicAnchorModel(rn,reference,prior,spacing_m=200.,epsilon_release=epsilon,
            epsilon_test=epsilon,cache_path=cache/f'belief-{epsilon:g}.npz')
        beliefs[epsilon]=ResponseAwareAnchorModel(base,reply)
    metadata={'native_sha256':sha(net),'catalogue':catalogue_summary(rn),
        'internal_states':sum(d['edge_id'].startswith(':') for _,d in rn.graph.nodes(data=True)),
        'reference_sha256':reference.sha256,'reply_sha256':reply.sha256,'poi_count':len(reference.pois),
        'belief_sha256':{str(k):v.base.sha256 if hasattr(v,'base') else v.sha256 for k,v in beliefs.items()},
        'source_sha256':{'artifacts/benchmarks/research_loop/resources.json':sha(ROOT/'artifacts/benchmarks/research_loop/resources.json')}}
    return rn,reference,reply,beliefs,metadata


def static_recall(rn,reference,reply,gps,queries):
    state,_=rn.nearest(*gps);truth=reference.signatures[state]
    union=set()
    for q in queries:
        ids=reply.query_indices(rn.nearest(*q)[0]);union.update(int(v) for v in ids.ravel() if v>=0)
    percat=[]
    for ids in truth:
        valid=ids[ids>=0]
        if len(valid):percat.append(len(set(map(int,valid))&union)/len(valid))
    return {'recall5':float(np.mean(percat)) if percat else None,'nonempty_categories':len(percat),
            'unique_reply_pois':len(union)}


def generate(work):
    if (OUT/'public_transcripts.json.gz').exists():raise FileExistsError('Preserve public transcript evidence')
    data=json.loads(gzip.decompress(DATA.read_bytes()));rn,reference,reply,beliefs,metadata=native_resources(data,work)
    private=work/'private_state';private.mkdir(parents=True,exist_ok=True);os.chmod(private,0o700)
    groups=[];accounting=[];start=time.perf_counter()
    for family in data['families']:
        specs=family['evaluator_only']['sessions'];contexts=family['public_context']
        key_path=private/f'{family["family_id"]}.key'
        if key_path.exists():key=key_path.read_bytes()
        else:
            fd=os.open(key_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'wb') as f:f.write(secrets.token_bytes(32))
            key=key_path.read_bytes()
        streams={m:[] for m in METHODS};details={};slots={}
        for method,total in [('geoi_session_reset',1.84),('geoi_epoch8',.23)]:
            policy=FixedEpochPolicy('native-eight-trip-public-epoch',0.,12000.,
                                    total_effective_epsilon_per_m=total,session_slots=8,horizon=12,read_interval_s=60.)
            ledger=PersistentEpochBudget(policy,private/f'{family["family_id"]}-{method}.sqlite',private_key=key)
            def factory(allocation,rngs):
                engine=PacedSlackProgressLaneDummy(rn,belief_model=beliefs[allocation.unit_epsilon_per_m],
                    k=5,budget=allocation.nominal_budget_per_m,horizon=allocation.horizon,theta_m=200.,
                    read_interval_s=allocation.read_interval_s,utility_slack=.03,rng=rngs.initialization)
                engine.anchor_rng=rngs.anchor;engine.dummy_rng=rngs.dummy;engine.reset()
                return engine
            slots[method]=(FixedEpochProtectedSessions(ledger,factory),ledger)
        session_truth=[]
        for spec in specs:
            trace=data['traces'][spec['session_id']];by_time={int(p['time_s']):p for p in trace}
            query=spec['day'] in (7,8)
            clock=sorted(set(range(0,601,20))|({family['clocks']['shared_fork_t'],family['clocks']['turn_visible_t']} if query else set()))
            events={m:[] for m in METHODS};utility={m:[] for m in METHODS};gps_calls={m:0 for m in slots}
            public_start=spec['depart_s']
            for method,(client,_) in slots.items():assert client.start_session(f'opaque-{spec["day"]}',public_start)
            for j,t in enumerate(clock):
                point=by_time[t];gps=(point['lat'],point['lon'])
                queries={'raw':(gps,)}
                for method,(client,_) in slots.items():
                    def supplier(m=method,g=gps):gps_calls[m]+=1;return g
                    queries[method]=client.protect_step(public_start+t,supplier)
                for method,qs in queries.items():
                    events[method].append({'event_id':f'e{j:04d}','timestamp_s':float(t),
                        'candidates':[{'candidate_id':f'q{k:04d}','lat':float(lat),'lon':float(lon)} for k,(lat,lon) in enumerate(qs)]})
                    utility[method].append({'t':t,**static_recall(rn,reference,reply,gps,qs)})
            ledger_info={}
            for method,(client,_) in slots.items():
                current=client.evaluator_current_session();assert gps_calls[method]==current['private_reads']
                ledger_info[method]=current;client.close_session(public_start+600.)
            for method in METHODS:streams[method].append({'events':events[method]})
            end=by_time[600];destination_xy=coordinates({'events':[{'event_id':'e','timestamp_s':0.,
                'candidates':[{'candidate_id':'q','lat':end['lat'],'lon':end['lon']}]}]})[0][0][0].tolist()
            future_labels={}
            if query:
                for stage,t in family['clocks'].items():
                    if stage not in ('shared_fork_t','turn_visible_t'):continue
                    next_external=next(p['edge_id'] for p in trace if p['time_s']>t and not p['edge_id'].startswith(':'))
                    # At shared fork, skip still-current incoming edge before departure.
                    if stage=='shared_fork_t':next_external=next(p['edge_id'] for p in trace if p['time_s']>t and
                        not p['edge_id'].startswith(':') and p['edge_id']!=family['fork_edge'])
                    assert next_external==contexts['choices'][spec['choice_index']]['edge_id']
                    future_labels[stage]=next_external
            session_truth.append({'day':spec['day'],'choice_index':spec['choice_index'],'destination_role':spec['destination_role'],
                'destination_xy':destination_xy,'next_edge_labels':future_labels,'ledger':ledger_info,'utility':utility})
        epoch={}
        for method,(client,ledger) in slots.items():
            epoch[method]=client.evaluator_summary();assert epoch[method]['spent_per_m']<=ledger.policy.total_effective_epsilon_per_m+1e-12
            ledger.close()
        groups.append({'public_scope':len(groups),'public_context':contexts,
            'public_clocks':{'shared_fork_t':family['clocks']['shared_fork_t'],'turn_visible_t':family['clocks']['turn_visible_t']},
            'streams':streams})
        accounting.append({'family_id':family['family_id'],'split':family['split'],
            'public_scope':len(groups)-1,'evaluator_sessions':session_truth,'epoch_accounting':epoch})
        print('Native GeoI generated',family['family_id'],'epoch spends',
              {m:round(v['spent_per_m'],5) for m,v in epoch.items()},flush=True)
    compressed_save(OUT/'public_transcripts.json.gz',{'schema':'public-native-eight-trip-windows-v1','groups':groups,
        'contract':'same-driver historical association and both public candidate geometries are auxiliary sideinformation; '
                   'streams contain coordinates/times only; future forecasting API slices queryprefix causally'})
    compressed_save(OUT/'private_accounting.json.gz',{'schema':'native-forecast-evaluator-accounting-v1',
        'dataset_sha256':sha(DATA),'public_transcript_sha256':sha(OUT/'public_transcripts.json.gz'),
        'resource_metadata':metadata,'rows':accounting,'generation_elapsed_s':time.perf_counter()-start,
        'private_keys_exported':False,'generation_source_sha256':{str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
            **{p:sha(ROOT/p) for p in ('core/session_budget.py','core/mechanisms.py','benchmark/engines/paced_slack.py',
                'benchmark/engines/paced_guard.py','benchmark/engines/filtered_cover.py','benchmark/engines/matched_filter.py',
                'benchmark/anchor_belief.py','benchmark/response_aware_belief.py','data/lane_states.py')}}})


def forecast_rows(method,stage):
    public=json.loads(gzip.decompress((OUT/'public_transcripts.json.gz').read_bytes()))
    private=json.loads(gzip.decompress((OUT/'private_accounting.json.gz').read_bytes()))
    rows=[]
    for g,truth in zip(public['groups'],private['rows']):
        clock=g['public_clocks'][stage+'_t'];streams=g['streams'][method]
        for slot in (6,7):
            y=truth['evaluator_sessions'][slot]
            prefix={'events':[e for e in streams[slot]['events'] if e['timestamp_s']<=clock]}
            rows.append({'family_id':truth['family_id'],'split':truth['split'],'query_slot':slot,
                'query':prefix,'histories':streams[:6],'public_context':g['public_context'],
                'choice_index':y['choice_index'],'destination_role':y['destination_role'],
                'destination_xy':y['destination_xy'],'actual_next_edge':y['next_edge_labels'][stage+'_t']})
    return rows


def fit_and_score(rows,use_history):
    parts={s:[r for r in rows if r['split']==s] for s in ('train','selection','test')}
    model=CandidateFutureAttack(parts['train'],use_history=use_history)
    banks={s:[model.predict(r['query'],r['public_context'],r['histories']) for r in part]
           for s,part in parts.items() if s!='train'}
    names=sorted(banks['selection'][0]);selected=min(names,key=lambda n:(
        -forecast_metrics(parts['selection'],[b[n] for b in banks['selection']])['balanced_accuracy'],
        forecast_metrics(parts['selection'],[b[n] for b in banks['selection']])['log_loss'],n))
    probabilities=[b[selected] for b in banks['test']]
    permutation=CandidateFutureAttack(parts['train'],use_history=use_history,permutation=True)
    negative={s:[permutation.predict(r['query'],r['public_context'],r['histories'])['candidate_trees']
                 for r in parts[s]] for s in ('selection','test')}
    predictions=[{'family_id':r['family_id'],'query_slot':r['query_slot'],'truth_choice':r['choice_index'],
        'predicted_choice':int(np.argmax(p)),'probabilities':p.tolist(),'true_next_edge':r['actual_next_edge'],
        'predicted_next_edge':r['public_context']['choices'][int(np.argmax(p))]['edge_id'],
        'destination_role':r['destination_role']} for r,p in zip(parts['test'],probabilities)]
    family={f:forecast_metrics([r for r in parts['test'] if r['family_id']==f],
        [p for r,p in zip(parts['test'],probabilities) if r['family_id']==f]) for f in sorted({r['family_id'] for r in parts['test']})}
    return {'selected_attacker':selected,'selection':forecast_metrics(parts['selection'],[b[selected] for b in banks['selection']]),
        'test':forecast_metrics(parts['test'],probabilities),
        'bank_test_descriptive_only':{n:forecast_metrics(parts['test'],[b[n] for b in banks['test']]) for n in names},
        'permuted_training_control':forecast_metrics(parts['test'],negative['test']),
        'test_family_metrics':family,'predictions':predictions,
        'counts':{s:len(p) for s,p in parts.items()}}


def score():
    results={}
    for method in METHODS:
        results[method]={}
        for stage in ('shared_fork','turn_visible'):
            rows=forecast_rows(method,stage)
            results[method][stage]={'S5_next_edge':fit_and_score(rows,False),
                                   'S6_history_destination':fit_and_score(rows,True)}
        print('Native forecast scored',method,flush=True)
    private=json.loads(gzip.decompress((OUT/'private_accounting.json.gz').read_bytes()))
    utility={}
    for method in METHODS:
        per_family={f['family_id']:float(np.mean([v['recall5'] for s in f['evaluator_sessions']
              for v in s['utility'][method] if v['recall5'] is not None])) for f in private['rows'] if f['split']=='test'}
        utility[method]={'test_static_recall5':float(np.mean(list(per_family.values()))),'test_family_recall5':per_family,
            'scope':'all600s public windows, allPOIsavailable, L10; not liveavailability or shortestpurpose-only benchmark'}
    raw=results['raw']
    assert np.isclose(raw['shared_fork']['S5_next_edge']['test']['exact_candidate_edge_accuracy'],.5)
    assert np.isclose(raw['shared_fork']['S6_history_destination']['test']['exact_candidate_edge_accuracy'],.5)
    gates={'shared_fork_raw_is_inherently_ambiguous':True,
        'raw_turn_positive_control':raw['turn_visible']['S5_next_edge']['test']['exact_candidate_edge_accuracy']>=.8,
        'raw_destination_turn_positive_control':raw['turn_visible']['S6_history_destination']['test']['destination_hit100']>=.8,
        'history_target_prior':'balanced one routine/one rare per family; history5/6 not substituted as test prevalence'}
    save(OUT/'results.json',{'schema':'fresh-native-candidate-edge-history-readout-v1','protocol_sha256':sha(OUT/'protocol.json'),
        'source_sha256':{str(DATA.relative_to(ROOT)):sha(DATA),
            'public_transcripts.json.gz':sha(OUT/'public_transcripts.json.gz'),
            'private_accounting.json.gz':sha(OUT/'private_accounting.json.gz'),
            str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
            'evaluation/candidate_future_attack.py':sha(ROOT/'evaluation/candidate_future_attack.py')},
        'results':results,'utility':utility,'validity_gates':gates,
        'scope':'Fresh six-family held-out native controlled test after predeclared candidate/attacker policies. '
            'Known-two-choice auxiliary context, synthetic native trips; not full open-world destination inference or originalroad reproduction.'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('generate','score','all'),default='all')
    parser.add_argument('--workdir',type=Path,default=WORK);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);declaration=declare()
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==json.loads(json.dumps(declaration))
    else:save(OUT/'protocol.json',declaration)
    if args.stage in ('generate','all'):generate(args.workdir)
    if args.stage in ('score','all'):score()


if __name__=='__main__':main()
