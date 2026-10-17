"""Frozen pilot: fixed-unit Geo-I behind unchanged whole-session bounded Q.

All 48 existing families, days7/8, one fresh draw each; no TEST tuning. First
600s reuse native GPS. The extra1200s are a separately labelled directed-road
synthetic continuation, NOT fresh native SUMO/field GPS. No published old source
or sealed evidence is changed. Private keys/SQLite remain outside the repo.
"""
import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import hmac
import json
import os
from pathlib import Path
import platform
import secrets
import traceback
from importlib.metadata import version

import numpy as np
from scipy.spatial import cKDTree

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.engines.continual_slack import ContinualPacedSlackProgressLaneDummy
from benchmark.defined_service_scores import DefinedPublicPurposeTable
from benchmark.exact_public_reference import ExactReferencePortfolio
from benchmark.targeted_cache_exploration import TargetedExploreLibrary, TargetedCachePolicy
from benchmark.cache_exploration import mask_bits
from benchmark.query_purpose import QueryPurpose
from benchmark.session_poi_cache import SessionCoverClient
from core.query_budget import PersistentQueryBudget
from evaluation.lane_travel import SparseTravel
from evaluation.live_comparison_attacks import LearnedAttack
from evaluation.ordered_endpoint_attacks import OrderedEndpointBank, ordered_endpoint_features
from evaluation.robust_endpoint_selection import robust_select
from benchmark.paper_comparators import PublicHistory
from experiments import main_depth_study_20261014 as old
from experiments import public_depth_resources_20261014 as public

ROOT = old.ROOT
OUT = ROOT/'artifacts/benchmarks/continual_geoi_20261017_v1'
WORK = Path('/private/tmp/trajectory-continual-geoi-20261017-v1')
METHODS = ('capped_g1', 'continual_g1', 'continual_g0')
WINDOWS = {'native_10min': (0, 600), 'full_30min': (0, 1800),
           'synthetic_after_12min': (720, 1800)}
CONFIG = dict(K=5, L_per_category=60, unit_epsilon_per_m=.01,
    capped_Cs_per_m=.23, private_interval_s=60, wire_interval_s=20,
    duration_s=1800, frames_per_segment=3, gamma_by_arm={'capped_g1':1, 'continual_g1':1, 'continual_g0':0},
    draws_per_trip=1, sessions=[7,8], bootstrap_seed=2026101747,
    bootstrap_replicates=10000, public_continuation_seed=2026101701,
    primary_comparison='continual_g1 minus capped_g1; no automatic promotion',
    retained_control='continual_g0 same public-base planner; do not hide negative gain',
    cache='static metadata; same current/60s-epoch/session cache and current local GPS',
    snapshot='first600s paired unchanged prefix, extra1200s directed public-goal stress',
    windows=WINDOWS, utility_gate_lower=-.02, no_private_outcome_selection=True)
sha, read, save, rng = old.sha, old.read, old.save, old.rng


def freeze(output):
    if output.exists():
        raise FileExistsError('Use a new protocol ID; retain all failed/completed attempts')
    extra = ['benchmark/engines/continual_slack.py',
             'experiments/continual_geoi_study_20261017.py', 'tests/test_continual_slack.py',
             'tests/test_continual_geoi_study.py',
             'experiments/run_continual_geoi_background_20261017.py',
             'requirements-continual-geoi.txt',
             'docs/research/2026-10-17_continual_geoi_protocol.md']
    names = sorted(set(old.source_closure()+extra))
    pins = {name:sha(ROOT/name) for name in names}
    save(output/'protocol.json', dict(schema='continual-geoi-pilot-v1', config=CONFIG,
        methods=METHODS, frozen_before_private_draws=True,
        reused_native_dataset_sha256=sha(old.DATA), source_sha256=pins,
        public_inputs_sha256={name:sha(ROOT/name) for name in (public.NETWORK, public.RESOURCES, public.DESIGN)},
        runtime=dict(python=platform.python_version(), packages={name:version(name) for name in
            ('numpy','scipy','networkx','scikit-learn','pyproj','sumolib','shapely','pytest')}),
        split_counts={'train':20,'selection':8,'test':20},
        interpretation='exploratory paired pilot; reused first600s geometry, synthetic extension; not independent long-trip generalization',
        attacker='TRAIN-only ExtraTrees64/kNN1/kNN5 and geometry; ordered endpoints; loss-specific SELECTION before TEST',
        promotion='never automatic; retain existing main release and report'))
    for name in names:
        target=output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    print('FROZEN protocol and source before private draws', flush=True)


def validate_sources(output):
    protocol=read(output/'protocol.json')
    # JSON turns tuple windows into lists.
    if protocol['config'] != json.loads(json.dumps(CONFIG)) or sha(old.DATA)!=protocol['reused_native_dataset_sha256']:
        raise ValueError('Frozen config/data changed')
    for name, digest in protocol['source_sha256'].items():
        if sha(ROOT/name)!=digest or sha(output/'source_snapshot'/name)!=digest:
            raise ValueError('Frozen source changed: '+name)
    for name, digest in protocol['public_inputs_sha256'].items():
        if sha(ROOT/name)!=digest:
            raise ValueError('Frozen public input changed: '+name)
    return protocol


def resources(work):
    rn, ref, reply, ids=public.resources(work/'public_resources')
    _, inv, counts=np.unique(np.floor(rn.xy/120.).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    prior=1./counts[inv]; prior/=prior.sum()
    base=PublicAnchorModel(rn,ref,prior,spacing_m=200.,epsilon_release=.01,epsilon_test=.01,
                          cache_path=work/'public_resources'/'belief01.npz')
    model=ResponseAwareAnchorModel(base,reply)
    lo,hi=rn.xy.min(axis=0),rn.xy.max(axis=0)
    destinations=sorted(set(map(int,rn.tree.query([lo+(hi-lo)*[a,b] for a in [.15,.5,.85] for b in [.15,.5,.85]])[1])))
    table=DefinedPublicPurposeTable(ref,reply,ids,destinations=destinations)
    portfolio=ExactReferencePortfolio(reply,ref,table)
    goals=public.certified_goals(portfolio,width=24)
    library=TargetedExploreLibrary(rn,goals,portfolio,travel_scale_s=60)
    policy=TargetedCachePolicy(library,table,portfolio,use_simulator=True,novelty_weight=4,long_range_weight=4)
    projection=cKDTree(rn.xy[ids]).query(base.xy)[1]
    return rn,ref,reply,ids,model,policy,projection


def continuation(native, session_id, rn, library):
    """Generate truth before defense randomness; never select on protection."""
    by_time={int(p['time_s']):p for p in native}
    truth=[dict(t=t,state=int(rn.nearest(by_time[t]['lat'],by_time[t]['lon'])[0]),
                xy=list(rn.point_xy(by_time[t]['lat'],by_time[t]['lon'])),
                gps=[by_time[t]['lat'],by_time[t]['lon']], source='reused_native_GPS')
           for t in range(0,601,20)]
    seed=int.from_bytes(hashlib.sha256((str(CONFIG['public_continuation_seed'])+session_id).encode()).digest()[:8],'big')
    generator=np.random.default_rng(seed)
    goals=sorted(set(s for group in library.goals for s in group))
    state=truth[-1]['state']; goal=state
    for t in range(620,1801,20):
        if state==goal or (t-620)%120==0:
            feasible=[g for g in goals if g!=state and np.isfinite(library.to_goal(g)[0][state])]
            if feasible:
                goal=feasible[int(generator.integers(len(feasible)))]
        state=int(library.advance(state,goal))
        gps=list(rn.latlon(state))
        truth.append(dict(t=t,state=state,xy=list(rn.point_xy(*gps)),gps=gps,
                          source='synthetic_directed_road_continuation'))
    return truth


def generate(output,work):
    validate_sources(output)
    if (output/'generation_started.json').exists():
        raise FileExistsError('Never restart/reseed a partly generated study')
    save(output/'generation_started.json',dict(protocol_sha256=sha(output/'protocol.json')))
    rn,ref,reply,ids,model,policy,projection=resources(work)
    data=read(old.DATA)
    prepared=[]
    for family in data['families']:
        trips=[]
        for spec in family['evaluator_only']['sessions']:
            if spec['day'] in CONFIG['sessions']:
                trips.append(dict(session_id=spec['session_id'],truth=continuation(data['traces'][spec['session_id']],spec['session_id'],rn,policy.library)))
        prepared.append(dict(family_id=family['family_id'],split=family['split'],trips=trips))
    save(output/'stress_dataset.json.gz',prepared)
    print('Prepared ALL96 truth traces before private defense keys',flush=True)
    private=work/'private';private.mkdir(parents=True,exist_ok=True,mode=0o700);os.chmod(private,0o700)
    utility=old.Utility(ref,reply)
    for family in prepared:
        trips=[]
        for sample in family['trips']:
            token=sample['session_id'];key=secrets.token_bytes(32)
            fd=os.open(private/(token+'.key'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'wb') as stream:stream.write(key)
            options=dict(belief_model=model,k=5,theta_m=200.,read_interval_s=60.,utility_slack=.03)
            engines={'capped':PacedSlackProgressLaneDummy(rn,**options,budget=.24,horizon=12,rng=rng(key,'geo-init')),
                     'continual':ContinualPacedSlackProgressLaneDummy(rn,**options,unit_epsilon=.01,rng=rng(key,'geo-init'))}
            for engine in engines.values():
                engine.anchor_rng=rng(key,'geo-anchor');engine.dummy_rng=rng(key,'geo-dummy');engine.reset()
            clients={m:SessionCoverClient(reply.categories,len(reply.pois),k=5,response_l=60,
                session_token=token,static_catalogue_id=reply.sha256) for m in METHODS}
            ledgers={m:PersistentQueryBudget(private/(token+'--'+m+'.sqlite'),session_token=token,
                context_id=sha(output/'protocol.json'),gamma=CONFIG['gamma_by_arm'][m],joint=True,
                private_key=hmac.new(key,b'matched-Q-stream',hashlib.sha256).digest()) for m in METHODS}
            events={m:[] for m in METHODS};scores={m:[] for m in METHODS}
            states={m:[] for m in METHODS};certs={m:[] for m in METHODS}
            cost={m:dict(requests=0,request_bytes=0,reply_bytes=0) for m in METHODS}
            cached={m:[] for m in METHODS}
            def server(request):
                s=rn.nearest(*request['coordinate'])[0]
                return [list(map(int,row[row>=0])) for row in reply.query_indices(s)]
            destination=sample['truth'][-1]['state']
            for point in sample['truth']:
                t=point['t']; beliefs={}
                for label,engine in engines.items():
                    engine.protect_step(*point['gps'],float(t))
                    b=np.bincount(projection,weights=engine.belief.weights,minlength=len(ids));b/=b.sum()
                    beliefs[label]=b
                if t<=600:
                    assert engines['capped'].last_anchor==engines['continual'].last_anchor
                    assert np.array_equal(beliefs['capped'],beliefs['continual'])
                for m in METHODS:
                    belief=beliefs['capped' if m=='capped_g1' else 'continual']
                    def choose(previous,epsilon,private_rng):
                        return policy.select(previous,epsilon,private_rng,belief=belief,
                            known=mask_bits(clients[m].session_known),frames=3)
                    frame,certificate=ledgers[m].frame(float(t),choose)
                    coordinates=tuple(rn.latlon(s) for s in frame)
                    wire=clients[m].step(float(t),coordinates,server)
                    states[m].append(list(frame));certs[m].append(dict(t=t,**certificate))
                    events[m].append(dict(timestamp_s=float(t),coordinates=[list(c) for c in coordinates]))
                    scores[m].append(dict(t=t,**{cache:utility.score(point['state'],destination,wire[cache]) for cache in ('current','known','session')}))
                    cached[m].append(dict(t=t,unique=int(wire['session'].sum())))
                    cost[m]['requests']+=len(wire['requests'])
                    for field,items in [('request_bytes',wire['requests']),('reply_bytes',wire['replies'])]:
                        cost[m][field]+=sum(len(json.dumps(x,separators=(',',':')).encode()) for x in items)
                if t<=600:assert events['capped_g1'][-1]==events['continual_g1'][-1]
            for ledger in ledgers.values():ledger.close()
            trips.append(dict(session_id=token,draw=1,truth=sample['truth'],events=events,utility=scores,
                sampled_states=states,Q_certificates=certs,cost=cost,cache_counts=cached,
                GeoI={label:dict(ledger=e.evaluator_ledger,anchors=e.evaluator_anchors,
                    diagnostic_spent_per_m=e.spent_bound,public_prefix_bound_per_m=(.23 if label=='capped' else e.prefix_epsilon_bound)) for label,e in engines.items()}))
        save(output/'families'/(family['family_id']+'.json.gz'),dict(family_id=family['family_id'],split=family['split'],trips=trips))
        print('Generated',family['family_id'],family['split'],flush=True)
    save(output/'generation.json',dict(family_files_sha256={p.name:sha(p) for p in sorted((output/'families').glob('*.gz'))},
        stress_dataset_sha256=sha(output/'stress_dataset.json.gz'),private_keys_exported=False))


def families(output):
    receipt=read(output/'generation.json'); result=[]
    for name,digest in receipt['family_files_sha256'].items():
        path=output/'families'/name
        if sha(path)!=digest:raise ValueError('Evidence changed: '+name)
        result.append(read(path))
    if sha(output/'stress_dataset.json.gz')!=receipt['stress_dataset_sha256']:
        raise ValueError('Stress truth changed')
    return result


def utility_readout(groups):
    test=[f for f in groups if f['split']=='test']
    bootstrap=np.random.default_rng(CONFIG['bootstrap_seed']).integers(0,len(test),(CONFIG['bootstrap_replicates'],len(test)))
    results=[]
    for window,(start,end) in WINDOWS.items():
        for cache in ('current','known','session'):
            values={}
            for method in METHODS:
                rows=[]
                for family in test:
                    row={}
                    for purpose in (p.value for p in QueryPurpose):
                        v=[s[cache][purpose]['recall5'] for trip in family['trips'] for s in trip['utility'][method]
                           if start<=s['t']<=end and s[cache][purpose]['recall5'] is not None]
                        row[purpose]=float(np.mean(v)) if v else None
                    row['macro']=float(np.mean([v for v in row.values() if v is not None]))
                    rows.append(row)
                values[method]=rows
            for method in METHODS:
                metrics={}
                for purpose in values[method][0]:
                    x=np.array([r[purpose] for r in values[method]],dtype=float)
                    b=np.array([r[purpose] for r in values['capped_g1']],dtype=float)
                    if not (np.isfinite(x).all() and np.isfinite(b).all()):
                        raise ValueError('Missing family/purpose; report explicitly rather than drop')
                    difference=x-b
                    metrics[purpose]=dict(mean=float(x.mean()),paired_delta=float(difference.mean()),
                        paired95=np.quantile(difference[bootstrap].mean(axis=1),[.025,.975]).tolist(),
                        paired_families=len(test))
                results.append(dict(window=window,cache=cache,method=method,metrics=metrics,
                    utility_gate=all(v['paired95'][0]>=CONFIG['utility_gate_lower'] for v in metrics.values())))
    return results


def attack_sample(trip,scenario,method,rn,history):
    if scenario in ('S9','S10'):
        x,geometry=ordered_endpoint_features(trip['events'][method],scenario,rn,history,observable_close_s=1800.)
        return x,geometry,np.array([trip['truth'][0 if scenario=='S9' else -1]['xy']])
    return old.attack_sample(trip,scenario,method,rn,history)


def attacks(output,groups,rn):
    train=[f for f in groups if f['split']=='train'];selection=[f for f in groups if f['split']=='selection'];test=[f for f in groups if f['split']=='test']
    history=PublicHistory(rn,[[p['state'] for p in trip['truth']] for f in train for trip in f['trips']])
    results=[]
    for method in METHODS:
        for scenario in ('S1','S2','S3','S9','S10'):
            x=defaultdict(list);y=[]
            for f in train:
                for trip in f['trips']:
                    features,_,target=attack_sample(trip,scenario,method,rn,history)
                    for name,rows in features.items():x[name].extend(rows)
                    y.extend(target)
            bank=OrderedEndpointBank(x,y) if scenario in ('S9','S10') else LearnedAttack(np.array(x['shadow']),np.array(y))
            def errors(trip):
                features,geometry,target=attack_sample(trip,scenario,method,rn,history)
                prediction=(bank.predictions(trip['events'][method],scenario,rn,history,observable_close_s=1800.)
                    if scenario in ('S9','S10') else geometry|{'shadow_'+n:v for n,v in bank.predict(features['shadow']).items()})
                return {name:np.linalg.norm(v-target,axis=1) for name,v in prediction.items()}
            rows=[]
            for f in selection:
                for trip in f['trips']:
                    e=errors(trip)
                    for i in range(len(next(iter(e.values())))):
                        rows.append(dict(family_id=f['family_id'],session_id=trip['session_id']+f'-e{i}',seed=1,errors={n:float(v[i]) for n,v in e.items()}))
            chosen,stats=robust_select(rows,[f['family_id'] for f in selection])
            save(output/'attacker_selection'/(method+'--'+scenario+'.json'),dict(selected=chosen,
                selection_statistics=stats,fit_split='train',selection_split='selection',before_test_predictions=True))
            rows=[]
            for f in test:
                for trip in f['trips']:
                    rows.append(dict(family_id=f['family_id'],session_id=trip['session_id'],errors={n:v.tolist() for n,v in errors(trip).items()}))
            save(output/'attacker_predictions'/(method+'--'+scenario+'.json.gz'),rows)
            family_metrics=[]
            for f in test:
                subset=[r for r in rows if r['family_id']==f['family_id']]
                family_metrics.append(dict(family_id=f['family_id'],MAE_m=float(np.mean([np.mean(r['errors'][chosen['mae']]) for r in subset])),
                    Hit100=float(np.mean([np.mean(np.array(r['errors'][chosen['hit100']])<=100) for r in subset]))))
            results.append(dict(method=method,scenario=scenario,selected=chosen,family_metrics=family_metrics,
                MAE_m=float(np.mean([r['MAE_m'] for r in family_metrics])),Hit100=float(np.mean([r['Hit100'] for r in family_metrics]))))
            print('Scored attackers',method,scenario,flush=True)
    return results


def audit(groups,rn):
    travel=SparseTravel(rn,cache_limit=4096);transitions=0;prefix_checks=0
    for family in groups:
        for trip in family['trips']:
            assert len(trip['truth'])==91
            for label,geo in trip['GeoI'].items():
                spent=0;reads=0
                for point,row in zip(trip['truth'],geo['ledger']):
                    spent+=row['cost_units'];reads+=int(row['private_read'])
                    assert spent==row['spent_units']
                    if label=='capped':assert spent<=23
                    elif point['t']%60==0:assert row['private_read']
                if label=='continual':assert reads==31 and spent<=61
            for method in METHODS:
                assert len(trip['events'][method])==91 and trip['cost'][method]['requests']==455
                previous_count=0
                for event,frame,certificate,count in zip(trip['events'][method],trip['sampled_states'][method],trip['Q_certificates'][method],trip['cache_counts'][method]):
                    assert [list(rn.latlon(s)) for s in frame]==event['coordinates']
                    block=int(event['timestamp_s']//60)+1
                    gamma=Fraction(CONFIG['gamma_by_arm'][method])
                    assert Fraction.from_float(certificate['epsilon_Q_upper'])<=gamma/(block*(block+1))
                    assert certificate['reserved_prefix_epsilon_Q']==float(gamma*block/(block+1))
                    assert count['unique']>=previous_count;previous_count=count['unique'];prefix_checks+=1
                for a,b in zip(trip['sampled_states'][method],trip['sampled_states'][method][1:]):
                    assert all(y in travel.reachable(x,20.) for x,y in zip(a,b));transitions+=5
            assert trip['events']['capped_g1'][:31]==trip['events']['continual_g1'][:31]
            synthetic=[p['state'] for p in trip['truth'][30:]]
            assert all(b in travel.reachable(a,20.) for a,b in zip(synthetic,synthetic[1:]))
    return dict(status='pass',Q_prefix_checks=prefix_checks,directed_Q_transitions=transitions,
                unchanged_native_prefix_verified=True,numerical_sampler_certified=False)


def verify(output,work):
    validate_sources(output);groups=families(output);result=read(output/'results.json')
    rn,*_=public.resources(work/'public_resources')
    assert len(groups)==48 and {s:sum(f['split']==s for f in groups) for s in ('train','selection','test')}=={'train':20,'selection':8,'test':20}
    assert result['utility']==utility_readout(groups)
    assert result['audit']==audit(groups,rn)
    for entry in result['attackers']:
        selection=read(output/'attacker_selection'/(entry['method']+'--'+entry['scenario']+'.json'))
        assert selection['before_test_predictions'] and selection['fit_split']=='train'
        rows=read(output/'attacker_predictions'/(entry['method']+'--'+entry['scenario']+'.json.gz'))
        for metric in ('MAE_m','Hit100'):
            values=[]
            for family in [f for f in groups if f['split']=='test']:
                group=[r for r in rows if r['family_id']==family['family_id']]
                assert len(group)==2
                key=selection['selected']['mae' if metric=='MAE_m' else 'hit100']
                values.append(float(np.mean([np.mean(r['errors'][key]) if metric=='MAE_m' else np.mean(np.array(r['errors'][key])<=100) for r in group])))
            assert entry[metric]==float(np.mean(values))
    receipt=dict(status='pass',protocol_sha256=sha(output/'protocol.json'),results_sha256=sha(output/'results.json'),
                 source_and_evidence_pins=True,all_declared_arms_retained=True,main_release_changed=False)
    if (output/'validation.json').exists():assert read(output/'validation.json')==receipt
    else:save(output/'validation.json',receipt)
    print('VERIFIED sources, Q prefixes, paths, paired utility and attacker summaries',flush=True)


def score(output,work):
    validate_sources(output);groups=families(output);rn,*_=public.resources(work/'public_resources')
    checks=audit(groups,rn);utility=utility_readout(groups)
    save(output/'utility_results.json',dict(utility=utility,audit=checks))
    print('Utility and road/Q checks saved; fitting TRAIN attackers next',flush=True)
    attacker=attacks(output,groups,rn)
    test=[f for f in groups if f['split']=='test']
    geo={label:dict(mean_private_reads=float(np.mean([sum(r['private_read'] for r in t['GeoI'][label]['ledger']) for f in test for t in f['trips']])),
        mean_diagnostic_spent_per_m=float(np.mean([t['GeoI'][label]['diagnostic_spent_per_m'] for f in test for t in f['trips']])),
        worst_case_prefix_per_m=.23 if label=='capped' else .61) for label in ('capped','continual')}
    cost={m:{k:float(np.mean([t['cost'][m][k] for f in test for t in f['trips']])) for k in ('requests','request_bytes','reply_bytes')} for m in METHODS}
    save(output/'results.json',dict(protocol_sha256=sha(output/'protocol.json'),utility=utility,attackers=attacker,
        audit=checks,GeoI=geo,cost=cost,main_release_changed=False,automatic_promotion=False,
        interpretation='Pilot evidence only: repeated native first600s plus public directed-road synthetic extension; one fresh draw/trip; static418-POI cache may saturate',
        privacy='Q global bound unchanged conditional on common public support/clock and protected-only wire; Geo-I .23 full-session cap NOT claimed for continual arm; float/PCG not certified'))
    verify(output,work)
    write_readout(output)


def write_readout(output):
    result=read(output/'results.json');receipt=read(output/'validation.json')
    assert receipt['status']=='pass' and receipt['results_sha256']==sha(output/'results.json')
    lines=['# Continual Geo-I pilot: paired results', '',
        'Exploratory pilot: one fresh draw per trip, all48 families, reused native first10min plus a directed-road synthetic extension to30min. The main release/report are unchanged. No automatic promotion.', '',
        'All percentages below are empirical. Intervals are paired-family percentile95% bootstrap intervals; delta is against capped_g1 in percentage points. Static418-POI cache saturation can mask gains.', '',
        '| Window | Arm | Session Recall (%) | Delta (pp), 95% CI | -2pp gate, all purposes |',
        '|---|---|---:|---|---|']
    for entry in result['utility']:
        if entry['cache']!='session':continue
        metric=entry['metrics']['macro'];lo,hi=metric['paired95']
        lines.append(f"| {entry['window']} | {entry['method']} | {100*metric['mean']:.3f} | {100*metric['paired_delta']:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}] | {entry['utility_gate']} |")
    lines+=['','## Full30min utility by purpose and cache','',
        '| Cache | Arm | Nearest (%) | Fastest (%) | Radius (%) | Detour (%) |',
        '|---|---|---:|---:|---:|---:|']
    for entry in result['utility']:
        if entry['window']!='full_30min':continue
        values=' | '.join(f"{100*entry['metrics'][p.value]['mean']:.3f}" for p in QueryPurpose)
        lines.append(f"| {entry['cache']} | {entry['method']} | {values} |")
    lines+=['','## Localization attack bank','',
        'TRAIN fits and SELECTION chooses loss-specific attacks before TEST. S2 is late-window coordinate recovery here, not semantic stop detection. Zero empirical Hit100 is not zero inference probability.', '',
        '| Arm | Scenario | Hit100 (%) | MAE (m) |','|---|---|---:|---:|']
    for entry in result['attackers']:
        lines.append(f"| {entry['method']} | {entry['scenario']} | {100*entry['Hit100']:.3f} | {entry['MAE_m']:.1f} |")
    lines+=['','## Upstream and wire cost','',
        '| Upstream | Mean GPS updates | Mean hidden-branch spend (/m) | Public prefix bound (/m) |',
        '|---|---:|---:|---:|']
    for label,geo in result['GeoI'].items():
        lines.append(f"| {label} | {geo['mean_private_reads']:.2f} | {geo['mean_diagnostic_spent_per_m']:.4f} | {geo['worst_case_prefix_per_m']:.2f} |")
    lines+=['','| Arm | Point requests | Request JSON KB | Reply-index JSON KB |',
        '|---|---:|---:|---:|']
    for method,cost in result['cost'].items():
        lines.append(f"| {method} | {cost['requests']:.0f} | {cost['request_bytes']/1000:.2f} | {cost['reply_bytes']/1000:.2f} |")
    lines+=['','The continual arm has no .23/m whole-session upstream cap. The separate bounded-Q theorem retains its all-duration logical-wire bound under the documented assumptions; private adaptation decreases over time. Numerical REM/PCG are not certified pure Geo-I implementations. Byte counts omit full metadata/HTTP/TLS. This pilot does not establish real-world or arbitrary-duration high utility.','',
        f"Validation: {receipt['status']}; Q prefix checks: {result['audit']['Q_prefix_checks']}; directed Q transitions: {result['audit']['directed_Q_transitions']}.",'']
    target=output/'readout.md'
    with target.open('x') as stream:stream.write('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('run','freeze','generate','score','verify'))
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--work',type=Path,default=WORK)
    args=parser.parse_args();args.work.mkdir(parents=True,exist_ok=True)
    if args.work.resolve().is_relative_to(ROOT):raise ValueError('Private work must remain outside repo')
    try:
        if args.stage=='run':
            freeze(args.output);generate(args.output,args.work);score(args.output,args.work)
        elif args.stage=='freeze':freeze(args.output)
        elif args.stage=='generate':generate(args.output,args.work)
        elif args.stage=='score':score(args.output,args.work)
        else:verify(args.output,args.work)
    except Exception as exc:
        if args.output.exists() and not (args.output/'failure.json').exists():
            save(args.output/'failure.json',dict(stage=args.stage,error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
