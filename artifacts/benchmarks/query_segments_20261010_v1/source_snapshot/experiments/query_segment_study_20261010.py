"""Write-once matched-Geo-I multistep study on a new native SUMO cohort.

Freeze SOURCE/config BEFORE generating the cohort. Neither private choice keys
nor Q/Geo-I RNG keys are published. All frontier arms/failures are retained.
Utility is paired by family; attack fits TRAIN and selection only, never TEST.
"""
import argparse
import ast
from collections import defaultdict
import gzip
import hashlib
import hmac
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
import pickle
import secrets
import traceback

import numpy as np
from scipy.spatial import cKDTree

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from benchmark.public_segment_scores import FixedPublicPurposeTable, local_sorted_pois
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec, PurposeIndependentCoverClient
from benchmark.query_segments import PublicSegmentLibrary, SegmentPqbPolicy
from core.query_budget import PersistentQueryBudget
from benchmark.paper_comparators import PublicHistory
from evaluation.live_comparison_attacks import LearnedAttack, features_and_geometry
from evaluation.ordered_endpoint_attacks import OrderedEndpointBank, ordered_endpoint_features
from evaluation.robust_endpoint_selection import robust_select
from experiments import query_bundle_audit_20261010 as public
from experiments import build_query_segment_fresh_native_20261010 as cohort

ROOT=public.ROOT
OUT=ROOT/'artifacts/benchmarks/query_segments_20261010_v1'
DATA=cohort.OUT/'dataset.json.gz'
WORK=Path('/private/tmp/trajectory-query-segments-study-20261010-v1')
METHODS=('baseline',)+tuple(f'{mode}_g{gamma:g}' for mode in ('step','segment') for gamma in (.5,1.,2.))
CONFIG=dict(K=5,L=30,reference_k=5,GPS_interval_s=60.,server_interval_s=20.,
    nominal_GeoI_B_per_m=.24,effective_GeoI_Cs_per_m=.23,H=12,u_per_m=.01,
    theta_m=200.,slack=.03,Q_Gamma_frontier=[.5,1.,2.],primary='segment_g1',
    allocation='Gamma/[j(j+1)]; step divides each block allowance equally among3 frames',
    frames_per_segment=3,public_floor=.75,purpose_weights='fixed equal four; N/A planner availability0 with validity mask',
    radius_m=1000.,public_destination_quantiles=[.15,.5,.85],
    cache='same current and within-public60s-epoch cumulative replies for every arm',
    evaluation_sessions=[7,8],draws=1,window_s=[0,600],
    utility_gate='paired-family percentile95 lower >= -.02 for macro AND each of four purposes; current primary',
    bootstrap_replicates=10000,bootstrap_seed=2026101047,
    privacy_scope='ideal-real coordinate/request stream conditional on public activation/close; no account/IP anonymity',
    utility_scope='nearest/fastest/radius1000/actual-local-endpoint detour; static same-native-map synthetic cohort',
    score_domain='36 public nearest-grid states; actual utility is unverified analytically outside this table',
    promotion='ONLY primary segment_g1 passes ALL privacy/path and utility gates; otherwise keep opt-in')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    path=Path(path); content=path.read_bytes()
    return json.loads(gzip.decompress(content) if path.suffix=='.gz' else content)
def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if path.suffix=='.gz':payload=gzip.compress(payload,mtime=0)
    with path.open('xb') as f:f.write(payload)


def source_closure():
    pending=[str(Path(__file__).relative_to(ROOT)),'tests/test_query_segments.py'];found=set()
    def add(module):
        for p in (module.replace('.','/')+'.py',module.replace('.','/')+'/__init__.py'):
            if (ROOT/p).is_file() and p not in found:pending.append(p)
    while pending:
        name=pending.pop()
        if name in found:continue
        found.add(name);tree=ast.parse((ROOT/name).read_text());package=name[:-3].split('/')[:-1]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for a in node.names:add(a.name)
            if isinstance(node,ast.ImportFrom):
                parent=package[:len(package)-node.level+1] if node.level else []
                module='.'.join(parent+([node.module] if node.module else []))
                if module:add(module)
                for a in node.names:
                    if a.name!='*':add('.'.join(filter(None,(module,a.name))))
    return sorted(found|{'tests/test_query_segments.py','requirements.txt','requirements-sumo.txt','requirements-query-segments.txt'})


def freeze(output=OUT):
    if output.exists() or DATA.exists():raise FileExistsError('Freeze requires NEW output and not-yet-generated cohort')
    pins={name:sha(ROOT/name) for name in source_closure()}
    p=dict(schema='query-segment-study-v1',config=CONFIG,methods=list(METHODS),source_sha256=pins,
        runtime=dict(python=platform.python_version(),platform=platform.platform(),
                     packages={name:version(name) for name in ('numpy','scipy','networkx','scikit-learn','pyproj','sumolib','eclipse-sumo','sumo-data')}),
        public_inputs_sha256={p:sha(ROOT/p) for p in (public.NETWORK,public.RESOURCES)},
        cohort_generator_public_seed=cohort.PUBLIC_SEED,cohort_splits=cohort.SPLIT_COUNTS,
        frozen_before_cohort_generation=True,baseline='unchanged Geo-I Slack engine adapted to L30, per-session Cs.23/m; not relabelled historical L10 scores',
        pairing='ONE baseline Geo-I engine per trip supplies protected history/belief to all arms; domain-separated private Q streams',
        attacker='TRAIN only ExtraTrees64/kNN1/kNN5 + public geometry; S9/S10 ordered track bank; loss-specific robust selection mean+/-SE',
        failure='retain partial outputs and failure receipt; no replace/reseed/reselect after TEST')
    save(output/'protocol.json',p)
    for name in pins:
        target=output/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    print('Frozen source/config BEFORE fresh cohort generation',flush=True)


def validate_sources(output):
    p=read(output/'protocol.json')
    if p['config']!=CONFIG or p['methods']!=list(METHODS):raise ValueError('Frozen configuration changed')
    for name,digest in p['source_sha256'].items():
        if sha(ROOT/name)!=digest or sha(output/'source_snapshot'/name)!=digest:raise ValueError('Source changed: '+name)
    for name,digest in p['public_inputs_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Public input changed: '+name)
    return p


class Utility:
    def __init__(self,reference,reply):
        self.ranking=MultiPurposeRoadRanking(reference,cache_limit=4096)
        self.reply=reply;self.references={}
        self.all=np.ones(len(reply.pois),bool)
    def refs(self,state,destination):
        key=(state,destination)
        if key not in self.references:
            result={}
            for purpose in QueryPurpose:
                rows=[]
                for category in self.ranking.categories:
                    spec=QuerySpec(purpose,category,k=5,
                        radius_m=1000. if purpose==QueryPurpose.WITHIN_RADIUS else None,
                        destination_state=destination if purpose==QueryPurpose.MIN_DETOUR else None)
                    ids=local_sorted_pois(self.ranking,state,self.all,spec)
                    rows.append((ids,ids[:5]))
                result[purpose.value]=rows
            self.references[key]=result
        return self.references[key]
    def score(self,state,destination,mask):
        # Entire local sorted answer, metric truncation ONLY in evaluator.
        result={}
        for purpose,rows in self.refs(state,destination).items():
            recalls=[];answer_size=0
            for full,reference in rows:
                answer=[i for i in full if mask[i]];answer_size+=len(answer)
                if reference:recalls.append(len(set(reference)&set(answer[:5]))/len(reference))
            result[purpose]=dict(recall5=float(np.mean(recalls)) if recalls else None,
                defined_categories=len(recalls),NA_categories=len(rows)-len(recalls),full_sorted_answer_count=answer_size)
        return result


def rng(key,domain):
    return np.random.default_rng(np.frombuffer(hmac.new(key,domain.encode(),hashlib.sha256).digest(),dtype='<u4'))


def generate(output=OUT, work=WORK):
    validate_sources(output)
    if (output/'generation.json').exists() or (output/'generation_started.json').exists():raise FileExistsError('Generation cannot restart or replace draws')
    save(output/'generation_started.json',dict(dataset_sha256=sha(DATA),protocol_sha256=sha(output/'protocol.json')))
    data=read(DATA)
    rn,reference,reply,ids=public.resources(work/'public_resources')
    goals=[a.states for a in public.library(reference,reply,ids) if len(a.states)==5]
    _,inv,counts=np.unique(np.floor(rn.xy/120.).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    prior=1./counts[inv];prior/=prior.sum()
    base=PublicAnchorModel(rn,reference,prior,spacing_m=200.,epsilon_release=.01,epsilon_test=.01,
                           cache_path=work/'public_resources'/'belief01.npz')
    model=ResponseAwareAnchorModel(base,reply)
    q=[.15,.5,.85];lo,hi=rn.xy.min(axis=0),rn.xy.max(axis=0)
    destinations=sorted(set(map(int,rn.tree.query([lo+(hi-lo)*[a,b] for a in q for b in q])[1])))
    table=FixedPublicPurposeTable(reference,reply,ids,destinations=destinations)
    library=PublicSegmentLibrary(rn,goals)
    selector=SegmentPqbPolicy(library,table)
    projection=cKDTree(rn.xy[ids]).query(base.xy)[1]
    utility=Utility(reference,reply)
    private=work/'private';private.mkdir(parents=True,exist_ok=True,mode=0o700)
    os.chmod(private,0o700)
    save(output/'resources.json',dict(native_states=len(rn),latent_states=ids.tolist(),public_goals=goals,
        public_destinations=destinations,public_purpose_validity=table.validity.tolist(),
        reference_sha256=reference.sha256,reply_sha256=reply.sha256,belief_sha256=base.sha256,
        catalogue_sha256=rn.catalogue_sha256,score_formula='mean(frame, fixed four purposes), with fixed public prototypes',
        empirical_NA='excluded from Recall, retained in count; planner scores availability0',
        analytical_actual_utility_status='unverified: no certified TV/grid/provider allowances'))
    for family in data['families']:
        trips=[]
        for spec in family['evaluator_only']['sessions']:
            if spec['day'] not in (7,8):continue
            token=spec['session_id'];key=secrets.token_bytes(32)
            fd=os.open(private/(token+'.key'),os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'wb') as f:f.write(key)
            engine=PacedSlackProgressLaneDummy(rn,belief_model=model,k=5,budget=.24,horizon=12,
                theta_m=200.,read_interval_s=60.,utility_slack=.03,rng=rng(key,'geo-init'))
            engine.anchor_rng=rng(key,'geo-anchor');engine.dummy_rng=rng(key,'geo-dummy');engine.reset()
            clients={m:PurposeIndependentCoverClient(reply.categories,len(reply.pois),k=5,response_l=30) for m in METHODS}
            ledgers={m:PersistentQueryBudget(private/(token+'--'+m+'.sqlite'),session_token=token,
                context_id=sha(output/'protocol.json'),joint=m.startswith('segment'),gamma=float(m.split('_g')[1]),
                private_key=hmac.new(key,m.encode(),hashlib.sha256).digest()) for m in METHODS if m!='baseline'}
            by_time={int(p['time_s']):p for p in data['traces'][token]}
            destination=int(rn.nearest(by_time[600]['lat'],by_time[600]['lon'])[0])
            events={m:[] for m in METHODS};scores={m:[] for m in METHODS};cost={m:dict(requests=0,request_bytes=0,reply_bytes=0) for m in METHODS}
            certs={m:[] for m in ledgers};truth=[];sampled_states={m:[] for m in METHODS}
            def server(request):
                state=rn.nearest(*request['coordinate'])[0]
                return [list(map(int,row[row>=0])) for row in reply.query_indices(state)]
            for t in range(0,601,20):
                point=by_time[t];gps=(point['lat'],point['lon'])
                baseline=engine.protect_step(*gps,float(t))
                belief=np.bincount(projection,weights=engine.belief.weights,minlength=len(ids))
                belief/=belief.sum()
                state=int(rn.nearest(*gps)[0]);truth.append(dict(t=t,xy=list(rn.point_xy(*gps)),state=state))
                for m in METHODS:
                    if m=='baseline':
                        coordinates=baseline
                        sampled_states[m].append(list(engine.evaluator_states[-1]))
                    else:
                        frames=3 if m.startswith('segment') else 1
                        def choose(previous,epsilon,private_rng):
                            return selector.select(previous,epsilon,private_rng,belief=belief,frames=frames)
                        frame,certificate=ledgers[m].frame(float(t),choose)
                        coordinates=tuple(rn.latlon(s) for s in frame)
                        sampled_states[m].append(list(frame))
                        certs[m].append(dict(t=t,**certificate))
                    wire=clients[m].step(float(t),coordinates,server)
                    events[m].append(dict(timestamp_s=float(t),coordinates=[list(c) for c in coordinates]))
                    scores[m].append(dict(t=t,**{cache:utility.score(state,destination,wire[cache]) for cache in ('current','known')}))
                    cost[m]['requests']+=len(wire['requests'])
                    for field,items in [('request_bytes',wire['requests']),('reply_bytes',wire['replies'])]:
                        cost[m][field]+=sum(len(json.dumps(i,separators=(',',':')).encode()) for i in items)
            for m,ledger in ledgers.items():
                assert ledger.reserved <= FractionGamma(m)
                ledger.close()
            assert engine.spent_bound<=.23+1e-12
            trips.append(dict(session_id=token,truth=truth,events=events,sampled_states=sampled_states,utility=scores,cost=cost,
                Q_certificates=certs,GeoI_anchors=engine.evaluator_anchors,GeoI_ledger=engine.evaluator_ledger,
                GeoI_spent_per_m=engine.spent_bound,GeoI_pairing='one shared protected history for all arms'))
        save(output/'families'/(family['family_id']+'.json.gz'),dict(family_id=family['family_id'],split=family['split'],trips=trips))
        print('Generated',family['family_id'],family['split'],flush=True)
    save(output/'generation.json',dict(dataset_sha256=sha(DATA),family_files_sha256={p.name:sha(p) for p in sorted((output/'families').glob('*.gz'))},private_keys_exported=False,all_declared_methods_retained=True))


def FractionGamma(method):
    from fractions import Fraction
    return Fraction(method.split('_g')[1])


def load_families(output):
    receipt=read(output/'generation.json')
    result=[]
    for name,digest in receipt['family_files_sha256'].items():
        p=output/'families'/name
        if sha(p)!=digest:raise ValueError('Family evidence changed')
        result.append(read(p))
    return result


def utility_readout(families):
    purposes=[p.value for p in QueryPurpose];test=[f for f in families if f['split']=='test']
    means={};counts={}
    for cache in ('current','known'):
        for m in METHODS:
            rows=[];NA=0
            for f in test:
                values={}
                for p in purposes:
                    valid=[s[cache][p]['recall5'] for trip in f['trips'] for s in trip['utility'][m] if s[cache][p]['recall5'] is not None]
                    NA+=sum(s[cache][p]['NA_categories'] for trip in f['trips'] for s in trip['utility'][m])
                    values[p]=float(np.mean(valid)) if valid else None
                values['macro']=float(np.mean([v for v in values.values() if v is not None])) if any(v is not None for v in values.values()) else None
                rows.append(values)
            means[cache,m]=rows;counts[cache,m]=NA
    result=[]
    bootstrap=np.random.default_rng(CONFIG['bootstrap_seed']).integers(0,len(test),(10000,len(test)))
    for cache in ('current','known'):
        for m in METHODS:
            metrics={};eligible=True
            for p in purposes+['macro']:
                x=np.array([v[p] if v[p] is not None else np.nan for v in means[cache,m]])
                b=np.array([v[p] if v[p] is not None else np.nan for v in means[cache,'baseline']])
                valid=np.isfinite(x)&np.isfinite(b)
                if not valid.all():
                    eligible=False
                diff=x-b
                ci=np.quantile(diff[bootstrap].mean(axis=1),[.025,.975]).tolist() if valid.all() else None
                metrics[p]=dict(mean=float(np.mean(x[valid])) if valid.any() else None,
                    paired_delta=float(np.mean(diff[valid])) if valid.any() else None,paired95=ci,
                    paired_families=int(valid.sum()))
                eligible &= ci is not None and ci[0]>=-.02
            result.append(dict(method=m,cache=cache,metrics=metrics,NA_category_events=counts[cache,m],utility_gate=bool(eligible)))
    return result


def attack_sample(trip, scenario, method, rn, history):
    events=trip['events'][method];xy=np.array([t['xy'] for t in trip['truth']])
    if scenario in ('S9','S10'):
        features,geometry=ordered_endpoint_features(events,scenario,rn,history,observable_close_s=600.)
        return features,geometry,xy[[0 if scenario=='S9' else -1]]
    if scenario=='S2':
        events=events[-4:];xy=xy[-4:].mean(axis=0,keepdims=True)
        features,geometry=features_and_geometry(events,'S2',rn,history)
    elif scenario=='S1':
        parts=[features_and_geometry([e],'S1',rn,history) for e in events]
        features=np.concatenate([a for a,b in parts]);geometry={n:np.concatenate([b[n] for a,b in parts]) for n in parts[0][1]}
    else:
        features,geometry=features_and_geometry(events,'S3',rn,history)
    return {'shadow':features},geometry,xy


def attacks(output,families,rn):
    train=[f for f in families if f['split']=='train'];selection=[f for f in families if f['split']=='selection'];test=[f for f in families if f['split']=='test']
    history=PublicHistory(rn,[[p['state'] for p in t['truth']] for f in train for t in f['trips']])
    summaries=[]
    for m in METHODS:
        for scenario in ('S1','S2','S3','S9','S10'):
            x=defaultdict(list);y=[]
            for f in train:
                for trip in f['trips']:
                    features,_,target=attack_sample(trip,scenario,m,rn,history)
                    for name,values in features.items():x[name].extend(values)
                    y.extend(target)
            bank=OrderedEndpointBank(x,y) if scenario in ('S9','S10') else LearnedAttack(np.array(x['shadow']),np.array(y))
            def predictions(trip):
                features,geo,target=attack_sample(trip,scenario,m,rn,history)
                if scenario in ('S9','S10'):
                    pred=bank.predictions(trip['events'][m],scenario,rn,history,observable_close_s=600.)
                else:pred=geo|{'shadow_'+n:v for n,v in bank.predict(features['shadow']).items()}
                return target,{name:np.linalg.norm(v-target,axis=1) for name,v in pred.items()}
            sel=[]
            for f in selection:
                for trip in f['trips']:
                    _,errs=predictions(trip)
                    # Each service event a row; family averaging gives equal
                    # weight to paired trips and complete native windows.
                    for i in range(len(next(iter(errs.values())))):
                        sel.append(dict(family_id=f['family_id'],session_id=trip['session_id']+f'-e{i}',seed=1,errors={n:float(e[i]) for n,e in errs.items()}))
            chosen,stats=robust_select(sel,[f['family_id'] for f in selection])
            model_path=output/'attacker_models'/(m+'--'+scenario+'.pkl.gz');model_path.parent.mkdir(exist_ok=True)
            with model_path.open('xb') as stream:stream.write(gzip.compress(pickle.dumps(bank,protocol=5),mtime=0))
            save(output/'attacker_selection'/(m+'--'+scenario+'.json'),dict(selected=chosen,selection_statistics=stats,model_sha256=sha(model_path),fit_split='train',selection_split='selection',before_test_predictions=True))
            rows=[]
            for f in test:
                for trip in f['trips']:
                    target,errs=predictions(trip)
                    rows.append(dict(family_id=f['family_id'],session_id=trip['session_id'],truth_xy=target.tolist(),errors={n:e.tolist() for n,e in errs.items()}))
            save(output/'attacker_predictions'/(m+'--'+scenario+'.json.gz'),rows)
            family_metrics=[]
            for f in test:
                group=[r for r in rows if r['family_id']==f['family_id']]
                family_metrics.append(dict(family_id=f['family_id'],MAE_m=float(np.mean([np.mean(r['errors'][chosen['mae']]) for r in group])),
                    Hit100=float(np.mean([np.mean(np.array(r['errors'][chosen['hit100']])<=100) for r in group]))))
            summaries.append(dict(method=m,scenario=scenario,selected_attacker=chosen,
                MAE_m=float(np.mean([r['MAE_m'] for r in family_metrics])),Hit100=float(np.mean([r['Hit100'] for r in family_metrics])),family_metrics=family_metrics))
            print('Scored attackers',m,scenario,flush=True)
    return summaries


def audit_paths(families,rn):
    from evaluation.lane_travel import SparseTravel
    travel=SparseTravel(rn,cache_limit=4096);checked=0
    for f in families:
        for trip in f['trips']:
            for m in METHODS:
                states=trip['sampled_states'][m];assert len(states)==31
                for frame,event in zip(states,trip['events'][m]):
                    assert len(frame)==5
                    assert [list(rn.latlon(s)) for s in frame]==event['coordinates']
                for previous,current in zip(states,states[1:]):
                    assert all(y in travel.reachable(x,20.) for x,y in zip(previous,current)),(f['family_id'],m,previous,current)
                    checked+=5
    return dict(status='pass',directed_track_transitions_checked=checked,coincident_lane_states_not_resnapped=True)


def verify(output=OUT):
    validate_sources(output);families=load_families(output)
    dataset=read(DATA)
    assert read(output/'generation.json')['dataset_sha256']==sha(DATA)
    assert {f['family_id']:f['split'] for f in families}=={f['family_id']:f['split'] for f in dataset['families']}
    rn,_,_,_=public.resources(WORK/'public_resources')
    path_report=audit_paths(families,rn)
    for f in families:
        assert len(f['trips'])==2
        for trip in f['trips']:
            assert trip['GeoI_spent_per_m']<=.23+1e-12
            for m in METHODS:
                ev=trip['events'][m];assert len(ev)==31 and [e['timestamp_s'] for e in ev]==list(range(0,601,20))
                assert trip['cost'][m]['requests']==155
                # Coordinate-to-lane snapping is ambiguous at coincident lane
                # coordinates; validate actual sampled states separately below.
                if m!='baseline':
                    certificates=trip['Q_certificates'][m]
                    assert all(c['epsilon_Q_upper']<=c['allocated_epsilon_Q'] and c['reserved_prefix_epsilon_Q']<=float(FractionGamma(m)) for c in certificates)
                    assert all(c['actual_expected_utility_lower'] is None for c in certificates)
            assert all(l['spent_units']<=23 for l in trip['GeoI_ledger'])
    summary=read(output/'results.json')
    assert summary['utility']==utility_readout(families)
    primary=next(r for r in summary['utility'] if r['method']=='segment_g1' and r['cache']=='current')
    assert summary['path_audit']==path_report
    assert summary['promotion']['utility_gate']==primary['utility_gate']
    assert summary['promotion']['promote']==(primary['utility_gate'] and summary['promotion']['privacy_path_checks'])
    certificate=dict(status='pass',protocol_sha256=sha(output/'protocol.json'),results_sha256=sha(output/'results.json'),
        family_count=len(families),all_frontier_arms_retained=True,numerical_sampler_certified=False)
    if (output/'validation.json').exists():assert read(output/'validation.json')==certificate
    else:save(output/'validation.json',certificate)
    print('Verified all families, prefix caps, utility gates and evidence hashes',flush=True)


def score(output=OUT):
    validate_sources(output);families=load_families(output)
    rn,_,_,_=public.resources(WORK/'public_resources')
    utility=utility_readout(families)
    path_report=audit_paths(families,rn)
    attacker=attacks(output,families,rn)
    costs={m:{key:float(np.mean([t['cost'][m][key] for f in families if f['split']=='test' for t in f['trips']])) for key in ('requests','request_bytes','reply_bytes')} for m in METHODS}
    primary=next(r for r in utility if r['method']=='segment_g1' and r['cache']=='current')
    save(output/'results.json',dict(protocol_sha256=sha(output/'protocol.json'),dataset_sha256=sha(DATA),utility=utility,
        attackers=attacker,cost=costs,path_audit=path_report,promotion=dict(primary='segment_g1',utility_gate=primary['utility_gate'],
            privacy_path_checks=True,promote=primary['utility_gate'],default_engine_changed=False),
        limitations=['single private draw per synthetic trip','same native map/generator; no real GPS/cross-city confirmation',
            'S2 parked-window coordinate proxy; no clinical/sensitive-place semantic labels',
            'S1 isolated-event localization; S3 whole-window reconstruction; S9/S10 observed public start/close',
            'public table utility only analytically; actual measured utility has no certified belief error',
            'finite attacker bank is empirical evidence, not proof of identity anonymity',
            'float/PCG64 sampler remains uncertified']))
    verify(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('freeze','generate','score','verify'))
    parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--work',type=Path,default=WORK)
    args=parser.parse_args()
    try:
        if args.stage=='freeze':freeze(args.output)
        elif args.stage=='generate':generate(args.output,args.work)
        elif args.stage=='score':score(args.output)
        else:verify(args.output)
    except Exception as exc:
        if args.output.exists() and not (args.output/'failure.json').exists():
            save(args.output/'failure.json',dict(stage=args.stage,error=repr(exc),traceback=traceback.format_exc()))
        raise
