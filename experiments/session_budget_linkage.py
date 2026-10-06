"""Predeclared development readout of persistent multi-session Geo-I accounting.

Existing cohorts/transcripts stay immutable. A research family bundles six
sessions in one conservative accounting scope, covering all same-person and
same-vehicle subsets. This is NOT a human/account identity privacy experiment.
Secret sampler keys and live SQLite ledgers stay outside public evidence files.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import hmac
from itertools import combinations
import json
import os
from pathlib import Path
import secrets
import time

import numpy as np

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.endpoint_noise import EndpointNoiseProgressLaneDummy
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from core.session_budget import FixedEpochPolicy, PersistentEpochBudget, FixedEpochProtectedSessions
from evaluation.identity_future import pair_features, ClassifierBank, classification_metrics
from evaluation.lane_travel import LanePoiService
from evaluation.live_poi import AvailabilityWorld, LivePointService, EpochResponseCache, score_returned
from experiments.identity_future_eval import DATA, S4_ROLES, SPLITS, SEED
from experiments.identity_future_refine import shape_features
from experiments.public_research_resources import ROOT, POIS, load_public_research_resources, sha

DEFAULT_OUT = ROOT/'artifacts/benchmarks/session_budget_20261005/round2'
DEFAULT_CACHE = Path('/private/tmp/trajectory-research-20261005-public-map')
POLICIES = [dict(id='global_cap_H12', cap=.23, horizon=12),
            dict(id='global_cap_H8', cap=.23, horizon=8),
            dict(id='equal_total_reset_control_H12', cap=.345, horizon=12)]
METHODS = ['raw', 'per_session_reset_H12']+[p['id'] for p in POLICIES]
REPS = (0, 1)
BACKBONE = ['core/mechanisms.py', 'benchmark/engines/filtered_cover.py',
            'benchmark/engines/matched_filter.py', 'benchmark/engines/paced_guard.py',
            'benchmark/engines/paced_slack.py', 'benchmark/engines/budgeted.py']


def write(path, payload):
    if path.exists():
        raise FileExistsError(f'Never overwrite evidence: {path}')
    encoded = (json.dumps(payload, indent=2, allow_nan=False)+'\n').encode()
    if path.suffix == '.gz':
        path.write_bytes(gzip.compress(encoded, mtime=0))
    else:
        path.write_bytes(encoded)


def protocol():
    files = BACKBONE+['core/session_budget.py', 'benchmark/engines/endpoint_noise.py',
        'benchmark/anchor_belief.py', 'benchmark/response_aware_belief.py',
        'evaluation/live_poi.py', 'evaluation/identity_future.py',
        'experiments/identity_future_refine.py', str(Path(__file__).relative_to(ROOT)),
        'experiments/public_research_resources.py', str(DATA.relative_to(ROOT)),
        'artifacts/benchmarks/identity_future_20261005/linkage_results.json',
        'artifacts/benchmarks/identity_future_20261005/linkage_public_transcripts.json']
    return dict(schema='fixed-epoch-linked-coordinate-protocol-v1', splits=SPLITS,
        split_unit='Entire route family, all linked sessions and RNG repetitions',
        policies=[dict(p, public_parameters=FixedEpochPolicy('public-research-day', 0., 86400.,
            p['cap'], 6, p['horizon'], 60.).public_parameters()) for p in POLICIES],
        per_session_reference=dict(nominal_B=.06, effective_cap=.0575, horizon=12,
            unit_epsilon_per_m=.0025, total_six_session_effective_cap=.345),
        allocation='Six public starts, equal fixed shares; consume full cap at start, no recycling; '
            'all six family sessions conservatively share one local accounting scope',
        no_private_schedule='Public role order base/repeat/new_device/shared_device/companion/partial; '
            'slots start i*3600 seconds, one fixed day, relative observations every60s; '
            'neither GPS, actual travel duration nor private branch chooses H or slot budget',
        clock='Range(0, len(trace),60), all ticks; true duration remains PUBLIC and observable',
        roles=S4_ROLES, repetitions=list(REPS), K=5, L=20, theta_m=200., utility_slack=.03,
        public_world=dict(seed=SEED+1703, available_probability=.8, refresh_s=60),
        private_rng='32-byte OS secret key outside repository; HMAC family/rep domain, then '
            'epoch/slot/anchor|dummy|initialization domains; no public seed/session-id-only sampler',
        paired_control='Same secret streams and primitive ε=.0025 for reset and equal-total .345 '
            'epoch cap; assert event coordinates, private branch ledger and states exactly equal',
        comparable_budget='Global H8 versus H12 both effective .23/day; reset reference .345/day '
            'is explicitly unmatched. Equal-total control tests whether accounting alone changes output.',
        attacker='Public summary and Hungarian matched K-track shape x ExtraTrees96/leaf2/depth16 '
            'and standardized weighted kNN1/5/15. Threshold grid0:.05:1 on selection only. '
            'Separate selection-best balanced accuracy and selection-best ROC-AUC; retain full test bank/family AUC.',
        negative_control='Independently shuffle training pair labels within each family and repetition, '
            'never shuffle rows across train/selection/test',
        utility='Available top5 per POI category at true local GPS after union of top20/category/Q replies; '
            'nearest road-distance purpose only, not S7 multipurpose evidence; macro categories then events/sessions/rep/family',
        time_windows='Public first8 sampled ticks and t>=480s long tail, same windows for every configuration; '
            'also retain every actual post-budget-exhaustion tick separately as evaluator-only diagnostic',
        bytes='Compact JSON candidate coordinates plus L; POI-ID category lists only; no HTTP/TLS; '
            'per public input tick including denied events',
        claims='Geo-I REM plus noisy reuse are unchanged; composition under ideal kernels and fixed public '
            'clock/length only. Coordinate protection cannot hide account/IP or deterministic identity metadata. '
            'Finite float samplers are numerical approximations, not secure pure-DP implementations.',
        scope='Previously inspected synthetic SUMO family cohorts on reconstructed public graph; '
            'development only, no independent confirmation and no comparison to faithful external models',
        source_sha256={f:sha(ROOT/f) for f in files})


def resources(cache, policies):
    rn, _, reference, _, old, ranking, metadata = load_public_research_resources(cache)
    pois = [{k:v for k,v in p.items() if k not in ('vertex', 'access_offset_m')}
            for p in json.loads(POIS.read_text())['pois_used']]
    context = PublicPoiContext(LanePoiService(rn, pois, k=20),
                              cache/f'endpoint-depth-{rn.catalogue_sha256[:12]}-L20.npz')
    # The anchor model's .prior is over latent cells, not full lane states.
    # Reconstruct the same public 120m-cell mass before new emission models.
    _, inv, counts = np.unique(np.floor(rn.xy/120.).astype(np.int64), axis=0,
                               return_inverse=True, return_counts=True)
    prior = 1./counts[inv]
    prior /= prior.sum()
    beliefs = {}
    for p in policies:
        eps = p.unit_epsilon_per_m
        if eps in beliefs:
            continue
        base = PublicAnchorModel(rn, reference, prior, spacing_m=200.,
            epsilon_release=eps, epsilon_test=eps,
            cache_path=cache/f'session-budget-belief-{rn.catalogue_sha256[:12]}-{eps:.17g}.npz')
        beliefs[eps] = ResponseAwareAnchorModel(base, context)
    metadata = dict(metadata, response_l=20, reply20_sha256=context.sha256,
                    actual_emission_beliefs={str(e):b.sha256 for e,b in beliefs.items()})
    return rn, beliefs, ranking, metadata


def factory(rn, belief):
    def build(allocation, streams):
        engine = EndpointNoiseProgressLaneDummy(rn, belief_model=belief, privacy_scale=1.,
            k=5, budget=allocation.nominal_budget_per_m, horizon=allocation.horizon,
            theta_m=200., read_interval_s=allocation.read_interval_s, utility_slack=.03,
            rng=streams.initialization)
        engine.anchor_rng, engine.dummy_rng = streams.anchor, streams.dummy
        engine.reset()
        return engine
    return build


def event(j, t, coordinates):
    return dict(event_id=f'e{j:04d}', timestamp_s=float(t), candidates=[
        dict(candidate_id=f'candidate_{i:04d}', lat=float(lat), lon=float(lon))
        for i, (lat, lon) in enumerate(coordinates)])


def service_readout(public, gps, rn, ranking, world, ledger=None):
    server, cache = LivePointService(ranking, world, response_l=20), EpochResponseCache(ranking.n)
    rows, request_bytes, response_bytes = [], 0, 0
    for j, (ev, point) in enumerate(zip(public['events'], gps)):
        t = ev['timestamp_s']; epoch = world.epoch(t)
        coordinates = [[c['lat'],c['lon']] for c in ev['candidates']]
        replies = [server.query(rn.nearest(*q)[0], epoch) for q in coordinates]
        _, known = cache.receive(epoch, replies)
        state, distance = rn.nearest(point['lat'], point['lon'])
        available = world.at_epoch(epoch)
        score = score_returned(ranking.top(state, available, 5), ranking.top(state, known, 5), available)
        req = len(json.dumps(dict(timestamp_s=t, coordinates=coordinates, L=20), separators=(',', ':')).encode()) if coordinates else 0
        resp = len(json.dumps([[[ranking.pois[i]['id'] for i in category] for category in reply]
                             for reply in replies], separators=(',', ':')).encode()) if coordinates else 0
        request_bytes += req; response_bytes += resp
        rows.append(dict(timestamp_s=t, recall=score['recall'], category_recall=score['category_recall'],
            mapping_distance_m=float(distance), request_bytes=req, response_bytes=resp,
            public_first8=j<8, public_tail_480s=t>=480.,
            evaluator_filter_stopped=(ledger is not None and ledger[j]['branch']=='postprocess')))
    def mean(where):
        values = [r['recall'] for r in rows if where(r) and r['recall'] is not None]
        return float(np.mean(values)) if values else None
    return dict(rows=rows, recall=mean(lambda r:True), first8_recall=mean(lambda r:r['public_first8']),
        tail_recall=mean(lambda r:r['public_tail_480s']),
        after_filter_recall=mean(lambda r:r['evaluator_filter_stopped']),
        request_bytes=request_bytes, response_bytes=response_bytes,
        bytes_per_tick=(request_bytes+response_bytes)/len(rows), eligible_ticks=sum(r['recall'] is not None for r in rows),
        total_ticks=len(rows), private_read_count=sum(r['private_read'] for r in ledger) if ledger else None)


def nested_family_mean(items, getter):
    family = defaultdict(lambda: defaultdict(list))
    for item in items:
        value = getter(item)
        if value is not None:
            family[item['family_id']][item['rep']].append(value)
    return float(np.mean([np.mean([np.mean(v) for v in reps.values()]) for reps in family.values()])) if family else None


def classify(rows, label):
    parts = {s:[r for r in rows if r['split']==s] for s in SPLITS}
    truth = {s:np.array([r[label] for r in rs]) for s,rs in parts.items()}
    features = [('summary', pair_features), ('shape', shape_features)]
    selections, test_predictions, shuffled_predictions = {}, {}, {}
    for fname, feature in features:
        x = {s:np.array([feature(*r['public_pair']) for r in rs]) for s,rs in parts.items()}
        bank = ClassifierBank(x['train'], truth['train'], SEED)
        ps, pt = bank.probability(x['selection']), bank.probability(x['test'])
        # Preserve family/rep blocks; no heldout rows participate in fitting.
        shuffled = truth['train'].copy()
        for family in SPLITS['train']:
            for rep in REPS:
                ids = [i for i,r in enumerate(parts['train']) if r['family_id']==family and r['rep']==rep]
                rng = np.random.default_rng(SEED+int(hashlib.sha256(f'{family}/{rep}/{label}'.encode()).hexdigest()[:8],16))
                shuffled[ids] = rng.permutation(shuffled[ids])
        negative = ClassifierBank(x['train'], shuffled, SEED)
        ns, nt = negative.probability(x['selection']), negative.probability(x['test'])
        for name, probabilities in ps.items():
            key = fname+'/'+name
            threshold = min(np.linspace(0.,1.,21), key=lambda t:(
                -classification_metrics(truth['selection'], probabilities>=t, probabilities)['balanced_accuracy'], abs(t-.5),t))
            selections[key] = dict(threshold=float(threshold),
                metrics=classification_metrics(truth['selection'], probabilities>=threshold, probabilities))
            test_predictions[key] = (pt[name]>=threshold, pt[name])
            nt_threshold = min(np.linspace(0.,1.,21), key=lambda t:(
                -classification_metrics(truth['selection'], ns[name]>=t, ns[name])['balanced_accuracy'],abs(t-.5),t))
            shuffled_predictions[key] = (nt[name]>=nt_threshold, nt[name],
                classification_metrics(truth['selection'],ns[name]>=nt_threshold,ns[name]))
    by_accuracy = min(selections, key=lambda n:(-selections[n]['metrics']['balanced_accuracy'],
        -selections[n]['metrics']['roc_auc'],n))
    by_auc = min(selections, key=lambda n:(-selections[n]['metrics']['roc_auc'],
        -selections[n]['metrics']['balanced_accuracy'],n))
    def metrics(prediction, subset=None):
        pred, prob = prediction
        ids = list(range(len(parts['test']))) if subset is None else subset
        return classification_metrics(truth['test'][ids], pred[ids], prob[ids])
    def result(name):
        p = test_predictions[name]
        families = {f:metrics(p,[i for i,r in enumerate(parts['test']) if r['family_id']==f]) for f in SPLITS['test']}
        return dict(attacker=name, threshold=selections[name]['threshold'], selection=selections[name]['metrics'],
            test=metrics(p), test_families=families,
            family_macro_auc=float(np.mean([v['roc_auc'] for v in families.values()])),
            maximum_family_auc=max(v['roc_auc'] for v in families.values()))
    negative_name = min(shuffled_predictions, key=lambda n:(-shuffled_predictions[n][2]['balanced_accuracy'],
                            -shuffled_predictions[n][2]['roc_auc'],n))
    return dict(selection_best_balanced_accuracy=result(by_accuracy), selection_best_auc=result(by_auc),
        selection_bank=selections, test_bank_descriptive_only={n:metrics(p) for n,p in test_predictions.items()},
        test_family_bank_descriptive_only={f:{n:metrics(p,[i for i,r in enumerate(parts['test']) if r['family_id']==f])
            for n,p in test_predictions.items()} for f in SPLITS['test']},
        shuffled_training_control=dict(attacker=negative_name, test=metrics(shuffled_predictions[negative_name][:2])),
        test_rows=[dict(family_id=r['family_id'], rep=r['rep'], session_pair=r['session_pair'],
            truth=int(y), **{n:float(prob[i]) for n,(_,prob) in test_predictions.items()})
            for i,(r,y) in enumerate(zip(parts['test'],truth['test']))],
        pair_counts={s:len(rs) for s,rs in parts.items()})


def run(out, cache, private_dir):
    if out.exists():
        raise FileExistsError('Choose a new artifact folder; retain every prior run')
    out.mkdir(parents=True)
    sealed = protocol(); write(out/'protocol.json', sealed)
    policy_objects = [FixedEpochPolicy('public-research-day', 0., 86400., p['cap'], 6, p['horizon'], 60.) for p in POLICIES]
    # Private key is OS-generated, never a published evaluation seed. Preserve
    # it locally for exact replay without exposing it through attack/evidence.
    private_dir.mkdir(parents=True, exist_ok=False)
    os.chmod(private_dir, 0o700)
    key_file = private_dir/'master.key'
    fd = os.open(key_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    master = secrets.token_bytes(32)
    with os.fdopen(fd, 'wb') as file:
        file.write(master)
    rn, beliefs, ranking, metadata = resources(cache, policy_objects)
    write(out/'resources.json', metadata)
    data = json.loads(DATA.read_text())
    world = AvailabilityWorld(ranking.n, SEED+1703, probability=.8, epoch_seconds=60)
    executions, accounting = [], []
    started = time.perf_counter()
    for family in sorted(data['families'], key=lambda f:f['family_id']):
        fid = family['family_id']
        if not any(fid in fs for fs in SPLITS.values()):
            continue
        selected = {s['role']:s for s in family['sessions'] if s['role'] in S4_ROLES}
        assert len(selected)==6
        for rep in REPS:
            secret = hmac.new(master, f'research-budget-family-v1/{fid}/{rep}'.encode(), hashlib.sha256).digest()
            ledgers, streams = {}, {}
            for config, policy in zip(POLICIES, policy_objects):
                ledgers[config['id']] = PersistentEpochBudget(policy,
                    private_dir/f'{fid}-rep{rep}-{config["id"]}.sqlite', private_key=secret)
                streams[config['id']] = FixedEpochProtectedSessions(ledgers[config['id']],
                    factory(rn, beliefs[policy.unit_epsilon_per_m]))
            for slot, role in enumerate(S4_ROLES):
                session = selected[role]; sid=session['session_id']; trace=data['traces'][sid]
                inputs=[trace[i] for i in range(0,len(trace),60)]
                start=slot*3600.; relative=[float(p['time_s']-trace[0]['time_s']) for p in inputs]
                assert relative[-1]<3600.
                events={m:[] for m in METHODS}
                for name, stream in streams.items():
                    assert stream.start_session(sid,start)
                control = streams['equal_total_reset_control_H12']
                reset_engine = factory(rn, beliefs[control._allocation.unit_epsilon_per_m])(
                    control._allocation, control.ledger.private_rng_streams(control._allocation))
                for j,(point,t) in enumerate(zip(inputs,relative)):
                    events['raw'].append(event(j,t,[(point['lat'],point['lon'])]))
                    q_reset=reset_engine.protect_step(point['lat'],point['lon'],t)
                    events['per_session_reset_H12'].append(event(j,t,q_reset))
                    for name, stream in streams.items():
                        q=stream.protect_step(start+t, lambda point=point:(point['lat'],point['lon']))
                        events[name].append(event(j,t,q))
                        if name=='equal_total_reset_control_H12':
                            assert q==q_reset, 'Equal-total accounting must not change coordinates'
                    assert reset_engine.spent_bound<=.0575+1e-12
                diagnostics={name:stream.evaluator_current_session() for name,stream in streams.items()}
                assert diagnostics['equal_total_reset_control_H12']['ledger']==reset_engine.evaluator_ledger
                assert diagnostics['equal_total_reset_control_H12']['states']==reset_engine.evaluator_states
                diagnostics['per_session_reset_H12']=dict(spent_per_m=reset_engine.spent_bound,
                    ledger=reset_engine.evaluator_ledger, states=reset_engine.evaluator_states,
                    private_reads=sum(r['private_read'] for r in reset_engine.evaluator_ledger))
                for name in METHODS:
                    public=dict(events=events[name])
                    ledger=None if name=='raw' else diagnostics[name]['ledger']
                    utility=service_readout(public,inputs,rn,ranking,world,ledger)
                    executions.append(dict(family_id=fid, rep=rep, session_id=sid, role=role,
                        split=next(s for s,fs in SPLITS.items() if fid in fs), method=name,
                        public=public, utility=utility, evaluator_only=dict(
                            person_id=session['person_id'], physical_vehicle_id=session['physical_vehicle_id'],
                            privacy=None if name=='raw' else diagnostics[name])))
                close=start+relative[-1]+1.
                for stream in streams.values():
                    stream.close_session(close)
            for name, stream in streams.items():
                summary=stream.evaluator_summary()
                assert np.isclose(summary['reserved_cap_per_m'],summary['epoch_cap_per_m'],rtol=1e-12,atol=0)
                accounting.append(dict(family_id=fid,rep=rep,method=name,**summary))
                ledgers[name].close()
            reference_spent=sum(e['evaluator_only']['privacy']['spent_per_m'] for e in executions
                if e['family_id']==fid and e['rep']==rep and e['method']=='per_session_reset_H12')
            accounting.append(dict(family_id=fid,rep=rep,method='per_session_reset_H12',
                spent_per_m=reference_spent, epoch_cap_per_m=.345, reserved_cap_per_m=.345,
                note='Reference resets .0575/session, not a .0575 user-level cap'))
        print('Budget generated',fid,'executions',len(executions),'elapsed_s',round(time.perf_counter()-started,1),flush=True)
    write(out/'evaluator_executions.json.gz',executions)
    write(out/'epoch_accounting.json',accounting)
    write(out/'public_transcripts.json.gz',dict(schema='attacker-visible-events-only-v1',
        transcripts=[dict(public_index=i, method=e['method'], public=e['public']) for i,e in enumerate(executions)]))
    results={}
    for method in METHODS:
        subset=[e for e in executions if e['method']==method]
        pair_rows=[]
        for fid in sorted(sum(SPLITS.values(),[])):
            for rep in REPS:
                group=[e for e in subset if e['family_id']==fid and e['rep']==rep]
                assert len(group)==6
                for a,b in combinations(group,2):
                    pair_rows.append(dict(family_id=fid, rep=rep, split=a['split'],
                        session_pair=[a['session_id'],b['session_id']], public_pair=[a['public'],b['public']],
                        same_person=int(a['evaluator_only']['person_id']==b['evaluator_only']['person_id']),
                        same_vehicle=int(a['evaluator_only']['physical_vehicle_id']==b['evaluator_only']['physical_vehicle_id'])))
        utility={split:{key:nested_family_mean([e for e in subset if e['split']==split],lambda e,key=key:e['utility'][key])
            for key in ('recall','first8_recall','tail_recall','after_filter_recall','bytes_per_tick','private_read_count')}
            for split in SPLITS}
        counts={s:dict(sessions=sum(e['split']==s for e in subset),
            ticks=sum(e['utility']['total_ticks'] for e in subset if e['split']==s),
            tail_ticks=sum(sum(r['public_tail_480s'] for r in e['utility']['rows']) for e in subset if e['split']==s),
            after_filter_ticks=sum(sum(r['evaluator_filter_stopped'] for r in e['utility']['rows']) for e in subset if e['split']==s))
            for s in SPLITS}
        results[method]=dict(S4={label:classify(pair_rows,label) for label in ('same_person','same_vehicle')},
                            utility=utility,counts=counts)
        print('Budget scored',method,flush=True)
    for f,digest in sealed['source_sha256'].items():
        assert sha(ROOT/f)==digest, f'Pinned historical source changed during experiment: {f}'
    write(out/'readout.json',dict(schema='fixed-epoch-linked-coordinate-readout-v1',
        protocol_sha256=sha(out/'protocol.json'), public_transcripts_sha256=sha(out/'public_transcripts.json.gz'),
        evaluator_executions_sha256=sha(out/'evaluator_executions.json.gz'),
        epoch_accounting_sha256=sha(out/'epoch_accounting.json'), results=results,
        source_invariants=dict(backbone_files_unchanged=True, previous72session_evidence_unchanged=True),
        equal_total_control=dict(all_coordinates_states_and_private_ledgers_equal=True,
            compared_sessions=72*len(REPS), baseline_epoch_cap_per_m=.345),
        total_executions=len(executions), unique_input_sessions=72, rng_repetitions=len(REPS),
        private_replay=dict(master_key_external_only=True, key_sha256=hashlib.sha256(master).hexdigest(),
            ledgers_not_in_public_artifacts=True), elapsed_s=time.perf_counter()-started,
        conclusion_gate='No superiority claim is inferred. Report family residual AUC, utility, '
            'stronger coordinate cap and equal-budget null control together. Test families were already inspected development.'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--cache-dir',type=Path,default=DEFAULT_CACHE)
    parser.add_argument('--private-state-dir',type=Path)
    args=parser.parse_args()
    private=args.private_state_dir or Path('/private/tmp')/f'geo-i-private-budget-{secrets.token_hex(8)}'
    run(args.out,args.cache_dir,private)


if __name__=='__main__':
    main()
