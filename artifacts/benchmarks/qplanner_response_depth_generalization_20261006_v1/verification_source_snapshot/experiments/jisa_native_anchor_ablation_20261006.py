"""Development primitive ablation: matched REM/planar Geo-I on frozen native trips.

No old source evidence is changed. The native test was previously inspected;
new secret-keyed draws do not make it a new confirmation dataset. A write-once
protocol precedes generation and all attacker selections precede test scoring.
"""
import argparse
import ast
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import pickle
import secrets
import time
import traceback

import numpy as np

from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.engines.planar_paced import PlanarPacedLaneDummy
from benchmark.planar_anchor import PlanarAnchorModel
from benchmark.public_poi_context import PublicPoiContext
from benchmark.versioned_static_poi_cache import VersionedStaticPoiCache
from core.session_budget import FixedEpochPolicy, PersistentEpochBudget, FixedEpochProtectedSessions
from evaluation.candidate_future_attack import CandidateFutureAttack, forecast_metrics, coordinates
from evaluation.lane_travel import LanePoiService
from experiments.build_future_sumo_cohort import ROOT, sha, save
from experiments.future_sumo_eval import DATA, WORK as RESOURCE_WORK, native_resources, compressed_save
from experiments.native_future_retrieval_depth import OUT as DEPTH, PHASES, load
from experiments.native_static_cache_20261006 import aggregate

OUT = ROOT/'artifacts/benchmarks/jisa_native_anchor_ablation_20261006_v1'
WORK = Path('/private/tmp/jisa-native-anchor-20261006-v1')
METHODS = ('raw', 'rem_epoch8', 'planar_epoch8')
PROTECTED = METHODS[1:]
POLICIES = ('current_only', 'versioned_epoch8')
SPLITS = {'train': [f'native-{i:02d}' for i in range(1, 13)],
          'selection': [f'native-{i:02d}' for i in range(13, 19)],
          'test': [f'native-{i:02d}' for i in range(19, 25)]}
OLD_EVIDENCE = (
    'artifacts/benchmarks/future_native_20261005_v1/protocol.json',
    'artifacts/benchmarks/future_native_20261005_v1/public_transcripts.json.gz',
    'artifacts/benchmarks/future_native_20261005_v1/results.json',
    'artifacts/benchmarks/native_versioned_static_20261006_v1/results.json',
)
SOURCE_FILES = (
    'experiments/jisa_native_anchor_ablation_20261006.py',
    'experiments/future_sumo_eval.py', 'experiments/native_static_cache_20261006.py',
    'experiments/native_future_retrieval_depth.py', 'experiments/build_future_sumo_cohort.py',
    'core/session_budget.py', 'core/mechanisms.py',
    'benchmark/planar_anchor.py', 'benchmark/engines/planar_paced.py',
    'benchmark/engines/paced_slack.py', 'benchmark/engines/paced_guard.py',
    'benchmark/engines/matched_filter.py', 'benchmark/engines/slack_progress.py',
    'benchmark/engines/progress_cover.py', 'benchmark/engines/filtered_cover.py',
    'benchmark/engines/quotient_cover.py', 'benchmark/anchor_belief.py',
    'benchmark/response_aware_belief.py', 'benchmark/public_poi_context.py',
    'benchmark/static_poi_cache.py', 'benchmark/versioned_static_poi_cache.py',
    'data/lane_states.py', 'core/road_network.py', 'evaluation/lane_travel.py',
    'evaluation/candidate_future_attack.py', 'evaluation/identity_future.py',
    'requirements.txt', 'requirements-sumo.txt',
)


def project_sources():
    """Pin the static transitive local import closure, plus dependency lists."""
    pending = list(SOURCE_FILES); found = set()
    while pending:
        name = pending.pop()
        if name in found: continue
        path = ROOT/name
        if not path.is_file(): raise FileNotFoundError(name)
        found.add(name)
        if path.suffix != '.py': continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else ([node.module] if isinstance(node, ast.ImportFrom) and not node.level and node.module else [])
            for module in modules:
                candidate = module.replace('.', '/')
                for relative in (candidate+'.py', candidate+'/__init__.py'):
                    if (ROOT/relative).is_file() and relative not in found: pending.append(relative)
    return sorted(found)


def policy():
    return FixedEpochPolicy('jisa-native-matched-eight-trip-epoch', 0., 12000.,
                           total_effective_epsilon_per_m=.23, session_slots=8,
                           horizon=12, read_interval_s=60.)


def configuration():
    return {'budget': policy().public_parameters(), 'K': 5, 'reply_depth_L': 20,
            'planner_reply_depth_L': 10, 'reference_top_k': 5,
            'theta_m': 200., 'utility_slack': .03, 'state_spacing_m': 40.,
            'policies': POLICIES, 'methods': METHODS}


def declaration():
    source_paths = {str(DATA.relative_to(ROOT)),
                    'artifacts/datasets/future_controlled_20261005_v2/manifest.json',
                    'artifacts/datasets/future_controlled_20261005_v1/public_native.net.xml.gz',
                    'artifacts/benchmarks/research_loop/resources.json',
                    str((DEPTH/'public_reply40.npz').relative_to(ROOT)),
                    str((DEPTH/'results.json').relative_to(ROOT)), *project_sources()}
    return {'schema': 'jisa-native-anchor-development-protocol-v1',
            'configuration': configuration(), 'splits': SPLITS, 'phases_s': PHASES,
            'source_sha256': {p: sha(ROOT/p) for p in sorted(source_paths)},
            'preserve_old_evidence_sha256': {p: sha(ROOT/p) for p in OLD_EVIDENCE},
            'comparison': 'REM vs full-plane planar Laplace with private reuse and respective matched emissions; same public planner/reference10,cap,clock,K5,L20,map,static service',
            'draws': 'NEW secret32-byte OS key per family retained privately0700/0600; same family key and public epoch ID pair methods; independent HMAC initialization/anchor/dummy streams per session slot; no key or seed exported',
            'cache': 'same current-only and same causal versioned static ID cache per family across eight publicly timed sessions; reset/version invalidation; no future replies',
            'attacker': 'same CandidateFutureAttack geometry/ExtraTrees finite bank trained per mechanism,task,stage; fit12 families,select6,report6; six histories plus one query causal prefix, TWO known public candidates',
            'selection': 'maximum selection balancedaccuracy,then minimum logloss,then lexicalname; write every selection before any test score; no defense/model candidate selected by this ablation',
            'controls': 'raw positive control; within-family TRAIN-label permutation; shared-fork ambiguity retained',
            'utility': 'static all-POI distance category-top5 union coverage; no liveavailability/traffic measurement; conditional recall and N/A/coverage retained',
            'cost': 'per-Q UTF8 compactJSON request and full id/category/lat/lon reply estimate including repeated records; not HTTP,TLS,latency or measured server traffic',
            'privacy': 'coordinate-only declared epoch; ideal kernel/float approximation caveat; no account/IP concealment; same .23 composed cap,not actual-spend matching',
            'status': 'DEVELOPMENT on previously inspected native families; new draws are not fresh dataset confirmation; primitive ablation,not faithful CCS2013/PETS2014 reproduction',
            'failure_policy': 'write failure.json with traceback if any stage fails; preserve partial outputs and do not replace family/seed; new run requires new artifact/version'}


def validate_protocol(out):
    stored = json.loads((out/'protocol.json').read_text())
    expected = json.loads(json.dumps(declaration()))
    if stored != expected:
        raise ValueError('Sealed protocol/source has changed; use a new version')
    if (out/'protocol.sha256').read_text().strip() != sha(out/'protocol.json'):
        raise ValueError('Protocol receipt mismatch')
    return stored


def declare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'protocol.json').exists():
        return validate_protocol(out)
    save(out/'protocol.json', declaration())
    (out/'protocol.sha256').write_text(sha(out/'protocol.json')+'\n')
    snapshots = out/'source_snapshot'; snapshots.mkdir()
    for name in project_sources():
        destination = snapshots/name; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT/name).read_bytes())
    return validate_protocol(out)


def make_engine(method, rn, belief, allocation, streams):
    if method not in PROTECTED:
        raise ValueError('Declared protected method required')
    if (belief.rn is not rn or belief.context.k != 10 or
            not np.isclose(belief.epsilon_release, allocation.unit_epsilon_per_m, rtol=1e-12) or
            not np.isclose(belief.epsilon_test, allocation.unit_epsilon_per_m, rtol=1e-12) or
            belief.theta_m != 200.):
        raise ValueError('Same public map/L10 and allocation-matched emission required')
    model = PlanarAnchorModel(belief) if method == 'planar_epoch8' else belief
    cls = PlanarPacedLaneDummy if method == 'planar_epoch8' else PacedSlackProgressLaneDummy
    engine = cls(rn, belief_model=model, k=5, budget=allocation.nominal_budget_per_m,
                 horizon=allocation.horizon, theta_m=200., read_interval_s=allocation.read_interval_s,
                 utility_slack=.03, rng=streams.initialization)
    engine.anchor_rng, engine.dummy_rng = streams.anchor, streams.dummy
    engine.reset()
    return engine


def private_key(path):
    if path.exists():
        if path.stat().st_mode & 0o077:
            raise ValueError('Existing key permissions too broad')
        value = path.read_bytes()
        if len(value) != 32: raise ValueError('Invalid private key')
        return value
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    value = secrets.token_bytes(32)
    with os.fdopen(fd, 'wb') as stream: stream.write(value)
    return value


def generate(out, work):
    validate_protocol(out)
    if any((out/name).exists() for name in ('public_transcripts.json.gz', 'private_accounting.json.gz', 'generation_started.json')):
        raise FileExistsError('Generation is write-once; retain existing/partial evidence')
    save(out/'generation_started.json', {'protocol_sha256': sha(out/'protocol.json'),
                                       'started_utc': datetime.now(timezone.utc).isoformat()})
    data = load(DATA)
    started = time.perf_counter()
    rn, reference, planner_reply, beliefs, metadata = native_resources(data, RESOURCE_WORK)
    reply = PublicPoiContext(LanePoiService(rn, list(reference.pois), k=40), DEPTH/'public_reply40.npz')
    depth_metadata = json.loads((DEPTH/'results.json').read_text())['resources']
    assert rn.catalogue_sha256 == depth_metadata['catalogue_sha256']
    assert reply.sha256 == depth_metadata['reply40_sha256']
    assert np.array_equal(planner_reply.signatures, reply.signatures[:, :, :10])
    assert np.array_equal(reference.signatures, reply.signatures[:, :, :5])
    assert len(reply.pois) == 418
    resource_elapsed = time.perf_counter()-started
    private = work/'private_state'; private.mkdir(parents=True, exist_ok=True); os.chmod(private, 0o700)
    groups, accounting = [], []
    started = time.perf_counter()
    for family in data['families']:
        key = private_key(private/f'{family["family_id"]}.key')
        clients, ledgers = {}, {}
        for method in PROTECTED:
            ledger = PersistentEpochBudget(policy(), private/f'{family["family_id"]}-{method}.sqlite', private_key=key)
            def factory(allocation, streams, m=method):
                return make_engine(m, rn, beliefs[allocation.unit_epsilon_per_m], allocation, streams)
            ledgers[method] = ledger
            clients[method] = FixedEpochProtectedSessions(ledger, factory)
        streams = {m: [] for m in METHODS}; session_truth = []
        for slot, spec in enumerate(family['evaluator_only']['sessions']):
            assert spec['depart_s'] == slot*1500.
            trace = data['traces'][spec['session_id']]; by_time = {int(p['time_s']): p for p in trace}
            clock = sorted(set(range(0, 601, 20)) | ({family['clocks']['shared_fork_t'], family['clocks']['turn_visible_t']} if slot >= 6 else set()))
            events = {m: [] for m in METHODS}; reads = {m: [] for m in PROTECTED}
            for client in clients.values(): assert client.start_session(f'opaque-{slot}', spec['depart_s'])
            for j, t in enumerate(clock):
                point = by_time[t]; gps = point['lat'], point['lon']; outputs = {'raw': (gps,)}
                for method, client in clients.items():
                    def supplier(m=method, g=gps, when=t):
                        reads[m].append(when); return g
                    outputs[method] = client.protect_step(spec['depart_s']+t, supplier)
                    assert len(outputs[method]) == 5
                for method, qs in outputs.items():
                    events[method].append({'event_id': f'e{j:04d}', 'timestamp_s': float(t),
                        'candidates': [{'candidate_id': f'q{k:04d}', 'lat': float(lat), 'lon': float(lon)} for k, (lat, lon) in enumerate(qs)]})
            ledger_info = {}
            for method, client in clients.items():
                current = client.evaluator_current_session()
                assert len(reads[method]) == current['private_reads'] <= 11
                assert all(b-a >= 60 for a, b in zip(reads[method], reads[method][1:]))
                assert current['spent_per_m'] <= .02875+1e-12
                current['supplier_times_s'] = reads[method]
                current['anchor_primitive'] = client._engine.anchor.name
                ledger_info[method] = current
                client.close_session(spec['depart_s']+600.)
            end = by_time[600]
            destination_xy = coordinates({'events': [{'event_id': 'e', 'timestamp_s': 0.,
                'candidates': [{'candidate_id': 'q', 'lat': end['lat'], 'lon': end['lon']}]}]})[0][0][0].tolist()
            labels = {}
            if slot >= 6:
                for stage in ('shared_fork_t', 'turn_visible_t'):
                    t = family['clocks'][stage]
                    future = next(p['edge_id'] for p in trace if p['time_s'] > t and not p['edge_id'].startswith(':')
                        and (stage != 'shared_fork_t' or p['edge_id'] != family['fork_edge']))
                    assert future == family['public_context']['choices'][spec['choice_index']]['edge_id']
                    labels[stage] = future
            session_truth.append({'slot': slot, 'choice_index': spec['choice_index'], 'destination_role': spec['destination_role'],
                                  'destination_xy': destination_xy, 'next_edge_labels': labels, 'ledger': ledger_info})
            for method in METHODS: streams[method].append({'events': events[method]})
        epoch = {method: client.evaluator_summary() for method, client in clients.items()}
        assert all(v['reserved_cap_per_m'] == .23 and v['spent_per_m'] <= .23+1e-12 for v in epoch.values())
        for ledger in ledgers.values(): ledger.close()
        group = {'public_scope': len(groups), 'public_context': family['public_context'],
                 'public_clocks': {name: family['clocks'][name] for name in ('shared_fork_t', 'turn_visible_t')}, 'streams': streams}
        truth = {'family_id': family['family_id'], 'split': family['split'], 'public_scope': len(groups),
                 'evaluator_sessions': session_truth, 'epoch_accounting': epoch}
        groups.append(group); accounting.append(truth)
        # Preserve every completed family if a later family fails; partial logs are not the final result.
        partial = out/'partial_families'; partial.mkdir(exist_ok=True)
        compressed_save(partial/f'{family["family_id"]}.json.gz', {'public': group, 'evaluator_only': truth})
        print('Matched anchor family complete', family['family_id'], {m: round(v['spent_per_m'], 5) for m, v in epoch.items()}, flush=True)
    compressed_save(out/'public_transcripts.json.gz', {'schema': 'jisa-matched-public-native-v1', 'groups': groups,
        'contract': 'Q/times plus declared known TWO-candidate context; subject association is public sideinformation; no keys,GPS truth,branch ledger,private labels in features'})
    compressed_save(out/'private_accounting.json.gz', {'schema': 'jisa-matched-evaluator-only-v1', 'rows': accounting,
        'protocol_sha256': sha(out/'protocol.json'), 'public_transcript_sha256': sha(out/'public_transcripts.json.gz'),
        'resources': {**metadata, 'reply40_sha256': reply.sha256, 'planar_belief_sha256': PlanarAnchorModel(beliefs[.00125]).sha256},
        'resource_setup_elapsed_s': resource_elapsed, 'generation_elapsed_s': time.perf_counter()-started,
        'private_keys_exported': False})


def forecast_rows(data, public, private, method, stage):
    rows = []
    for group, truth in zip(public['groups'], private['rows']):
        cut = group['public_clocks'][stage+'_t']; streams = group['streams'][method]
        for slot in (6, 7):
            y = truth['evaluator_sessions'][slot]
            rows.append({'family_id': truth['family_id'], 'split': truth['split'], 'query_slot': slot,
                'query': {'events': [e for e in streams[slot]['events'] if e['timestamp_s'] <= cut]},
                'histories': streams[:6], 'public_context': group['public_context'],
                'choice_index': y['choice_index'], 'destination_role': y['destination_role'],
                'destination_xy': y['destination_xy'], 'actual_next_edge': y['next_edge_labels'][stage+'_t']})
    return rows


def select_attacks(out, data, public, private):
    selections = {}; fitted = {}; directory = out/'attack_models'; directory.mkdir()
    for method in METHODS:
        for stage in ('shared_fork', 'turn_visible'):
            rows = forecast_rows(data, public, private, method, stage)
            train = [r for r in rows if r['split'] == 'train']; selection = [r for r in rows if r['split'] == 'selection']
            for task, history in (('S5_next_edge', False), ('S6_history_destination', True)):
                name = f'{method}--{stage}--{task}'
                model = CandidateFutureAttack(train, use_history=history)
                negative = CandidateFutureAttack(train, use_history=history, permutation=True)
                predictions = [model.predict(r['query'], r['public_context'], r['histories']) for r in selection]
                scores = {n: forecast_metrics(selection, [p[n] for p in predictions]) for n in sorted(predictions[0])}
                selected = min(scores, key=lambda n: (-scores[n]['balanced_accuracy'], scores[n]['log_loss'], n))
                files = {}
                for label, instance in (('model', model), ('permutation', negative)):
                    path = directory/f'{name}--{label}.pickle'
                    with path.open('xb') as stream: pickle.dump(instance, stream, protocol=5)
                    files[label] = {'path': str(path.relative_to(out)), 'sha256': sha(path)}
                selections[name] = {'selected_attacker': selected, 'selection_scores': scores,
                    'selection_predictions': [{n: p.tolist() for n, p in bank.items()} for bank in predictions],
                    'models': files, 'train_family_ids': SPLITS['train'], 'selection_family_ids': SPLITS['selection']}
                fitted[name] = (model, negative, rows)
    save(out/'attack_selection.json', selections)  # Before any test metric is evaluated.
    return fitted, selections


def utility(out, data, public, private):
    rn, reference, _, _, metadata = native_resources(data, RESOURCE_WORK)
    reply = PublicPoiContext(LanePoiService(rn, list(reference.pois), k=40), DEPTH/'public_reply40.npz')
    assert reply.sha256 == private['resources']['reply40_sha256']
    families = {f['family_id']: f for f in data['families']}
    poi_ids = [p['id'] for p in reply.pois]; index = {name: i for i, name in enumerate(poi_ids)}
    responses = {}; rows = []; wire_rows = []
    def response(state):
        if state not in responses:
            ids = reply.signatures[reply.access[state], :, :20].ravel(); ids = ids[ids >= 0]
            records = [{k: reply.pois[int(i)][k] for k in ('id', 'category', 'lat', 'lon')} for i in ids]
            size = len(json.dumps({'results': records}, separators=(',', ':'), ensure_ascii=False).encode())
            responses[state] = list(map(int, ids)), records, size
        return responses[state]
    for group, truth in zip(public['groups'], private['rows']):
        family = families[truth['family_id']]
        caches = {m: VersionedStaticPoiCache(poi_ids, catalogue_version=reply.sha256,
            epoch_id=policy().epoch_id, start_s=0., end_s=12000.) for m in METHODS}
        for slot, spec in enumerate(family['evaluator_only']['sessions']):
            trace = {int(p['time_s']): p for p in data['traces'][spec['session_id']]}
            for method in METHODS:
                for event in group['streams'][method][slot]['events']:
                    t = int(event['timestamp_s']); absolute = spec['depart_s']+t
                    received = [response(rn.nearest(q['lat'], q['lon'])[0]) for q in event['candidates']]
                    current = {i for ids, _, _ in received for i in ids}
                    cached_names = caches[method].receive(absolute, [p for _, records, _ in received for p in records],
                        catalogue_version=reply.sha256, epoch_id=policy().epoch_id)
                    cached = {index[name] for name in cached_names}
                    gps = trace[t]; state, _ = rn.nearest(gps['lat'], gps['lon'])
                    refs = [set(map(int, ids[ids >= 0])) for ids in reference.signatures[state] if np.any(ids >= 0)]
                    request_payloads = [{'timestamp_s': absolute, 'lat': q['lat'], 'lon': q['lon'],
                        'categories': list(reply.categories), 'L': 20} for q in event['candidates']]
                    request_bytes = sum(len(json.dumps(p, separators=(',', ':'), ensure_ascii=False).encode()) for p in request_payloads)
                    costs = {'requests': len(received), 'request_bytes': request_bytes, 'reply_bytes': sum(r[2] for r in received)}
                    wire_rows.append({'family_id': truth['family_id'], 'split': truth['split'], 'slot': slot,
                        'method': method, 'event_id': event['event_id'], 't': t, 'absolute_public_time_s': absolute,
                        'reply_poi_indices_by_Q': [r[0] for r in received], **costs})
                    for name, ids in (('current_only', current), ('versioned_epoch8', cached)):
                        recall = float(np.mean([len(ref & ids)/len(ref) for ref in refs])) if refs else None
                        rows.append({'family_id': truth['family_id'], 'split': truth['split'], 'slot': slot,
                            'method': method, 'policy': name, 'event_id': event['event_id'], 't': t,
                            'absolute_public_time_s': absolute, 'recall5': recall, 'cached_poi_count': len(ids),
                            'cached_poi_ids': sorted(ids), 'nonempty_reference_categories': len(refs), **costs})
        print('Matched static utility complete', truth['family_id'], flush=True)
    compressed_save(out/'utility_rows.json.gz', {'schema': 'jisa-matched-utility-v1', 'rows': rows})
    compressed_save(out/'wire_rows.json.gz', {'schema': 'jisa-matched-static-wire-v1', 'rows': wire_rows})
    summaries = {}
    for method in METHODS:
        summaries[method] = {}
        for split in SPLITS:
            summaries[method][split] = {}
            for name in POLICIES:
                subset = [r for r in rows if r['method'] == method and r['split'] == split and r['policy'] == name]
                summaries[method][split][name] = {phase: aggregate([r for r in subset if start <= r['t'] <= end])
                    for phase, (start, end) in PHASES.items()}
                summaries[method][split][name]['cold_first_session'] = aggregate([r for r in subset if r['slot'] == 0])
                summaries[method][split][name]['warm_later_sessions'] = aggregate([r for r in subset if r['slot'] > 0])
                overall = summaries[method][split][name]['all_0_600']
                overall['request_bytes_total'] = sum(r['request_bytes'] for r in subset)
    return summaries


def score(out):
    protocol = validate_protocol(out)
    if any((out/name).exists() for name in ('results.json', 'attack_selection.json')):
        raise FileExistsError('Preserve completed or partial scoring evidence')
    data, public, private = load(DATA), load(out/'public_transcripts.json.gz'), load(out/'private_accounting.json.gz')
    fitted, selections = select_attacks(out, data, public, private)
    attack_results = {}; predictions = []
    for name, (model, negative, rows) in fitted.items():
        test = [r for r in rows if r['split'] == 'test']; selected = selections[name]['selected_attacker']
        banks = [model.predict(r['query'], r['public_context'], r['histories']) for r in test]
        negative_predictions = [negative.predict(r['query'], r['public_context'], r['histories'])['candidate_trees'] for r in test]
        selected_metrics = forecast_metrics(test, [p[selected] for p in banks])
        attack_results[name] = {'selected_attacker': selected, 'test': selected_metrics,
            'bank_test_descriptive_only': {n: forecast_metrics(test, [p[n] for p in banks]) for n in sorted(banks[0])},
            'permuted_training_control': forecast_metrics(test, negative_predictions),
            'test_family_metrics': {f: forecast_metrics([r for r in test if r['family_id'] == f],
                [p[selected] for r, p in zip(test, banks) if r['family_id'] == f]) for f in SPLITS['test']}}
        for r, bank, permuted in zip(test, banks, negative_predictions):
            predictions.append({'attack_key': name, 'family_id': r['family_id'], 'slot': r['query_slot'],
                'truth_choice': r['choice_index'], 'banks': {n: p.tolist() for n, p in bank.items()},
                'permuted_candidate_trees': permuted.tolist()})
    compressed_save(out/'attack_predictions.json.gz', {'schema': 'jisa-matched-attacks-v1', 'rows': predictions})
    utilities = utility(out, data, public, private)
    receipts = {name: sha(out/name) for name in ('protocol.json', 'public_transcripts.json.gz',
        'private_accounting.json.gz', 'attack_selection.json', 'attack_predictions.json.gz',
        'utility_rows.json.gz', 'wire_rows.json.gz')}
    save(out/'results.json', {'schema': 'jisa-matched-anchor-readout-v1', 'configuration': configuration(),
        'status': protocol['status'], 'file_sha256': receipts, 'resources': private['resources'],
        'attacks': attack_results, 'utility': utilities, 'no_old_evidence_mutation': True,
        'liveavailability_evaluated': False, 'defense_winner_selected': False,
        'runtime': {k: private[k] for k in ('resource_setup_elapsed_s', 'generation_elapsed_s')}})
    print('Saved matched anchor development readout', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('declare', 'generate', 'score', 'all'), default='all')
    parser.add_argument('--output', type=Path, default=OUT); parser.add_argument('--workdir', type=Path, default=WORK)
    args = parser.parse_args(); declare(args.output)
    if args.stage == 'declare': return
    try:
        if args.stage in ('all', 'generate'): generate(args.output, args.workdir)
        if args.stage in ('all', 'score'): score(args.output)
    except Exception as error:
        path = args.output/'failure.json'
        if not path.exists(): save(path, {'stage': args.stage, 'exception_type': type(error).__name__,
                                         'message': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
