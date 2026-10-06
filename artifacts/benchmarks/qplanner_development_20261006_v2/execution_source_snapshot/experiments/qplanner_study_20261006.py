"""Write-once development/generalization study of public Geo-I Q planners.

Every alternative uses the same secret-keyed REM anchor/ledger realization.
The only intervention is public/protected postprocessing. Frozen artifacts are
replayable without exporting private keys. Source/config changes require a new
output directory; partial families and failed runs remain inspectable.
"""
import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import pickle
import time
import traceback

import numpy as np

from benchmark.engines.public_service_planner import make_service_planner_engine
from benchmark.public_service_profiles import build_public_service_profiles
from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from benchmark.paper_comparators import PublicHistory
from core.session_budget import PersistentEpochBudget, FixedEpochProtectedSessions
from evaluation.candidate_future_attack import CandidateFutureAttack, forecast_metrics, coordinates
from evaluation.ordered_endpoint_attacks import OrderedEndpointBank, ordered_endpoint_features
from evaluation.robust_endpoint_selection import robust_select
from evaluation.lane_travel import LanePoiService
from experiments.build_future_sumo_cohort import ROOT, sha, save
from experiments.future_sumo_eval import DATA, native_resources, compressed_save
from experiments.jisa_native_anchor_ablation_20261006 import policy, private_key


OUT = ROOT/'artifacts/benchmarks/qplanner_development_20261006_v1'
WORK = Path('/private/tmp/qplanner-study-20261006-v1')
METHOD_CONFIGS = {
    'legacy_l10': {'mode': 'legacy_l10'},
    'aligned_nearest': {'mode': 'aligned_nearest'},
    'multi_mean': {'mode': 'aligned_mean_multi'},
    'tail25': {'mode': 'risk_multi', 'risk_weight': .25},
    'tail50': {'mode': 'risk_multi', 'risk_weight': .5},
}
PURPOSES = tuple(p.value for p in QueryPurpose)


def read(path):
    path = Path(path)
    content = path.read_bytes()
    return json.loads(gzip.decompress(content) if path.suffix == '.gz' else content)


def source_closure():
    """Static absolute AND relative local imports, including from-pkg modules."""
    pending = [str(Path(__file__).relative_to(ROOT))]
    found = set()
    def add(module):
        for relative in (module.replace('.', '/')+'.py', module.replace('.', '/')+'/__init__.py'):
            if (ROOT/relative).is_file() and relative not in found:
                pending.append(relative)
    while pending:
        name = pending.pop()
        if name in found: continue
        found.add(name)
        tree = ast.parse((ROOT/name).read_text())
        package = name.removesuffix('.py').split('/')[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names: add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                parts = package[:len(package)-node.level+1] if node.level else []
                module = '.'.join(parts + ([node.module] if node.module else []))
                if module: add(module)
                for alias in node.names:
                    if alias.name != '*': add('.'.join(filter(None, (module, alias.name))))
    return sorted(found | {'requirements.txt', 'requirements-sumo.txt'})


def configuration(methods):
    if not methods or len(set(methods)) != len(methods) or any(m not in METHOD_CONFIGS for m in methods):
        raise ValueError('Distinct declared methods required')
    return {'methods': {m: METHOD_CONFIGS[m] for m in methods}, 'budget': policy().public_parameters(),
            'K': 5, 'server_L': 20, 'reference_k': 5, 'public_radius_m': 1000.,
            'public_destination_rule': 'one latent public road state nearest each of nine fixed map-bounding-box grid points; no private endpoint',
            'public_destination_grid_quantiles': [.15, .5, .85],
            'theta_m': 200., 'utility_slack': .03, 'tail_mass': .25,
            'mean_slack': .01, 'max_risk_exchanges': 3,
            'utility_purposes': PURPOSES, 'private_utility_destination': 'actual session endpoint; local evaluator only',
            'cache': 'current replies plus separate same-version cumulative static epoch cache for every method'}


def public_input_pins(dataset):
    network = read(dataset)['network']['compressed_path']
    names = [network, 'artifacts/benchmarks/research_loop/resources.json']
    return {name: sha(ROOT/name) for name in sorted(set(names))}


def declare(out, dataset, methods, draws, splits, status, draws_by_split=None):
    config = configuration(methods)
    if draws < 1 or len(set(splits)) != len(splits) or any(s not in ('train', 'selection', 'test') for s in splits):
        raise ValueError('Positive draw count and valid distinct splits required')
    schedule = {split: draws if draws_by_split is None else draws_by_split[split] for split in splits}
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in schedule.values()):
        raise ValueError('Positive integer draws per declared split required')
    sources = source_closure()
    value = {'schema': 'qplanner-matched-study-v1', 'configuration': config,
        'dataset_path': str(dataset.resolve().relative_to(ROOT)), 'dataset_sha256': sha(dataset),
        'source_sha256': {p: sha(ROOT/p) for p in sources}, 'draw_count': draws,
        'draws_by_split': schedule, 'splits': splits, 'public_inputs_sha256': public_input_pins(dataset),
        'status': status, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'pairing': 'same private32byte family/draw key and public epoch/session domains; assert bitwise anchors/ledger/GPS calls across methods; alternatives are separate deployments',
        'privacy_scope': 'same ideal .23/m eight-session cap; pure public/protected postprocessing, no new private read; float/PCG64 simulator approximation, no new theorem',
        'utility': 'equal-family, equal-purpose conditional category Recall@5; retain empty reference N/A and completion; current-only primary; cache separate',
        'endpoint_bank': 'method-adapted invariant+observed-slot+Hungarian nearest/velocity aggregate/sequence ExtraTrees64/kNN1/kNN5 and geometric/history decoders; fit TRAIN only; per-loss robust mean+SE MAE / mean-SE Hit SELECTION rule; models/decoders durable before test predictions/errors',
        'future_bank': 'same finite candidate geometry+ExtraTrees96 bank, TWO public alternatives; six complete linked history windows and causal query prefix; no joint pair decoder',
        'cost': 'actual generated public request/reply compactJSON byte estimates; no HTTP/TLS/latency assertion; local processing milliseconds measured separately',
        'failure_policy': 'retain partials, write failure receipt, never replace bad draws/families; source/config changes require new version',
        'generation_order': 'selection groups first, then train, then test; fixed family order within split; draw order1..declared count; development may inspect partial selection but all planned groups/draws retained',
        'selection_rule': 'development utility/attacker tables only; any adopted defense must be recorded and source/config frozen before fresh TEST scoring; fresh synthetic same-map validation is not real-data confirmation'}
    if (out/'protocol.json').exists():
        previous = read(out/'protocol.json')
        value['created_utc'] = previous['created_utc']
        if previous != json.loads(json.dumps(value)): raise ValueError('Sealed protocol changed; use new output version')
        if sha(out/'protocol.json') != (out/'protocol.sha256').read_text().strip(): raise ValueError('Protocol hash mismatch')
        return previous
    out.mkdir(parents=True, exist_ok=True)
    save(out/'protocol.json', value)
    (out/'protocol.sha256').write_text(sha(out/'protocol.json')+'\n')
    for name in sources:
        path = out/'source_snapshot'/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    return read(out/'protocol.json')


def validate(out):
    protocol = read(out/'protocol.json')
    if sha(out/'protocol.json') != (out/'protocol.sha256').read_text().strip(): raise ValueError('Protocol hash mismatch')
    for name, digest in protocol['source_sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Source changed: '+name)
    for name, digest in protocol['public_inputs_sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Public input changed: '+name)
    dataset = ROOT/protocol['dataset_path']
    if sha(dataset) != protocol['dataset_sha256']: raise ValueError('Dataset changed')
    return protocol


def resources(data, work):
    started = time.perf_counter()
    rn, reference, legacy, beliefs, metadata = native_resources(data, work)
    reply = PublicPoiContext(LanePoiService(rn, list(reference.pois), k=20), work/'public_resources'/'reply20.npz')
    base = beliefs[.00125].base
    qs = [.15, .5, .85]; low, high = rn.xy.min(axis=0), rn.xy.max(axis=0)
    points = np.array([low+(high-low)*[a, b] for a in qs for b in qs])
    destination_ids = sorted(set(map(int, base.state_ids[base.tree.query(points)[1]])))
    profiles = build_public_service_profiles(base, reply, public_destination_states=destination_ids,
        public_radius_m=1000., reference_k=5)
    ranking = MultiPurposeRoadRanking(reference, cache_limit=4096)
    metadata = dict(metadata, reply20_sha256=reply.sha256, profiles_sha256=profiles.sha256,
                    profiles_metadata=profiles.metadata, setup_elapsed_s=time.perf_counter()-started)
    return rn, reference, legacy, reply, base, profiles, ranking, metadata


class UtilityEvaluator:
    """GPS/destination are evaluator-local, never accepted by planner factory."""
    def __init__(self, rn, reply, ranking):
        self.rn, self.reply, self.ranking = rn, reply, ranking
        self.refs, self.responses = {}, {}
        self.all_ids = np.ones(len(reply.pois), bool)

    def references(self, state, destination):
        key = int(state), int(destination)
        if key not in self.refs:
            result = {}
            for purpose in QueryPurpose:
                result[purpose.value] = []
                for category in self.ranking.categories:
                    spec = QuerySpec(purpose, category, k=5,
                        radius_m=1000. if purpose == QueryPurpose.WITHIN_RADIUS else None,
                        destination_state=destination if purpose == QueryPurpose.MIN_DETOUR else None)
                    scores = self.ranking.scores(state, spec)
                    eligible = [i for i, poi in enumerate(self.reply.pois)
                                if poi['category'] == category and np.isfinite(scores[i])]
                    eligible.sort(key=lambda i: (float(scores[i]), self.reply.pois[i]['id']))
                    result[purpose.value].append((spec, eligible[:5], eligible))
            self.refs[key] = result
        return self.refs[key]

    def response(self, state):
        if state not in self.responses:
            ids = self.reply.query_indices(state).ravel(); ids = list(map(int, ids[ids >= 0]))
            records = [{k: self.reply.pois[i][k] for k in ('id', 'category', 'lat', 'lon')} for i in ids]
            size = len(json.dumps({'results': records}, separators=(',', ':'), ensure_ascii=False).encode())
            self.responses[state] = ids, size
        return self.responses[state]

    def score(self, state, destination, available):
        refs = self.references(state, destination); result = {}
        for purpose, entries in refs.items():
            recalls, completions, hits, sizes = [], [], [], []
            for spec, ids, eligible in entries:
                if not ids: continue
                answer = [i for i in eligible if i in available][:5]
                overlap = len(set(answer) & set(ids)); recalls.append(overlap/len(ids))
                completions.append(len(answer)/len(ids)); hits.append(overlap); sizes.append(len(ids))
            result[purpose] = {'recall5': float(np.mean(recalls)) if recalls else None,
                'completion': float(np.mean(completions)) if completions else None,
                'reference_category_count': len(recalls), 'all_category_count': len(entries),
                'overlap_total': sum(hits), 'reference_poi_total': sum(sizes)}
        return result


def run_family(family, data, draw, methods, objects, private_dir):
    rn, reference, legacy, reply, base, profiles, ranking, _ = objects
    key = private_key(private_dir/f'{family["family_id"]}--draw{draw}.key')
    clients, ledgers = {}, {}
    for method in methods:
        ledger = PersistentEpochBudget(policy(), private_dir/f'{family["family_id"]}--draw{draw}--{method}.sqlite', private_key=key)
        def factory(allocation, streams, name=method):
            config = METHOD_CONFIGS[name]
            engine = make_service_planner_engine(config['mode'], rn, base, reply, profiles,
                legacy_context=legacy, risk_weight=config.get('risk_weight', .5), tail_mass=.25,
                mean_slack=.01, max_risk_exchanges=3, k=5,
                budget=allocation.nominal_budget_per_m, horizon=allocation.horizon, theta_m=200.,
                read_interval_s=allocation.read_interval_s, utility_slack=.03, rng=streams.initialization)
            engine.anchor_rng, engine.dummy_rng = streams.anchor, streams.dummy
            engine.reset(); return engine
        ledgers[method] = ledger; clients[method] = FixedEpochProtectedSessions(ledger, factory)
    streams = {m: [] for m in ('raw', *methods)}; truths = []; utilities = []; wire = []
    evaluator = UtilityEvaluator(rn, reply, ranking)
    cached = {m: set() for m in streams}; timing = defaultdict(list)
    for slot, spec in enumerate(family['evaluator_only']['sessions']):
        trace = {int(p['time_s']): p for p in data['traces'][spec['session_id']]}
        clocks = sorted(set(range(0, 601, 20)) | ({family['clocks']['shared_fork_t'], family['clocks']['turn_visible_t']} if slot >= 6 else set()))
        events = {m: [] for m in streams}; reads = {m: [] for m in methods}
        end = trace[600]; destination = rn.nearest(end['lat'], end['lon'])[0]
        for client in clients.values():
            assert client.start_session(f'opaque-{slot}', spec['depart_s'])
        for j, t in enumerate(clocks):
            point = trace[t]; gps = point['lat'], point['lon']; qs = {'raw': (gps,)}
            for method, client in clients.items():
                def supplier(m=method): reads[m].append(t); return gps
                started = time.perf_counter(); qs[method] = client.protect_step(spec['depart_s']+t, supplier)
                timing[method].append((time.perf_counter()-started)*1000.)
                assert len(qs[method]) == 5
            state = rn.nearest(*gps)[0]
            for method, positions in qs.items():
                event = {'event_id': f'e{j:04d}', 'timestamp_s': float(t), 'candidates': [
                    {'candidate_id': f'q{i:04d}', 'lat': float(lat), 'lon': float(lon)} for i, (lat, lon) in enumerate(positions)]}
                events[method].append(event)
                responses = [evaluator.response(rn.nearest(*q)[0]) for q in positions]
                current = {i for ids, _ in responses for i in ids}; cached[method].update(current)
                payloads = [{'timestamp_s': spec['depart_s']+t, 'lat': lat, 'lon': lon,
                    'categories': list(reply.categories), 'L': 20} for lat, lon in positions]
                costs = {'requests': len(positions), 'request_bytes': sum(len(json.dumps(p, separators=(',', ':')).encode()) for p in payloads),
                         'reply_bytes': sum(size for _, size in responses)}
                fields = {'family_id': family['family_id'], 'split': family['split'], 'draw': draw,
                    'slot': slot, 'method': method, 'event_id': event['event_id'], 't': t}
                wire.append(dict(fields, reply_poi_ids_by_Q=[ids for ids, _ in responses], **costs))
                for name, available in (('current', current), ('static_epoch_cache', cached[method])):
                    scores = evaluator.score(state, destination, available)
                    utilities.append(dict(fields, cache=name, purposes=scores, available_count=len(available)))
        info = {}; anchor_tapes = []
        for method, client in clients.items():
            ledger_row = client.evaluator_current_session()
            assert ledger_row['private_reads'] == len(reads[method]) <= 11
            assert ledger_row['spent_per_m'] <= .02875+1e-12
            assert all(b-a >= 60 for a, b in zip(reads[method], reads[method][1:]))
            anchors = client._engine.evaluator_anchors
            anchor_tapes.append((anchors, ledger_row['ledger'], reads[method]))
            info[method] = dict(ledger_row, anchors=anchors, supplier_times_s=reads[method],
                step_ms=timing[method][-len(clocks):], planner_objectives=client._engine.evaluator_objective)
            client.close_session(spec['depart_s']+600.)
        assert all(tape == anchor_tapes[0] for tape in anchor_tapes[1:]), 'Planner changed the Geo-I private transcript!'
        labels = {}
        if slot >= 6:
            for stage in ('shared_fork_t', 'turn_visible_t'):
                cut = family['clocks'][stage]
                labels[stage] = next(p['edge_id'] for when, p in sorted(trace.items()) if when > cut
                    and not p['edge_id'].startswith(':') and (stage != 'shared_fork_t' or p['edge_id'] != family['fork_edge']))
                assert labels[stage] == family['public_context']['choices'][spec['choice_index']]['edge_id']
        truths.append({'slot': slot, 'choice_index': spec['choice_index'], 'destination_role': spec['destination_role'],
            'origin_xy': list(rn.point_xy(trace[0]['lat'], trace[0]['lon'])),
            'destination_xy': coordinates({'events': events['raw']})[0][-1][0].tolist(),
            'next_edge_labels': labels, 'ledger': info})
        for method in streams: streams[method].append({'events': events[method]})
    summaries = {m: c.evaluator_summary() for m, c in clients.items()}
    assert all(v['reserved_cap_per_m'] == .23 and v['spent_per_m'] <= .23+1e-12 for v in summaries.values())
    for ledger in ledgers.values(): ledger.close()
    return {'public': {'public_context': family['public_context'],
                'public_clocks': {k: family['clocks'][k] for k in ('shared_fork_t', 'turn_visible_t')}, 'streams': streams},
            'evaluator_only': {'family_id': family['family_id'], 'split': family['split'], 'draw': draw,
                'sessions': truths, 'epoch_accounting': summaries}, 'utility': utilities, 'wire': wire}


def generate(out, work):
    p = validate(out)
    if work.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Private RNG keys/ledgers require workdir outside repository')
    if (out/'generation_started.json').exists(): raise FileExistsError('Write-once generation already started; retain partials')
    data = read(ROOT/p['dataset_path'])
    families = [f for f in data['families'] if f['split'] in p['splits']]
    families.sort(key=lambda f: (('selection', 'train', 'test').index(f['split']), f['family_id']))
    if not families or any(not any(f['split'] == s for f in families) for s in p['splits']): raise ValueError('Every declared split required')
    save(out/'generation_started.json', {'started_utc': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': sha(out/'protocol.json')})
    objects = resources(data, work); save(out/'resources.json', objects[-1])
    private_dir = work/'private_state'; private_dir.mkdir(parents=True, exist_ok=True); os.chmod(private_dir, 0o700)
    directory = out/'families'; directory.mkdir(); receipts = {}; started = time.perf_counter()
    for family in families:
        for draw in range(1, p['draws_by_split'][family['split']]+1):
            begin = time.perf_counter()
            value = run_family(family, data, draw, tuple(p['configuration']['methods']), objects, private_dir)
            name = f'{family["family_id"]}--draw{draw}.json.gz'
            compressed_save(directory/name, value); receipts[name] = sha(directory/name)
            print('Q planner completed', family['family_id'], 'draw', draw, 'seconds', round(time.perf_counter()-begin, 2), flush=True)
    save(out/'generation.json', {'family_files_sha256': receipts, 'family_count': len(families),
        'draw_count': p['draw_count'], 'draws_by_split': p['draws_by_split'], 'elapsed_s': time.perf_counter()-started,
        'identical_private_transcript_asserted': True, 'private_keys_exported': False})


def lower_tail(values, mass=.25):
    if not values: return None
    ordered = np.sort(np.asarray(values, float)); amount = len(ordered)*mass
    whole = int(amount); fraction = amount-whole
    return float((ordered[:whole].sum()+(fraction*ordered[whole] if fraction else 0.))/amount)


def summarize_utility(rows):
    """Draws/ticks are nested within family; all four purposes equal macro weight."""
    families = sorted({r['family_id'] for r in rows}); result = {}
    for purpose in PURPOSES:
        values = {f: [r['purposes'][purpose]['recall5'] for r in rows if r['family_id'] == f
            and r['purposes'][purpose]['recall5'] is not None] for f in families}
        means = {f: float(np.mean(v)) if v else None for f, v in values.items()}
        valid = [v for v in means.values() if v is not None]
        result[purpose] = {'family_mean': float(np.mean(valid)) if valid else None,
            'family_values': means, 'family_lower_quartile_cvar': lower_tail(valid),
            'minimum_family_mean': min(valid) if valid else None,
            'event_lower_quartile_cvar_family_macro': float(np.mean([lower_tail(v) for v in values.values() if v])) if valid else None,
            'defined_windows': sum(len(v) for v in values.values()), 'total_windows': len(rows),
            'defined_categories': sum(r['purposes'][purpose]['reference_category_count'] for r in rows),
            'total_categories': sum(r['purposes'][purpose]['all_category_count'] for r in rows)}
    overall = {f: float(np.mean([result[p]['family_values'][f] for p in PURPOSES if result[p]['family_values'][f] is not None]))
        for f in families if any(result[p]['family_values'][f] is not None for p in PURPOSES)}
    result['equal_purpose_macro'] = {'family_mean': float(np.mean(list(overall.values()))) if overall else None,
        'family_values': overall, 'family_lower_quartile_cvar': lower_tail(list(overall.values()))}
    return result


def utility_readout(out):
    p = validate(out); generation = read(out/'generation.json'); rows = []; wires = []; timings = defaultdict(list)
    for name, digest in generation['family_files_sha256'].items():
        if sha(out/'families'/name) != digest: raise ValueError('Family changed')
        bundle = read(out/'families'/name); rows.extend(bundle['utility']); wires.extend(bundle['wire'])
        for session in bundle['evaluator_only']['sessions']:
            for method, ledger in session['ledger'].items(): timings[method].extend(ledger['step_ms'])
    summaries = {}
    for method in ('raw', *p['configuration']['methods']):
        summaries[method] = {}
        for split in p['splits']:
            summaries[method][split] = {}
            for cache in ('current', 'static_epoch_cache'):
                subset = [r for r in rows if r['method'] == method and r['split'] == split and r['cache'] == cache]
                summaries[method][split][cache] = {
                    'all': summarize_utility(subset),
                    'cold': summarize_utility([r for r in subset if r['slot'] == 0]),
                    'early_0_180': summarize_utility([r for r in subset if r['t'] <= 180]),
                    'temporal_tail_400_600': summarize_utility([r for r in subset if r['t'] >= 400])}
            subset = [r for r in wires if r['method'] == method and r['split'] == split]
            summaries[method][split]['cost'] = {k: sum(r[k] for r in subset) for k in ('requests', 'request_bytes', 'reply_bytes')}
    save(out/'utility_readout.json', {'schema': 'qplanner-utility-readout-v1', 'summary': summaries,
        'protocol_sha256': sha(out/'protocol.json'), 'generation_sha256': sha(out/'generation.json'),
        'processing_ms_all_splits_descriptive': {m: {'median': float(np.median(v)), 'p95': float(np.quantile(v, .95)), 'max': max(v)} for m, v in timings.items()}})
    print('Saved utility readout', out, flush=True)


def future_rows(bundles, method, stage):
    rows = []
    for bundle in bundles:
        group, truth = bundle['public'], bundle['evaluator_only']; streams = group['streams'][method]
        for slot in (6, 7):
            target = truth['sessions'][slot]
            rows.append({'family_id': truth['family_id'], 'split': truth['split'], 'draw': truth['draw'], 'query_slot': slot,
                'query': {'events': [e for e in streams[slot]['events'] if e['timestamp_s'] <= group['public_clocks'][stage+'_t']]},
                'histories': streams[:6], 'public_context': group['public_context'],
                'choice_index': target['choice_index'], 'destination_role': target['destination_role'],
                'destination_xy': target['destination_xy'], 'actual_next_edge': target['next_edge_labels'][stage+'_t']})
    return rows


def attack_readout(out, work):
    p = validate(out); generation = read(out/'generation.json')
    if not {'train', 'selection'} <= set(p['splits']): raise ValueError('Attacks require disjoint fit/selection splits')
    bundles = [read(out/'families'/name) for name in generation['family_files_sha256']]
    data = read(ROOT/p['dataset_path']); rn, *_ = native_resources(data, work)
    train_ids = {b['evaluator_only']['family_id'] for b in bundles if b['evaluator_only']['split'] == 'train'}
    tracks = [[rn.nearest(pt['lat'], pt['lon'])[0] for pt in data['traces'][spec['session_id']][::20]]
        for f in data['families'] if f['family_id'] in train_ids for spec in f['evaluator_only']['sessions']]
    history = PublicHistory(rn, tracks); models = {}; selections = {}; predictions = {}; all_endpoint_rows = {}
    directory = out/'attack_models'; directory.mkdir()
    for method in ('raw', *p['configuration']['methods']):
        for stage in ('shared_fork', 'turn_visible'):
            rows = future_rows(bundles, method, stage)
            for task, use_history in (('S5', False), ('S6', True)):
                key = f'{method}--{stage}--{task}'; train = [r for r in rows if r['split'] == 'train']
                model = CandidateFutureAttack(train, use_history=use_history)
                banks = [(r, model.predict(r['query'], r['public_context'], r['histories'])) for r in rows if r['split'] == 'selection']
                validation = [(r, bank) for r, bank in banks if r['split'] == 'selection']
                scores = {name: forecast_metrics([r for r, _ in validation], [bank[name] for _, bank in validation]) for name in validation[0][1]}
                selected = min(scores, key=lambda n: (-scores[n]['balanced_accuracy'], scores[n]['log_loss'], n))
                models[key] = model; selections[key] = {'selected': selected, 'selection_scores': scores}
                predictions[key] = (rows, banks)
        for scenario in ('S9', 'S10'):
            key = f'{method}--{scenario}'; samples = []
            for bundle in bundles:
                truth = bundle['evaluator_only']
                if truth['split'] == 'test': continue
                for slot, session in enumerate(bundle['public']['streams'][method]):
                    events = [{'timestamp_s': e['timestamp_s'], 'coordinates': [[q['lat'], q['lon']] for q in e['candidates']]} for e in session['events']]
                    features, geometry = ordered_endpoint_features(events, scenario, rn, history, observable_close_s=600.)
                    target = truth['sessions'][slot]['origin_xy' if scenario == 'S9' else 'destination_xy']
                    samples.append({'family_id': truth['family_id'], 'split': truth['split'], 'draw': truth['draw'], 'slot': slot,
                        'events': events, 'features': {name: value[0] for name, value in features.items()}, 'target': np.asarray(target)})
            train = [r for r in samples if r['split'] == 'train']
            model = OrderedEndpointBank({name: [r['features'][name] for r in train] for name in train[0]['features']},
                                        [r['target'] for r in train])
            rows = []
            for r in samples:
                if r['split'] == 'train': continue
                bank = model.predictions(r['events'], scenario, rn, history, observable_close_s=600.)
                rows.append({k: r[k] for k in ('family_id', 'split', 'draw', 'slot')} | {
                    'session_id': str(r['slot']), 'seed': r['draw'], 'errors': {
                    name: float(np.linalg.norm(np.asarray(point)[0]-r['target'])) for name, point in bank.items()}})
            selection_rows = [r for r in rows if r['split'] == 'selection']
            selected, statistics = robust_select(selection_rows, sorted({r['family_id'] for r in selection_rows}))
            models[key] = model; selections[key] = {'selected': selected, 'selection_statistics': statistics}; all_endpoint_rows[key] = rows
        print('Fitted and selected public attacks', method, flush=True)
    for name, model in models.items():
        path = directory/(name+'.pickle')
        with path.open('xb') as stream: pickle.dump(model, stream, protocol=5)
        selections[name]['model_sha256'] = sha(path)
    save(out/'attack_selection.json', {'protocol_sha256': sha(out/'protocol.json'), 'selections': selections})
    # Selection/model file is durable BEFORE calculating a test metric.
    result = {}; bank_errors = {}; future_predictions = []
    for key, (rows, banks) in predictions.items():
        model = models[key]
        banks += [(r, model.predict(r['query'], r['public_context'], r['histories'])) for r in rows if r['split'] == 'test']
        future_predictions.extend({'attack_key': key, 'family_id': r['family_id'], 'split': r['split'],
            'draw': r['draw'], 'slot': r['query_slot'], 'truth_choice': r['choice_index'],
            'banks': {name: value.tolist() for name, value in bank.items()}} for r, bank in banks)
        selected = selections[key]['selected']; result[key] = {}
        for split in ('selection', 'test'):
            subset = [(r, b) for r, b in banks if r['split'] == split]
            if subset:
                result[key][split] = forecast_metrics([r for r, _ in subset], [b[selected] for _, b in subset])
                result[key][split]['family_values'] = {f: forecast_metrics([r for r, _ in subset if r['family_id'] == f],
                    [b[selected] for r, b in subset if r['family_id'] == f]) for f in sorted({r['family_id'] for r, _ in subset})}
    for key, rows in all_endpoint_rows.items():
        method, scenario = key.split('--'); model = models[key]
        for bundle in bundles:
            truth = bundle['evaluator_only']
            if truth['split'] != 'test': continue
            for slot, session in enumerate(bundle['public']['streams'][method]):
                events = [{'timestamp_s': e['timestamp_s'], 'coordinates': [[q['lat'], q['lon']] for q in e['candidates']]} for e in session['events']]
                target = np.asarray(truth['sessions'][slot]['origin_xy' if scenario == 'S9' else 'destination_xy'])
                bank = model.predictions(events, scenario, rn, history, observable_close_s=600.)
                rows.append({'family_id': truth['family_id'], 'split': 'test', 'draw': truth['draw'], 'slot': slot,
                    'session_id': str(slot), 'seed': truth['draw'], 'errors': {
                        name: float(np.linalg.norm(np.asarray(point)[0]-target)) for name, point in bank.items()}})
        result[key] = {}; bank_errors[key] = rows
        for split in ('selection', 'test'):
            subset = [r for r in rows if r['split'] == split]
            if not subset: continue
            families = sorted({r['family_id'] for r in subset}); selected = selections[key]['selected']
            family_values = {f: {'mae_m': float(np.mean([r['errors'][selected['mae']] for r in subset if r['family_id'] == f])),
                **{h: float(np.mean([r['errors'][selected[h]] <= int(h[3:]) for r in subset if r['family_id'] == f])) for h in ('hit100', 'hit500')}} for f in families}
            result[key][split] = {'family_values': family_values,
                **{metric: float(np.mean([v[metric] for v in family_values.values()])) for metric in ('mae_m', 'hit100', 'hit500')}}
    compressed_save(out/'endpoint_bank_errors.json.gz', bank_errors)
    compressed_save(out/'future_bank_predictions.json.gz', {'rows': future_predictions})
    save(out/'attack_readout.json', {'protocol_sha256': sha(out/'protocol.json'), 'selection_sha256': sha(out/'attack_selection.json'), 'results': result})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=OUT); parser.add_argument('--workdir', type=Path, default=WORK)
    parser.add_argument('--methods', nargs='+', choices=tuple(METHOD_CONFIGS), default=list(METHOD_CONFIGS))
    parser.add_argument('--draws', type=int, default=2)
    parser.add_argument('--train-draws', type=int); parser.add_argument('--selection-draws', type=int)
    parser.add_argument('--test-draws', type=int)
    parser.add_argument('--splits', nargs='+', choices=('train', 'selection', 'test'), default=['train', 'selection'])
    parser.add_argument('--status', default='DEVELOPMENT: old native groups inspected; no fresh confirmation')
    parser.add_argument('--stage', choices=('declare', 'generate', 'utility', 'attacks', 'all'), default='all')
    args = parser.parse_args()
    schedule = {split: getattr(args, split+'_draws') or args.draws for split in args.splits}
    declare(args.output, args.dataset, args.methods, args.draws, args.splits, args.status, schedule)
    if args.stage == 'declare': return
    try:
        if args.stage in ('all', 'generate'): generate(args.output, args.workdir)
        if args.stage in ('all', 'utility'): utility_readout(args.output)
        if args.stage in ('all', 'attacks'): attack_readout(args.output, args.workdir)
    except Exception as error:
        if not (args.output/'failure.json').exists(): save(args.output/'failure.json', {'error': type(error).__name__,
            'message': str(error), 'traceback': traceback.format_exc(), 'stage': args.stage})
        raise


if __name__ == '__main__': main()
