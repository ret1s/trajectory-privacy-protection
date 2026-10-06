"""Write-once DEVELOPMENT replay of service depth on frozen legacy L10 Q.

Depth changes public response size and the public L request field. Coordinates,
clocks, K, protected anchors, reads and ledger are reused, never regenerated.
The fixed criterion is declared before response-depth utility is calculated.
This is a static-service utility/cost configuration experiment, not a new
location mechanism or a privacy comparison against other algorithms.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking
from evaluation.lane_travel import LanePoiService
from experiments import qplanner_study_20261006_v2 as common
from experiments.qplanner_paired_readout_20261006 import (
    PUBLIC_CACHE_FILES, declared_clock_inventory, validate_study_sources,
)

ROOT = common.ROOT
DEPTHS = (20, 30, 40, 60)
METHOD = 'legacy_l10'
CRITERION = {
    'split': 'selection', 'cache': 'current', 'phase': 'all',
    'candidate_depths_in_order': [30, 40, 60], 'baseline_depth': 20,
    'minimum_equal_family_equal_purpose_recall_gain': .02,
    'minimum_nearest_distance_recall_gain': 0.,
    'maximum_total_reply_json_byte_ratio': 2.5,
    'choice': 'smallest eligible depth; NONE if no candidate passes all gates',
    'undefined': 'N/A is never zero; require identical complete family pairs and reference coverage',
    'status': 'exploratory development selection; fresh confirmation requires a separate freeze',
}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def save_new(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def audit_input(source):
    """Validate metadata/hashes only; never calculate a response-depth score."""
    source = Path(source).resolve()
    p = common.read(source/'protocol.json')
    if not p['splits'] or 'test' in p['splits'] or any(s not in ('train', 'selection') for s in p['splits']):
        raise ValueError('This DEVELOPMENT-only replay refuses fresh TEST cohorts')
    c = p['configuration']
    if METHOD not in c['methods'] or c['methods'][METHOD]['mode'] != METHOD:
        raise ValueError('Frozen legacy_l10 Q source required')
    if (c['K'], c['server_L'], c['reference_k'], c['public_radius_m']) != (5, 20, 5, 1000.):
        raise ValueError('Source service/private controls differ from fixed experiment')
    generation = common.read(source/'generation.json')
    validate_study_sources(source, p, generation, root=ROOT)
    data = common.read(ROOT/p['dataset_path'])
    jobs = {f'{f["family_id"]}--draw{d}.json.gz' for f in data['families'] if f['split'] in p['splits']
        for d in range(1, p['draws_by_split'][f['split']]+1)}
    pins = generation['family_files_sha256']
    if set(pins) != jobs or {f.name for f in (source/'families').iterdir()} != jobs:
        raise ValueError('Completed source must retain every planned family/draw with no extras')
    for name, digest in pins.items():
        if common.sha(source/'families'/name) != digest:
            raise ValueError('Frozen family block changed: '+name)
    return p, generation


def depth_source_closure():
    return sorted(set(common.source_closure()) | {
        str(Path(__file__).relative_to(ROOT)),
        'experiments/qplanner_paired_readout_20261006.py',
    })


def declare_depth_study(source, output):
    """Write immutable selection policy and input/source pins before scoring."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('New empty output required; retain any previous attempt')
    p, generation = audit_input(source)
    source_names = ['protocol.json', 'protocol.sha256', 'generation.json', 'resources.json']
    source_names += [n for n in ('execution_protocol.json', 'execution_protocol.sha256') if (source/n).exists()]
    value = {
        'schema': 'qplanner-response-depth-development-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_output': str(source), 'source_files_sha256': {n: common.sha(source/n) for n in source_names},
        'family_files_sha256': generation['family_files_sha256'],
        'source_sha256': {n: common.sha(ROOT/n) for n in depth_source_closure()},
        'dataset_path': p['dataset_path'], 'dataset_sha256': p['dataset_sha256'],
        'splits': p['splits'], 'draws_by_split': p['draws_by_split'],
        'fixed_configuration': p['configuration'], 'frozen_coordinate_method': METHOD,
        'depths': list(DEPTHS), 'selection_criterion': CRITERION,
        'service': 'static public all-category directed-distance replies; stable POI-ID ties; exact ordered prefixes of L60',
        'utility': 'local four-purpose reference top5; conditional nonempty categories; equal-purpose then equal-family; nested draws retained',
        'cost': 'actual compact JSON application payload estimates, no HTTP/TLS/latency/battery measurement',
        'boundary': 'no Q regeneration, private key, new GPS read, purpose-dependent request or budget change; L and response bytes change',
        'claim_scope': 'same-map synthetic static utility/cost ablation; no new algorithm, privacy superiority or fresh confirmation',
    }
    output.mkdir(parents=True, exist_ok=True)
    save_new(output/'protocol.json', value)
    (output/'protocol.sha256').write_text(common.sha(output/'protocol.json')+'\n')
    for name in value['source_sha256']:
        target = output/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    return value


def validate_depth_study(output):
    output = Path(output)
    p = common.read(output/'protocol.json')
    if common.sha(output/'protocol.json') != (output/'protocol.sha256').read_text().strip():
        raise ValueError('Response-depth protocol changed')
    if p['selection_criterion'] != CRITERION or p['depths'] != list(DEPTHS):
        raise ValueError('Fixed depth policy differs')
    for name, digest in p['source_sha256'].items():
        if common.sha(ROOT/name) != digest or common.sha(output/'source_snapshot'/name) != digest:
            raise ValueError('Response-depth source/snapshot changed: '+name)
    source = Path(p['source_output'])
    for name, digest in p['source_files_sha256'].items():
        if common.sha(source/name) != digest:
            raise ValueError('Source study metadata changed: '+name)
    old, generation = audit_input(source)
    if generation['family_files_sha256'] != p['family_files_sha256'] or old['configuration'] != p['fixed_configuration']:
        raise ValueError('Source cohort or fixed controls changed')
    return p


class PrefixContext:
    """Read-only exact category-wise prefix, with the same coordinate access."""
    def __init__(self, full, depth):
        if isinstance(depth, bool) or not isinstance(depth, int) or not 1 <= depth <= full.k:
            raise ValueError('Integer depth within public full response required')
        self.rn, self.pois, self.categories, self.k = full.rn, full.pois, full.categories, depth
        self.access, self.signatures = full.access, full.signatures[:, :, :depth]
        self.sha256 = canonical_sha({'parent_sha256': full.sha256, 'depth': depth})

    def query_indices(self, state):
        return self.signatures[self.access[int(state)]]


def build_depth_resources(dataset, workdir, public_cache=None):
    work = Path(workdir).resolve()
    if work.is_relative_to(ROOT.resolve()):
        raise ValueError('Use a new temporary workdir outside the repository')
    work.mkdir(parents=True, exist_ok=True)
    cache = work/'public_resources'; cache.mkdir(exist_ok=True)
    copied = {}
    if public_cache is not None:
        source = Path(public_cache)
        for name in PUBLIC_CACHE_FILES:
            path, target = source/name, cache/name
            if path.is_symlink() or not path.is_file():
                raise ValueError('Only complete safe public cache may be copied')
            if target.exists():
                raise FileExistsError('New workdir/public-cache copy required')
            shutil.copyfile(path, target); copied[name] = common.sha(target)
    data = common.read(dataset)
    rn, reference, _, _, metadata = common.native_resources(data, work)
    old20 = PublicPoiContext(LanePoiService(rn, list(reference.pois), k=20), cache/'reply20.npz')
    full = PublicPoiContext(LanePoiService(rn, list(reference.pois), k=60), cache/'reply60.npz')
    if (full.pois != old20.pois or not np.array_equal(full.access, old20.access)
            or not np.array_equal(full.signatures[:, :, :20], old20.signatures)):
        raise ValueError('L60 prefix does not reproduce exact frozen L20 catalogue/access/ties')
    ranking = MultiPurposeRoadRanking(reference, cache_limit=256)
    contexts = {depth: PrefixContext(full, depth) for depth in DEPTHS}
    evaluators = {depth: common.UtilityEvaluator(rn, ctx, ranking) for depth, ctx in contexts.items()}
    shared_refs = {}
    for evaluator in evaluators.values(): evaluator.refs = shared_refs
    metadata = dict(metadata, full_reply60_sha256=full.sha256, frozen_reply20_sha256=old20.sha256,
        exact_l20_prefix_asserted=True, public_cache_copies_sha256=copied,
        depth_context_sha256={str(d): ctx.sha256 for d, ctx in contexts.items()})
    return rn, evaluators, metadata


def _assert_baseline(expected, observed):
    if set(expected) != set(observed):
        raise ValueError('L20 baseline row inventory differs')
    for key, value in expected.items():
        if isinstance(value, dict):
            _assert_baseline(value, observed[key])
        elif isinstance(value, float):
            if (not isinstance(observed[key], (float,int))
                    or not np.isclose(value, observed[key], atol=1e-12, rtol=0)):
                raise ValueError('L20 baseline floating score differs')
        elif value != observed[key]:
            raise ValueError('L20 baseline score/reply/cost differs')


def replay_bundle(bundle, family, rn, evaluators):
    """Replay static replies; private truth is evaluator-only local ranking."""
    truth = bundle['evaluator_only']; group = bundle['public']
    if 20 not in evaluators or any(d not in DEPTHS or evaluator.reply.k != d for d,evaluator in evaluators.items()):
        raise ValueError('Exact baselineL20 and correctly labeled fixed depth evaluators required')
    methods = tuple(group['streams'])
    inventory = declared_clock_inventory(bundle, methods)
    if len(family['evaluator_only']['sessions']) != 8:
        raise ValueError('Fixed eight-session source required')
    old_util = {(r['slot'], r['t'], r['cache']): r for r in bundle['utility'] if r['method'] == METHOD}
    old_wire = {(r['slot'], r['t']): r for r in bundle['wire'] if r['method'] == METHOD}
    if set(old_wire) != inventory or set(old_util) != {(s,t,c) for s,t in inventory for c in ('current','static_epoch_cache')}:
        raise ValueError('Source baseline utility/wire inventory is incomplete')
    rows, wires = [], []; cached = {depth: set() for depth in evaluators}
    for slot, session in enumerate(group['streams'][METHOD]):
        raw = group['streams']['raw'][slot]['events']
        if [e['timestamp_s'] for e in raw] != [e['timestamp_s'] for e in session['events']]:
            raise ValueError('Evaluator raw/Q clocks differ')
        if any(len(e['candidates']) != 1 for e in raw) or raw[-1]['timestamp_s'] != 600:
            raise ValueError('Singleton evaluator truth and fixed final clock required')
        last = raw[-1]['candidates'][0]; destination = rn.nearest(last['lat'], last['lon'])[0]
        departure = family['evaluator_only']['sessions'][slot]['depart_s']
        for event, original in zip(session['events'], raw):
            # Original requests used integer loop clocks, whereas public events
            # serialize timestamp_s as floats. Preserve that byte representation.
            t = old_wire[(slot,event['timestamp_s'])]['t']; point = original['candidates'][0]
            state = rn.nearest(point['lat'], point['lon'])[0]
            if len(event['candidates']) != 5:
                raise ValueError('Fixed protected K5 required')
            positions = [(q['lat'], q['lon']) for q in event['candidates']]
            previous_pool = set(); previous_scores = None
            for depth, evaluator in sorted(evaluators.items()):
                responses = [evaluator.response(rn.nearest(*q)[0]) for q in positions]
                pool = {i for ids, _ in responses for i in ids}
                if not previous_pool <= pool: raise ValueError('Deeper public replies lost POIs')
                previous_pool = pool; cached[depth].update(pool)
                fields = {k: truth[k] for k in ('family_id', 'split', 'draw')}
                fields.update(slot=slot, method=f'service_l{depth}', event_id=event['event_id'], t=t)
                payloads = [dict(timestamp_s=departure+t, lat=lat, lon=lon,
                    categories=list(evaluator.reply.categories), L=depth) for lat, lon in positions]
                wire = dict(fields, requests=5,
                    request_bytes=sum(len(json.dumps(p, separators=(',', ':')).encode()) for p in payloads),
                    reply_bytes=sum(size for _, size in responses), reply_poi_ids_by_Q=[ids for ids, _ in responses])
                wires.append(wire)
                for name, available in (('current', pool), ('static_epoch_cache', cached[depth])):
                    score = evaluator.score(state, destination, available)
                    row = dict(fields, cache=name, purposes=score, available_count=len(available)); rows.append(row)
                    if depth == 20:
                        _assert_baseline(old_util[(slot,t,name)], dict(row, method=METHOD))
                    if name == 'current':
                        if previous_scores is not None:
                            for purpose, value in score.items():
                                earlier = previous_scores[purpose]
                                if (value['recall5'] is None) != (earlier['recall5'] is None):
                                    raise ValueError('Depth changed reference N/A coverage')
                                if value['recall5'] is not None and value['recall5']+1e-12 < earlier['recall5']:
                                    raise ValueError('Ordered pool extension decreased static Recall')
                        current_score = score
                previous_scores = current_score
                if depth == 20: _assert_baseline(old_wire[(slot,t)], dict(wire, method=METHOD))
    ledger = [{k:v for k,v in session['ledger'][METHOD].items() if k != 'step_ms'} for session in truth['sessions']]
    return {'schema':'qplanner-service-depth-replay-family-v1',
        'evaluator_only':{k:truth[k] for k in ('family_id','split','draw')},
        'frozen_controls': {'Q_stream_sha256':canonical_sha(group['streams'][METHOD]),
            'ledger_and_anchors_sha256':canonical_sha(ledger), 'Q_not_regenerated':True,
            'private_reads_not_performed':True}, 'utility':rows, 'wire':wires}


def summarize_depth(rows, wires, splits):
    summaries = {}
    for depth in DEPTHS:
        method = f'service_l{depth}'; summaries[str(depth)] = {}
        for split in splits:
            local = [r for r in rows if r['method'] == method and r['split'] == split]
            cells = {}
            for cache in ('current','static_epoch_cache'):
                subset = [r for r in local if r['cache'] == cache]
                cells[cache] = {
                    'all': common.summarize_utility(subset),
                    'cold': common.summarize_utility([r for r in subset if r['slot'] == 0]),
                    'temporal_tail_400_600': common.summarize_utility([r for r in subset if r['t'] >= 400]),
                }
            selected = [r for r in wires if r['method'] == method and r['split'] == split]
            cells['cost'] = {k:sum(r[k] for r in selected) for k in ('requests','request_bytes','reply_bytes')}
            summaries[str(depth)][split] = cells
    return summaries


def replay_depth_study(output, workdir, public_cache=None):
    output = Path(output); p = validate_depth_study(output)
    save_new(output/'replay_started.json', {'protocol_sha256':common.sha(output/'protocol.json'),
        'started_utc':datetime.now(timezone.utc).isoformat()})
    rn, evaluators, resources = build_depth_resources(ROOT/p['dataset_path'], workdir, public_cache)
    source_resources = common.read(Path(p['source_output'])/'resources.json')
    if resources['frozen_reply20_sha256'] != source_resources['reply20_sha256']:
        raise ValueError('Replay public L20 context differs from frozen generated source')
    save_new(output/'resources.json', resources)
    data = common.read(ROOT/p['dataset_path']); families = {f['family_id']:f for f in data['families']}
    target = output/'families'; target.mkdir(); pins = {}; rows = []; wires = []
    for name, digest in p['family_files_sha256'].items():
        path = Path(p['source_output'])/'families'/name
        if common.sha(path) != digest: raise ValueError('Frozen Q tape changed during replay')
        bundle = common.read(path)
        value = replay_bundle(bundle, families[bundle['evaluator_only']['family_id']], rn, evaluators)
        value['source_bundle_sha256'] = digest
        common.compressed_save(target/name, value); pins[name] = common.sha(target/name)
        rows.extend(value['utility']); wires.extend(value['wire'])
        print('Replayed exact Q at depths20/30/40/60:', name, flush=True)
    save_new(output/'readout.json', {'schema':'qplanner-response-depth-readout-v1',
        'protocol_sha256':common.sha(output/'protocol.json'), 'resources_sha256':common.sha(output/'resources.json'),
        'family_files_sha256':pins, 'summary':summarize_depth(rows,wires,p['splits']),
        'L20_source_rows_exactly_reproduced':True, 'static_event_recall_monotonicity_asserted':True,
        'no_private_generation':True})
    return common.read(output/'readout.json')


def selection_from_summary(summary):
    """Pure fixed gate: same complete family pairs; no test data accepted."""
    baseline = summary['20']['selection']; candidates = []; selected = None
    for depth in CRITERION['candidate_depths_in_order']:
        other = summary[str(depth)]['selection']; gates = {}; differences = {}
        for purpose in ('equal_purpose_macro','nearest_distance'):
            left = other['current']['all'][purpose]['family_values']
            right = baseline['current']['all'][purpose]['family_values']
            if set(left) != set(right) or not left or any(left[f] is None or right[f] is None for f in left):
                raise ValueError('Depth selection needs identical complete defined family pairs')
            delta = float(np.mean([left[f]-right[f] for f in sorted(left)])); differences[purpose] = delta
        for purpose in common.PURPOSES:
            left, right = other['current']['all'][purpose], baseline['current']['all'][purpose]
            if any(left[k] != right[k] for k in ('defined_windows','total_windows','defined_categories','total_categories')):
                raise ValueError('Depth reference coverage differs')
        denominator = baseline['cost']['reply_bytes']
        if denominator <= 0: raise ValueError('Positive baseline application reply bytes required')
        ratio = other['cost']['reply_bytes']/denominator
        if other['cost']['requests'] != baseline['cost']['requests']:
            raise ValueError('Service depth changed request count')
        gates['macro_gain_at_least_2pp'] = differences['equal_purpose_macro'] >= .02-1e-12
        gates['nearest_no_loss'] = differences['nearest_distance'] >= -1e-12
        gates['reply_byte_ratio_at_most_2_5'] = ratio <= 2.5+1e-12
        eligible = all(gates.values())
        candidates.append({'depth':depth,'differences':differences,'reply_json_byte_ratio':ratio,
            'gates':gates,'eligible':eligible})
        if selected is None and eligible: selected = depth
    return {'schema':'qplanner-response-depth-selection-v1','selected_depth':selected,
        'criterion':CRITERION,'candidates':candidates,'fresh_confirmation':'PENDING; not scored by this development replay'}


def select_depth(output):
    output = Path(output); p = validate_depth_study(output); path = output/'readout.json'
    value = common.read(path)
    if value['protocol_sha256'] != common.sha(output/'protocol.json'):
        raise ValueError('Readout belongs to another policy')
    if value['resources_sha256'] != common.sha(output/'resources.json'):
        raise ValueError('Replay public resources changed')
    if set(value['family_files_sha256']) != set(p['family_files_sha256']):
        raise ValueError('Readout omitted or added family/draw blocks')
    rows, wires = [], []
    for name,digest in value['family_files_sha256'].items():
        path = output/'families'/name
        if common.sha(path) != digest: raise ValueError('Replay family changed')
        block = common.read(path)
        if block['source_bundle_sha256'] != p['family_files_sha256'][name]:
            raise ValueError('Replay source block mismatch')
        rows.extend(block['utility']); wires.extend(block['wire'])
    reconstructed = summarize_depth(rows,wires,p['splits'])
    _assert_baseline(reconstructed,value['summary'])
    result = selection_from_summary(reconstructed)
    result.update(protocol_sha256=common.sha(output/'protocol.json'),readout_sha256=common.sha(output/'readout.json'))
    save_new(output/'depth_selection.json',result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=common.OUT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--public-cache', type=Path)
    parser.add_argument('--stage',choices=('declare','replay','select','all'),default='all')
    args = parser.parse_args(argv)
    if args.stage in ('replay','all') and args.workdir is None: parser.error('--workdir required for replay')
    if args.stage in ('declare','all'): declare_depth_study(args.input,args.output)
    if args.stage in ('replay','all'): replay_depth_study(args.output,args.workdir,args.public_cache)
    if args.stage in ('select','all'): print(json.dumps(select_depth(args.output),indent=2))


if __name__ == '__main__': main()
