"""Controlled local-ranking GPS sensitivity on immutable inspected Geo-I Q.

The frozen privacy inputs/anchors/ledger and network replies are NOT perturbed.
An independent public 60s LOCAL sensor clock feeds hold/extrapolation ranking.
This is an inspected synthetic diagnostic, not fresh confirmation, real GPS
error calibration, total device GNSS reads, energy evidence or a privacy proof.
"""
import argparse
import ast
from collections import defaultdict, OrderedDict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

import numpy as np

from benchmark.public_poi_context import PublicPoiContext
from benchmark.query_purpose import MultiPurposeRoadRanking, QueryPurpose, QuerySpec
from data.lane_states import build_lane_states, catalogue_summary
from evaluation.lane_travel import LanePoiService

ROOT = Path(__file__).resolve().parents[1]
THIS = 'experiments/local_gps_robustness_20261007.py'
TEST = 'tests/test_local_gps_robustness.py'
SOURCE = 'artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1'
PUBLIC = 'artifacts/benchmarks/research_loop/resources.json'
DEFAULT_CACHE = ROOT/'artifacts/benchmarks/dynamic_provider_status_20261006_v1/public_reply60.npz'
OUT = ROOT/'artifacts/benchmarks/local_gps_robustness_20261007_v1'
SEED = 2026100701
FIX_CLOCKS = tuple(float(t) for t in range(0, 601, 60))
DEPTHS = (20, 30)
PURPOSES = tuple(p.value for p in QueryPurpose)
VARIANTS = {'exact_event_oracle': {'mode': 'exact_event_oracle', 'sigma_m': 0.}}
for _mode in ('hold_last60', 'velocity_twofix60'):
    for _sigma in (0, 5, 15):
        VARIANTS[f'{_mode}_s{_sigma}'] = {'mode': _mode, 'sigma_m': float(_sigma)}
ARMS = tuple(f'{name}--L{depth}' for name in VARIANTS for depth in DEPTHS)


def read(path):
    path = Path(path); raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    if path.suffix == '.gz': raw = gzip.compress(raw, mtime=0)
    with path.open('xb') as stream: stream.write(raw)


def standardized_noise(family, draw, slot, fix_t):
    """Public synthetic standard-normal pair, never a protection RNG seed."""
    domain = ['synthetic-local-gps-gaussian-v1', SEED, str(family), int(draw), int(slot), float(fix_t)]
    digest = hashlib.sha256(json.dumps(domain, separators=(',', ':')).encode()).digest()
    uniforms = []
    for chunk in (digest[:8], digest[8:16]):
        value = math.ldexp((int.from_bytes(chunk, 'big') >> 11)+.5, -53)
        uniforms.append(min(math.nextafter(1., 0.), max(math.nextafter(0., 1.), value)))
    radius = math.sqrt(-2.*math.log(uniforms[0])); angle = 2.*math.pi*uniforms[1]
    return [radius*math.cos(angle), radius*math.sin(angle)]


def estimate_xy(fixes, event_t, mode, sigma_m, max_speed=8.):
    """Past fixes only. No truth event/lane/route/heading/destination argument."""
    if (mode not in ('hold_last60', 'velocity_twofix60') or not math.isfinite(event_t) or event_t < 0
            or not math.isfinite(sigma_m) or sigma_m < 0 or not math.isfinite(max_speed) or max_speed <= 0):
        raise ValueError('Valid fixed local-estimator parameters required')
    # Filter before touching future coordinates, including poisoned future data.
    past = sorted((fix for fix in fixes if float(fix['t']) <= event_t), key=lambda fix: fix['t'])
    if not past or len({fix['t'] for fix in past}) != len(past):
        raise ValueError('At least one distinct past local fix required')
    last = past[-1]; previous = past[-2] if len(past) > 1 else None
    def noisy(fix):
        xy, normal = np.asarray(fix['xy'], float), np.asarray(fix['normal'], float)
        if xy.shape != (2,) or normal.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(normal).all():
            raise ValueError('Finite projected fix and synthetic offset required')
        return xy+sigma_m*normal
    position = noisy(last); speed = 0.; clipped = False
    fallback = mode == 'velocity_twofix60' and previous is None
    if mode == 'velocity_twofix60' and previous is not None:
        velocity = (position-noisy(previous))/(last['t']-previous['t'])
        speed = float(np.linalg.norm(velocity)); clipped = speed > max_speed
        if clipped: velocity *= max_speed/speed
        position = position+velocity*(event_t-last['t'])
    return dict(estimated_xy=position.tolist(), last_fix_t=last['t'], previous_fix_t=None if previous is None else previous['t'],
                local_fixes_observed=len(past), stored_fix_count=min(len(past), 2 if mode == 'velocity_twofix60' else 1),
                raw_velocity_m_s=speed, velocity_clipped=clipped, first_fix_fallback=fallback)


def score_rankings(true_orders, estimated_orders, pool, invalid_estimate=False):
    """True-event references never move to the stale/noisy estimate."""
    pool = set(pool); result = {}
    for purpose in PURPOSES:
        entries = true_orders[purpose]
        predicted = [[] for _ in entries] if invalid_estimate else estimated_orders[purpose]
        if len(entries) != len(predicted): raise ValueError('Category inventory differs')
        recalls, completions = [], []
        counts = dict(reference_category_count=0, all_category_count=len(entries), overlap_total=0,
                      reference_poi_total=0, empty_reference_categories=0, zero_answer_reference_categories=0,
                      estimated_empty_categories=0, invalid_estimate_reference_categories=0,
                      returned_items=0, returned_outside_true_domain=0, answered_empty_reference_categories=0)
        for true_order, estimated_order in zip(entries, predicted):
            reference = list(true_order[:5]); answer = [int(i) for i in estimated_order if int(i) in pool][:5]
            domain = set(map(int, true_order)); overlap = len(set(reference) & set(answer))
            counts['returned_items'] += len(answer)
            counts['returned_outside_true_domain'] += len(set(answer)-domain)
            counts['estimated_empty_categories'] += not estimated_order
            if not reference:
                counts['empty_reference_categories'] += 1
                counts['answered_empty_reference_categories'] += bool(answer)
                continue
            recalls.append(overlap/len(reference))
            completions.append(min(len(set(answer) & domain), len(reference))/len(reference))
            counts['reference_category_count'] += 1; counts['overlap_total'] += overlap
            counts['reference_poi_total'] += len(reference)
            counts['zero_answer_reference_categories'] += not answer
            counts['invalid_estimate_reference_categories'] += bool(invalid_estimate)
        result[purpose] = dict(recall5=statistics.mean(recalls) if recalls else None,
                               completion=statistics.mean(completions) if completions else None, **counts)
    return result


class LocalOrders:
    """Same public graph objectives as the prior exact-event evaluator."""
    def __init__(self, context):
        self.context = context; self.ranking = MultiPurposeRoadRanking(LanePoiService(context.rn, list(context.pois), k=5), cache_limit=256)
        self.cache = OrderedDict()

    def ordered(self, state, destination):
        key = int(state), int(destination)
        if key not in self.cache:
            result = {}
            for purpose in QueryPurpose:
                spec = QuerySpec(purpose, self.context.categories[0], k=5,
                    radius_m=1000. if purpose == QueryPurpose.WITHIN_RADIUS else None,
                    destination_state=destination if purpose == QueryPurpose.MIN_DETOUR else None)
                costs = self.ranking.scores(state, spec)
                result[purpose.value] = []
                for category in self.context.categories:
                    ids = [i for i, poi in enumerate(self.context.pois) if poi['category'] == category and math.isfinite(costs[i])]
                    ids.sort(key=lambda i: (float(costs[i]), self.context.pois[i]['id']))
                    result[purpose.value].append(ids)
            self.cache[key] = result
            if len(self.cache) > 512: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


def source_closure():
    pending, found = [THIS], set()
    def add(module):
        for name in (module.replace('.', '/')+'.py', module.replace('.', '/')+'/__init__.py'):
            if (ROOT/name).is_file() and name not in found: pending.append(name)
    while pending:
        name = pending.pop()
        if name in found: continue
        found.add(name); package = name.removesuffix('.py').split('/')[:-1]
        for node in ast.walk(ast.parse((ROOT/name).read_text())):
            if isinstance(node, ast.Import):
                for item in node.names: add(item.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package)-node.level+1] if node.level else []
                module = '.'.join(prefix+([node.module] if node.module else []))
                if module: add(module)
                for item in node.names:
                    if item.name != '*': add('.'.join(filter(None, (module, item.name))))
    return sorted(found | {TEST, 'requirements.txt', 'requirements-sumo.txt'})


def audit_source():
    source = ROOT/SOURCE; p = read(source/'protocol.json'); cert = read(source/'validation.json'); old = read(source/'readout.json')
    assert cert['status'] == 'pass' and cert['protocol_sha256'] == sha(source/'protocol.json')
    assert cert['readout_sha256'] == sha(source/'readout.json') and cert['paired_readout_sha256'] == sha(source/'paired_readout.json')
    assert cert['no_new_Q_GPS_anchors_or_ledger_reads'] is True and p['depths'] == [20, 30]
    base = ROOT/p['base_q_output']; generation = read(base/'generation.json')
    assert sha(base/'generation.json') == old['base_generation_sha256'] and sha(base/'validation.json') == old['base_validation_sha256']
    assert sha(ROOT/p['dataset_path']) == p['dataset_sha256']
    data = read(ROOT/p['dataset_path']); families = sorted(f['family_id'] for f in data['families'] if f['split'] == 'test')
    assert len(families) == 24
    names = [f'{f}--draw{draw}.json.gz' for f in families for draw in (1, 2, 3)]
    for name in names:
        assert sha(source/'families'/name) == old['family_files_sha256'][name]
        assert sha(base/'families'/name) == generation['family_files_sha256'][name]
    return p, data, families, names, old, generation


def declare(output, public_cache=DEFAULT_CACHE):
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists(): raise FileExistsError('New repository output path required')
    p, data, families, names, old, generation = audit_source()
    base, source = ROOT/p['base_q_output'], ROOT/SOURCE
    original_sources = read(base/'protocol.json')['source_sha256']
    privacy_names = ('core/mechanisms.py','core/session_budget.py','benchmark/anchor_belief.py',
        'benchmark/engines/filtered_cover.py','benchmark/engines/matched_filter.py','benchmark/engines/paced_guard.py')
    privacy_pins = {name:original_sources[name] for name in privacy_names}
    for name, pin in privacy_pins.items(): assert sha(ROOT/name) == pin
    family_index = {f['family_id']: f for f in data['families']}
    tape, inventory, controls = {}, {}, {}
    for name in names:
        bundle = read(base/'families'/name); depth = read(source/'families'/name)
        family, draw = bundle['evaluator_only']['family_id'], bundle['evaluator_only']['draw']
        tape[name], inventory[name] = [], []
        assert depth['source_bundle_sha256'] == generation['family_files_sha256'][name]
        controls[name] = depth['frozen_controls']
        for slot, session in enumerate(family_index[family]['evaluator_only']['sessions']):
            trace = {float(row['time_s']): row for row in data['traces'][session['session_id']]}
            fixes = []
            for t in FIX_CLOCKS:
                point = trace[t]
                fixes.append(dict(t=t, lat=point['lat'], lon=point['lon'], normal=standardized_noise(family, draw, slot, t)))
            tape[name].append(fixes)
            events = bundle['public']['streams']['legacy_l10'][slot]['events']
            inventory[name].extend([slot, row['timestamp_s'], row['event_id']] for row in events)
            assert events[0]['timestamp_s'] == 0 and events[-1]['timestamp_s'] == 600
    with np.load(public_cache, allow_pickle=False) as a:
        metadata = json.loads(str(a['metadata'])); digest = hashlib.sha256(str(a['metadata']).encode())
        digest.update(a['signatures'].astype('<i4').tobytes()); digest.update(a['access'].astype('<i4').tobytes())
        assert digest.hexdigest() == p['service_kernel']['full_reply60_sha256'] and metadata['k'] == 60
    output.mkdir(parents=True)
    save(output/'local_sensor_tape.json.gz', dict(schema='public-clock-synthetic-local-gps-tape-v1', seed=SEED, tapes=tape))
    shutil.copyfile(public_cache, output/'public_reply60.npz')
    protocol = dict(schema='fixed-Q-local-gps-robustness-v1', created_utc=datetime.now(timezone.utc).isoformat(),
        source_output=SOURCE, base_Q_output=p['base_q_output'], dataset_path=p['dataset_path'], dataset_sha256=p['dataset_sha256'],
        source_files_sha256={n:sha(source/n) for n in ('protocol.json','readout.json','paired_readout.json','resources.json','validation.json')},
        base_files_sha256={n:sha(base/n) for n in ('protocol.json','generation.json','resources.json','validation.json')},
        source_family_files_sha256={n:old['family_files_sha256'][n] for n in names},
        base_family_files_sha256={n:generation['family_files_sha256'][n] for n in names},
        source_sha256={n:sha(ROOT/n) for n in source_closure()}, public_resource_sha256=sha(ROOT/PUBLIC),
        frozen_privacy_source_sha256=privacy_pins,
        public_reply60_sha256=p['service_kernel']['full_reply60_sha256'], reply_cache_sha256=sha(output/'public_reply60.npz'),
        local_sensor_tape_sha256=sha(output/'local_sensor_tape.json.gz'), included_blocks=names, families=families,
        split='test', draws=[1,2,3], variants=VARIANTS, arms=ARMS, depths=DEPTHS, purposes=PURPOSES,
        local_fix_clocks=FIX_CLOCKS, sensor_interval_s=60., speed_ceiling_m_s=8., radius_m=1000., reference_k=5,
        noise_seed=SEED, noise='Controlled per-axis Gaussian sigma0/5/15m via explicit SHA256/Box-Muller; shared standardized offset across sigma/mode/depth; not a calibrated GPS model',
        estimate='Hold or straight Cartesian last-two-noisy-fix velocity, clip8m/s, first-fix hold; nearest public graph vertex only; no true lane/route/heading/future or purpose-driven retry',
        snap='Always nearest public vertex for a finite estimate; no truth lane or maximum-distance censoring; report pre/post error and snap distance; invalid estimate scores0 for defined reference',
        destination='Known actual session destination is retained ONLY as a local/evaluator oracle for detour; also report three-purpose macro excluding detour',
        utility='True event GPS top5 reference; estimated local position ranks received pool; true reference-empty N/A, no estimated answer for defined reference is0; explicit true-domain violation counts; current-only',
        clocks_boundary='11 virtual LOCAL sensor reads/session are independent of protected supplier/cap; frozen privacy trace used earlier exact synthetic GPS. Not total device GNSS reads or energy measurements',
        exact_control='Exact-event GPS sigma0 oracle at every public event; MUST recover saved historical current-only scores before diagnostic output',
        scope='SECONDARY already-inspected24TESTfamilies x3draws; no fresh confirmation, true-input-noise Geo-I test, privacy improvement, retuning, extra Q/requests or private key',
        frozen_controls=controls, public_event_inventory=inventory, parameter_selection='All fixed arms retained; no winner selection or threshold tuning',
        runtime=dict(python=__import__('sys').version, numpy=np.__version__))
    save(output/'protocol.json', protocol); (output/'protocol.sha256').write_text(sha(output/'protocol.json')+'\n')
    for name in protocol['source_sha256']:
        target = output/'source_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes((ROOT/name).read_bytes())
    return protocol


def validate(output):
    output = Path(output); p = read(output/'protocol.json')
    assert sha(output/'protocol.json') == (output/'protocol.sha256').read_text().strip()
    assert p['variants'] == VARIANTS and p['arms'] == list(ARMS) and p['local_fix_clocks'] == list(FIX_CLOCKS)
    for name, pin in p['source_sha256'].items(): assert sha(ROOT/name) == sha(output/'source_snapshot'/name) == pin
    for name, pin in p['frozen_privacy_source_sha256'].items(): assert sha(ROOT/name) == pin
    for folder, field in ((ROOT/p['source_output'],'source_files_sha256'),(ROOT/p['base_Q_output'],'base_files_sha256')):
        for name, pin in p[field].items(): assert sha(folder/name) == pin
    assert sha(ROOT/p['dataset_path']) == p['dataset_sha256'] and sha(ROOT/PUBLIC) == p['public_resource_sha256']
    assert sha(output/'public_reply60.npz') == p['reply_cache_sha256']
    assert sha(output/'local_sensor_tape.json.gz') == p['local_sensor_tape_sha256']
    return p


def build_context(output, p):
    data = read(ROOT/p['dataset_path']); network = ROOT/data['network']['compressed_path']
    assert sha(network) == data['network']['compressed_sha256']
    assert hashlib.sha256(gzip.decompress(network.read_bytes())).hexdigest() == data['network']['native_sha256']
    rn = build_lane_states(network, spacing_m=40.)
    pois = [{k:v for k,v in row.items() if k not in ('vertex','access_offset_m')} for row in read(ROOT/PUBLIC)['pois_used']]
    context = PublicPoiContext(LanePoiService(rn, pois, k=60), Path(output)/'public_reply60.npz')
    assert context.sha256 == p['public_reply60_sha256']
    return context


def _baseline(score, expected):
    for purpose in PURPOSES:
        for key in ('recall5','completion','reference_category_count','all_category_count','overlap_total','reference_poi_total'):
            left, right = score[purpose][key], expected[purpose][key]
            if isinstance(left, float): assert math.isclose(left, right, rel_tol=0, abs_tol=1e-12), (purpose,key,left,right)
            else: assert left == right, (purpose,key,left,right)


def recover_exact_block(bundle, depth, context, orders):
    """Recover the old exact oracle BEFORE calculating any sensor arm."""
    streams, rn = bundle['public']['streams'], context.rn
    old_rows = {(r['method'],r['slot'],r['t']):r for r in depth['utility'] if r['cache'] == 'current'}
    old_wire = {(r['method'],r['slot'],r['t']):r for r in depth['wire']}
    events_checked = 0
    for slot, session in enumerate(streams['legacy_l10']):
        raw = streams['raw'][slot]['events']
        assert [e['timestamp_s'] for e in raw] == [e['timestamp_s'] for e in session['events']]
        endpoint = raw[-1]['candidates'][0]; destination = rn.nearest(endpoint['lat'],endpoint['lon'])[0]
        for event, original in zip(session['events'],raw):
            t = event['timestamp_s']; point = original['candidates'][0]
            reference = orders.ordered(rn.nearest(point['lat'],point['lon'])[0],destination)
            for L in DEPTHS:
                replies = [[int(i) for i in context.query_indices(rn.nearest(q['lat'],q['lon'])[0])[:,:L].ravel() if i >= 0] for q in event['candidates']]
                assert replies == old_wire[f'service_l{L}',slot,t]['reply_poi_ids_by_Q']
                pool = {i for reply in replies for i in reply}
                _baseline(score_rankings(reference,reference,pool),old_rows[f'service_l{L}',slot,t]['purposes'])
            events_checked += 1
    assert len(old_rows) == len(old_wire) == events_checked*len(DEPTHS)
    return events_checked


def replay_block(bundle, depth, context, orders, tape):
    truth, streams = bundle['evaluator_only'], bundle['public']['streams']; rn = context.rn
    old_rows = {(r['method'],r['slot'],r['t']):r for r in depth['utility'] if r['cache'] == 'current'}
    old_wire = {(r['method'],r['slot'],r['t']):r for r in depth['wire']}
    rows, estimates = [], []
    for slot, session in enumerate(streams['legacy_l10']):
        raw = streams['raw'][slot]['events']; assert [e['timestamp_s'] for e in raw] == [e['timestamp_s'] for e in session['events']]
        endpoint = raw[-1]['candidates'][0]; destination = rn.nearest(endpoint['lat'], endpoint['lon'])[0]
        fixes = [dict(t=f['t'],xy=rn.point_xy(f['lat'],f['lon']),normal=f['normal']) for f in tape[slot]]
        for event, original in zip(session['events'], raw):
            t = event['timestamp_s']; point = original['candidates'][0]; true_xy = np.array(rn.point_xy(point['lat'],point['lon']))
            true_state = rn.nearest(point['lat'],point['lon'])[0]; reference = orders.ordered(true_state,destination)
            positions = [(q['lat'],q['lon']) for q in event['candidates']]; assert len(positions) == 5
            pools = {}
            for L in DEPTHS:
                wire = old_wire[f'service_l{L}',slot,t]
                replies = [[int(i) for i in context.query_indices(rn.nearest(*q)[0])[:,:L].ravel() if i >= 0] for q in positions]
                assert replies == wire['reply_poi_ids_by_Q']
                pools[L] = {i for reply in replies for i in reply}
            assert pools[20] <= pools[30]
            for variant, config in VARIANTS.items():
                if config['mode'] == 'exact_event_oracle':
                    state = dict(estimated_xy=true_xy.tolist(),last_fix_t=t,previous_fix_t=None,
                        local_fixes_observed=None,stored_fix_count=1,raw_velocity_m_s=0.,velocity_clipped=False,first_fix_fallback=False)
                else: state = estimate_xy(fixes,t,config['mode'],config['sigma_m'])
                xy = np.asarray(state['estimated_xy']); valid = bool(np.isfinite(xy).all())
                if valid:
                    distance, idx = rn.tree.query(xy); index = int(idx)
                    predicted = orders.ordered(index,destination)
                    before, after = float(np.linalg.norm(xy-true_xy)), float(np.linalg.norm(rn.xy[index]-true_xy))
                else: index = None; distance = before = after = None; predicted = {}
                estimate = dict(slot=slot,t=t,event_id=event['event_id'],variant=variant,**state,
                    estimated_state=index,snap_distance_m=None if distance is None else float(distance),
                    error_before_snap_m=before,error_after_snap_m=after,invalid_reason=None if valid else 'nonfinite_local_estimate')
                estimates.append(estimate)
                for L in DEPTHS:
                    score = score_rankings(reference,predicted,pools[L],invalid_estimate=not valid)
                    if variant == 'exact_event_oracle': _baseline(score,old_rows[f'service_l{L}',slot,t]['purposes'])
                    rows.append(dict(family_id=truth['family_id'],draw=truth['draw'],split='test',slot=slot,t=t,
                        event_id=event['event_id'],variant=variant,depth=L,arm=f'{variant}--L{L}',purposes=score))
    return dict(schema='local-gps-robustness-block-v1',family_id=truth['family_id'],draw=truth['draw'],split='test',
                estimates=estimates,rows=rows,wire=depth['wire'],frozen_controls=depth['frozen_controls'],
                local_sensor_fix_count_per_session=[len(fixes) for fixes in tape],private_GPS_or_Q_regeneration=False,
                source_exact_current_scores_recovered=True)


def summarize(blocks):
    groups, costs, error_cells = defaultdict(list), defaultdict(lambda:defaultdict(int)), defaultdict(list)
    for block in blocks:
        for row in block['rows']:
            phases = ['all']+(['cold'] if row['slot'] == 0 else [])+(['temporal_tail_400_600'] if row['t'] >= 400 else [])
            for phase in phases:
                for purpose in PURPOSES: groups[row['arm'],phase,purpose,row['family_id'],row['draw']].append(row['purposes'][purpose])
        for row in block['wire']:
            for variant in VARIANTS:
                arm = f"{variant}--L{row['method'].removeprefix('service_l')}"
                for field in ('requests','request_bytes','reply_bytes'): costs[arm][field] += row[field]
        for row in block['estimates']: error_cells[row['variant']].append(row)
    summary, local, local_index = {}, [], {}
    for arm in ARMS:
        summary[arm] = {}
        for phase in ('all','cold','temporal_tail_400_600'):
            result = {}
            for purpose in PURPOSES:
                family, draws, counts = defaultdict(list), defaultdict(list), defaultdict(int)
                for (a,ph,pu,f,draw), values in sorted(groups.items()):
                    if (a,ph,pu) != (arm,phase,purpose): continue
                    defined = [v['recall5'] for v in values if v['recall5'] is not None]; mean = statistics.mean(defined) if defined else None
                    family[f].append(mean); draws[draw].append(mean)
                    local.append(dict(arm=arm,phase=phase,purpose=purpose,family_id=f,draw=draw,recall5=mean,total_events=len(values),defined_events=len(defined)))
                    local_index[arm,phase,f,draw,purpose] = mean
                    counts['total_events'] += len(values); counts['defined_events'] += len(defined)
                    for value in values:
                        for field, count in value.items():
                            if field not in ('recall5','completion'): counts[field] += count
                family_values = {f:statistics.mean([v for v in vlist if v is not None]) if any(v is not None for v in vlist) else None for f,vlist in family.items()}
                finite = [v for v in family_values.values() if v is not None]
                result[purpose] = dict(family_mean=statistics.mean(finite) if finite else None,family_values=family_values,
                    within_draw_family_mean={str(d):statistics.mean([v for v in vlist if v is not None]) if any(v is not None for v in vlist) else None for d,vlist in draws.items()},explicit_denominators=dict(counts))
            for name, purposes in (('equal_purpose_macro',PURPOSES),('three_purpose_macro',PURPOSES[:3])):
                macros = {}
                for f in result[PURPOSES[0]]['family_values']:
                    values = [result[p]['family_values'][f] for p in purposes if result[p]['family_values'][f] is not None]
                    if values: macros[f] = statistics.mean(values)
                per_draw = {}
                for draw in (1,2,3):
                    draw_values = []
                    for f in macros:
                        values = [local_index.get((arm,phase,f,draw,purpose)) for purpose in purposes]
                        values = [value for value in values if value is not None]
                        if values: draw_values.append(statistics.mean(values))
                    per_draw[str(draw)] = statistics.mean(draw_values) if draw_values else None
                result[name] = dict(family_mean=statistics.mean(macros.values()) if macros else None,family_values=macros,
                    minimum_family_mean=min(macros.values()) if macros else None,
                    within_draw_family_mean=per_draw)
            summary[arm][phase] = result
    errors = {}
    for variant, values in error_cells.items():
        fields = {}
        for name in ('error_before_snap_m','error_after_snap_m','snap_distance_m'):
            finite = [v[name] for v in values if v[name] is not None]
            fields[name] = dict(mean=statistics.mean(finite) if finite else None,p50=float(np.percentile(finite,50)) if finite else None,
                p95=float(np.percentile(finite,95)) if finite else None,max=max(finite) if finite else None,defined_events=len(finite))
        errors[variant] = dict(events=len(values),invalid_events=sum(v['invalid_reason'] is not None for v in values),
            velocity_clipped_events=sum(v['velocity_clipped'] for v in values),first_fix_fallback_events=sum(v['first_fix_fallback'] for v in values),**fields)
    return dict(summary=summary,family_draw_purpose_values=local,cost={k:dict(v) for k,v in costs.items()},position_error=errors)


def replay(output):
    output = Path(output); p = validate(output)
    if (output/'readout.json').exists(): raise FileExistsError('Completed diagnostic cannot be replaced')
    verifier = read(output/'verification_protocol.json')
    assert verifier['workload_protocol_sha256'] == sha(output/'protocol.json')
    for name,pin in verifier['source_sha256'].items(): assert sha(ROOT/name) == pin
    save(output/'replay_started.json',dict(schema='local-gps-replay-start-v1',started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(output/'protocol.json'),verification_protocol_sha256=sha(output/'verification_protocol.json'),
        local_sensor_tape_sha256=p['local_sensor_tape_sha256'],stage='before_public_context_and_scoring'))
    context = build_context(output,p); orders = LocalOrders(context); tape = read(output/'local_sensor_tape.json.gz')['tapes']
    baseline_events = 0
    for index,name in enumerate(p['included_blocks'],1):
        base, source = ROOT/p['base_Q_output']/'families'/name, ROOT/p['source_output']/'families'/name
        assert sha(base) == p['base_family_files_sha256'][name] and sha(source) == p['source_family_files_sha256'][name]
        baseline_events += recover_exact_block(read(base),read(source),context,orders)
        print(f'Exact-event baseline recovery {index}/{len(p["included_blocks"])} blocks',flush=True)
    save(output/'baseline_recovery.json',dict(schema='local-gps-exact-baseline-recovery-v1',
        protocol_sha256=sha(output/'protocol.json'),blocks=len(p['included_blocks']),public_events=baseline_events,
        depths=list(DEPTHS),metrics=['recall5','completion','reference_category_count','all_category_count','overlap_total','reference_poi_total'],
        completed_before_sensor_arm_scoring=True,all_exact_current_scores_recovered=True))
    blocks, pins = [], {}
    for index,name in enumerate(p['included_blocks'],1):
        base, source = ROOT/p['base_Q_output']/'families'/name, ROOT/p['source_output']/'families'/name
        assert sha(base) == p['base_family_files_sha256'][name] and sha(source) == p['source_family_files_sha256'][name]
        block = replay_block(read(base),read(source),context,orders,tape[name])
        block.update(source_bundle_sha256=sha(base),source_depth_bundle_sha256=sha(source))
        save(output/'blocks'/name,block); pins[name] = sha(output/'blocks'/name); blocks.append(block)
        print(f'Local GPS diagnostic {index}/{len(p["included_blocks"])} blocks; exact baseline recovered',flush=True)
    result = dict(schema='local-gps-robustness-readout-v1',protocol_sha256=sha(output/'protocol.json'),
        verification_protocol_sha256=sha(output/'verification_protocol.json'),replay_started_sha256=sha(output/'replay_started.json'),
        baseline_recovery_sha256=sha(output/'baseline_recovery.json'),
        local_sensor_tape_sha256=p['local_sensor_tape_sha256'],block_files_sha256=pins,**summarize(blocks),
        fixed_Q_clocks_anchors_ledger_wire=True,no_private_key_or_protection_sampler_calls=True,
        source_already_inspected_secondary_only=True,total_public_events=sum(len(b['estimates'])//len(VARIANTS) for b in blocks),
        virtual_local_sensor_fixes=sum(sum(b['local_sensor_fix_count_per_session']) for b in blocks),
        local_clock_is_not_total_device_GNSS_reads=True,all_exact_current_scores_recovered=True)
    save(output/'resources.json',dict(catalogue=catalogue_summary(context.rn),public_reply60_sha256=context.sha256,pois=len(context.pois)))
    save(output/'readout.json',result); validate(output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('declare','replay','all'),required=True)
    parser.add_argument('--output',type=Path,default=OUT); parser.add_argument('--public-cache',type=Path,default=DEFAULT_CACHE)
    args = parser.parse_args()
    if args.stage in ('declare','all'): declare(args.output,args.public_cache); print('Local sensor/noise/source protocol frozen',args.output,flush=True)
    if args.stage in ('replay','all'): replay(args.output); print('Completed secondary local-GPS diagnostic',args.output,flush=True)
