"""Write-once, independently grouped native SUMO cohort for Q-planner evaluation.

All route eligibility uses public geometry and counterfactual native calibration,
before hidden synthetic choices. No defense, attacker or utility score is read.
The same archived native map and public POIs are retained. This is a new
same-generator/same-map generalization cohort, never a real GPS or cross-city
confirmation dataset.
"""
from pathlib import Path
from itertools import combinations
import argparse
import gzip
import hashlib
import hmac
import json
import math
import os
import random
import secrets
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import sumolib

from data.scenario_suite.mobility import parse_fcd, successors
from experiments.build_future_sumo_cohort import (
    ROOT, sha, save, prefix, tail, tangent, via_lanes,
    clocks_from_public_calibration,
)

SOURCE = ROOT/'artifacts/datasets/future_controlled_20261005_v2/dataset.json.gz'
PUBLIC_POIS = ROOT/'artifacts/benchmarks/research_loop/resources.json'
OUT = ROOT/'artifacts/datasets/qplanner_fresh_native_20261006_v2'
WORK = Path('/private/tmp/trajectory-qplanner-fresh-native-20261006-v2')
PUBLIC_SEED = 2026100671
SPLIT_COUNTS = {'train': 24, 'selection': 12, 'test': 24}
SPEEDS = (6., 8.)
SOURCE_FILES = (
    'experiments/build_qplanner_fresh_native_20261006_v2.py',
    'experiments/build_future_sumo_cohort.py',
    'data/scenario_suite/mobility.py',
    'requirements-sumo.txt',
    'tests/test_qplanner_fresh_native_cohort_v2.py',
)


def compressed_save(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Preserve completed evidence: {path}')
    with path.open('xb') as stream:
        stream.write(gzip.compress(json.dumps(payload, separators=(',', ':'),
            allow_nan=False).encode(), mtime=0))


def assignments(seed=PUBLIC_SEED):
    """Balanced public speed labels and source-family splits; no GPS inputs."""
    result = []
    rng = random.Random(seed)
    for split, count in SPLIT_COUNTS.items():
        if count % len(SPEEDS):
            raise ValueError('Each split must balance all public speed strata')
        speeds = list(SPEEDS)*(count//len(SPEEDS))
        rng.shuffle(speeds)
        for speed in speeds:
            result.append({'family_id': f'freshqp-{len(result)+1:03d}',
                'split': split, 'public_speed_m_s': speed})
    return result


def route_overlap(a, b):
    """Public edge-set Jaccard guard against effectively copied routes."""
    a, b = set(a), set(b)
    return len(a & b)/len(a | b) if a or b else 1.


def independent_route_pair(proposal, retained_routes, threshold=.8):
    return all(route_overlap(route, old) < threshold
        for route in proposal['routes'] for old in retained_routes)


def route_proposal(network, fork, attempt):
    rng = random.Random(PUBLIC_SEED+attempt*1009)
    choices = [e for e in successors(fork)
        if sum(l.getLength() for l in via_lanes(network, fork, e)) >= 12.]
    pairs = [(a, b) for a, b in combinations(choices, 2)
        if float(tangent(a) @ tangent(b)) < .6]
    rng.shuffle(pairs)
    common = prefix(fork, rng)
    for a, b in pairs[:16]:
        try:
            tails = [tail(e, common, rng) for e in (a, b)]
            destinations = [p[-1].getShape()[-1] for p in tails]
            if math.dist(*destinations) < 600.:
                continue
            if any(sum(e.getLength() for e in common+t) > 4000. for t in tails):
                continue
            ordered = sorted(zip((a, b), tails), key=lambda v: v[0].getID())
            return {'fork_edge': fork.getID(),
                'public_junction_group': fork.getToNode().getID(),
                'common_route': [e.getID() for e in common],
                'routes': [[e.getID() for e in common+t] for _, t in ordered],
                'public_choices': [{'edge_id': e.getID(),
                    'via_lane_ids': [l.getID() for l in via_lanes(network, fork, e)],
                    'via_shape_xy': [list(p) for l in via_lanes(network, fork, e) for p in l.getShape()],
                    'outgoing_shape_xy': [list(p) for p in e.getShape()],
                    'destination_xy': list(t[-1].getShape()[-1]),
                    'destination_edge_id': t[-1].getID()} for e, t in ordered]}
        except ValueError:
            continue
    raise ValueError('no_public_separated_branches')


def simulate(net_path, network, specs, directory, seed, *, park):
    if not 0 <= seed <= 2**31-1:
        raise ValueError('SUMO seed must fit signed32-bit nonnegative range')
    directory.mkdir(parents=True, exist_ok=False)
    xml = ET.Element('routes')
    for speed in SPEEDS:
        ET.SubElement(xml, 'vType', id=f'controlled-{int(speed)}', vClass='passenger',
            maxSpeed=str(speed), speedFactor='1', speedDev='0', sigma='0', length='4', minGap='2')
    for spec in specs:
        depart = spec['depart_s']
        vehicle = ET.SubElement(xml, 'vehicle', id=spec['session_id'],
            type=f'controlled-{int(spec["public_speed_m_s"])}',
            depart=str(depart), departLane='0', departPos='5',
            departSpeed=str(spec['public_speed_m_s']))
        ET.SubElement(vehicle, 'route', edges=' '.join(spec['route_edges']))
        if park:
            lane = network.getEdge(spec['route_edges'][-1]).getLanes()[0]
            ET.SubElement(vehicle, 'stop', lane=lane.getID(),
                endPos=str(lane.getLength()-5), until=str(depart+650.), parking='false')
    routes = directory/'routes.rou.xml'
    ET.ElementTree(xml).write(routes, encoding='utf-8', xml_declaration=True)
    fcd, vehicle_routes = directory/'fcd.xml', directory/'vehicle_routes.xml'
    executable = Path(shutil.which('sumo') or Path(sys.executable).parent/'sumo')
    command = [str(executable), '--net-file', str(net_path), '--route-files', str(routes),
        '--seed', str(seed), '--begin', '0', '--end', str(max(s['depart_s'] for s in specs)+1400.),
        '--step-length', '1', '--device.fcd.period', '1', '--fcd-output', str(fcd),
        '--fcd-output.geo', 'true', '--fcd-output.attributes', 'id,x,y,speed,lane,pos,angle',
        '--vehroute-output', str(vehicle_routes), '--vehroute-output.write-unfinished', 'true',
        '--time-to-teleport', '-1', '--no-step-log', 'true', '--no-warnings', 'true']
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
    traces = parse_fcd(fcd, network)
    vehicles = {v.attrib['id']: v.attrib for v in ET.parse(vehicle_routes).getroot().findall('vehicle')}
    if set(traces) != {s['session_id'] for s in specs} or set(vehicles) != set(traces):
        raise ValueError('native_sumo_missing_trip')
    if any(not 0 <= float(v.get('arrival', -1))-float(v['depart']) < 1400.
        for v in vehicles.values()):
        raise ValueError('native_sumo_incomplete_or_unisolated_trip')
    if not park and any(float(v['arrival'])-float(v['depart']) > 550. for v in vehicles.values()):
        raise ValueError('public_counterfactual_trip_not_parkable_before600')
    if park:
        for trace in traces.values():
            origin = trace[0]['time_s']
            by_time = {p['time_s']-origin: p for p in trace}
            if not set(range(601)).issubset(by_time):
                raise ValueError('native_fixed_window_missing_1Hz_fix')
            if by_time[600.]['speed_m_s'] != 0.:
                raise ValueError('native_destination_not_parked_at600')
    return traces, {'command': command, 'stderr': result.stderr,
        'files': {p.name: sha(p) for p in (routes, fcd, vehicle_routes)}, 'arrivals': vehicles}


def archive_native_sources(directory, target):
    target.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name in ('routes.rou.xml', 'fcd.xml', 'vehicle_routes.xml'):
        source = directory/name
        path = target/f'{name}.gz'
        with path.open('xb') as stream:
            stream.write(gzip.compress(source.read_bytes(), mtime=0))
        entries[name] = {'compressed_path': str(path.relative_to(ROOT)),
            'compressed_sha256': sha(path), 'native_sha256': sha(source)}
    return entries


def private_choice_key(work):
    private = work/'private_state'
    private.mkdir(mode=0o700)
    os.chmod(private, 0o700)
    key = secrets.token_bytes(32)
    fd = os.open(private/'cohort_choice.key', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(key)
    return key


def choice_pattern(key, family_id):
    digest = hmac.new(key, ('fresh-qplanner-choice-v1\0'+family_id).encode(), hashlib.sha256).digest()
    routine = digest[0] % 2
    query = [routine, 1-routine]
    if digest[1] % 2:
        query.reverse()
    return routine, [routine, routine, 1-routine, routine, routine, routine]+query


def declare(output, original):
    source_pins = {name: sha(ROOT/name) for name in SOURCE_FILES}
    return {'schema': 'fresh-qplanner-native-cohort-protocol-v1',
        'public_geometry_seed': PUBLIC_SEED, 'assignments': assignments(),
        'split_counts': SPLIT_COUNTS, 'source_sha256': source_pins,
        'old_dataset_sha256': sha(SOURCE), 'public_poi_source_sha256': sha(PUBLIC_POIS),
        'old_public_calibration_sha256': sha(ROOT/original['public_calibration_source']),
        'network_sha256': original['network']['native_sha256'],
        'max_public_forks_screened': 800, 'route_near_duplicate_jaccard_reject_at': .8,
        'grouping': 'public incoming-fork terminal junction; all original24 junction groups excluded, '
            'all new families use distinct junction groups; any public route Jaccard>=.8 to old/new routes rejected',
        'public_eligibility': 'incoming>=180m, sharedprefix>=700m, nativevia>=12m, '
            'branchdirectiondot<.6, tails>=1000m, separatedendpoints>=600m, routes<=4000m',
        'public_calibration': 'both branches at BOTH6/8m/s, all arrive<=550s, raw prefix equality, '
            'shared internal turn clock; calibrate before hidden choices and speed allocation',
        'speed_allocation': 'public seeded balanced6/8m/s within each split, no privateGPS/protection/score input',
        'native_history': 'eight authentic SUMO trips,5routine1rare history; balanced routine/rare queries in private order; '
            'fixed public0..600s 1Hz native fixes, real destination lane stop until650s,1500s departure spacing',
        'choice_randomness': 'OS32byte HMAC master retained only outside repository; no sampler/choice seed in dataset',
        'test_access': 'dataset integrity only; root must freeze selected method/source before any fresh test readout',
        'public_pois': 'same complete public418-record catalogue from archived research_loop resources, '
            'never constructed from generatedGPS or heldout labels',
        'prohibited_selection': 'no defense/attacker/utility scores, no rawGPS edits/interpolation, no winner-directed filtering',
        'scope': 'independent same-native-map synthetic generalization/stability after source/config freeze; '
            'not realGPS, crosscity, dynamic availability or complete JISA confirmation',
        'output': str(output.relative_to(ROOT))}


def build(output=OUT, work=WORK):
    output, work = Path(output).resolve(), Path(work).resolve()
    if not output.is_relative_to(ROOT) or work.is_relative_to(ROOT):
        raise ValueError('Evidence output belongs in repository; private work directory must remain outside it')
    if output.exists() or work.exists():
        raise FileExistsError('Use new evidence/private-work paths; never resume or overwrite a sealed cohort')
    original = json.loads(gzip.decompress(SOURCE.read_bytes()))
    output.mkdir(parents=True)
    work.mkdir(parents=True, mode=0o700)
    protocol = declare(output, original)
    save(output/'protocol.json', protocol)
    (output/'protocol.sha256').write_text(sha(output/'protocol.json')+'\n')
    for name in SOURCE_FILES:
        snapshot = output/'source_snapshot'/name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT/name).read_bytes())
    compressed = ROOT/original['network']['compressed_path']
    if sha(compressed) != original['network']['compressed_sha256']:
        raise ValueError('Original public native network archive changed')
    net_path = work/'native.net.xml'
    net_path.write_bytes(gzip.decompress(compressed.read_bytes()))
    if sha(net_path) != protocol['network_sha256']:
        raise ValueError('Native network source mismatch')
    network = sumolib.net.readNet(str(net_path), withInternal=True)
    key = private_choice_key(work)
    excluded_junctions = {network.getEdge(f['fork_edge']).getToNode().getID()
        for f in original['families']}
    old_calibration = json.loads(gzip.decompress((ROOT/original['public_calibration_source']).read_bytes()))
    retained_routes = [route for calibration in old_calibration['public_calibrations']
        for route in calibration['proposal']['routes']]
    forks = sorted([e for e in network.getEdges(withInternal=False)
        if e.allows('passenger') and e.getLength() >= 180. and len(successors(e)) >= 2
        and e.getToNode().getID() not in excluded_junctions], key=lambda e: e.getID())
    random.Random(PUBLIC_SEED).shuffle(forks)
    families, public_calibrations, traces, rejects = [], [], {}, []
    used_junctions = set(excluded_junctions)
    native_dir = output/'native_sources'
    start = time.perf_counter()
    for attempt, fork in enumerate(forks[:protocol['max_public_forks_screened']]):
        if len(families) == sum(SPLIT_COUNTS.values()):
            break
        junction = fork.getToNode().getID()
        try:
            if junction in used_junctions:
                raise ValueError('public_junction_group_already_retained')
            proposal = route_proposal(network, fork, attempt)
            if not independent_route_pair(proposal, retained_routes):
                raise ValueError('public_route_near_duplicate_old_or_new')
            calibration_specs = [{'session_id': f'calibration_s{int(speed)}_b{branch}',
                'route_edges': route, 'depart_s': (2*speed_index+branch)*1500.,
                'public_speed_m_s': speed} for speed_index, speed in enumerate(SPEEDS)
                for branch, route in enumerate(proposal['routes'])]
            directory = work/f'calibration_{attempt:03d}'
            calibration, cal_run = simulate(net_path, network, calibration_specs, directory,
                PUBLIC_SEED+attempt, park=False)
            clocks_by_speed = {}
            for speed in SPEEDS:
                pair = {f'calibration_{b}': calibration[f'calibration_s{int(speed)}_b{b}'] for b in (0, 1)}
                clocks_by_speed[str(int(speed))] = clocks_from_public_calibration(proposal, pair, network)
        except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            rejects.append({'public_fork': fork.getID(), 'public_junction_group': junction,
                'attempt': attempt, 'reason': str(exc)})
            continue
        assigned = protocol['assignments'][len(families)]
        family_id, speed = assigned['family_id'], assigned['public_speed_m_s']
        clocks = clocks_by_speed[str(int(speed))]
        routine, pattern = choice_pattern(key, family_id)
        specs = [{'session_id': f'{family_id}_day{day}', 'day': day,
            'depart_s': (day-1)*1500., 'public_speed_m_s': speed,
            'route_edges': proposal['routes'][branch], 'choice_index': branch,
            'destination_role': 'routine' if branch == routine else 'rare'}
            for day, branch in enumerate(pattern, 1)]
        actual_dir = work/family_id
        actual, actual_run = simulate(net_path, network, specs, actual_dir,
            PUBLIC_SEED+10000+len(families), park=True)
        for spec in specs:
            trace = actual[spec['session_id']]
            origin = trace[0]['time_s']
            for p in trace:
                p['time_s'] -= origin
            by_time = {int(p['time_s']): p for p in trace}
            if by_time[clocks['turn_visible_t']]['lane_id'] not in proposal['public_choices'][spec['choice_index']]['via_lane_ids']:
                raise ValueError('Native actual turn changed after public calibration; retain failure, do not replace target')
        queries = [actual[s['session_id']] for s in specs if s['day'] in (7, 8)]
        cut = clocks['shared_fork_t']
        if [(p['lat'], p['lon']) for p in queries[0] if p['time_s'] <= cut] != [
                (p['lat'], p['lon']) for p in queries[1] if p['time_s'] <= cut]:
            raise ValueError('Native actual shared prefix unequal; retain failure, do not replace target')
        cal_archive = archive_native_sources(directory, native_dir/f'{family_id}-calibration')
        actual_archive = archive_native_sources(actual_dir, native_dir/family_id)
        families.append({**assigned, 'public_attempt': attempt, 'fork_edge': fork.getID(),
            'public_junction_group': junction, 'clocks': clocks,
            'public_context': {'choices': proposal['public_choices']},
            'evaluator_only': {'routine_choice': routine, 'sessions': specs, 'actual_run': actual_run,
                'calibration_run': cal_run, 'native_source_archives': actual_archive}})
        public_calibrations.append({'family_id': family_id, 'proposal': proposal,
            'clocks_by_speed': clocks_by_speed, 'source': cal_run,
            'native_source_archives': cal_archive})
        traces.update(actual)
        used_junctions.add(junction)
        retained_routes.extend(proposal['routes'])
        print('Fresh native source accepted', family_id, assigned['split'],
            'public speed', speed, 'public fork attempt', attempt, flush=True)
    save(output/'public_eligibility.json', {'available_public_forks': len(forks),
        'retained_family_count': len(families), 'screened_fork_count': len(families)+len(rejects),
        'rejections': rejects, 'excluded_old_public_junctions': sorted(excluded_junctions)})
    if len(families) != sum(SPLIT_COUNTS.values()):
        save(output/'generation_failure.json', {'reason': 'insufficient_eligible_public_groups',
            'retained': len(families), 'required': sum(SPLIT_COUNTS.values()),
            'no_protection_score_inspected': True})
        raise RuntimeError('Insufficient public eligible families; preserve failed cohort')
    new_net = output/'public_native.net.xml.gz'
    new_net.write_bytes(compressed.read_bytes())
    native_meta = {**original['network'], 'compressed_path': str(new_net.relative_to(ROOT)),
        'compressed_sha256': sha(new_net), 'same_archived_native_map': True}
    for name, expected in protocol['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Sealed generator source changed during generation: {name}')
    if sha(SOURCE) != protocol['old_dataset_sha256'] or sha(PUBLIC_POIS) != protocol['public_poi_source_sha256']:
        raise ValueError('Pinned old dataset/public POIs changed during generation')
    payload = {'schema': 'fresh-qplanner-native-cohort-v1',
        'source': 'authentic native SUMO1Hz FCD; fresh public grouped routes with synthetic private choices',
        'protocol_sha256': sha(output/'protocol.json'), 'source_sha256': protocol['source_sha256'],
        'network': native_meta, 'families': families, 'traces': traces,
        'public_calibrations': public_calibrations, 'family_count': len(families),
        'session_count': len(traces), 'public_eligibility_rejections': rejects,
        'scope': protocol['scope'], 'choice_key_exported': False}
    target = output/'dataset.json.gz'
    compressed_save(target, payload)
    version = subprocess.run([str(Path(sys.executable).parent/'sumo'), '--version'],
        check=True, capture_output=True, text=True).stdout
    save(output/'manifest.json', {'schema': 'fresh-qplanner-native-manifest-v1',
        'dataset_sha256': sha(target), 'protocol_sha256': payload['protocol_sha256'],
        'source_sha256': protocol['source_sha256'], 'network': native_meta,
        'family_count': len(families), 'session_count': len(traces), 'split_counts': SPLIT_COUNTS,
        'public_speed_counts_by_split': {s: {str(int(v)): sum(f['split'] == s and f['public_speed_m_s'] == v
            for f in families) for v in SPEEDS} for s in SPLIT_COUNTS},
        'native_source_files_archived': True, 'private_choice_key_exported': False,
        'protection_or_attacker_scores_inspected': False, 'sumo_version': version,
        'generation_elapsed_s': time.perf_counter()-start})
    print('Saved fresh native cohort', target, len(traces), 'native trips; no defense score read', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--workdir', type=Path, default=WORK)
    args = parser.parse_args()
    build(args.output, args.workdir)
