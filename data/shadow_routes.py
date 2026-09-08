"""Public-map auxiliary route design; no benchmark targets are consulted."""
from collections import Counter, defaultdict
import math
import random

from data.scenario_suite.mobility import successors
from data.sumo_demo import _load_sumolib

PROFILES = ('cruise20', 'cruise60', 'stop5', 'return5')


def plans(network, seeds=range(1001, 1081)):
    edges = sorted((e for e in network.getEdges(withInternal=False)
                    if e.allows('passenger') and e.getLength() >= 40), key=lambda e: e.getID())
    cells = defaultdict(list)
    for edge in edges:
        lane = next(l for l in edge.getLanes() if l.allows('passenger'))
        geometry = _load_sumolib().geomhelper
        xy = geometry.positionAtShapeOffset(lane.getShape(), geometry.polyLength(lane.getShape()) / 2)
        cells[tuple(math.floor(v / 120.) for v in xy)].append(edge)
    used_cells, used_routes, result = set(), set(), []
    used_stop_lanes, used_stop_cells = set(), set()
    for seed in seeds:
        rng = random.Random(seed)
        rejected = Counter()
        for _ in range(3000):
            available = sorted(cells.keys() - used_cells)
            if not available:
                raise ValueError('Not enough distinct public departure cells')
            cell = rng.choice(available)
            origin, destination = rng.choice(cells[cell]), rng.choice(edges)
            path, length = network.getShortestPath(origin, destination, vClass='passenger')
            if not path or not 2500 <= length <= 4000 or len(path) < 12:
                rejected['length_or_connectivity'] += 1
                continue
            route = tuple(e.getID() for e in path)
            if route in used_routes:
                rejected['duplicate_route'] += 1
                continue
            candidates = [i for i in range(2, len(path) // 2) if path[i].getLength() >= 60]
            if not candidates:
                rejected['no_stop_lane'] += 1
                continue
            stop_i = rng.choice(candidates)
            back, _ = network.getShortestPath(path[-1], path[stop_i], vClass='passenger')
            if not back or len(back) < 3:
                rejected['no_return'] += 1
                continue
            loop = list(path) + list(back[1:]) + list(path[stop_i + 1:])
            if any(b not in successors(a) for a, b in zip(loop, loop[1:])):
                rejected['passenger_connection'] += 1
                continue
            lane = next(l for l in path[stop_i].getLanes() if l.allows('passenger'))
            turn = next(l for l in path[-1].getLanes() if l.allows('passenger'))
            stop = {'lane': lane.getID(), 'endPos': round(lane.getLength() * .7, 2), 'duration': 180}
            stop_xy = geometry.positionAtShapeOffset(lane.getShape(), stop['endPos'])
            stop_cell = tuple(math.floor(v) for v in stop_xy)
            if lane.getID() in used_stop_lanes or stop_cell in used_stop_cells:
                rejected['duplicate_public_stop'] += 1
                continue
            waypoint = {'lane': turn.getID(), 'endPos': round(turn.getLength() * .7, 2), 'duration': 1}
            sessions = []
            for role, r, depart, stops in (('cruise', path, 0, []), ('return', loop, 2000, [stop, waypoint, stop.copy()])):
                sid = f'shadow-{seed}-{role}'
                sessions.append({'session_id': sid, 'role': role, 'depart_s': depart,
                    'person_id': sid + '/person', 'device_id': sid + '/device',
                    'physical_vehicle_id': sid + '/car', 'route_edges': [e.getID() for e in r], 'stops': stops})
            result.append({'seed': seed, 'family_id': f'shadow-family-{seed}',
                'split': 'development_train' if seed <= 1064 else 'development_validation',
                'role': 'auxiliary_training' if seed <= 1064 else 'auxiliary_holdout',
                'departure_cell': list(cell), 'base_route_length_m': length,
                'planning_rejections': dict(rejected), 'sessions': sessions})
            used_cells.add(cell)
            used_routes.add(route)
            used_stop_lanes.add(lane.getID())
            used_stop_cells.add(stop_cell)
            break
        else:
            raise ValueError(f'No feasible public route for seed {seed}')
    return result


def stationary_runs(trace, lane):
    result, run = [], []
    for i, p in enumerate(trace):
        if p['lane_id'] == lane and p['speed_m_s'] <= .05:
            run.append(i)
        else:
            if len(run) >= 121:
                result.append(run)
            run = []
    if len(run) >= 121:
        result.append(run)
    return result


def records(family, traces):
    cruise, returned = family['sessions']
    t = traces[cruise['session_id']]
    start = min(len(t) // 4, max(0, len(t) - 221))
    indices = {'cruise20': list(range(start, min(len(t), start + 221), 20)),
               'cruise60': list(range(0, min(len(t), 661), 60))}
    stop_trace = traces[returned['session_id']]
    stops = stationary_runs(stop_trace, returned['stops'][0]['lane'])
    if len(stops) != 2 or stop_trace[stops[1][10]]['time_s'] - stop_trace[stops[0][35]]['time_s'] <= 120:
        raise ValueError('Native SUMO return does not realize two separated long stops')
    indices.update(stop5=stops[0][10:66:5], return5=stops[0][10:36:5] + stops[1][10:36:5])
    result = []
    for profile in PROFILES:
        sid = cruise['session_id'] if profile.startswith('cruise') else returned['session_id']
        idx = indices[profile]
        if len(idx) < 2 or len(idx) > 12:
            raise ValueError('Auxiliary query schedule outside declared horizon')
        result.append({'record_id': f"shadow-{family['seed']}-{profile}",
            'case_id': 'AUX.' + profile, 'scenario': 'AUX', 'profile': profile,
            'family_id': family['family_id'], 'split': family['split'],
            'session_ids': [sid], 'observed_indices': [idx], 'labels': {},
            'observation_policy': {'clock': 'relative_to_first_allowed_sample_per_session',
                'coordinates': 'device_only_until_protected', 'route_plan': 'evaluator_only',
                'full_trip_duration': 'withheld'},
            'evidence': {'purpose': family['role'], 'not_a_core_scenario_gate': True}})
    return result


def summarize(bundle):
    counts = Counter(r['case_id'] for r in bundle['records'])
    return {'families': len(bundle['families']), 'sessions': len(bundle['traces']),
        'raw_fcd_samples': sum(map(len, bundle['traces'].values())),
        'scenario_records': len(bundle['records']), 'generated_subcases': len(counts),
        'declared_subcases': len(PROFILES), 'generated_scenarios': 1,
        'by_case': dict(counts), 'by_scenario': {'AUX': len(bundle['records'])}}
