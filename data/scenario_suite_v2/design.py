"""Public map/POI-driven demand, fixed before observing protection results."""
from collections import Counter
from copy import deepcopy
import math
import random

from data.scenario_suite.mobility import plan as base_plan, successors


def rarity_context(pois):
    counts = Counter(p['category'] for p in pois)
    return {'definition':'OSM_category_share_not_visit_frequency',
            'max_category_share':.05, 'max_observed_distance_m':150.,
            'category_counts':dict(sorted(counts.items())), 'pois':pois,
            'rare_ids':[p['id'] for p in pois if counts[p['category']]/len(pois) <= .05]}


def plan(network, seed, context):
    # Public geometric constraints only; no simulation/privacy outcome search.
    for attempt in range(64):
        proposal = base_plan(network, seed + 10000*attempt)
        base = proposal['sessions'][0]['route_edges']
        if len(successors(network.getEdge(base[0]))) == 1 and network.getEdge(base[-1]).getLength() <= 400:
            break
    else:
        raise ValueError('No public route meeting v2 origin/destination constraints')
    result = deepcopy(proposal)
    result.update(seed=seed, family_id=f'family-{seed}', route_plan_seed=proposal['seed'])
    for i, s in enumerate(result['sessions']):
        s['session_id'] = f'u{seed}_{i:02d}'
        for key in ('person_id','device_id','physical_vehicle_id'):
            s[key] = f'{seed}/' + s[key].split('/')[1]
        s['day_index'] = 0
    roles = {s['role']:s for s in result['sessions']}
    # Extend the alternative on legal roads, with a public endpoint separation
    # constraint. This creates an observable partial-companionship challenge.
    rng = random.Random(seed)
    alt = roles['partial']['route_edges']
    end_xy = network.getEdge(base[-1]).getShape()[-1]
    endpoints = sorted((e for e in network.getEdges() if not e.getID().startswith(':') and e.allows('passenger') and
                        math.dist(e.getShape()[-1],end_xy) >= 700), key=lambda e:e.getID())
    rng.shuffle(endpoints)
    for target in endpoints[:300]:
        tail, length = network.getShortestPath(network.getEdge(alt[-1]), target, vClass='passenger')
        if tail and 300 <= length <= 1800:
            alt = alt + [e.getID() for e in tail[1:]]
            break
    else:
        raise ValueError('No separated public alternative continuation')
    roles['partial']['route_edges'] = list(alt)
    roles['incidental']['route_edges'] = list(alt)

    def add(role, route, depart, person=None, day=0):
        index = len(result['sessions'])
        identity = person or f'extra{index}'
        s = {'session_id':f'u{seed}_{index:02d}', 'role':role, 'family':role,
             'person_id':f'{seed}/{identity}', 'device_id':f'{seed}/{identity}-device',
             'physical_vehicle_id':f'{seed}/{identity}-car', 'depart_s':depart,
             'route_edges':list(route), 'stops':[], 'day_index':day}
        result['sessions'].append(s)
        return s

    add('near_destination',base[:-1],180)
    rare = [p for p in context['pois'] if p['id'] in context['rare_ids']]
    rng.shuffle(rare)
    for poi in rare:
        x,y = network.convertLonLat2XY(poi['lon'],poi['lat'])
        nearby = sorted(network.getNeighboringEdges(x,y,120),key=lambda item:(item[1],item[0].getID()))
        for edge, access in nearby:
            if not edge.allows('passenger') or edge.getID().startswith(':'):
                continue
            prefix, _ = network.getShortestPath(network.getEdge(base[0]),edge,vClass='passenger')
            suffix, _ = network.getShortestPath(edge,network.getEdge(base[-1]),vClass='passenger')
            if prefix and suffix:
                visit = add('rare_poi_visit',[e.getID() for e in prefix]+[e.getID() for e in suffix[1:]],210)
                visit.update(planned_poi_id=poi['id'], public_access_distance_m=access)
                break
        else:
            continue
        break
    else:
        raise ValueError('No public rare POI with a connected passenger route')

    pattern = ['routine','routine','rare','routine','routine','routine','routine','rare']
    for day, destination in enumerate(pattern,1):
        s = add(f'activity_day_{day}',base if destination=='routine' else alt,
                day*86400+8*3600,'activity-person',day)
        s['planned_destination_class'] = destination
    result['activity_schedule'] = {'history_days':list(range(1,7)), 'query_days':[7,8],
                                   'pattern':pattern, 'source':'synthetic_diagnostic_schedule'}
    return result
