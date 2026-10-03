"""Inspect one unchanged SUMO trip with the active engine and boundary wrapper.

The exact historical lane cache is unavailable. Reconstruct a separate, public,
directed demonstration graph from archived road polylines, never from GPS truth.
This is a new explanatory run, not a rerun or replacement of existing benchmarks.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import sys

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from core.road_network import RoadNetwork, LocalProjection
from core.boundary_release import BoundaryPolicy, BoundaryProtectedStream
from benchmark.anchor_belief import PublicAnchorModel
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from evaluation.lane_travel import LanePoiService
from evaluation.live_poi import (RankedRoadPois, AvailabilityWorld, LivePointService,
                                 EpochResponseCache, score_returned)

DATA = ROOT / 'artifacts/datasets/research_loop_expanded_v1/dataset.json'
ROADS = ROOT / 'artifacts/benchmarks/paper_benchmark/results.json'
POIS = ROOT / 'artifacts/benchmarks/research_loop/resources.json'
SOURCE_PATHS = [DATA, ROADS, POIS] + [ROOT / s for s in (
    'core/mechanisms.py', 'core/boundary_release.py', 'core/road_network.py',
    'benchmark/engines/budgeted.py', 'benchmark/engines/contextual_lane.py',
    'benchmark/engines/filtered_cover.py', 'benchmark/engines/matched_filter.py',
    'benchmark/engines/paced_guard.py', 'benchmark/engines/paced_slack.py',
    'benchmark/engines/fair_cover.py', 'benchmark/engines/quotient_cover.py',
    'benchmark/engines/progress_cover.py', 'benchmark/engines/slack_progress.py',
    'benchmark/anchor_belief.py', 'benchmark/response_aware_belief.py',
    'benchmark/public_poi_context.py', 'evaluation/lane_travel.py',
    'evaluation/live_poi.py')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InspectRandom:
    """Observe Laplace draws without changing RNG consumption or values."""
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.laplace_draws = []

    def laplace(self, *args, **kwargs):
        value = self.rng.laplace(*args, **kwargs)
        self.laplace_draws.append(float(value))
        return value

    def __getattr__(self, name):
        return getattr(self.rng, name)


def public_graph(roads):
    """Build a separate SUMO-format demonstration network from public geometry."""
    import xml.etree.ElementTree as ET
    from pyproj import Transformer
    from data.lane_states import build_lane_states
    bbox = (116.29, 39.98, 116.315, 40.0)
    lines = [r for r in roads if all(bbox[0] <= x <= bbox[2] and
             bbox[1] <= y <= bbox[3] for x, y in r)]
    project = Transformer.from_crs(4326, 32650, always_xy=True)
    shapes = [np.array([project.transform(x,y) for x,y in line]) for line in lines]
    endpoints = np.array([p for shape in shapes for p in (shape[0],shape[-1])])
    parents = np.arange(len(endpoints))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return int(i)
    for i,j in sorted(cKDTree(endpoints).query_pairs(35.)):
        a,b = root(i),root(j)
        if a != b: parents[max(a,b)] = min(a,b)
    labels = [root(i) for i in range(len(endpoints))]
    centers = {i: endpoints[np.array(labels)==i].mean(axis=0) for i in sorted(set(labels))}
    # Endpoint clustering can collapse a short polyline into one junction.
    # Such self-loop edges are invalid in SUMO; remove by a public geometry rule.
    keep = [j for j in range(len(lines)) if labels[2*j] != labels[2*j+1]]
    labels = [labels[2*j+k] for j in keep for k in (0,1)]
    shapes = [shapes[j] for j in keep]; lines = [lines[j] for j in keep]
    incoming, outgoing = {i:[] for i in centers}, {i:[] for i in centers}
    for j in range(len(lines)):
        outgoing[labels[2*j]].append(j); incoming[labels[2*j+1]].append(j)
    net = ET.Element('net', version='1.20', junctionCornerDetail='0')
    bounds = [*endpoints.min(axis=0),*endpoints.max(axis=0)]
    ET.SubElement(net,'location',netOffset='0.00,0.00',
        convBoundary=','.join(map(str,bounds)),origBoundary=','.join(map(str,bbox)),
        projParameter='+proj=utm +zone=50 +datum=WGS84 +units=m +no_defs')
    for j,shape in enumerate(shapes):
        edge = ET.SubElement(net,'edge',id=f'demo_r{j}',
            **{'from':f'n{labels[2*j]}','to':f'n{labels[2*j+1]}','priority':'1'})
        length = float(np.linalg.norm(np.diff(shape,axis=0),axis=1).sum())
        ET.SubElement(edge,'lane',id=f'demo_r{j}_0',index='0',speed='8.0',
            length=str(max(length,.1)),shape=' '.join(f'{x},{y}' for x,y in shape),allow='passenger')
    for i,(x,y) in centers.items():
        ET.SubElement(net,'junction',id=f'n{i}',type='priority',x=str(x),y=str(y),
            incLanes=' '.join(f'demo_r{j}_0' for j in incoming[i]),intLanes='',
            shape=f'{x-1},{y-1} {x+1},{y-1} {x+1},{y+1} {x-1},{y+1}')
    for i in centers:
        for a in incoming[i]:
            for b in outgoing[i]:
                if a != b:
                    ET.SubElement(net,'connection',**{'from':f'demo_r{a}','to':f'demo_r{b}',
                        'fromLane':'0','toLane':'0','dir':'s','state':'M'})
    path = OUT/'demo.net.xml'
    ET.ElementTree(net).write(path,encoding='utf-8',xml_declaration=True)
    rn = build_lane_states(path)
    return rn, lines, {'bbox':bbox,'spacing_m':20.,'endpoint_cluster_radius_m':35.,
        'speed_cap_m_s':8.,'states':len(rn),'arcs':rn.graph.number_of_edges(),
        'net_sha256':sha(path),'catalogue_sha256':rn.catalogue_sha256,
        'limitations':'Separate SUMO-format public demonstration network from archived polylines; '
            'junction connectivity and turns are reconstructed assumptions. '
            'Original SUMO lane and turn restrictions are NOT recovered or verified.'}


def run(rn, belief, ranking, trace, head, delay):
    model = PacedSlackProgressLaneDummy(rn, belief_model=belief, k=5, budget=.24,
                 horizon=12, theta_m=200., read_interval_s=60., utility_slack=.03,
                 rng=np.random.default_rng(20261003))
    random = InspectRandom(20261003)
    model.anchor_rng, model.dummy_rng = random, np.random.default_rng(20261004)
    stream = BoundaryProtectedStream(model, BoundaryPolicy(head, delay), session_start_s=0.)
    world = AvailabilityWorld(ranking.n, seed=24092801, probability=.8, epoch_seconds=60)
    server, cache = LivePointService(ranking, world, response_l=10), EpochResponseCache(ranking.n)
    rows, publications, fifo, public_events = [], [], [], []
    indices = sorted(set(range(0, len(trace), 20)) | {len(trace)-1})
    start = trace[0]['time_s']
    for i in indices:
        point = trace[i]; t = point['time_s']-start
        old_anchor = model.last_anchor
        old_states, old_t = copy.copy(model.previous), model.last_t
        old_read = model.last_private_read_s
        draw_count = len(random.laplace_draws)
        before = stream.generated
        released = stream.ingest(t, point['lat'], point['lon'])
        row = {'index': i, 't': t, 'gps': [point['lat'], point['lon']],
               'speed_m_s': point['speed_m_s'], 'head_skipped': stream.generated == before,
               'released_source_times': [], 'spent': model.spent_units*.01,
               'remaining': .23-model.spent_units*.01}
        if not row['head_skipped']:
            fifo.append(len(rows))
            ledger = model.evaluator_ledger[-1]
            private_test = ledger['private_read'] and old_anchor is not None
            distance = (float(np.linalg.norm(np.array(rn.point_xy(point['lat'], point['lon']))-
                         np.array(rn.point_xy(*old_anchor)))) if private_test else None)
            eta = random.laplace_draws[-1] if len(random.laplace_draws)>draw_count else None
            row.update(branch=ledger['branch'], private_read=ledger['private_read'],
                cost=ledger['cost_units']*.01, old_anchor=old_anchor,
                distance_m=distance, test_noise_m=eta,
                noisy_distance_m=distance+eta if private_test else None,
                anchor=model.last_anchor,
                anchor_error_m=float(np.linalg.norm(np.array(rn.point_xy(*model.last_anchor))-
                    np.array(rn.point_xy(point['lat'], point['lon'])))),
                since_last_read_s=None if old_read is None else t-old_read,
                states=list(map(int, model.previous)),
                queries=[list(rn.latlon(v)) for v in model.previous],
                objective=copy.deepcopy(model.evaluator_objective[-1]),
                region_weights=model.belief.weights.tolist(),
                region_mean_xy=model.belief.mean_xy().tolist(),
                reachable_counts=[] if old_states is None else
                    [sum(model.viable[v] for v in model.travel.reachable(s,t-old_t)) for s in old_states],
                previous_states=old_states, previous_query_t=old_t,
                posterior_updated=bool(ledger['private_read']))
            assert math.isclose(sum(row['region_weights']), 1., abs_tol=1e-10)
            if private_test:
                assert (row['noisy_distance_m'] <= 200.) == (ledger['branch']=='reuse')
            for slot, s in enumerate(row['states']):
                if old_states is not None:
                    assert s in model.travel.reachable(old_states[slot], t-old_t)
            o = row['objective']
            if 'objective_loss' in o:
                assert o['objective_loss'] <= .03+1e-10
        replies = []
        epoch = world.epoch(t)
        for event in released:
            public_events.append(event.to_dict())
            source_index = fifo.pop(0)
            source_t = (row if source_index==len(rows) else rows[source_index])['t']
            src = row if source_index==len(rows) else rows[source_index]
            src.update(publication_t=t, publication_delay_s=t-source_t)
            row['released_source_times'].append(source_t)
            assert t-source_t >= delay
            candidates = [c.to_dict() for c in event.candidates]
            per_point = [server.query(rn.nearest(c['lat'],c['lon'])[0], epoch) for c in candidates]
            replies.extend(per_point)
            publications.append({'source_t': source_t, 'publication_t': t,
                'coordinates': [[c['lat'],c['lon']] for c in candidates], 'replies': per_point})
        current, known = cache.receive(epoch, replies)
        truth_state = rn.nearest(point['lat'],point['lon'])[0]
        available = world.at_epoch(epoch)
        reference = ranking.top(truth_state, available, k=5)
        returned = ranking.top(truth_state, known, k=5)
        score = score_returned(reference, returned, available)
        from evaluation.lane_travel import matrix
        road_dist = dijkstra(matrix(rn), directed=True, indices=truth_state)
        row['service'] = {'epoch': epoch, 'replies': replies, 'fresh_union_count': int(current.sum()),
            'cached_union_count': int(known.sum()), 'reference': reference,
            'returned': returned, **score,
            'returned_distances_m': [[float(road_dist[ranking.pois[v]['vertex']]) for v in cat]
                                     for cat in returned]}
        rows.append(row)
    for idx in fifo:
        rows[idx]['tail_cancelled'] = True
    stream.close(trace[-1]['time_s']-start)
    account = stream.evaluator_summary()
    assert account['protected_events'] == account['released_events']+account['tail_cancelled']
    assert account['tail_cancelled'] == len(fifo)
    assert model.spent_units <= 23
    return {'policy': {'head_s': head, 'delay_s': delay}, 'rows': rows,
            'publications': publications, 'accounting': account, 'budget_spent': model.spent_units*.01,
            'public_transcript':{'events':public_events},
            'private_reads': sum(bool(r.get('private_read')) for r in rows),
            'reuse_count': sum(r.get('branch')=='reuse' for r in rows),
            'recall_all_input_times': float(np.mean([r['service']['recall'] for r in rows
                                                    if r['service']['recall'] is not None])),
            'recall_delivery_times': float(np.mean([r['service']['recall'] for r in rows
                   if r['released_source_times'] and r['service']['recall'] is not None]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', type=Path, default=Path('/private/tmp/trajectory-walkthrough'))
    args = parser.parse_args(); args.workdir.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in SOURCE_PATHS}
    data = json.loads(DATA.read_text()); roads = json.loads(ROADS.read_text())['roads']
    sid = sorted(data['traces'])[0]  # Freeze selection before scores; do not choose the nicest run.
    trace = data['traces'][sid]
    rn, lines, graph_metadata = public_graph(roads)
    args.workdir = args.workdir / rn.catalogue_sha256[:12]
    args.workdir.mkdir(parents=True,exist_ok=True)
    print('Public demo graph', graph_metadata, flush=True)
    pois = [{k: v for k, v in p.items() if k not in ('vertex','access_offset_m')}
            for p in json.loads(POIS.read_text())['pois_used']]
    reference_service = LanePoiService(rn, pois, k=5)
    reference = PublicPoiContext(reference_service, args.workdir/'demo-poi5.npz')
    reply = PublicPoiContext(LanePoiService(rn, pois, k=10), args.workdir/'demo-poi10.npz')
    _, inv, counts = np.unique(np.floor(rn.xy/120.).astype(np.int64),axis=0,
                               return_inverse=True,return_counts=True)
    prior = 1./counts[inv]; prior /= prior.sum()
    base = PublicAnchorModel(rn,reference,prior,cache_path=args.workdir/'demo-belief.npz')
    belief = ResponseAwareAnchorModel(base, reply)
    ranking = RankedRoadPois(reference_service,args.workdir/'demo-ranking.npy')
    print('POIs',ranking.n,'latent cells',len(base.xy),flush=True)
    runs = {}
    for name,head,delay in [('plain',0.,0.),('boundary',60.,60.)]:
        runs[name]=run(rn,belief,ranking,trace,head,delay)
        print(name,runs[name]['accounting'],'budget',runs[name]['budget_spent'],
              'recall',runs[name]['recall_all_input_times'],flush=True)
    for p in SOURCE_PATHS:
        assert sha(p)==hashes[str(p.relative_to(ROOT))], p
    points=[{'t':p['time_s']-trace[0]['time_s'],'lat':p['lat'],'lon':p['lon'],
             'speed_m_s':p['speed_m_s']} for p in trace]
    payload={'schema':'algorithm-walkthrough-v1','session_id':sid,
        'selection':'Lexically first session ID, rep/anchor seed 20261003, dummy seed 20261004; fixed before scores',
        'scope':'New explanatory integration run on unchanged SUMO GPS and a separate reconstructed public graph. '
            'Not a rerun of frozen benchmarks, not a privacy-attack evaluation or superiority claim.',
        'dataset_source':data['source'],'source_hashes':hashes,
        'builder_sha256':sha(Path(__file__)),
        'parameters':{'nominal_B_per_m':.24,'reference_H':12,'epsilon_test_per_m':.01,
            'epsilon_release_per_m':.01,'effective_cap_per_m':.23,'theta_m':200.,
            'private_read_interval_s':60.,'public_query_interval_s':20.,'K':5,'L':10,
            'slack':.03,'response_epoch_s':60.,'availability_p':.8,'world_seed':24092801},
        'graph':graph_metadata,'roads_lonlat':lines,'poi_count':ranking.n,
        'pois':ranking.pois,'categories':ranking.categories,
        'latent_latlon':[list(rn.latlon(s)) for s in base.state_ids],
        'projection_lat0':rn.proj.lat0,'true_trace':points,'duration_s':points[-1]['t'],
        'runs':runs,'validation':{'source_hashes_unchanged':True,'budget_and_noise_checks':True,
            'normalized_belief':True,'directed_reachability_in_demo_graph':True,
            'slack_objective_loss_at_most_03':True,'boundary_conservation':True,
            'delayed_current_service_evaluated':True,'original_turn_rules_verified':False,
            'privacy_attacker_evaluated':False}}
    def numpy_value(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        raise TypeError(type(v).__name__)
    (OUT/'walkthrough.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,
                                                   default=numpy_value)+'\n')
    print('Saved',OUT/'walkthrough.json',flush=True)


if __name__=='__main__':
    main()
