"""Separate reconstructed public map for development when exact cache is absent.

All archived polylines are used, before consulting any private GPS or score.
Endpoint clustering/turns and speed are assumptions, not recovery of SUMO's
original lane restrictions. Never use these resources to overwrite frozen runs.
"""
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
from pyproj import Transformer
from scipy.spatial import cKDTree

from benchmark.anchor_belief import PublicAnchorModel
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from data.lane_states import build_lane_states, catalogue_summary
from evaluation.lane_travel import LanePoiService
from evaluation.live_poi import RankedRoadPois

ROOT = Path(__file__).resolve().parents[1]
ROADS = ROOT/'artifacts/benchmarks/paper_benchmark/results.json'
POIS = ROOT/'artifacts/benchmarks/research_loop/resources.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct_public_network(path):
    lines = json.loads(ROADS.read_text())['roads']
    transform = Transformer.from_crs(4326, 32650, always_xy=True)
    shapes = [np.asarray([transform.transform(x, y) for x, y in line]) for line in lines]
    endpoints = np.asarray([p for shape in shapes for p in (shape[0], shape[-1])])
    parents = np.arange(len(endpoints))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return int(i)

    for a, b in sorted(cKDTree(endpoints).query_pairs(35.)):
        x, y = find(a), find(b)
        if x != y:
            parents[max(x, y)] = min(x, y)
    labels = np.asarray([find(i) for i in range(len(endpoints))])
    centers = {int(i): endpoints[labels == i].mean(axis=0) for i in sorted(set(labels))}
    # Public geometry rule, required for valid SUMO network; no score filtering.
    keep = [j for j in range(len(shapes)) if labels[2*j] != labels[2*j+1]
            and np.linalg.norm(np.diff(shapes[j], axis=0), axis=1).sum() > .1]
    incoming = {i: [] for i in centers}
    outgoing = {i: [] for i in centers}
    for j in keep:
        outgoing[int(labels[2*j])].append(j)
        incoming[int(labels[2*j+1])].append(j)
    net = ET.Element('net', version='1.20', junctionCornerDetail='0')
    lonlat = np.concatenate([np.asarray(line) for line in lines])
    ET.SubElement(net, 'location', netOffset='0.00,0.00',
        convBoundary=','.join(map(str, [*endpoints.min(axis=0), *endpoints.max(axis=0)])),
        origBoundary=','.join(map(str, [*lonlat.min(axis=0), *lonlat.max(axis=0)])),
        projParameter='+proj=utm +zone=50 +datum=WGS84 +units=m +no_defs')
    for j in keep:
        shape = shapes[j]
        edge = ET.SubElement(net, 'edge', id=f'public_r{j}', priority='1',
            **{'from': f'n{labels[2*j]}', 'to': f'n{labels[2*j+1]}'})
        ET.SubElement(edge, 'lane', id=f'public_r{j}_0', index='0', speed='8.0',
            length=str(float(np.linalg.norm(np.diff(shape, axis=0), axis=1).sum())),
            shape=' '.join(f'{x},{y}' for x, y in shape), allow='passenger')
    for i, (x, y) in centers.items():
        ET.SubElement(net, 'junction', id=f'n{i}', type='priority', x=str(x), y=str(y),
            incLanes=' '.join(f'public_r{j}_0' for j in incoming[i]), intLanes='',
            shape=f'{x-1},{y-1} {x+1},{y-1} {x+1},{y+1} {x-1},{y+1}')
        for a in incoming[i]:
            for b in outgoing[i]:
                if a != b:
                    ET.SubElement(net, 'connection', **{'from': f'public_r{a}',
                        'to': f'public_r{b}', 'fromLane': '0', 'toLane': '0',
                        'dir': 's', 'state': 'M'})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(net).write(path, encoding='utf-8', xml_declaration=True)
    return len(lines), len(keep)


def load_public_research_resources(cache_dir, *, spacing_m=40., latent_spacing_m=200., epsilon=.01):
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    net, provenance_path = cache/'public_reconstructed.net.xml', cache/'public_resources.json'
    source = {str(p.relative_to(ROOT)): sha(p) for p in (ROADS, POIS, Path(__file__))}
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        if source != provenance['source_sha256'] or sha(net) != provenance['net_sha256']:
            raise ValueError('Stale public reconstructed resources; choose fresh cache')
    else:
        original_lines, retained_lines = reconstruct_public_network(net)
        provenance = {'schema': 'public-research-reconstruction-v1', 'source_sha256': source,
            'net_sha256': sha(net), 'archived_polylines': original_lines,
            'retained_non_self_loop_polylines': retained_lines, 'endpoint_cluster_m': 35.,
            'speed_m_s': 8., 'original_turns_verified': False,
            'scope': 'Full archived public polylines, reconstructed junctions/turns; development only; '
                     'not an exact rerun of original SUMO network or frozen benchmark'}
        provenance_path.write_text(json.dumps(provenance, indent=2)+'\n')
    rn = build_lane_states(net, spacing_m=spacing_m)
    state_cache = cache/rn.catalogue_sha256[:16]
    state_cache.mkdir(exist_ok=True)
    pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')}
            for p in json.loads(POIS.read_text())['pois_used']]
    service = LanePoiService(rn, pois, k=5)
    reference = PublicPoiContext(service, state_cache/'poi5.npz')
    reply = PublicPoiContext(LanePoiService(rn, pois, k=10), state_cache/'poi10.npz')
    _, inv, counts = np.unique(np.floor(rn.xy/120.).astype(np.int64),
                               axis=0, return_inverse=True, return_counts=True)
    prior = 1./counts[inv]
    prior /= prior.sum()
    base = PublicAnchorModel(rn, reference, prior, spacing_m=latent_spacing_m,
        epsilon_release=epsilon, epsilon_test=epsilon,
        cache_path=state_cache/f'belief-{latent_spacing_m:g}-{epsilon:g}.npz')
    belief = ResponseAwareAnchorModel(base, reply)
    ranking = RankedRoadPois(service, state_cache/'ranking.npy')
    metadata = dict(provenance, catalogue=catalogue_summary(rn), poi_count=ranking.n,
                    latent_cells=len(base.xy), belief_sha256=belief.sha256,
                    reference_sha256=reference.sha256, reply_sha256=reply.sha256)
    return rn, service, reference, reply, belief, ranking, metadata
