"""Public SUMO lane-progress catalogue for dummy motion, independent of trips.

No nearest-junction collapse and no turn invented by merging junction nodes.
This is a free-flow, no-lane-change subset of SUMO motion, not a traffic replay.
"""
import hashlib
import json
import math
from copy import copy

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point

from core.road_network import RoadNetwork
from data.sumo_demo import _load_sumolib


def build_lane_states(network_path, spacing_m=20., max_speed_m_s=8.):
    if not math.isfinite(spacing_m) or spacing_m<=0 or not math.isfinite(max_speed_m_s) or max_speed_m_s<=0:
        raise ValueError('Positive finite public spacing and speed required')
    net=_load_sumolib().net.readNet(str(network_path),withInternal=True)
    lanes={l.getID():l for e in net.getEdges(withInternal=True) for l in e.getLanes() if l.allows('passenger')}
    graph=nx.DiGraph(schema='sumo-lane-progress-v1',spacing_m=spacing_m,max_speed_m_s=max_speed_m_s)
    indices={}
    for lane_id,lane in sorted(lanes.items()):
        shape=LineString(lane.getShape())
        length=lane.getLength()
        if length<=0:
            raise ValueError(f'Degenerate public lane: {lane_id}')
        # Use actual geometry length as well as SUMO's declared longitudinal
        # length, so spatial steps are bounded even if the two differ slightly.
        n=max(1,math.ceil(max(length,shape.length)/spacing_m))
        ids=[]
        for j in range(n+1):
            p=shape.interpolate(j/n,normalized=True) if shape.length>0 else Point(lane.getShape()[0])
            lon,lat=net.convertXY2LonLat(p.x,p.y)
            i=len(graph)
            graph.add_node(i,x=float(lon),y=float(lat),lane_id=lane_id,edge_id=lane.getEdge().getID(),
                           lane_pos_m=length*j/n)
            ids.append(i)
        for a,b in zip(ids,ids[1:]):
            graph.add_edge(a,b,length=max(length,shape.length)/n,
                           speed=min(max_speed_m_s,lane.getSpeed()),kind='lane_progress')
        indices[lane_id]=ids
    # Each connection follows the official via lane (including internal
    # junction continuations), or the target lane when no via exists.
    for lane_id,lane in sorted(lanes.items()):
        for c in lane.getOutgoing():
            if not c.allows('passenger'):
                continue
            target=c.getViaLaneID() or c.getToLane().getID()
            if target not in indices:
                continue
            a,b=indices[lane_id][-1],indices[target][0]
            p,q=lane.getShape()[-1],lanes[target].getShape()[0]
            gap=math.dist(p,q)
            graph.add_edge(a,b,length=gap,speed=min(max_speed_m_s,lane.getSpeed(),lanes[target].getSpeed()),
                           kind='sumo_connection')
    rn=RoadNetwork(graph)
    payload={'spacing_m':spacing_m,'max_speed_m_s':max_speed_m_s,
             'nodes':[(i,d['lane_id'],d['lane_pos_m'],d['y'],d['x']) for i,d in graph.nodes(data=True)],
             'edges':[(a,b,d['length'],d['speed'],d['kind']) for a,b,d in graph.edges(data=True)]}
    rn.catalogue_sha256=hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()
    rn.lane_indices=indices
    return rn


def catalogue_summary(rn):
    return {'schema':rn.graph.graph['schema'],'states':len(rn),'arcs':rn.graph.number_of_edges(),
            'lanes':len(rn.lane_indices),'spacing_m':rn.graph.graph['spacing_m'],
            'max_speed_m_s':rn.graph.graph['max_speed_m_s'],'sha256':rn.catalogue_sha256,
            'lane_changes':False,'traffic_lights_and_congestion':False}


def coordinate_catalogue(rn):
    """Selection-only view: one representative per EXACT public coordinate.

    Different lanes may share endpoints. They must remain distinct motion
    states, but are not distinct dummy locations for a location-set mechanism.
    No geometry rounding, private-data filtering or new output location is used.
    Preserve the full catalogue's fixed projection and first-state tie rule.
    The graph is deliberately absent: route queries must use the full rn.
    """
    seen=set()
    ids=[]
    for i,coordinate in enumerate(zip(rn.lats,rn.lons)):
        if coordinate not in seen:
            seen.add(coordinate)
            ids.append(i)
    sites=copy(rn)
    sites.state_indices=np.asarray(ids,dtype=int)
    for field in ('node_ids','lats','lons','xs','ys','xy'):
        setattr(sites,field,getattr(rn,field)[sites.state_indices])
    sites.tree=cKDTree(sites.xy)
    sites.graph=None
    return sites
