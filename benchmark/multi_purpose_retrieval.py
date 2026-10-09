"""Opt-in fixed public retrieval templates; no private QuerySpec enters the wire.

The detour template covers a fixed PUBLIC destination bank by round-robin
rank, not the user's destination. Radius uses a fixed public radius around Q.
These templates diversify candidates, but do not guarantee local exactness.
Geo-I anchors, reads and Q generation are outside this service layer.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse.csgraph import dijkstra

from benchmark.query_purpose import PurposeIndependentCoverClient
from evaluation.lane_travel import matrix


TYPES = ('nearest_distance', 'fastest_travel', 'within_radius', 'public_detour_bank')


@dataclass(frozen=True)
class PublicRetrievalPlan:
    channels: tuple[tuple[str, int], ...]
    radius_m: float = 1000.
    destination_states: tuple[int, ...] = ()

    def __post_init__(self):
        if (not isinstance(self.channels, tuple) or not self.channels
                or any(not isinstance(c, tuple) or len(c) != 2 for c in self.channels)):
            raise ValueError('Immutable nonempty public channel tuple required')
        names = [c[0] for c in self.channels]
        if len(set(names)) != len(names) or any(n not in TYPES for n in names):
            raise ValueError('Distinct supported public templates required')
        if any(isinstance(l, bool) or not isinstance(l, int) or l < 1 for _, l in self.channels):
            raise ValueError('Positive integer response depth required')
        if not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError('Fixed positive public radius required')
        if (not isinstance(self.destination_states, tuple)
                or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.destination_states)
                or len(set(self.destination_states)) != len(self.destination_states)):
            raise ValueError('Distinct immutable public destination states required')
        if 'public_detour_bank' in names and not self.destination_states:
            raise ValueError('Detour cover needs a public destination bank')

    def requests(self, timestamp_s, coordinate, categories, epoch):
        result=[]
        for purpose,depth in self.channels:
            r=dict(schema='fixed_public_purpose_cover_v1', timestamp_s=timestamp_s,
                coordinate=list(coordinate),categories=list(categories),epoch=epoch,
                retrieval_type=purpose,response_l=depth)
            if purpose=='within_radius':r['public_radius_m']=self.radius_m
            if purpose=='public_detour_bank':r['public_destination_states']=list(self.destination_states)
            result.append(r)
        return result


class FixedMultiPurposeCoverClient(PurposeIndependentCoverClient):
    """Compatible with GeoILbsClient; private demand is only used after step."""
    def __init__(self, categories, poi_count, plan, **kwargs):
        if not isinstance(plan,PublicRetrievalPlan):raise ValueError('Public retrieval plan required')
        self.plan=plan
        super().__init__(categories,poi_count,response_l=sum(l for _,l in plan.channels),**kwargs)

    def step(self,timestamp_s,protected_coordinates,server):
        coordinates=tuple(tuple(c) for c in protected_coordinates)
        # Reuse the fixed client's coordinate/clock validation and TTL handling.
        expanded=[];returned=[]
        def expand(base):
            requests=self.plan.requests(base['timestamp_s'],base['coordinate'],self.categories,base['epoch'])
            replies=[server(r) for r in requests]
            expanded.extend(requests);returned.extend(replies)
            # Existing cache consumes list-of-category lists of POI indices.
            return [list(row) for reply in replies for row in reply]
        result=super().step(timestamp_s,coordinates,expand)
        result.update(requests=expanded,replies=returned)
        return result


def round_robin_unique(rankings,depth):
    """Stable equal-rank interleave of public prototypes, capped after dedup."""
    result=[];seen=set()
    for rank in range(max(map(len,rankings),default=0)):
        for row in rankings:
            if rank<len(row) and row[rank] not in seen:
                seen.add(row[rank]);result.append(int(row[rank]))
                if len(result)==depth:return result
    return result


class PublicPurposePoiService:
    """Exact static graph scores, fixed public inputs and lexical ID ties.

    Matrices are built by reverse shortest paths from POIs. They contain no
    GPS trace, session endpoint or private intent. The state access mapping
    matches PublicPoiContext, including its lane-direction approximation.
    """
    def __init__(self,rn,context,distance_to_pois,time_to_pois,*,radius_m,destination_states):
        self.rn,self.context=rn,context
        self.pois,self.categories=context.pois,context.categories
        self.distance=np.asarray(distance_to_pois);self.travel_time=np.asarray(time_to_pois)
        shape=(len(self.pois),len(rn))
        if self.distance.shape!=shape or self.travel_time.shape!=shape:
            raise ValueError('Public POI-to-state reverse cost matrices required')
        PublicRetrievalPlan((('public_detour_bank',1),),radius_m,tuple(destination_states))
        if any(s>=len(rn) for s in destination_states):raise ValueError('Public destination outside graph')
        self.radius_m=radius_m;self.destination_states=tuple(destination_states)
        self.to_dest=dijkstra(matrix(rn).transpose().tocsr(),directed=True,indices=self.destination_states)
        self.vertices=np.array([p['vertex'] for p in self.pois],int)
        self.distance_graph=matrix(rn)
        self.detour_forward_cache={}
        self.category_ids=[np.array([i for i,p in enumerate(self.pois) if p['category']==cat],int)
                           for cat in self.categories]
        self.cache={}

    def query(self,state,purpose,depth):
        if purpose not in TYPES or not isinstance(depth,int) or isinstance(depth,bool) or depth<1:
            raise ValueError('Supported template and positive depth required')
        if not 0<=state<len(self.rn):raise ValueError('Valid public access state required')
        key=int(state),purpose,depth
        if key in self.cache:return self.cache[key]
        access=int(self.context.access[state]);dist=self.distance[:,access]
        if purpose=='public_detour_bank':
            # Preserve the local reference implementation's forward distance
            # arithmetic, including near-zero detour ties after cancellation.
            # Only retain POI and PUBLIC prototype costs; no trajectory input.
            if access not in self.detour_forward_cache:
                forward=dijkstra(self.distance_graph,directed=True,indices=access)
                self.detour_forward_cache[access]=(forward[self.vertices],forward[list(self.destination_states)])
            dist,directs=self.detour_forward_cache[access]
        scores=self.travel_time[:,access] if purpose=='fastest_travel' else dist
        result=[]
        for category,ids in enumerate(self.category_ids):
            if purpose=='nearest_distance':
                # Retain the existing service's exact nearest prefixes/ties.
                row=self.context.query_indices(state)[category,:depth]
                result.append([int(v) for v in row if v>=0]);continue
            if purpose=='public_detour_bank':
                rows=[]
                for j,target in enumerate(self.to_dest):
                    if not np.isfinite(directs[j]):rows.append([]);continue
                    values=dist[ids]+target[self.vertices[ids]]-directs[j]
                    valid=np.flatnonzero(np.isfinite(values))
                    order=valid[np.argsort(np.maximum(values[valid],0.),kind='stable')]
                    rows.append(ids[order[:depth]].tolist())
                result.append(round_robin_unique(rows,depth));continue
            valid=np.flatnonzero(np.isfinite(scores[ids]))
            if purpose=='within_radius':valid=valid[scores[ids[valid]]<=self.radius_m]
            order=valid[np.argsort(scores[ids[valid]],kind='stable')]
            result.append(ids[order[:depth]].tolist())
        self.cache[key]=result
        return result

    def serve(self,request):
        purpose=request['retrieval_type']
        required={'schema','timestamp_s','coordinate','categories','epoch','retrieval_type','response_l'}
        if purpose=='within_radius':required.add('public_radius_m')
        if purpose=='public_detour_bank':required.add('public_destination_states')
        if (set(request)!=required or request['schema']!='fixed_public_purpose_cover_v1'
                or tuple(request['categories'])!=self.categories
                or (purpose=='within_radius' and request['public_radius_m']!=self.radius_m)
                or (purpose=='public_detour_bank' and tuple(request['public_destination_states'])!=self.destination_states)):
            raise ValueError('Only the declared public schema/parameters are accepted')
        state=self.rn.nearest(*request['coordinate'])[0]
        return self.query(state,purpose,request['response_l'])
