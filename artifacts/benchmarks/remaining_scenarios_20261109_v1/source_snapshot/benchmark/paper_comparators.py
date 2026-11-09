"""Source-mapped local paper comparators for the common live-POI experiment.

All are explicitly adaptations. RDG follows arXiv:1805.06104, Algorithm 6;
fake insertion follows DOI 10.1007/s44443-025-00438-z, Sections 4.1--4.2.
Road discretisation, sparse-history smoothing and timer choices are recorded
in the protocol. No evaluation route or label is used to fit a comparator.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import math
import time

import networkx as nx
import numpy as np

from benchmark.engines.dls import DLSGraph, entropy
from benchmark.engines.anotherme import RoadNetworkRouteProvider
from benchmark.methods import AnotherMeAdaptation, SemanticCorrelationComparator, TransProtectAdaptation
from core.demo_protocol import TrajectoryPoint


class PreparedDLS(DLSGraph):
    """Exact existing DLS pool/tie policy, with probability order precomputed."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.order = np.lexsort((np.arange(len(self.q)), self.q))
        self.sorted_q = self.q[self.order]

    @lru_cache(maxsize=8192)
    def candidate_pool(self, true_index):
        a = np.searchsorted(self.sorted_q, self.q[true_index], side='left')
        b = np.searchsorted(self.sorted_q, self.q[true_index], side='right')
        ties = self.order[a:b]
        ties = ties[ties != true_index]
        half = len(ties)//2
        order = np.r_[self.order[:a], ties[:half], true_index, ties[half:], self.order[b:]]
        center = a+half
        left, right = max(0, center-self.k), min(len(order), center+self.k+1)
        if right-left < min(len(order), 2*self.k+1):
            left = max(0, right-min(len(order), 2*self.k+1))
            right = min(len(order), left+min(len(order), 2*self.k+1))
        return order[left:right][order[left:right] != true_index]


def rdg_choice(previous, posterior, true_index, pool, transition, k):
    """Algorithm 6: greedy entropy of normalized max-product path weights.

    transition(a,b) returns empirical weights, normalized over each *trial*
    release set as in the source's Eq. 9. Forward posterior (Eq. 12), not
    max-product scores, is propagated between calls.
    """
    chosen = [int(true_index)]
    pool = [int(x) for x in pool if x != true_index]
    if len(set(pool)) != len(pool) or len(pool) < k-1:
        raise ValueError('Distinct dummy pool with at least K-1 members required')
    posterior = np.asarray(posterior, float)
    if np.shape(posterior) != (len(previous),) or np.any(posterior <= 0):
        raise ValueError('Positive previous posterior required')
    while len(chosen) < k:
        scores = []
        for candidate in pool:
            weights = np.asarray(transition(previous, chosen+[candidate]), float)
            if np.any(weights <= 0) or weights.shape != (len(previous), len(chosen)+1):
                raise ValueError('Positive transition weights required')
            kernel = weights/weights.sum(axis=1, keepdims=True)
            scores.append(entropy(np.max(posterior[:, None]*kernel, axis=0)))
        winner = int(np.argmax(scores))  # public deterministic first-pool tie
        chosen.append(pool.pop(winner))
    weights = transition(previous, chosen)
    kernel = weights/weights.sum(axis=1, keepdims=True)
    next_posterior = posterior @ kernel
    return np.array(chosen), next_posterior/next_posterior.sum()


class PublicHistory:
    """100-m public cells and disjoint background empirical query transitions."""
    def __init__(self, rn, training, cell_m=100., smoothing=.1):
        self.rn = rn
        self.cell_m, self.smoothing = cell_m, smoothing
        cells = np.floor(rn.xy/cell_m).astype(int)
        _, self.cells, sizes = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        visits = np.ones(len(sizes))
        self.counts = defaultdict(Counter)
        self.training_sequences = 0
        for sequence in training:
            ids = np.array(sequence, dtype=int)
            cs = self.cells[ids]
            visits += np.bincount(cs, minlength=len(visits))
            for a,b in zip(cs, cs[1:]): self.counts[int(a)][int(b)] += 1
            self.training_sequences += 1
        self.q = visits[self.cells]/sizes[self.cells]
        self.q /= self.q.sum()

    def transition(self, previous, current):
        return np.array([[self.counts[int(self.cells[a])][int(self.cells[b])]+self.smoothing
                          for b in current] for a in previous], float)

    @lru_cache(maxsize=8192)
    def reachable(self, start, seconds):
        # Free-flow SUMO passenger-lane travel, no straight-line teleportation.
        seconds = float(seconds)
        lengths = nx.single_source_dijkstra_path_length(
            self.rn.graph, self.rn.node_ids[int(start)], cutoff=max(seconds, .01),
            weight=lambda u,v,d: d['length']/max(d.get('speed',8.), .01))
        ids = [int(i) for i,dt in lengths.items() if i != int(start) and dt >= .4*seconds]
        # Lane-state IDs equal array indices in the pinned catalogue.
        return tuple(sorted(ids, key=lambda i: (-self.transition([start],[i])[0,0], i)))


def path_similarity(reference, candidate, floor=1.):
    """Relative length and direction deviations; explicit local feature choice."""
    ref, cand = np.asarray(reference,float), np.asarray(candidate,float)
    nr, nc = np.linalg.norm(ref), np.linalg.norm(cand)
    relative = abs(nc-nr)/max(nr, floor)
    angle = 0. if nr < floor and nc < floor else math.acos(float(np.clip(np.dot(ref,cand)/max(nr*nc,1e-12),-1,1)))/math.pi
    return math.hypot(relative, angle)


class FakeQuerySequence:
    """Causal timer/continuation adapter; fake flags never enter public events.

    R=2, random waits 5--15 seconds; section 4.1's historical graph uses public
    background cells, constrained by road reachability. Infeasible cover sets
    are suppressed and counted, not fabricated. Real fallback is DLS (§4.2).
    """
    def __init__(self, rn, history, dls, rng, *, max_fake=2, sigma=.75):
        self.rn,self.history,self.dls,self.rng = rn,history,dls,rng
        self.max_fake,self.sigma = max_fake,sigma
        self.previous=None;self.anchor=None;self.last_time=None;self.next_time=math.inf
        self.remaining=0;self.failed_fake=0;self.real_fallback=0;self.next_reference=None

    def continuation(self, elapsed, fixed_reference=None):
        old=self.previous;anchor=self.anchor
        candidates=self.history.reachable(anchor, round(float(elapsed),3))[:24]
        if fixed_reference is not None:
            if int(fixed_reference) not in self.history.reachable(anchor,round(float(elapsed),3)):
                return None
            candidates=(int(fixed_reference),)
        for target in candidates:
            if target in old:continue
            ref=self.rn.xy[target]-self.rn.xy[anchor]
            result=[int(target)]
            for start in old:
                if start==anchor:continue
                options=[v for v in self.history.reachable(int(start),round(float(elapsed),3))[:24]
                         if v not in result and v not in old]
                if not options:break
                scored=[path_similarity(ref,self.rn.xy[v]-self.rn.xy[start]) for v in options]
                best=int(np.argmin(scored))
                if scored[best]>self.sigma:break
                result.append(options[best])
            if len(result)==self.dls.k:return np.array(result)
        return None

    def step(self, timestamp, real_index):
        emitted=[]
        while self.next_time < timestamp and self.remaining:
            t=self.next_time
            selected=self.continuation(t-self.last_time)
            if selected is not None:
                emitted.append((t,selected,False))
                self.previous=selected;self.anchor=int(selected[0]);self.last_time=t
            else:self.failed_fake+=1
            self.remaining-=1
            self.next_time=t+float(self.rng.uniform(5,15)) if self.remaining else math.inf
        selected=None
        if self.previous is not None and timestamp>self.last_time:
            selected=self.continuation(timestamp-self.last_time,real_index)
        if selected is None:
            selected,_=self.dls.select(real_index);self.real_fallback+=1
        self.previous=selected;self.anchor=int(real_index);self.last_time=timestamp
        emitted.append((timestamp,selected,True))
        self.remaining=int(self.rng.integers(self.max_fake+1))
        self.next_time=timestamp+float(self.rng.uniform(5,15)) if self.remaining else math.inf
        return emitted


class PassengerRouteProvider(RoadNetworkRouteProvider):
    """Vehicle benchmark route domain; no unsupported walking/bicycle claims."""
    name='sumo_passenger_route_adapter'
    def _mode_graph(self, mode):
        self.last_mode_policy='passenger_graph_for_vehicle_dataset_all_speed_modes'
        return self.road_network.graph


class PoiSemanticLabeller:
    """Public nearest-POI categories instead of absent lane highway attributes."""
    def __init__(self,rn,pois):
        from scipy.spatial import cKDTree
        self.rn=rn
        self.categories=np.array([p['category'] for p in pois])
        self.tree=cKDTree(rn.xy[[p['vertex'] for p in pois]])
    @lru_cache(maxsize=8192)
    def label(self,index):return str(self.categories[int(self.tree.query(self.rn.xy[index])[1])])


def shared_models(rn, history, training, pois):
    """Setup once; private per-session state is reset by each protect_run."""
    trans=TransProtectAdaptation.from_road_network(rn,training_trajectories=training,
        candidate_k=10,target_count=8,alpha=10000,epsilon=.005,rng=np.random.default_rng(0))
    semantic=SemanticCorrelationComparator(rn,k=5,rng=np.random.default_rng(0))
    semantic.labeller=PoiSemanticLabeller(rn,pois)
    return {'transprotect_markov':trans,'semantic_poi':semantic,
            'dls':PreparedDLS(rn,history.q,k=5,rng=np.random.default_rng(0)),
            'rdg_pool':PreparedDLS(rn,history.q,k=21,rng=np.random.default_rng(0))}


def generate(method, points, rn, history, shared, seed):
    """Return public coordinate events plus evaluator-only index bookkeeping."""
    rng=np.random.default_rng(seed)
    times=[float(p.timestamp_s) for p in points]
    started=time.perf_counter();ledger={};diagnostics={}
    if method in ('transprotect_markov','semantic_poi'):
        model=shared[method];model.rng=rng
        events=model.protect_run(points).to_attacker_dict()['events']
        public=[{'timestamp_s':e['timestamp_s'],'coordinates':[[c['lat'],c['lon']] for c in e['candidates']]} for e in events]
        ledger={i:i for i in range(len(points))}
    elif method=='anotherme_offline':
        from benchmark.engines.anotherme import RoadNetworkVirtualEndpointMapper
        model=AnotherMeAdaptation(route_provider=PassengerRouteProvider(rn),
            endpoint_mapper=RoadNetworkVirtualEndpointMapper(rn,seed=seed),seed=seed)
        events=model.protect_run(points).to_attacker_dict()['events']
        public=[{'timestamp_s':e['timestamp_s'],'coordinates':[[c['lat'],c['lon']] for c in e['candidates']]} for e in events]
        ledger={i:i for i in range(len(points))}
        diagnostics['uses_future']=True
    elif method=='raw':
        public=[{'timestamp_s':p.timestamp_s,'coordinates':[[p.lat,p.lon]]} for p in points]
        ledger={i:i for i in range(len(points))}
    else:
        dls=shared['dls'];dls.rng=rng;pool=shared['rdg_pool'];pool.rng=rng
        fake=FakeQuerySequence(rn,history,dls,rng) if method=='fake_queries' else None
        public=[];previous=None;posterior=None
        for i,p in enumerate(points):
            true=int(rn.nearest(p.lat,p.lon)[0])
            if fake:
                emitted=fake.step(p.timestamp_s,true)
            else:
                if method=='dls' or previous is None:
                    ids,_=dls.select(true);posterior=history.q[ids];posterior/=posterior.sum()
                elif method=='rdg':
                    candidates,_=pool.select(true)
                    ids,posterior=rdg_choice(previous,posterior,true,candidates[candidates!=true],history.transition,5)
                else:raise ValueError(method)
                previous=ids
                emitted=[(p.timestamp_s,ids,True)]
            for t,ids,is_real in emitted:
                if is_real:ledger[i]=len(public)
                public.append({'timestamp_s':float(t),'coordinates':[list(rn.latlon(int(j))) for j in rng.permutation(ids)]})
        if fake:diagnostics.update(failed_fake=fake.failed_fake,real_fallback=fake.real_fallback,
                                   fake_events=len(public)-len(points))
    elapsed=time.perf_counter()-started
    return {'events':public,'service_event_positions':ledger,'generation_ms':1000*elapsed,
            'diagnostics':diagnostics}
