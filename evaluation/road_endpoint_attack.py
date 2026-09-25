"""Road-aware endpoint estimates from permitted public coordinates only.

No hidden elapsed time, destination, original sample index or route enters
this API. A/C use the public 60-second masking rule, B has no such assumption.
Lane-change links are a public approximation, not SUMO replay or a DP theorem.
"""
from collections import defaultdict,OrderedDict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from evaluation.live_comparison_attacks import public_arrays,viterbi


class RoadEndpointAttack:
    def __init__(self,rn,history,cell_m=100.):
        self.rn,self.history,self.cell_m=rn,history,float(cell_m)
        low=np.floor(rn.xy.min(axis=0)/cell_m).astype(int)-1
        high=np.floor(rn.xy.max(axis=0)/cell_m).astype(int)+1
        self.low=low;self.shape=high-low+1
        grid=np.indices(tuple(self.shape)).reshape(2,-1).T+low
        self.centers=(grid+.5)*cell_m;self.size=len(grid)
        self.node_cells=self.cells(rn.xy)
        self.counts=np.bincount(self.node_cells,minlength=self.size)
        road=(self.counts>0).astype(float)
        self.prior=.9*road/road.sum()+.1/self.size
        # Existing directed lane progress plus bounded public lane changes.
        graph=rn.graph;edges=[(int(u),int(v),max(.1,d['length'])) for u,v,d in graph.edges(data=True)]
        lanes=defaultdict(list)
        for lane,ids in rn.lane_indices.items():
            edge=graph.nodes[ids[0]]['edge_id'];lanes[edge].append(np.array(ids,int))
        for group in lanes.values():
            for left,right in zip(sorted(group,key=lambda x:int(x[0])),sorted(group,key=lambda x:int(x[0]))[1:]):
                tree=cKDTree(rn.xy[right]);dist,j=tree.query(rn.xy[left]);valid=dist<=25
                for a,b,d in zip(left[valid],right[j[valid]],dist[valid]):
                    edges.extend([(int(a),int(b),max(1.,float(d))),(int(b),int(a),max(1.,float(d)))])
        self.graph=csr_matrix(([d for _,_,d in edges],([a for a,_,_ in edges],[b for _,b,_ in edges])),shape=(len(rn),len(rn)))
        self.reverse=self.graph.T.tocsr();self.cache=OrderedDict()
        tree=cKDTree(self.centers);neighbors=tree.query_ball_tree(tree,100.)
        rr=np.repeat(np.arange(self.size),[len(v) for v in neighbors]);cc=np.concatenate(neighbors)
        self.hit_neighborhood=csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(self.size,self.size))

    def cells(self,xy):
        indices=np.floor(np.asarray(xy)/self.cell_m).astype(int)-self.low
        if np.any(indices<0) or np.any(indices>=self.shape):raise ValueError('Outside public map grid')
        return indices[...,0]*self.shape[1]+indices[...,1]

    def distances(self,state,reverse):
        key=int(state),bool(reverse)
        if key not in self.cache:
            d=dijkstra(self.reverse if reverse else self.graph,directed=True,indices=state,limit=6000.)
            ids=np.flatnonzero(np.isfinite(d));self.cache[key]=ids,d[ids]
            if len(self.cache)>128:self.cache.popitem(last=False)
        self.cache.move_to_end(key);return self.cache[key]

    def road_density(self,point,velocity,reverse,horizon):
        state=int(self.rn.tree.query(point)[1]);ids,d=self.distances(state,reverse)
        speed=float(np.clip(np.linalg.norm(velocity),1.,8.))
        length=speed*horizon;sigma=max(80.,.35*length)
        weights=np.exp(-.5*((d-length)/sigma)**2)
        # Soft direction evidence; allow curves and turns instead of projecting
        # straight through blocks. Reverse movement for hidden origins.
        delta=self.rn.xy[ids]-point;norm=np.linalg.norm(delta,axis=1)
        heading=velocity*(-1 if reverse else 1)
        cosine=(delta@heading)/np.maximum(norm*np.linalg.norm(heading),1.)
        weights*=np.exp(np.clip(cosine,-1,1))
        weights/=self.counts[self.node_cells[ids]]
        density=np.bincount(self.node_cells[ids],weights=weights,minlength=self.size)
        if density.sum()==0:return self.prior.copy()
        return .9*density/density.sum()+.1*self.prior

    def posteriors(self,events,case_id):
        xy,times=public_arrays(events,self.rn);reverse=case_id.startswith('S9.');at=0 if reverse else -1
        streams={'motion':viterbi(xy,times,self.rn,self.history,True),
                 'history':viterbi(xy,times,self.rn,self.history),
                 'centroid':np.array([x.mean(axis=0) for x in xy])}
        horizons=(30.,60.,90.) if case_id[-1] in 'AC' else (60.,180.,360.,600.)
        result={'map_prior':self.prior.copy()}
        for name,points in streams.items():
            n=min(4,len(times));sl=slice(0,n) if reverse else slice(-n,None)
            t=times[sl];design=np.c_[np.ones(n),t-t[0]]
            velocity=np.linalg.lstsq(design,points[sl]-points[sl][0],rcond=None)[0][1]
            for horizon in horizons:
                result[f'road_{name}_{int(horizon)}s']=self.road_density(points[at],velocity,reverse,horizon)
        return result

    def estimates(self,density):
        """Common posterior decoders; 100-m loss does not use the true endpoint."""
        return {'mean':density@self.centers,
                'map':self.centers[int(np.argmax(density))],
                'hit100':self.centers[int(np.argmax(self.hit_neighborhood@density))]}

    def joint(self,densities):
        """Linked same-endpoint sessions: combine likelihoods, count prior once."""
        logp=np.log(self.prior)+sum(np.log(p)-np.log(self.prior) for p in densities)
        p=np.exp(logp-logp.max());return p/p.sum()

    def distribution_metrics(self,density,truth):
        cells=self.cells(np.asarray(truth));order=np.argsort(-density,kind='stable')
        cumulative=np.cumsum(density[order]);n=min(self.size,int(np.searchsorted(cumulative,.9))+1)
        rank=np.empty(self.size,int);rank[order]=np.arange(self.size)
        return {'log_gain_bits':float(np.mean(np.log2(density[cells]/self.prior[cells]))),
                'nll_bits':float(np.mean(-np.log2(density[cells]))),
                'entropy_bits':float(-np.sum(density*np.log2(density))),
                'credible90_cells':n,'credible90_coverage':float(np.mean(rank[cells]<n))}
