"""Permutation-invariant attacks for unequal-size public query outputs.

Only relative allowed-window times and public coordinates enter features.
Source/session IDs, method secrets, real/fake labels and endpoints are absent.
The evaluator alone maps per-emission predictions to genuine service events.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree


def public_view(ex,indices):
    ledger={int(k):int(v) for k,v in ex['service_event_positions'].items()}
    clock_to_slot={int(i):j for j,i in enumerate(ex['clock_indices'])}
    real_positions={ledger[clock_to_slot[i]] for i in indices}
    all_real=set(ledger.values())
    first,last=min(real_positions),max(real_positions)
    selected=[j for j in range(first,last+1) if j in real_positions or j not in all_real]
    t0=ex['events'][selected[0]]['timestamp_s']
    events=[];positions={}
    for j in selected:
        e=ex['events'][j]
        xy=e.get('coordinates') or [q[1:] for q in e['queries']]
        events.append({'timestamp_s':e['timestamp_s']-t0,'coordinates':xy})
        positions[j]=len(events)-1
    evaluator_slots=[positions[ledger[clock_to_slot[i]]] for i in indices]
    return events,evaluator_slots


def public_arrays(events,rn):
    times=np.array([e['timestamp_s'] for e in events],float)
    if not len(times) or np.any(np.diff(times)<=0):raise ValueError('Strict public clock required')
    coordinates=[np.unique(np.array([rn.point_xy(*p) for p in e['coordinates']]),axis=0) for e in events]
    return coordinates,times-times[0]


def descriptors(xy,times):
    # All set statistics are invariant to event-local candidate permutation.
    z=np.array([np.r_[p.mean(axis=0),np.median(p,axis=0),p.std(axis=0),
                      np.quantile(p,[0,.25,.75,1],axis=0).ravel(),np.log1p(len(p))*1000] for p in xy])/1000
    means=np.array([p.mean(axis=0) for p in xy])
    past=np.cumsum(z[:,:4],axis=0)/np.arange(1,len(z)+1)[:,None]
    future=(np.cumsum(z[::-1,:4],axis=0)/np.arange(1,len(z)+1)[:,None])[::-1]
    features=np.c_[z,past,future,times/60,(times[-1]-times)/60,np.full(len(z),np.log1p(len(z))) ]
    return features,means


def viterbi(xy,times,rn,history,spatial=False):
    states=[rn.tree.query(p)[1].astype(int) for p in xy]
    scores=np.log(history.q[states[0]])
    back=[]
    for j in range(1,len(states)):
        if spatial:
            distance=np.linalg.norm(xy[j-1][:,None,:]-xy[j][None,:,:],axis=2)
            logp=-distance/max(20.,8.*(times[j]-times[j-1]))
            logp-=np.max(logp,axis=1,keepdims=True)
            transition=np.exp(logp)
        else:transition=history.transition(states[j-1],states[j])
        transition/=transition.sum(axis=1,keepdims=True)
        candidate=scores[:,None]+np.log(np.maximum(transition,1e-300))
        choice=np.argmax(candidate,axis=0);back.append(choice)
        scores=candidate[choice,np.arange(len(states[j]))]
        scores-=scores.max()
    ids=[int(np.argmax(scores))]
    for b in back[::-1]:ids.append(int(b[ids[-1]]))
    return np.array([p[i] for p,i in zip(xy,ids[::-1])])


def features_and_geometry(events,scenario,rn,history):
    xy,times=public_arrays(events,rn);features,means=descriptors(xy,times)
    bank={'centroid':means,'median':np.array([np.median(p,axis=0) for p in xy]),
          'query_prior':np.array([p[np.argmax(history.q[rn.tree.query(p)[1]])] for p in xy]),
          'viterbi_history':viterbi(xy,times,rn,history),
          'viterbi_motion':viterbi(xy,times,rn,history,True)}
    if scenario in ('S2','S9','S10'):
        features=np.r_[features[0],features[-1],features.mean(axis=0),features.std(axis=0)][None,:]
        if scenario=='S2':bank={a:x.mean(axis=0,keepdims=True) for a,x in bank.items()}
        else:bank={a:x[[0 if scenario=='S9' else -1]] for a,x in bank.items()}
    return features,bank


class LearnedAttack:
    """Small common empirical bank, fitted without evaluation targets."""
    def __init__(self,x,y):
        self.mean=np.mean(x,axis=0);self.scale=np.std(x,axis=0)
        self.scale[self.scale<1e-9]=1
        self.x=(np.asarray(x)-self.mean)/self.scale;self.y=np.asarray(y)
        self.tree=ExtraTreesRegressor(n_estimators=64,min_samples_leaf=2,max_depth=16,
                                      random_state=20260925,n_jobs=1).fit(self.x,self.y)
        self.nn=NearestNeighbors(n_neighbors=min(5,len(y))).fit(self.x)
        self.prior=np.median(y,axis=0)
        # Training-label medoid is a stronger fixed-prior hit candidate than mean.
        support=cKDTree(self.y).query_ball_point(self.y,100,return_length=True)
        self.prior_mode=self.y[int(np.argmax(support))]

    def predict(self,x):
        z=(np.asarray(x)-self.mean)/self.scale
        ids=self.nn.kneighbors(z,return_distance=False)
        return {'tree':self.tree.predict(z),'knn1':self.y[ids[:,0]],'knn5':self.y[ids].mean(axis=1),
                'prior_median':np.repeat(self.prior[None,:],len(x),axis=0),
                'prior_hit100':np.repeat(self.prior_mode[None,:],len(x),axis=0)}
