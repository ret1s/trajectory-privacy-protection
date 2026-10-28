"""Existing endpoint OLS challenge generalized to unlinked/unequal-size sets.

Horizons/windows match experiments.contribution_stress.endpoint_predictions:
2/3/6 observed events and 30/60/120 seconds. Viterbi-decoded paths replace
unjustified stable candidate IDs. No endpoint labels or hidden cut duration.
"""
import numpy as np
from evaluation.live_comparison_attacks import public_arrays,viterbi


def extrapolations(events,scenario,rn,history):
    xy,times=public_arrays(events,rn)
    streams={'centroid':np.array([p.mean(axis=0) for p in xy]),
             'median':np.array([np.median(p,axis=0) for p in xy]),
             'viterbi_motion':viterbi(xy,times,rn,history,True),
             'viterbi_history':viterbi(xy,times,rn,history)}
    first=scenario=='S9';at=0 if first else -1;sign=-1 if first else 1
    result={}
    for name,points in streams.items():
        for count in (2,3,6):
            ids=np.arange(min(count,len(times))) if first else np.arange(max(0,len(times)-count),len(times))
            design=np.c_[np.ones(len(ids)),times[ids]-times[at]]
            coefficients=np.linalg.lstsq(design,points[ids],rcond=None)[0]
            for seconds in (30,60,120):
                result[f'{name}_ols{count}_{seconds}s']=(np.array([1.,sign*seconds])@coefficients)[None,:]
    return result
