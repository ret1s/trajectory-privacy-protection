"""Geometry-only fusion of two granted, causal public location channels."""
import numpy as np
from evaluation.remaining_scenarios import public_geometry, location_candidates


def fusion_estimates(target,partner,cut):
    a,_,_=public_geometry(target,cut);b,_,_=public_geometry(partner,cut)
    p,q=a[-1],b[-1]
    ta=location_candidates(target,cut);tb=location_candidates(partner,cut)
    result={}
    for key in ('centroid','median','velocity'):
        for weight in (.25,.5,.75):
            result[f'fusion_{key}_{weight:g}']=weight*ta[key]+(1-weight)*tb[key]
    # A shared real member, when present, can be exposed by cross-set proximity.
    pairs=[]
    for left in p:
        for right in q:
            midpoint=(left+right)/2
            pairs.append((float(np.linalg.norm(left-right)),float(midpoint[0]),float(midpoint[1]),midpoint))
    pairs.sort(key=lambda v:v[:3])
    result['fusion_nearest_pair']=pairs[0][3]
    for count in (2,3):
        result[f'fusion_nearest_{count}_pairs_mean']=np.mean([v[3] for v in pairs[:count]],axis=0)
    assert all(np.shape(v)==(2,) and np.isfinite(v).all() for v in result.values())
    return result
