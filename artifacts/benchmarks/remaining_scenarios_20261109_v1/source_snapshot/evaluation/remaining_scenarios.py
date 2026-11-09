"""Public-only feature contracts for the frozen S4--S8 supplement.

Synthetic identity and joint-motion assumptions belong to evaluator metadata,
never to these feature APIs. Fixed protected outputs are not regenerated.
"""
import hashlib
import json

import numpy as np
from pyproj import Transformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.candidate_future_attack import CandidateFutureAttack, candidate_features

PROJECT = Transformer.from_crs(4326, 32650, always_xy=True)


def prefix(events, cut):
    """A strict public time window; reject bookkeeping rather than consume it."""
    if not np.isfinite(cut) or cut < 0:
        raise ValueError('Nonnegative public cutoff required')
    out = []
    previous = -np.inf
    for event in events:
        if set(event) != {'timestamp_s', 'coordinates'}:
            raise ValueError('Private/evaluator event fields forbidden')
        time = float(event['timestamp_s'])
        if not np.isfinite(time) or time <= previous:
            raise ValueError('Strict finite public timestamps required')
        previous = time
        if time > cut:
            break  # Do not inspect future coordinates.
        points = np.asarray(event['coordinates'], float)
        if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
                or not np.isfinite(points).all() or np.any(np.abs(points[:, 0]) > 90)
                or np.any(np.abs(points[:, 1]) > 180)):
            raise ValueError('Finite latitude/longitude query set required')
        out.append({'timestamp_s':time, 'coordinates':points.tolist()})
    if not out:
        raise ValueError('Nonempty public prefix required')
    return out


def xy(latlon):
    points = np.asarray(latlon, float)
    return np.array(PROJECT.transform(points[..., 1], points[..., 0])).T


def public_geometry(events, cut):
    events = prefix(events, cut)
    points = [xy(e['coordinates']) for e in events]
    times = np.array([e['timestamp_s'] for e in events])
    centers = np.array([p.mean(axis=0) for p in points])
    return points, times, centers


def prefix_vector(events, cut):
    points, times, centers = public_geometry(events, cut)
    spread = np.array([p.std(axis=0) for p in points])
    values = np.c_[centers, spread]/1000
    velocity = (centers[-1]-centers[0])/max(1., times[-1]-times[0])/10
    return np.r_[values[0], values[-1], values.mean(axis=0), values.std(axis=0),
        np.quantile(values, [.25,.5,.75], axis=0).ravel(), velocity,
        times[-1]/60, np.log1p(len(times))]


def linkage_features(left, right, cut=600.):
    a, at, ac = public_geometry(left, cut)
    b, bt, bc = public_geometry(right, cut)
    va, vb = prefix_vector(left, cut), prefix_vector(right, cut)
    # Differences generalize across absolute map locations and are symmetric.
    minimum = lambda p,q:float(np.linalg.norm(p[:,None,:]-q[None,:,:], axis=2).min())/1000
    return np.r_[np.abs(va-vb), np.linalg.norm(ac[0]-bc[0])/1000,
        np.linalg.norm(ac[-1]-bc[-1])/1000, minimum(a[0],b[0]),minimum(a[-1],b[-1])]


def candidate_public(events, cut):
    events = prefix(events, cut)
    # IDs encode only observable packet/slot order; candidate_features ignores them.
    return {'events':[{'event_id':f'packet-{i}', 'timestamp_s':e['timestamp_s'],
        'candidates':[{'candidate_id':f'slot-{k}','lat':p[0],'lon':p[1]}
                      for k,p in enumerate(e['coordinates'])]}
        for i,e in enumerate(events)]}


class FutureBank:
    def __init__(self, rows):
        self.model = CandidateFutureAttack(rows, seed=20261109)
        x = np.concatenate([candidate_features(r['query'],r['public_context']) for r in rows])
        y = np.array([int(k==r['choice_index']) for r in rows for k in range(2)])
        self.knn = {f'candidate_knn{k}':make_pipeline(StandardScaler(),
            KNeighborsClassifier(k, weights='distance')).fit(x,y) for k in (1,5)}

    def predict(self, row):
        result = self.model.predict(row['query'],row['public_context'])
        x = candidate_features(row['query'],row['public_context'])
        for name, model in self.knn.items():
            p = model.predict_proba(x)[:,list(model.classes_).index(1)]+1e-9
            result[name] = p/p.sum()
        return result


class LinkageBank:
    def __init__(self, features, labels):
        x,y = np.asarray(features),np.asarray(labels)
        self.models = {
            'extra_trees96':ExtraTreesClassifier(n_estimators=96,min_samples_leaf=2,
                class_weight='balanced',max_depth=12,random_state=20261109,n_jobs=1).fit(x,y),
            'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(x,y)}
        for k in (1,5):
            self.models[f'knn{k}'] = make_pipeline(StandardScaler(),
                KNeighborsClassifier(k,weights='distance')).fit(x,y)

    def predict(self, features):
        x = np.asarray(features)
        probabilities = {name:model.predict_proba(x)[:,list(model.classes_).index(1)]
                         for name,model in self.models.items()}
        probabilities['uniform'] = np.full(len(x), .5)
        # The final four columns are explicit public pair distances.
        for scale in (2.,10.,25.,50.,100.,200.,500.,1000.):
            probabilities[f'start_distance_{scale:g}m'] = np.exp(-1000*x[:,-4]/scale)
        return {name+'@'+str(threshold):np.c_[1-p,p]
            for name,p in probabilities.items() for threshold in (.1,.25,.5,.75,.9)}


def class_prediction(probability, threshold=None):
    p = np.asarray(probability,float)
    if p.ndim != 2 or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Normalized finite class probabilities required')
    if threshold is not None:
        return (p[:,1] >= threshold).astype(int)
    return np.argmax(p,axis=1)


def location_candidates(events, cut):
    points,times,centers = public_geometry(events, cut)
    result = {'centroid':centers[-1], 'median':np.median(points[-1],axis=0)}
    recent = min(3,len(times))
    if recent >= 2:
        t = times[-recent:]-times[-1]
        design = np.c_[np.ones(recent),t]
        coefficients = np.linalg.lstsq(design,centers[-recent:],rcond=None)[0]
        result['velocity'] = coefficients[0]+(cut-times[-1])*coefficients[1]
    else:
        result['velocity'] = centers[-1]
    return result


def joint_vector(target, partner, cut):
    return np.r_[prefix_vector(target,cut),prefix_vector(partner,cut),
                 linkage_features(target,partner,cut)]


def wire_digest(requests, replies):
    return hashlib.sha256(json.dumps({'requests':requests,'replies':replies},
        sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
