"""S4 linkage and S5/S6 inference from a strictly public observation window.

Truth, route plans, original clocks and person/vehicle identifiers are not
accepted by these feature functions. Pairwise linkage is not identification
of a real person. A grouped split must keep every linked session in one fold.
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

LAT0, LON0 = 40., 116.32  # Fixed public frame, never fitted to evaluation truth.
METRES_PER_DEGREE = np.pi / 180. * 6371000.


def xy_from_latlon(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError('Finite N x 2 latitude/longitude array required')
    if np.any(np.abs(p[:, 0]) > 90) or np.any(np.abs(p[:, 1]) > 180):
        raise ValueError('Invalid geographic coordinates')
    return np.c_[(p[:, 1]-LON0)*METRES_PER_DEGREE*np.cos(np.radians(LAT0)),
                 (p[:, 0]-LAT0)*METRES_PER_DEGREE]


def public_arrays(public):
    """Validate the public boundary; candidate count/order may vary by method."""
    if not isinstance(public, dict) or set(public) != {'events'}:
        raise ValueError('Only public events are permitted')
    events = public['events']
    if not events:
        raise ValueError('Nonempty public prefix required')
    centres, spreads, times = [], [], []
    for event in events:
        if set(event) != {'event_id', 'timestamp_s', 'candidates'}:
            raise ValueError('Evaluator/private fields forbidden in public event')
        candidates = event['candidates']
        if not candidates:
            raise ValueError('Nonempty public candidate set required')
        for c in candidates:
            if set(c) != {'candidate_id', 'lat', 'lon'}:
                raise ValueError('Evaluator/private fields forbidden in candidate')
        points = xy_from_latlon([[c['lat'], c['lon']] for c in candidates])
        centres.append(points.mean(axis=0)); spreads.append(points.std(axis=0))
        times.append(float(event['timestamp_s']))
    times = np.array(times)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Finite, strictly increasing public times required')
    return np.array(centres), np.array(spreads), times-times[0]


def prefix_features(public):
    """Order-invariant K features; no future events beyond the supplied prefix."""
    centre, spread, times = public_arrays(public)
    values = np.c_[centre, spread] / 1000.
    span = max(1., times[-1])
    velocity = (centre[-1]-centre[0])/span
    return np.r_[values[0], values[-1], values.mean(axis=0), values.std(axis=0),
                 np.quantile(values, [.25, .5, .75], axis=0).ravel(),
                 velocity, times[-1]/60., np.log1p(len(times))]


def pair_features(left, right):
    """Symmetric pair similarity features, with no person or vehicle IDs."""
    a, b = prefix_features(left), prefix_features(right)
    return np.r_[np.abs(a-b), (a+b)/2., np.minimum(a,b), np.maximum(a,b)]


def check_group_split(train, selection, test):
    groups = [set(g) for g in (train, selection, test)]
    if any(not g for g in groups) or any(groups[i] & groups[j]
            for i in range(3) for j in range(i+1,3)):
        raise ValueError('Nonempty group-disjoint train/selection/test required')
    return True


def classification_metrics(labels, prediction, probability=None):
    labels, prediction = np.asarray(labels), np.asarray(prediction)
    if labels.ndim != 1 or prediction.shape != labels.shape or not len(labels):
        raise ValueError('Matching nonempty classification labels required')
    result = {'accuracy':float(np.mean(labels==prediction)),
              'balanced_accuracy':float(balanced_accuracy_score(labels,prediction)),
              'macro_f1':float(f1_score(labels,prediction,average='macro',zero_division=0)),
              'label_count':int(len(np.unique(labels))), 'n':len(labels)}
    if probability is not None and len(np.unique(labels)) == 2:
        result['roc_auc'] = float(roc_auc_score(labels,np.asarray(probability)))
    return result


def location_metrics(truth, estimate):
    truth, estimate = np.asarray(truth,dtype=float), np.asarray(estimate,dtype=float)
    if truth.ndim != 2 or truth.shape[1] != 2 or estimate.shape != truth.shape:
        raise ValueError('Matching N x 2 position arrays required')
    if not len(truth) or not np.isfinite(truth).all() or not np.isfinite(estimate).all():
        raise ValueError('Finite, nonempty location arrays required')
    errors = np.linalg.norm(truth-estimate,axis=1)
    return {'mae_m':float(errors.mean()),'median_m':float(np.median(errors)),
            'rmse_m':float(np.sqrt(np.mean(errors**2))),
            **{f'hit{r}':float(np.mean(errors<=r)) for r in (50,100,200,500)},'n':len(errors)}


class ClassifierBank:
    """Fit on training only; choose one fixed attacker on selection only."""
    def __init__(self,x,y,seed=20261005):
        x,y=np.asarray(x,dtype=float),np.asarray(y)
        if x.ndim != 2 or len(x) != len(y) or len(np.unique(y)) < 2:
            raise ValueError('Training requires finite features and at least two classes')
        self.scale=StandardScaler().fit(x)
        self.models={'trees':ExtraTreesClassifier(n_estimators=96,min_samples_leaf=2,
            class_weight='balanced',max_depth=16,random_state=seed,n_jobs=1).fit(x,y)}
        for k in (1,5,15):
            if k<=len(y):
                self.models[f'knn{k}']=KNeighborsClassifier(k,weights='distance').fit(self.scale.transform(x),y)
        self.classes=np.unique(y)

    def predict(self,x):
        x=np.asarray(x,dtype=float)
        return {name:model.predict(self.scale.transform(x) if name.startswith('knn') else x)
                for name,model in self.models.items()}

    def probability(self,x):
        x=np.asarray(x,dtype=float)
        if set(self.classes) != {0,1}:raise ValueError('Binary 0/1 labels required')
        return {name:model.predict_proba(self.scale.transform(x) if name.startswith('knn') else x)[:,
                     list(model.classes_).index(1)] for name,model in self.models.items()}


class RegressorBank:
    def __init__(self,x,y,seed=20261005):
        x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
        self.scale=StandardScaler().fit(x)
        self.models={'trees':ExtraTreesRegressor(n_estimators=96,min_samples_leaf=2,
                       max_depth=16,random_state=seed,n_jobs=1).fit(x,y)}
        for k in (1,5,15):
            if k<=len(y):
                self.models[f'knn{k}']=KNeighborsRegressor(k,weights='distance').fit(self.scale.transform(x),y)
        self.prior=y.mean(axis=0)

    def predict(self,x):
        x=np.asarray(x,dtype=float)
        return {**{name:model.predict(self.scale.transform(x) if name.startswith('knn') else x)
                   for name,model in self.models.items()},'train_prior':np.repeat(self.prior[None,:],len(x),axis=0)}
