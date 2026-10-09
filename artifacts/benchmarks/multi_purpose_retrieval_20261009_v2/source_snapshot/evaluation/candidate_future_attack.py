"""Geometry-based S5/S6 candidate scoring, generalizing to unseen road IDs.

The TWO-choice context is public auxiliary knowledge fixed before the hidden
query choice. It contains both native turns/destinations, never the selected
route, future GPS, routine label or private seed. Literal IDs are output names,
not classifier features. Historical inputs are six protected public windows.
"""
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString,Point
from sklearn.ensemble import ExtraTreesClassifier
from evaluation.identity_future import public_arrays

ALLOWED_CHOICE={'edge_id','via_lane_ids','via_shape_xy','outgoing_shape_xy',
                'destination_xy','destination_edge_id'}
PROJECT=Transformer.from_crs(4326,32650,always_xy=True)


def coordinates(public):
    public_arrays(public)  # Enforces the public-only schema and causal clock.
    values=[]
    for event in public['events']:
        values.append(np.array([PROJECT.transform(c['lon'],c['lat']) for c in event['candidates']]))
    return values,np.array([float(e['timestamp_s']) for e in public['events']])


def validate_context(context):
    if not isinstance(context,dict) or set(context)!={'choices'} or len(context['choices'])!=2:
        raise ValueError('Public two-choice context required')
    for choice in context['choices']:
        if set(choice)!=ALLOWED_CHOICE:
            raise ValueError('Private choice/routine/route fields forbidden')
        for name in ('via_shape_xy','outgoing_shape_xy'):
            points=np.asarray(choice[name],dtype=float)
            if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
                raise ValueError('Finite public native polyline required')
        if np.shape(choice['destination_xy'])!=(2,) or not np.isfinite(choice['destination_xy']).all():
            raise ValueError('Finite public candidate destination required')
    if context['choices'][0]['edge_id']==context['choices'][1]['edge_id']:
        raise ValueError('Distinct public candidate edges required')
    return context['choices']


def candidate_features(query,context,histories=()):
    """Return 2x24 numerical features; no scenario/family/day/identity labels."""
    choices=validate_context(context)
    if histories and len(histories)!=6:raise ValueError('Exactly six public history windows required')
    xy,times=coordinates(query);centres=np.array([p.mean(axis=0) for p in xy])
    last=xy[-1];recent=np.concatenate(xy[-3:]);span=max(1.,times[-1]-times[max(0,len(times)-3)])
    velocity=(centres[-1]-centres[max(0,len(centres)-3)])/span
    unit=velocity/max(1.,np.linalg.norm(velocity))
    history=[]
    for h in histories:
        hx,_=coordinates(h);history.append(np.concatenate(hx[-3:]))
    ends=np.array([c['destination_xy'] for c in choices]);sep=max(1.,np.linalg.norm(ends[1]-ends[0]))
    hdist=np.array([[np.linalg.norm(h-end,axis=1).mean() for end in ends] for h in history])
    votes=np.bincount(np.argmin(hdist,axis=1),minlength=2) if len(history) else np.zeros(2)
    features=[]
    for i,c in enumerate(choices):
        curve=LineString(c['via_shape_xy']);out=LineString(c['outgoing_shape_xy'])
        distance=lambda points,line:np.array([Point(p).distance(line) for p in points])
        d=distance(last,curve);r=distance(recent,curve);o=distance(last,out)
        shape=np.asarray(c['via_shape_xy']);heading=shape[-1]-shape[0]
        heading=heading/max(1.,np.linalg.norm(heading))
        end=ends[i];delta=end-centres[-1];endheading=delta/max(1.,np.linalg.norm(delta))
        hist=hdist[:,i]/sep if len(history) else np.zeros(6)
        features.append(np.r_[d.mean()/1000.,d.min()/1000.,d.max()/1000.,r.mean()/1000.,r.min()/1000.,
            o.mean()/1000.,o.min()/1000.,np.linalg.norm(centres[-1]-shape.mean(axis=0))/1000.,
            unit@heading,np.linalg.norm(velocity)/10.,unit@endheading,
            np.linalg.norm(delta)/sep,np.linalg.norm(last-end,axis=1).min()/sep,
            np.abs(delta)/1000.,times[-1]/60.,len(times)/30.,
            hist.mean(),hist.min(),hist.max(),hist.std(),votes[i]/6.,
            float(len(history)),np.linalg.norm(ends[i]-ends[1-i])/1000.])
    result=np.array(features)
    assert result.shape==(2,24) and np.isfinite(result).all()
    return result


def normalize(scores):
    values=np.asarray(scores,dtype=float)
    if values.shape!=(2,) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('Two finite nonnegative candidate scores required')
    values=values+1e-9
    return values/values.sum()


def geometric_bank(query,context,histories=()):
    features=candidate_features(query,context,histories)
    result={'uniform':np.ones(2)/2}
    for temperature in (.02,.10,.30):
        for name,column in [('curve_mean',0),('curve_min',1),('curve_recent',3)]:
            logits=-features[:,column]/temperature;logits-=logits.max()
            result[f'{name}_{temperature:g}']=normalize(np.exp(logits))
    result['motion']=normalize(np.exp(2*features[:,8]-features[:,0]/.1))
    if histories:
        votes=features[:,21]*6
        result['history_prior']=normalize(votes+1.)
        result['history_query']=normalize(result['curve_mean_0.1']*(votes+1.))
    return result


class CandidateFutureAttack:
    """Train on candidate features/labels; never fit a closed road-ID classifier."""
    def __init__(self,rows,*,use_history=False,permutation=False,seed=2026100519):
        x=[];y=[];self.use_history=use_history
        flips={f:int(np.random.default_rng(seed+int(f.split('-')[-1])).integers(0,2))
               for f in sorted({r['family_id'] for r in rows})}
        for r in rows:
            history=r['histories'] if use_history else ()
            features=candidate_features(r['query'],r['public_context'],history)
            target=r['choice_index']^(flips[r['family_id']] if permutation else 0)
            x.extend(features);y.extend([int(i==target) for i in range(2)])
        self.trees=ExtraTreesClassifier(n_estimators=96,max_depth=12,min_samples_leaf=2,
                                       random_state=seed,n_jobs=1).fit(np.array(x),np.array(y))

    def predict(self,query,context,histories=()):
        history=histories if self.use_history else ()
        features=candidate_features(query,context,history)
        probabilities=self.trees.predict_proba(features)[:,list(self.trees.classes_).index(1)]
        return {**geometric_bank(query,context,history),'candidate_trees':normalize(probabilities)}


def forecast_metrics(rows,probabilities):
    if not rows or len(rows)!=len(probabilities):raise ValueError('Aligned forecast rows required')
    truth=np.array([r['choice_index'] for r in rows]);p=np.array(probabilities)
    if p.shape!=(len(rows),2) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(axis=1),1):
        raise ValueError('Normalized forecast distributions required')
    prediction=np.argmax(p,axis=1);correct=truth==prediction
    errors=[];chance=[]
    for r,index in zip(rows,prediction):
        ends=np.array([c['destination_xy'] for c in validate_context(r['public_context'])])
        errors.append(float(np.linalg.norm(ends[index]-r['destination_xy'])))
        chance.append(float(np.linalg.norm(ends-r['destination_xy'],axis=1).mean()))
    return {'exact_candidate_edge_accuracy':float(correct.mean()),
        'balanced_accuracy':float(np.mean([correct[truth==i].mean() for i in (0,1)])),
        'brier':float(np.mean((p[:,1]-truth)**2)),
        'log_loss':float(-np.log(np.maximum(1e-12,p[np.arange(len(rows)),truth])).mean()),
        'destination_mae_m':float(np.mean(errors)),
        'destination_hit100':float(np.mean(np.array(errors)<=100.)),
        'uniform_destination_mae_m':float(np.mean(chance)),
        'routine_accuracy':float(np.mean([ok for r,ok in zip(rows,correct) if r['destination_role']=='routine'])),
        'rare_accuracy':float(np.mean([ok for r,ok in zip(rows,correct) if r['destination_role']=='rare'])),
        'n':len(rows),'family_count':len({r['family_id'] for r in rows})}
