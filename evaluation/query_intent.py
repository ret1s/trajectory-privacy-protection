"""Attacker features and selection for explicitly observed query transcripts.

Features are whitelisted public requests/replies. Query truth, person/session IDs,
GPS, radius and destination are never read unless present in the wire payload.
"""
import json

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def wire_features(requests, replies, categories, purposes):
    if not requests:
        raise ValueError('Public request list required')
    coords=np.asarray([q['coordinate'] for q in requests],dtype=float)
    # Public common-coordinate geometry; no projection reference from truth.
    coords=(coords-np.array([39.99,116.32]))*np.array([111.2,85.2])
    categories=tuple(categories);purposes=tuple(purposes)
    cat=np.array([sum(c in q.get('categories',[q.get('category')]) for q in requests)
                  for c in categories],dtype=float)
    purpose=np.array([sum(q.get('purpose')==p for q in requests) for p in purposes],dtype=float)
    time=np.array([q['timestamp_s'] for q in requests],dtype=float)
    byte_count=len(json.dumps({'requests':requests,'replies':replies},
                             separators=(',',':'),sort_keys=True).encode())
    stats=np.r_[coords.mean(axis=0),coords.std(axis=0),coords.min(axis=0),coords.max(axis=0)]
    # Counts/presence of optional private fields are observable for controls.
    fields=np.array([sum(k in q for q in requests) for k in ['purpose','radius_m','destination']])
    reply_sizes=[len(catrow) for reply in replies for catrow in reply]
    return np.r_[stats,cat,purpose,fields,len(requests),byte_count/1000.,
                 time.min()/60.,time.max()/60.,sum(reply_sizes)]


def classifier_bank(train_x, train_y):
    x=np.asarray(train_x,float);y=np.asarray(train_y)
    if x.ndim!=2 or not len(x) or y.shape!=(len(x),) or not np.isfinite(x).all():
        raise ValueError('Finite attacker training rows and labels required')
    specs={'knn1':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=1)),
           'knn5':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=min(5,len(x)))),
           'knn15':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=min(15,len(x)))),
           'extra_trees_leaf1':ExtraTreesClassifier(n_estimators=128,min_samples_leaf=1,
               max_features=1.,random_state=31005,n_jobs=1),
           'extra_trees_leaf3':ExtraTreesClassifier(n_estimators=128,min_samples_leaf=3,
               max_features=1.,random_state=31005,n_jobs=1),
           'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=10.,max_iter=2000))}
    return {name:model.fit(x,y) for name,model in specs.items()}


def score_classification(truth, prediction, classes):
    truth=np.asarray(truth);prediction=np.asarray(prediction)
    if truth.shape!=prediction.shape or not len(truth):
        raise ValueError('Aligned nonempty truth and prediction required')
    return {'balanced_accuracy':float(balanced_accuracy_score(truth,prediction)),
            'macro_f1':float(f1_score(truth,prediction,labels=list(classes),average='macro',zero_division=0)),
            'accuracy':float(np.mean(truth==prediction)), 'records':len(truth),
            'classes_present':len(np.unique(truth))}


def selected_attack(train_x,train_y,selection_x,selection_y,test_x,test_y,classes):
    bank=classifier_bank(train_x,train_y)
    selection={name:score_classification(selection_y,model.predict(selection_x),classes)
               for name,model in bank.items()}
    winner=min(bank,key=lambda name:(-selection[name]['balanced_accuracy'],
                                   -selection[name]['macro_f1'],name))
    results={name:score_classification(test_y,model.predict(test_x),classes)
             for name,model in bank.items()}
    return {'selected_attack':winner,'selection_scores':selection,
            'test':results[winner],'all_test_descriptive':results}
