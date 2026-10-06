"""Preserved development iteration: public-road attacks and linkage thresholds.

This follows inspection of initial controls, so its reused test families are
development readouts, not independent confirmation. Selection still never fits
on those test rows. Initial failed controls are retained beside this output.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from evaluation.identity_future import (public_arrays,xy_from_latlon,prefix_features,pair_features,
    ClassifierBank,classification_metrics,location_metrics)
from experiments.identity_future_eval import ROOT,OUT,SPLITS,SEED,save,sha


def shape_features(left,right):
    """Set-track matching on public coordinates; invariant to candidate names."""
    def paths(public):
        centre,spread,times=public_arrays(public)
        events=public['events'];k=len(events[0]['candidates'])
        if any(len(e['candidates'])!=k for e in events):raise ValueError('Stable public K required')
        xy=np.array([xy_from_latlon([[c['lat'],c['lon']] for c in e['candidates']]) for e in events])
        clock=times/max(1.,times[-1]);grid=np.linspace(0,1,12)
        tracks=np.array([[np.interp(grid,clock,xy[:,j,axis]) for axis in range(2)] for j in range(k)]).transpose(0,2,1)
        return tracks/1000.,centre/1000.,spread/1000.,times
    a,ac,asp,at=paths(left);b,bc,bsp,bt=paths(right)
    distance=np.linalg.norm(a[:,None,:,:]-b[None,:,:,:],axis=3).mean(axis=2)
    i,j=linear_sum_assignment(distance);matched=distance[i,j]
    # Absolute geography removed except pair differences; transfer to new families.
    return np.r_[matched.mean(),matched.min(),matched.max(),np.quantile(matched,[.25,.5,.75]),
        np.linalg.norm(ac.mean(axis=0)-bc.mean(axis=0)),
        np.abs(ac.std(axis=0)-bc.std(axis=0)),np.abs(asp.mean(axis=0)-bsp.mean(axis=0)),
        abs(at[-1]-bt[-1])/60.,abs(len(at)-len(bt)),
        np.abs((a[:,-1]-a[:,0]).mean(axis=0)-(b[:,-1]-b[:,0]).mean(axis=0))]


def calibrated_linkage(rows,label):
    parts={s:[r for r in rows if r['split']==s] for s in SPLITS}
    truth={s:np.array([r[label] for r in v]) for s,v in parts.items()}
    predictions={};selections={};banks={}
    for feature_name,feature in [('summary',pair_features),('shape',shape_features)]:
        x={s:np.array([feature(*r['public_pair']) for r in p]) for s,p in parts.items()}
        model=ClassifierBank(x['train'],truth['train'],SEED);banks[feature_name]=(model,x)
        for n,p in model.probability(x['selection']).items():
            key=feature_name+'/'+n
            # Fixed public grid and tie-rule; no threshold picked on test values.
            threshold=min(np.linspace(0.,1.,21),key=lambda t:(
                -classification_metrics(truth['selection'],p>=t,p)['balanced_accuracy'],abs(t-.5),t))
            selections[key]={'threshold':float(threshold),
                'metrics':classification_metrics(truth['selection'],p>=threshold,p)}
            test=model.probability(x['test'])[n]
            predictions[key]=(test>=threshold,test)
    selected=min(selections,key=lambda n:(-selections[n]['metrics']['balanced_accuracy'],
                    -selections[n]['metrics']['roc_auc'],n))
    pred,prob=predictions[selected]
    result={'selected_attacker':selected,'selected_threshold':selections[selected]['threshold'],
        'selection':selections[selected]['metrics'],'test':classification_metrics(truth['test'],pred,prob),
        'bank_test_descriptive_only':{n:classification_metrics(truth['test'],v,p) for n,(v,p) in predictions.items()},
        'test_family_metrics':{f:classification_metrics(
            [r[label] for r in parts['test'] if r['family_id']==f],
            [v for r,v in zip(parts['test'],pred) if r['family_id']==f],
            [v for r,v in zip(parts['test'],prob) if r['family_id']==f]) for f in SPLITS['test']},
        'test_predictions':pred.astype(int).tolist(),'test_truth':truth['test'].tolist()}
    # Negative control uses one fixed block-label shuffle with all paired rows.
    labels=truth['train'].copy();rng=np.random.default_rng(SEED);rng.shuffle(labels)
    model,x=banks[selected.split('/')[0]]
    control=ClassifierBank(x['train'],labels,SEED)
    ps,pt=control.probability(x['selection']),control.probability(x['test'])
    controls=[]
    for n,p in ps.items():
        threshold=min(np.linspace(0.,1.,21),key=lambda t:(
            -classification_metrics(truth['selection'],p>=t,p)['balanced_accuracy'],abs(t-.5),t))
        controls.append((n,threshold,classification_metrics(truth['selection'],p>=threshold,p)))
    n,t,_=min(controls,key=lambda item:(-item[2]['balanced_accuracy'],-item[2]['roc_auc'],item[0]))
    result['permuted_training_control']={'selected_attacker':n,'threshold':float(t),
        'test':classification_metrics(truth['test'],pt[n]>=t,pt[n])}
    return result


def public_road_catalogue():
    """Densify all archived public lines at <=20m; no GPS truth input."""
    roads=json.loads((ROOT/'artifacts/benchmarks/paper_benchmark/results.json').read_text())['roads']
    points=[]
    for line in roads:
        xy=xy_from_latlon([[lat,lon] for lon,lat in line])
        for a,b in zip(xy,xy[1:]):
            steps=max(1,int(np.ceil(np.linalg.norm(b-a)/20.)))
            points.extend(a+(b-a)*t for t in np.linspace(0,1,steps+1))
    return cKDTree(np.unique(np.round(points,6),axis=0))


def geometry_future(rows):
    tree=public_road_catalogue();results={}
    for method in sorted({r['method'] for r in rows}):
        subset=[r for r in rows if r['method']==method and r['next_xy'] is not None]
        parts={s:[r for r in subset if r['split']==s] for s in ('selection','test')}
        candidates={};truth={s:np.array([r['next_xy'] for r in p]) for s,p in parts.items()}
        for s,p in parts.items():
            centres=[public_arrays(r['public']) for r in p]
            last=np.array([c[-1] for c,_,_ in centres])
            full=np.array([c[-1]+20*(c[-1]-c[0])/max(1.,t[-1]) for c,_,t in centres])
            recent=np.array([c[-1]+20*(c[-1]-c[-2])/max(1.,t[-1]-t[-2]) if len(c)>1 else c[-1]
                             for c,_,t in centres])
            bank={'last':last,'full_velocity':full,'recent_velocity':recent}
            candidates[s]={**bank,**{n+'_public_road':tree.data[tree.query(v)[1]] for n,v in bank.items()}}
        selected=min(candidates['selection'],key=lambda n:(location_metrics(truth['selection'],candidates['selection'][n])['mae_m'],n))
        results[method]={'selected_attacker':selected,
            'selection_mae_m':location_metrics(truth['selection'],candidates['selection'][selected])['mae_m'],
            'test':location_metrics(truth['test'],candidates['test'][selected]),
            'bank_test_descriptive_only':{n:location_metrics(truth['test'],v) for n,v in candidates['test'].items()},
            'selection_count':len(parts['selection']),'test_count':len(parts['test'])}
    return {'public_catalogue_points':len(tree.data),'spacing_m':20.,'results':results,
        'scope':'S5 next20s-location proxy only; public polyline projection cannot restore original next-edge IDs.'}


def main():
    declaration={'schema':'identity-future-refinement-protocol-v1','splits':SPLITS,
        'scope':'Preserved development iteration after initial controls, not independent confirmation',
        'S4_attacker':'summary and public track-shape matching x trees/kNN1/5/15',
        'threshold':'fixed grid0,.05,...1, maximum selection balancedaccuracy, thresholdclosest.5thenlowest',
        'S5_attacker':'last,wholeprefixvelocity,recentvelocity,+projectionto20mpublicpolylinecatalogue',
        'selection':'selection families only; test observations/labels never fit, every bank value reported'}
    save(OUT/'refinement_protocol.json',declaration)
    linkage=json.loads((OUT/'linkage_results.json').read_text());sessions=linkage['evaluator_sessions']
    results={}
    for method in ('raw','geoi_slack_reconstructed'):
        rows=[]
        for item in linkage['pair_rows']:
            if item['method']!=method:continue
            pair=[next(t for t in sessions if t['session_id']==sid)[method] for sid in item['session_pair']]
            rows.append({**item,'public_pair':pair})
        results[method]={label:calibrated_linkage(rows,label) for label in ('same_person','same_vehicle')}
    future=json.loads((OUT/'future_results_v2.json').read_text())
    save(OUT/'refinement_results.json',{'schema':'identity-future-refinement-v1',
        'protocol_sha256':sha(OUT/'refinement_protocol.json'),'source_sha256':{
            'linkage_results.json':sha(OUT/'linkage_results.json'),
            'future_results_v2.json':sha(OUT/'future_results_v2.json'),
            'evaluation/identity_future.py':sha(ROOT/'evaluation/identity_future.py'),
            'experiments/identity_future_refine.py':sha(Path(__file__)),
            'artifacts/benchmarks/paper_benchmark/results.json':sha(ROOT/'artifacts/benchmarks/paper_benchmark/results.json')},
        'linkage_results':results,'S5_geometry':geometry_future(future['rows']),
        'scope':'Bounded empirical development attack iteration; no new protection configuration selected on test.'})


if __name__=='__main__':main()
