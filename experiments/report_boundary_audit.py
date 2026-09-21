"""Reanalyse saved boundary transcripts; standard library only, no new training."""
from pathlib import Path
from collections import defaultdict
from statistics import mean, median
from copy import deepcopy
import hashlib, json, math
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'artifacts/benchmarks/paper_benchmark/results.json'
OUT=ROOT/'artifacts/benchmarks/report_boundary_audit'
PROTOCOL=ROOT/'docs/supervisor_meeting/2026-09-26_brief/boundary_audit_protocol.md'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cut_public(public,scenario,drop):
    if scenario not in ('S9','S10') or drop not in (0,2,4):raise ValueError('Predeclared cases/grid only')
    events=public['events']
    if len(events)<=drop:raise ValueError('Empty release not allowed in this diagnostic')
    indices=list(range(drop,len(events))) if scenario=='S9' else list(range(len(events)-drop))
    return {**deepcopy(public),'events':[deepcopy(events[i]) for i in indices]},indices

def xy(lat,lon,lat0):
    m=math.pi/180*6371000
    return (lon*m*math.cos(math.radians(lat0)),lat*m)

def predictions(public,scenario,lat0,hidden_seconds):
    # This function cannot access target truth: public coordinates and fixed policy only.
    events=public['events'];centers=[]
    for e in events:
        pts=[xy(c['lat'],c['lon'],lat0) for c in e['candidates']]
        centers.append(tuple(mean(p[j] for p in pts) for j in (0,1)))
    first=scenario=='S9';at=0 if first else -1
    boundary=centers[at]
    pair=(0,min(2,len(events)-1)) if first else (max(0,len(events)-3),len(events)-1)
    a,b=pair;dt=events[b]['timestamp_s']-events[a]['timestamp_s']
    v=tuple((centers[b][j]-centers[a][j])/dt if dt else 0 for j in (0,1))
    sign=-1 if first else 1
    return {'boundary_centroid':boundary,'window_mean':tuple(mean(c[j] for c in centers) for j in (0,1)),
            'linear_boundary':tuple(boundary[j]+sign*hidden_seconds*v[j] for j in (0,1))}

def utility(source_row,indices):
    n=len(source_row['public']['events']);kept=set(indices)
    details=source_row['utility']['poi_rows'];cats=source_row['public']['query_categories']
    assert len(details)==n*len(cats)
    full=[];retained=[];complete=[]
    for ci,c in enumerate(cats):
        rows=details[ci*n:(ci+1)*n]
        assert all(r['category']==c for r in rows)
        for i,r in enumerate(rows):
            ref=set(r['reference']);got=set(r['returned'])
            if not ref:assert r['recall'] is None;continue
            recall=len(ref&got)/len(ref)
            assert math.isclose(recall,r['recall'],abs_tol=1e-12)
            full.append(recall if i in kept else 0.)
            if i in kept:retained.append(recall);complete.append(len(got)==len(ref))
    return dict(recall_all_original_queries=mean(full),recall_retained=mean(retained),
                returned_complete_retained=mean(complete),original_evaluable_queries=len(full),
                retained_evaluable_queries=len(retained),events_before=n,events_after=len(indices),
                no_cache=True,includes_original_hidden_60s=False)

def nr(xs,q):return sorted(xs)[max(0,math.ceil(q*len(xs))-1)]

def run():
    data=json.loads(SOURCE.read_text());lat0={m['seed']:m['projection_lat0'] for m in data['manifests']}
    input_hash=digest(SOURCE);rows=[];checks=0
    for r in data['rows']:
        if r['scenario'] not in ('S9','S10') or r['method'] not in ('unprotected','br_private') or r['k']!=5:continue
        assert r['status']=='ok' and len(r['public']['events'])==12
        times=[e['timestamp_s'] for e in r['public']['events']]
        assert all(math.isclose(b-a,20.) for a,b in zip(times,times[1:])) and r['checks']['mask_s']==60
        # Independently recalculate existing report errors with the same projection.
        target=xy(*r['hidden_target'],lat0[r['seed']]);at=0 if r['scenario']=='S9' else -1
        e=math.dist(r['attack_xy'][at],target)
        h=math.dist(r['hit_attack_xy'][at],target)
        assert math.isclose(e,r['metrics']['location_mae_m'],abs_tol=1e-7)
        assert math.isclose(float(h<=100),r['metrics']['location_hit_100m'],abs_tol=1e-12)
        checks+=1
        for drop in (0,2,4):
            public,indices=cut_public(r['public'],r['scenario'],drop)
            assert len(public['events'])==12-drop
            if drop==0:assert public==r['public']
            pred=predictions(public,r['scenario'],lat0[r['seed']],60+20*drop)
            rows.append(dict(seed=r['seed'],record_id=r['record_id'],scenario=r['scenario'],method=r['method'],
                extra_cut_events=drop,extra_cut_s=20*drop,source_event_indices=indices,
                predictions_xy={k:list(v) for k,v in pred.items()},target_xy=list(target),
                error_m={k:math.dist(v,target) for k,v in pred.items()},utility=utility(r,indices)))
    assert len(rows)==144 and checks==48 and digest(SOURCE)==input_hash
    groups=defaultdict(list)
    for r in rows:groups[r['scenario'],r['method'],r['extra_cut_events']].append(r)
    summaries=[]
    for (s,m,d),group in sorted(groups.items()):
        assert len(group)==12 and len({r['seed'] for r in group})==3
        attacks={}
        for a in group[0]['error_m']:
            e=[r['error_m'][a] for r in group]
            attacks[a]={'mae_m':mean(e),'median_m':median(e),'p90_m':nr(e,.9),
                        **{f'hit{b}':mean(x<=b for x in e) for b in (50,100,200)}}
        summaries.append(dict(scenario=s,method=m,extra_cut_events=d,extra_cut_s=20*d,records=12,simulation_seeds=3,
            attacks=attacks,envelope_mae_m=min(v['mae_m'] for v in attacks.values()),
            envelope_hit100=max(v['hit100'] for v in attacks.values()),
            utility={k:mean(r['utility'][k] for r in group) for k in ['recall_all_original_queries','recall_retained','events_after']}))
    result={'schema':'report-boundary-audit-v1','status':'exploratory_saved_transcript_reanalysis_not_new_confirmation',
            'source':str(SOURCE.relative_to(ROOT)),'source_sha256':input_hash,'protocol_sha256':digest(PROTOCOL),
            'code_sha256':digest(Path(__file__)),'verified_source_rows':checks,'rows':rows,'summaries':summaries}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'results.sha256').write_text(digest(OUT/'results.json')+'\n')
    print(f'Verified {checks} original rows; evaluated {len(rows)} diagnostic rows.')
    for s in summaries:print(s['scenario'],s['method'],s['extra_cut_s'],round(s['envelope_mae_m'],1),round(100*s['envelope_hit100'],1),round(100*s['utility']['recall_all_original_queries'],1))
if __name__=='__main__':run()
