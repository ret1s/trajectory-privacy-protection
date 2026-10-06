"""Posthoc OLD-development proxy/Recall association; never open fresh artifacts.

No model, planner, threshold or private sample is selected or modified. Events
stay nested in draws and families; descriptive correlations carry no tick-level
confidence intervals and do not certify a calibrated posterior or causality.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
SOURCE='experiments/qplanner_proxy_diagnostic_20261006.py'
OLD='artifacts/benchmarks/qplanner_development_20261006_v3'
OUT='artifacts/benchmarks/qplanner_proxy_diagnostic_development_20261006_v1'
PURPOSES=('nearest_distance','fastest_travel','within_radius','minimum_detour')
NORMALIZED=('normalized_mean','normalized_tight','normalized_tail')
PHASES=('all','cold','early_0_180','temporal_tail_400_600')


def read(path):
    path=Path(path);data=path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix=='.gz' else data)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def mean(values):
    values=[float(v) for v in values if v is not None]
    return float(np.mean(values)) if values else None


def correlation(a,b,*,rank=False):
    pairs=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
    if len(pairs)<2:return None
    x,y=np.array(pairs,float).T
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite descriptive values')
    if rank:x,y=rankdata(x),rankdata(y)
    if np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])


def family_nested(rows,field):
    """Every family receives equal weight; its nested draws receive equal weight."""
    groups=defaultdict(list)
    for row in rows:groups[row['family_id']].append(row[field])
    values={f:mean(v) for f,v in sorted(groups.items())}
    return dict(mean=mean(values.values()),family_values=values,
                defined_families=sum(v is not None for v in values.values()),families=len(values))


def movement_m(previous,current):
    if previous is None:return None
    assert len(previous)==len(current)==5
    a,b=np.radians(np.asarray(previous,float)),np.radians(np.asarray(current,float))
    d=b-a;h=np.sin(d[:,0]/2)**2+np.cos(a[:,0])*np.cos(b[:,0])*np.sin(d[:,1]/2)**2
    return float(np.mean(6371008.8*2*np.arcsin(np.sqrt(np.clip(h,0,1)))))


def in_phase(row,phase):
    return phase=='all' or (phase=='cold' and row['slot']==0) or (phase=='early_0_180' and row['t']<=180) or (phase=='temporal_tail_400_600' and row['t']>=400)


def event_metrics(bundle):
    """Join saved objectives and current local utilities at the SAME public event."""
    truth=bundle['evaluator_only'];methods=tuple(truth['sessions'][0]['ledger'])
    scores={(r['method'],r['slot'],r['t']):r for r in bundle['utility'] if r['cache']=='current'}
    assert len(scores)==sum(r['cache']=='current' for r in bundle['utility'])
    rows=[]
    for method in methods:
        sessions=bundle['public']['streams'][method]
        assert len(sessions)==len(truth['sessions'])==8
        for slot,(public,private) in enumerate(zip(sessions,truth['sessions'])):
            ledger=private['ledger'][method];objectives=ledger['planner_objectives']
            assert len(objectives)==len(public['events'])==len(ledger['states'])
            previous=None
            for event,obj,states in zip(public['events'],objectives,ledger['states']):
                t=event['timestamp_s'];utility=scores[method,slot,t]
                assert utility['event_id']==event['event_id']
                purposes=utility['purposes'];assert tuple(purposes)==PURPOSES
                actual={p:purposes[p]['recall5'] for p in PURPOSES}
                for p in PURPOSES:
                    assert (actual[p] is None)==(purposes[p]['reference_category_count']==0)
                qs=[(q['lat'],q['lon']) for q in event['candidates']]
                normalized=method in NORMALIZED;history=obj.get('risk_history',[])
                if normalized:assert history and history[-1]['states']==states==obj['risk_states']
                rows.append(dict(family_id=truth['family_id'],split=truth['split'],draw=truth['draw'],
                    method=method,slot=slot,t=t,actual_macro=mean(actual.values()),**actual,
                    proxy_mean=obj['mean_value'] if normalized else None,
                    proxy_lower_tail=obj['lower_tail_cvar'] if normalized else None,
                    signed_actual_minus_proxy=mean(actual.values())-obj['mean_value'] if normalized else None,
                    absolute_actual_minus_proxy=abs(mean(actual.values())-obj['mean_value']) if normalized else None,
                    empty_profile_mass=obj.get('empty_profile_mass'),
                    reachable_count_mean=mean(obj['reachable_counts']),quotient_count_mean=mean(obj['quotient_counts']),
                    straight_line_Q_movement_mean_m=movement_m(previous,qs),
                    risk_accepted_exchanges=max(len(history)-1,0) if normalized else None,
                    risk_changed=float(history[0]['states']!=history[-1]['states']) if normalized else None,
                    risk_proxy_objective_gain=history[-1]['objective']-history[0]['objective'] if normalized else None,
                    risk_proxy_mean_gain=history[-1]['mean']-history[0]['mean'] if normalized else None,
                    risk_proxy_lower_tail_gain=history[-1]['lower_tail_cvar']-history[0]['lower_tail_cvar'] if normalized else None,
                    risk_candidate_evaluations=obj.get('risk_candidate_evaluations'),risk_termination=obj.get('risk_termination')))
                previous=qs
    return rows


FIELDS=('actual_macro',*PURPOSES,'proxy_mean','proxy_lower_tail','signed_actual_minus_proxy',
    'absolute_actual_minus_proxy','empty_profile_mass','reachable_count_mean','quotient_count_mean',
    'straight_line_Q_movement_mean_m','risk_accepted_exchanges','risk_changed','risk_proxy_objective_gain',
    'risk_proxy_mean_gain','risk_proxy_lower_tail_gain','risk_candidate_evaluations')


def aggregate(rows,methods):
    output={};family_draw=[]
    for split in ('train','selection'):
        for method in methods:
            for phase in PHASES:
                subset=[r for r in rows if r['split']==split and r['method']==method and in_phase(r,phase)]
                groups=defaultdict(list)
                for row in subset:groups[row['family_id'],row['draw']].append(row)
                local=[]
                for (family,draw),items in sorted(groups.items()):
                    metrics={field:mean(r[field] for r in items) for field in FIELDS}
                    metrics.update(proxy_actual_event_pearson=correlation([r['proxy_mean'] for r in items],[r['actual_macro'] for r in items]),
                        proxy_actual_event_spearman=correlation([r['proxy_mean'] for r in items],[r['actual_macro'] for r in items],rank=True))
                    record=dict(family_id=family,draw=draw,split=split,method=method,phase=phase,events=len(items),**metrics)
                    local.append(record);family_draw.append(record)
                summary={field:family_nested(local,field) for field in (*FIELDS,'proxy_actual_event_pearson','proxy_actual_event_spearman')}
                pa=summary['proxy_mean']['family_values'];aa=summary['actual_macro']['family_values']
                summary['between_family_proxy_actual_pearson']=correlation(list(pa.values()),list(aa.values()))
                summary['within_draw_actual_macro']={str(d):mean(r['actual_macro'] for r in local if r['draw']==d) for d in sorted(set(r['draw'] for r in local))}
                summary['event_inventory']={f'draw{d}':sum(r['events'] for r in local if r['draw']==d) for d in sorted(set(r['draw'] for r in local))}
                summary['interpretation']='Descriptive equal-family / equal-nested-draw statistics; event correlations averaged locally, no tick-level inference'
                output[f'{method}--{split}--{phase}']=summary
    return output,family_draw


def paired_diagnostics(rows):
    output={}
    for left,right in (('normalized_tight','normalized_mean'),('normalized_tail','normalized_tight')):
        for split in ('train','selection'):
            for phase in PHASES:
                def index(method):return {(r['family_id'],r['draw'],r['slot'],r['t']):r for r in rows if r['method']==method and r['split']==split and in_phase(r,phase)}
                a,b=index(left),index(right);assert a.keys()==b.keys()
                groups=defaultdict(list)
                for key in a:
                    x,y=a[key],b[key]
                    assert all((x[p] is None)==(y[p] is None) for p in PURPOSES)
                    dp=x['proxy_mean']-y['proxy_mean'];da=x['actual_macro']-y['actual_macro']
                    groups[key[:2]].append(dict(proxy_mean_delta=dp,actual_macro_delta=da,
                        proxy_tail_delta=x['proxy_lower_tail']-y['proxy_lower_tail'],
                        positive_proxy_nonpositive_actual=float(dp>1e-12 and da<=1e-12),positive_proxy=float(dp>1e-12),
                        nonzero_actual=float(abs(da)>1e-12)))
                local=[dict(family_id=f,draw=d,**{field:mean(r[field] for r in items) for field in items[0]}) for (f,d),items in sorted(groups.items())]
                summary={field:family_nested(local,field) for field in local[0] if field not in ('family_id','draw')}
                summary['within_draw_actual_delta']={str(d):mean(r['actual_macro_delta'] for r in local if r['draw']==d) for d in sorted(set(r['draw'] for r in local))}
                summary['family_draws']=local
                summary['scope']='Paired SAME public event; trajectories have different prior Q states/feasible buckets, so between-arm changes do not isolate a causal exchange effect'
                output[f'{left}--minus--{right}--{split}--{phase}']=summary
    return output


def run(output=ROOT/OUT):
    output=Path(output);source=ROOT/OLD
    if output.exists() and any(output.iterdir()):raise FileExistsError('New write-once diagnostic output required')
    protocol=read(source/'protocol.json');generation=read(source/'generation.json')
    assert protocol['splits']==['train','selection'] and protocol['draws_by_split']=={'train':1,'selection':3}
    execution=read(source/'execution_protocol.json')
    assert generation['execution_protocol_sha256']==sha(source/'execution_protocol.json')
    assert execution['common_protocol_sha256']==sha(source/'protocol.json')
    plan=dict(schema='qplanner-old-development-proxy-diagnostic-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        source_output=OLD,source_protocol_sha256=sha(source/'protocol.json'),source_generation_sha256=sha(source/'generation.json'),
        source_family_files_sha256=generation['family_files_sha256'],analysis_source=SOURCE,analysis_source_sha256=sha(ROOT/SOURCE),
        methods=list(protocol['configuration']['methods']),splits=protocol['splits'],draws_by_split=protocol['draws_by_split'],
        comparison_pairs=[['normalized_tight','normalized_mean'],['normalized_tail','normalized_tight']],phases=PHASES,
        definition='Actual per-event average of defined purpose Recall@5; compare to normalized public posterior/destination-mixture mean, aggregate events inside each draw then equal draws inside equal-weight family',
        correlation='Descriptive within family-draw Pearson/Spearman averaged equally; separate between-family association; no p-values or tick-independent CI',
        controls='Retain all controls actual Recall; their differently-defined objective values are not comparable and remain null',
        posthoc=True,fresh_data_or_scores_opened=False,defense_or_threshold_selected=False,
        limits=['Same already-inspected development cohorts; no calibrated posterior/cause proof',
            'Proxy public 9-destination prior differs from actual local destination; finite latent road belief differs from true current position',
            'No saved counterfactual local Recall for within-step risk baseline/alternative sets',
            'Q movement is mean same-index straight-line displacement, not road-path feasibility or identity matching'])
    save_new(output/'analysis_protocol.json',plan)
    snapshot=output/'source_snapshot'/SOURCE;snapshot.parent.mkdir(parents=True);snapshot.write_bytes((ROOT/SOURCE).read_bytes())
    rows=[]
    for name,digest in generation['family_files_sha256'].items():
        assert sha(source/'families'/name)==digest
        bundle=read(source/'families'/name);assert bundle['evaluator_only']['split'] in ('train','selection')
        rows.extend(event_metrics(bundle))
    summary,local=aggregate(rows,plan['methods']);paired=paired_diagnostics(rows)
    value=dict(schema='qplanner-old-development-proxy-diagnostic-readout-v1',analysis_protocol_sha256=sha(output/'analysis_protocol.json'),
        scope=plan['limits'],events=len(rows),family_draw_method_phase_metrics=local,summary=summary,paired_diagnostics=paired,
        fresh_data_or_scores_opened=False,defense_or_threshold_selected=False)
    save_new(output/'diagnostic.json',value)
    print(json.dumps({'output':str(output.relative_to(ROOT)),'events':len(rows),'diagnostic_sha256':sha(output/'diagnostic.json')},indent=2))
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/OUT)
    run(parser.parse_args().output)
