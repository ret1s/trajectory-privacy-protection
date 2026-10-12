"""Supplemental descriptive/paired analysis; never chooses or tunes the defense."""
from pathlib import Path
from fractions import Fraction
import argparse,json
import numpy as np
from experiments import cache_balance_study_20261012_v2 as study
from benchmark.late_prefix_bounds import late_change_ratio
from benchmark.finite_session_bounds import posterior_mass_bound


def analyze(output=study.OUT):
    output=Path(output);study.validate_sources(output);families=study.load_families(output)
    test=[f for f in families if f['split']=='test'];primary=study.CONFIG['primary'];purposes=[p.value for p in study.QueryPurpose]
    bootstrap=np.random.default_rng(study.CONFIG['bootstrap_seed']).integers(0,len(test),(10000,len(test)))
    def means(method,cache,lo=0,hi=600):
        result=[]
        for family in test:
            row={p:float(np.mean([v[cache][p]['recall5'] for t in family['trips'] for v in t['utility'][method] if lo<=v['t']<=hi and v[cache][p]['recall5'] is not None])) for p in purposes}
            row['macro']=float(np.mean(list(row.values())));result.append(row)
        return result
    comparisons=[];reference=means(primary,'session')
    for control in study.METHODS:
        if control==primary:continue
        other=means(control,'session');metrics={}
        for p in purposes+['macro']:
            delta=np.array([a[p]-b[p] for a,b in zip(reference,other)])
            metrics[p]=dict(delta=float(delta.mean()),paired95=np.quantile(delta[bootstrap].mean(axis=1),[.025,.975]).tolist())
        comparisons.append(dict(control=control,metrics=metrics))
    cache_deltas={}
    for cache in ('current','known'):
        other=means(primary,cache);delta=np.array([a['macro']-b['macro'] for a,b in zip(reference,other)])
        cache_deltas[cache]=dict(delta=float(delta.mean()),paired95=np.quantile(delta[bootstrap].mean(axis=1),[.025,.975]).tolist())
    intervals={p:np.quantile(np.array([r[p] for r in reference])[bootstrap].mean(axis=1),[.025,.975]).tolist() for p in purposes+['macro']}
    periods=[]
    for lo,hi in ((0,60),(80,300),(320,600)):
        periods.append(dict(window=[lo,hi],means={m:float(np.mean([r['macro'] for r in means(m,'session',lo,hi)])) for m in ('baseline','finite_dominance_g1',primary)}))
    coverage={}
    for method in study.METHODS:
        counts=[(len(t['retrieved_POI_indices'][method][0]['session']),len(t['retrieved_POI_indices'][method][-1]['session'])) for f in test for t in f['trips']]
        coverage[method]=dict(mean_initial_POIs=float(np.mean([a for a,b in counts])),mean_final_POIs=float(np.mean([b for a,b in counts])),mean_new_POIs=float(np.mean([b-a for a,b in counts])),
            minimum_new_POIs=min(b-a for a,b in counts),maximum_new_POIs=max(b-a for a,b in counts))
    data=study.read(study.DATA);late=[]
    for f in data['families']:
        a,b=[data['traces'][s['session_id']] for s in f['evaluator_only']['sessions'] if s['day'] in (7,8)]
        tau=next(int(x['time_s']) for x,y in zip(a,b) if (x['lat'],x['lon'])!=(y['lat'],y['lon']))
        j=1+(tau+59)//60;r=late_change_ratio(1,j,11)
        late.append(dict(family_id=f['family_id'],first_different_native_GPS_s=tau,first_affected_selection=j,exact_conditional_pair_ratio=str(r),
            illustrative_equal_prior_posterior_cap=float(posterior_mass_bound(Fraction(1,2),r)),
            scope='fixed shared-prefix pair; causal protected producer; ideal uniform Q bits; NOT inferred from attacker score'))
    payload=dict(schema='balanced-cache-secondary-no-tuning-v1',primary=primary,cache='session',comparisons=comparisons,
        primary_mean_recall_paired_family95=intervals,same_wire_cache_deltas=cache_deltas,periods=periods,POI_coverage=coverage,
        shared_prefix_pair_certificates=late,shared_prefix_scope='evaluator-only conditional pair proof; not a generic wire-specific posterior estimate',
        confirmed_belief_actual_utility_advantage=False,all_recorded_results_retained=True)
    p=output/'secondary_readout.json'
    if p.exists():assert study.read(p)==payload
    else:study.save(p,payload)
    print(json.dumps({k:v for k,v in payload.items() if k not in ('shared_prefix_pair_certificates',)},indent=2))
    return payload

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=study.OUT);analyze(parser.parse_args().output)
