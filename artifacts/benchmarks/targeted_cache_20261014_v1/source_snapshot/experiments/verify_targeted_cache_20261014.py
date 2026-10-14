"""Independent public-cache, support, exact-PMF and actual-utility audit."""
import argparse,json
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.spatial import cKDTree
from benchmark.anchor_belief import AnchorBelief,PublicAnchorModel
from benchmark.defined_service_scores import DefinedPublicPurposeTable
from benchmark.public_pool_portfolio import PublicPoolPortfolio,FiniteSegmentPolicy
from benchmark.query_segments import PublicSegmentLibrary
from benchmark.dominance_motion import DominanceMotionLibrary
from benchmark.cache_exploration import ExploreMotionLibrary,CacheExplorePolicy,mask_bits
from benchmark.finite_query_kernel import FiniteCenteredKernel,quantize_scores
from benchmark.targeted_cache_exploration import TargetedExploreLibrary,TargetedCachePolicy
from experiments import targeted_cache_study_20261014 as study
from experiments import query_bundle_audit_20261010 as public


def verify(output=study.OUT):
    output=Path(output);protocol=study.validate_sources(output);families=study.load_families(output)
    data=study.read(study.DATA);native=study.read(study.DATA.parent/'validation.json')
    assert native['passed'] and native['dataset_sha256']==study.sha(study.DATA)
    for path in (study.cohort.DEVELOPMENT,study.cohort.DEVELOPMENT_2):
        dev=study.read(path)
        assert not ({f['public_junction_group'] for f in dev['families']}&{f['public_junction_group'] for f in data['families']})
    rn,ref,reply,ids=public.resources(study.WORK/'public_resources');meta=study.read(output/'resources.json')
    assert meta['catalogue_sha256']==rn.catalogue_sha256 and meta['reply_sha256']==reply.sha256
    table=DefinedPublicPurposeTable(ref,reply,ids,destinations=meta['public_destinations'])
    pool=PublicPoolPortfolio(reply,public_poi_weights=table.public_poi_weights);goals=pool.goals(width=24)
    assert [list(g) for g in goals]==meta['public_portfolio_goals']
    initial=FiniteSegmentPolicy(PublicSegmentLibrary(rn,goals),table,pool,use_simulator=True)
    dominance=FiniteSegmentPolicy(DominanceMotionLibrary(rn,goals,pool),table,pool,use_simulator=True)
    exploration=CacheExplorePolicy(ExploreMotionLibrary(rn,goals,pool),table,pool,use_simulator=True,novelty_weight=4)
    targeted=TargetedCachePolicy(TargetedExploreLibrary(rn,goals,pool,travel_scale_s=60),table,pool,use_simulator=True,novelty_weight=4,long_range_weight=4)
    _,inv,counts=np.unique(np.floor(rn.xy/120.).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    prior=1./counts[inv];prior/=prior.sum()
    model=PublicAnchorModel(rn,ref,prior,spacing_m=200.,epsilon_release=.01,epsilon_test=.01,cache_path=study.WORK/'public_resources'/'belief01.npz')
    assert model.sha256==meta['belief_sha256'];projection=cKDTree(rn.xy[ids]).query(model.xy)[1]
    utility=study.Utility(ref,reply);decisions=0;cache_checks=0;utility_checks=0;relative_checks=0;ratios=set();strict_cache_expansions=0
    for family in families:
        assert len(family['trips'])==2*study.CONFIG['draws_by_split'][family['split']]
        for trip in family['trips']:
            assert all(trip['cache_restart_checks'].values())
            for m in ('cache_explore_g1','cache_target_g1'):assert trip['sampled_states'][m][:3]==trip['sampled_states']['finite_static_g1'][:3]
            b=AnchorBelief(model);last_read=None
            seen={m:0 for m in study.METHODS};epoch_seen={m:0 for m in study.METHODS};ratio={m:Fraction(1) for m in study.METHODS[1:]}
            by_time={int(p['time_s']):p for p in data['traces'][trip['native_session_id']]}
            destination=rn.nearest(by_time[600]['lat'],by_time[600]['lon'])[0]
            for tick,(anchor,ledger) in enumerate(zip(trip['GeoI_anchors'],trip['GeoI_ledger'])):
                t=tick*20;truth=trip['truth'][tick];point=by_time[t]
                assert truth['state']==rn.nearest(point['lat'],point['lon'])[0]
                assert truth['xy']==list(rn.point_xy(point['lat'],point['lon']))
                if ledger['private_read']:assert last_read is None or t-last_read>=60;last_read=t
                assert ledger['spent_units']<=23 and trip['GeoI_spent_per_m']<=.23+1e-12
                weights=b.update(anchor,float(t),observed=ledger['private_read']);projected=np.bincount(projection,weights=weights,minlength=len(ids));projected/=projected.sum()
                for method in study.METHODS:
                    if method!='baseline':
                        cert=trip['Q_certificates'][method][tick];block=tick//3+1
                        allowed=study.FractionGamma(method)/Fraction(block*(block+1))
                        assert 0<=Fraction.from_float(cert['epsilon_Q_upper'])<=allowed
                        assert cert['reserved_prefix_epsilon_Q']<=float(study.FractionGamma(method))
                        if tick%3:
                            first=trip['Q_certificates'][method][tick-tick%3]
                            assert {k:v for k,v in cert.items() if k!='t'}=={k:v for k,v in first.items() if k!='t'}
                        elif method=='finite_static_g1' and tick:
                            assert cert['epsilon_Q_upper']==0 and cert['exact_likelihood_ratio_upper']=='1'
                            assert trip['sampled_states'][method][tick]==trip['sampled_states'][method][0]
                        elif method!='legacy_segment_g1':
                            previous=tuple(trip['sampled_states'][method][tick-1]) if tick else None
                            if method.startswith('cache_') and tick:
                                actions,g,d,masses=(targeted if method.startswith('cache_target') else exploration).public_tables(previous,seen[method],3)
                                assert list(masses)==cert['public_base_masses'] and list(d)==cert['public_novel_POI_counts']
                                mu=sum(Fraction(x) for x in d)/len(d);var=sum((Fraction(x)-mu)**2 for x in d)/len(d)
                                if method=='cache_explore_g1':assert sum(Fraction(x*w,sum(masses)) for x,w in zip(d,masses))==mu+4*var/(1+4*mu)
                            else:
                                actions,g,_=(dominance if method=='finite_dominance_g1' else initial).public_tables(previous,3);masses=[1]*len(actions)
                            use_b=np.full(len(ids),1./len(ids)) if method=='cache_explore_prior_g1' else projected
                            values=np.einsum('i,ij->j',use_b,g,optimize=False);f=quantize_scores(values)
                            assert values.tolist()==cert['planning_scores_binary64'] and list(f)==cert['integer_scores']
                            kernel=FiniteCenteredKernel(masses,Fraction.from_float(cert['epsilon_Q_upper']))
                            assert sum(kernel.integer_weights(f))==kernel.public_total==int(cert['public_draw_total'])
                            assert str(kernel.alpha)==cert['alpha'] and Fraction(cert['exact_likelihood_ratio_upper'])==1+kernel.epsilon
                            for name,value in kernel.utility_certificate(f,planning_scores=values).items():assert cert[name]==value
                            assert Fraction(cert['exact_planning_score_gain'])>=0
                            observed=tuple(tuple(s) for s in trip['sampled_states'][method][tick:min(tick+3,31)])
                            assert any(a.frames[:len(observed)]==observed for a in actions)
                            assert cert['public_library_size']==len(actions);ratio[method]*=1+kernel.epsilon;decisions+=1
                    frame=trip['sampled_states'][method][tick];current=0
                    for state in frame:
                        snapped=rn.nearest(*rn.latlon(state))[0];assert pool.state_pool[state]==pool.state_pool[snapped]
                        current|=pool.state_pool[snapped]
                    if tick%3==0:epoch_seen[method]=0
                    epoch_seen[method]|=current;old_seen=seen[method];seen[method]|=current
                    strict_cache_expansions+=int(method=='cache_target_g1' and seen[method]!=old_seen)
                    for cache,bits in (('current',current),('known',epoch_seen[method]),('session',seen[method])):
                        indices=np.flatnonzero(pool.mask(bits)).tolist()
                        assert indices==trip['retrieved_POI_indices'][method][tick][cache];cache_checks+=1
                        actual=utility.score(truth['state'],destination,pool.mask(bits))
                        assert actual==trip['utility'][method][tick][cache];utility_checks+=1
                    if method=='cache_target_g1':
                        assert pool.quality_exact(seen[method])>=initial.quality_floor_exact
                        static=trip['utility']['finite_static_g1'][tick]['session']
                        for purpose,values in trip['utility'][method][tick]['session'].items():
                            x,y=values['recall5'],static[purpose]['recall5'];assert (x is None)==(y is None)
                            if x is not None:assert x>=y;relative_checks+=1
            assert ratio['cache_target_g0']==1;ratios.add(str(ratio['cache_target_g1']))
    results=study.read(output/'results.json')
    assert results['utility']==study.utility_readout(families) and results['path_audit']==study.audit_paths(families,rn)
    test_ids={f['family_id'] for f in families if f['split']=='test'}
    for row in results['attackers']:
        stem=row['method']+'--'+row['scenario'];selection=study.read(output/'attacker_selection'/(stem+'.json'))
        assert selection['fit_split']=='train' and selection['selection_split']=='selection' and selection['before_test_predictions']
        assert selection['model_sha256']==study.sha(output/'attacker_models'/(stem+'.pkl.gz')) and row['selected_attacker']==selection['selected']
        predictions=study.read(output/'attacker_predictions'/(stem+'.json.gz'));assert len(predictions)==120 and {p['family_id'] for p in predictions}==test_ids
        mae=[];hit=[]
        for fid in sorted(test_ids):
            group=[p for p in predictions if p['family_id']==fid];assert len(group)==6
            mae.append(np.mean([np.mean(p['errors'][selection['selected']['mae']]) for p in group]))
            hit.append(np.mean([np.mean(np.array(p['errors'][selection['selected']['hit100']])<=100) for p in group]))
        assert abs(row['MAE_m']-np.mean(mae))<1e-10 and abs(row['Hit100']-np.mean(hit))<1e-14
    receipt=dict(status='pass',protocol_sha256=study.sha(output/'protocol.json'),results_sha256=study.sha(output/'results.json'),verifier_sha256=study.sha(Path(__file__)),native_validation_sha256=study.sha(study.DATA.parent/'validation.json'),
        exact_selection_decisions_checked=decisions,cache_masks_rebuilt=cache_checks,actual_utility_recomputed=utility_checks,actual_utility_vs_static_checks=relative_checks,
        primary_strict_cache_expansions=strict_cache_expansions,primary_prefix_ratio=sorted(ratios),
        assumptions='static valid metadata; public clocks; trusted checkpoint; fresh independent uniform bits for deployment',PCG_information_theoretic_entropy_certified=False,GeoI_float_certified=False,
        files_sha256={str(p.relative_to(output)):study.sha(p) for folder in ('attacker_models','attacker_selection','attacker_predictions') for p in sorted((output/folder).iterdir())})
    path=output/'independent_validation.json'
    if path.exists():assert study.read(path)==receipt
    else:study.save(path,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='files_sha256'},indent=2));return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=study.OUT);verify(parser.parse_args().output)
