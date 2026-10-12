"""Independent frozen public-support, integer-PMF and heldout evidence audit."""
import argparse
from fractions import Fraction
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree

from benchmark.anchor_belief import AnchorBelief,PublicAnchorModel
from benchmark.finite_query_kernel import FiniteCenteredKernel,quantize_scores
from benchmark.defined_service_scores import DefinedPublicPurposeTable
from benchmark.public_pool_portfolio import PublicPoolPortfolio,FiniteSegmentPolicy
from benchmark.public_segment_scores import FixedPublicPurposeTable
from benchmark.query_segments import PublicSegmentLibrary
from experiments import finite_pool_study_20261012 as study
from experiments import query_bundle_audit_20261010 as public


def verify(output=study.OUT):
    output=Path(output);protocol=study.validate_sources(output)
    families=study.load_families(output);data=study.read(study.DATA)
    native=study.read(study.DATA.parent/'validation.json')
    assert native['passed'] and native['dataset_sha256']==study.sha(study.DATA)
    assert study.read(output/'generation.json')['dataset_sha256']==study.sha(study.DATA)
    assert {f['family_id']:f['split'] for f in families}=={f['family_id']:f['split'] for f in data['families']}
    rn,ref,reply,ids=public.resources(study.WORK/'public_resources')
    meta=study.read(output/'resources.json')
    assert meta['catalogue_sha256']==rn.catalogue_sha256 and meta['reply_sha256']==reply.sha256
    old=FixedPublicPurposeTable(ref,reply,ids,destinations=meta['public_destinations'])
    defined=DefinedPublicPurposeTable(ref,reply,ids,destinations=meta['public_destinations'])
    pool=PublicPoolPortfolio(reply,public_poi_weights=defined.public_poi_weights)
    old_goals=[a.states for a in public.library(ref,reply,ids) if len(a.states)==5]
    broad_goals=pool.goals(width=24)
    assert [list(g) for g in broad_goals]==meta['public_portfolio_goals']
    ordinary=FiniteSegmentPolicy(PublicSegmentLibrary(rn,old_goals),old,pool,broad=False,use_simulator=True,proximity_weight=0.)
    broad=FiniteSegmentPolicy(PublicSegmentLibrary(rn,broad_goals),defined,pool,use_simulator=True,proximity_weight=.1)
    assert str(broad.quality_floor_exact)==meta['exact_public_reference_mass_floor']
    _,inv,counts=np.unique(np.floor(rn.xy/120.).astype(np.int64),axis=0,return_inverse=True,return_counts=True)
    prior=1./counts[inv];prior/=prior.sum()
    model=PublicAnchorModel(rn,ref,prior,spacing_m=200.,epsilon_release=.01,epsilon_test=.01,
        cache_path=study.WORK/'public_resources'/'belief01.npz')
    assert model.sha256==meta['belief_sha256']
    projection=cKDTree(rn.xy[ids]).query(model.xy)[1]
    decisions=0;positive_gain=0;minimum_pool=Fraction(1);ratios=set()
    for family in families:
        assert len(family['trips'])==2*study.CONFIG['draws_by_split'][family['split']]
        for trip in family['trips']:
            belief=AnchorBelief(model);last_read=None
            expected_ratio={method:Fraction(1) for method in study.METHODS[1:]}
            for tick,(anchor,ledger) in enumerate(zip(trip['GeoI_anchors'],trip['GeoI_ledger'])):
                if ledger['private_read']:
                    assert last_read is None or tick*20-last_read>=60
                    last_read=tick*20
                assert ledger['spent_units']<=23 and trip['GeoI_spent_per_m']<=.23+1e-12
                b=belief.update(anchor,float(tick*20),observed=ledger['private_read'])
                projected=np.bincount(projection,weights=b,minlength=len(ids));projected/=projected.sum()
                for method in study.METHODS[1:]:
                    cert=trip['Q_certificates'][method][tick]
                    assert cert['reserved_prefix_epsilon_Q']<=float(study.FractionGamma(method))
                    if method=='legacy_segment_g1':continue
                    block=tick//3+1
                    allowed=study.FractionGamma(method)/Fraction(block*(block+1))
                    assert 0<=Fraction.from_float(cert['epsilon_Q_upper'])<=allowed
                    if tick%3:
                        first=trip['Q_certificates'][method][tick-tick%3]
                        assert {k:v for k,v in cert.items() if k!='t'}=={k:v for k,v in first.items() if k!='t'}
                        continue
                    policy=ordinary if method=='finite_original_g1' else broad
                    previous=tuple(trip['sampled_states'][method][tick-1]) if tick else None
                    actions,g,missing=policy.public_tables(previous,3)
                    use_b=np.full(len(ids),1./len(ids)) if method=='finite_portfolio_prior_g1' else projected
                    values=np.einsum('i,ij->j',use_b,g,optimize=False)
                    assert values.tolist()==cert['planning_scores_binary64']
                    f=quantize_scores(values,65536)
                    assert list(f)==cert['integer_scores']
                    kernel=FiniteCenteredKernel([1]*len(actions),Fraction.from_float(cert['epsilon_Q_upper']))
                    weights=kernel.integer_weights(f)
                    assert sum(weights)==kernel.public_total==int(cert['public_draw_total'])
                    assert min(weights)>0 and str(kernel.alpha)==cert['alpha']
                    assert Fraction(cert['exact_likelihood_ratio_upper'])==1+kernel.epsilon
                    expected_ratio[method]*=1+kernel.epsilon
                    cert_u=kernel.utility_certificate(f,planning_scores=values)
                    for name,value in cert_u.items():assert cert[name]==value
                    assert Fraction(cert['exact_planning_score_gain'])>=0
                    observed=tuple(tuple(s) for s in trip['sampled_states'][method][tick:min(tick+3,31)])
                    candidates=[a for a in actions if a.frames[:len(observed)]==observed]
                    assert candidates,'Observed Q not in PUBLIC filtered support'
                    assert cert['public_library_size']==len(actions)
                    if method!='finite_original_g1':
                        floor=Fraction(cert['exact_public_reference_mass_floor'])
                        assert floor==broad.quality_floor_exact
                        assert all(pool.quality_exact(pool.pool(frame))>=floor for a in candidates for frame in a.frames)
                        minimum_pool=min(minimum_pool,Fraction(cert['exact_selected_public_reference_mass']))
                    decisions+=1;positive_gain+=int(Fraction(cert['exact_planning_score_gain'])>0)
            assert expected_ratio['finite_portfolio_g0']==1
            ratios.add(str(expected_ratio['finite_portfolio_g1']))
    results=study.read(output/'results.json')
    assert results['utility']==study.utility_readout(families)
    assert results['path_audit']==study.audit_paths(families,rn)
    test_ids={f['family_id'] for f in families if f['split']=='test'}
    for row in results['attackers']:
        stem=row['method']+'--'+row['scenario']
        selection=study.read(output/'attacker_selection'/(stem+'.json'))
        assert selection['fit_split']=='train' and selection['selection_split']=='selection'
        assert selection['model_sha256']==study.sha(output/'attacker_models'/(stem+'.pkl.gz'))
        assert row['selected_attacker']==selection['selected']
        predictions=study.read(output/'attacker_predictions'/(stem+'.json.gz'))
        assert len(predictions)==144 and {p['family_id'] for p in predictions}==test_ids
        mae=[];hit=[]
        for family in sorted(test_ids):
            group=[p for p in predictions if p['family_id']==family];assert len(group)==6
            mae.append(np.mean([np.mean(p['errors'][selection['selected']['mae']]) for p in group]))
            hit.append(np.mean([np.mean(np.array(p['errors'][selection['selected']['hit100']])<=100) for p in group]))
        assert abs(row['MAE_m']-np.mean(mae))<1e-10 and abs(row['Hit100']-np.mean(hit))<1e-14
    names=[p for folder in ('attacker_models','attacker_selection','attacker_predictions') for p in (output/folder).iterdir()]
    receipt=dict(status='pass',protocol_sha256=study.sha(output/'protocol.json'),results_sha256=study.sha(output/'results.json'),
        verifier_sha256=study.sha(Path(__file__)),native_validation_sha256=study.sha(study.DATA.parent/'validation.json'),
        exact_integer_decisions_checked=decisions,positive_planning_gain_decisions=positive_gain,
        minimum_exact_public_reference_mass=str(minimum_pool),primary_exact_prefix_likelihood_ratios=sorted(ratios),
        bit_entropy_assumption='fresh independent uniform bits for deployment; PCG test streams are not certified',
        upstream_GeoI_float_certified=False,hardware_timing_certified=False,
        files_sha256={str(p.relative_to(output)):study.sha(p) for p in sorted(names)})
    path=output/'independent_validation.json'
    if path.exists():assert study.read(path)==receipt
    else:study.save(path,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='files_sha256'},indent=2))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=study.OUT)
    verify(parser.parse_args().output)
