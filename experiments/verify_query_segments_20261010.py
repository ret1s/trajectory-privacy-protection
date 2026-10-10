"""Independent multistep evidence audit, without private RNG keys.

Rebuild common public support/score/floors and check sampled road states, every
prefix allowance, empirical utility, attacker receipts and promotion. Point-mass
likelihood diagnostics are NOT a certified float sampler or exhaustive proof.
"""
from fractions import Fraction
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from benchmark.public_segment_scores import FixedPublicPurposeTable
from benchmark.query_segments import PublicSegmentLibrary, oscillation_upper
from experiments import query_segment_study_20261010 as study
from experiments import query_bundle_audit_20261010 as public


def verify(output=study.OUT, *, write=True):
    output=Path(output);protocol=study.validate_sources(output)
    families=study.load_families(output)
    dataset=study.read(study.DATA);generation=study.read(output/'generation.json')
    native=study.read(study.DATA.parent/'validation.json')
    assert native['passed'] and native['dataset_sha256']==study.sha(study.DATA)==generation['dataset_sha256']
    assert len(families)==60 and {f['family_id']:f['split'] for f in families}=={f['family_id']:f['split'] for f in dataset['families']}
    rn,reference,reply,ids=public.resources(study.WORK/'public_resources')
    meta=study.read(output/'resources.json')
    assert meta['catalogue_sha256']==rn.catalogue_sha256
    assert meta['reference_sha256']==reference.sha256 and meta['reply_sha256']==reply.sha256
    assert meta['public_goals']==[list(a.states) for a in public.library(reference,reply,ids) if len(a.states)==5]
    library=PublicSegmentLibrary(rn,meta['public_goals'])
    table=FixedPublicPurposeTable(reference,reply,ids,destinations=meta['public_destinations'])
    assert table.validity.tolist()==meta['public_purpose_validity']
    inspected={};checked=0;degraded=0
    for family in families:
        for trip in family['trips']:
            reads=[p['t'] for p,l in zip(trip['truth'],trip['GeoI_ledger']) if l['private_read']]
            assert all(b-a>=60 for a,b in zip(reads,reads[1:]))
            assert all(l['spent_units']<=23 for l in trip['GeoI_ledger'])
            assert trip['GeoI_spent_per_m']<=.23+1e-12
            for method in study.METHODS[1:]:
                frames=3 if method.startswith('segment') else 1
                states=trip['sampled_states'][method];certs=trip['Q_certificates'][method]
                assert len(certs)==len(states)==31
                gamma=study.FractionGamma(method)
                for tick,cert in enumerate(certs):
                    block,phase=tick//3+1,tick%3
                    allowed=gamma/Fraction(block*(block+1)*(1 if frames==3 else 3))
                    prefix=gamma*Fraction(block-1,block)+allowed*(1 if frames==3 else phase+1)
                    assert cert['block']==block and cert['allocated_epsilon_Q']==float(allowed)
                    assert cert['reserved_prefix_epsilon_Q']==float(prefix) and prefix<=gamma
                    assert 0<=cert['epsilon_Q_upper']<=float(allowed)
                    assert cert['actual_expected_utility_lower'] is None and cert['utility_status']=='public_table_only'
                    if frames==3 and phase:
                        assert {k:v for k,v in cert.items() if k!='t'}=={k:v for k,v in certs[tick-phase].items() if k!='t'}
                        continue
                    previous=tuple(states[tick-1]) if tick else None
                    cache_key=previous,frames
                    if cache_key not in inspected:
                        actions=library.build(previous,frames)
                        g=table(actions);floor=table.floor_table(actions)
                        keep=np.flatnonzero(floor.min(axis=0)>=.75)
                        fallback=not len(keep)
                        if not fallback:
                            actions=tuple(actions[i] for i in keep);g=g[:,keep];floor=floor[:,keep]
                        inspected[cache_key]=(actions,g,oscillation_upper(g),float(floor.min()),fallback)
                    actions,g,kappa,floor,fallback=inspected[cache_key]
                    assert cert['kappa_upper']==float(kappa) and cert['exact_zero_oscillation']==(kappa==0)
                    assert cert['public_library_size']==len(actions) and cert['floor_degraded']==fallback
                    assert cert['attained_public_floor']==floor
                    observed=tuple(tuple(f) for f in states[tick:min(tick+frames,len(states))])
                    assert any(a.frames[:len(observed)]==observed for a in actions),'Selected sequence outside PUBLIC support'
                    exact=Fraction.from_float(cert['beta'])*kappa/2
                    assert exact<=Fraction.from_float(cert['epsilon_Q_upper'])<=allowed
                    assert 0<=cert['beta']<=40
                    # Extreme public beliefs, WITHOUT ground truth or E. This
                    # validates numerics only; the theorem covers all beliefs.
                    logits=.5*cert['beta']*g/1.25
                    logs=logits-logsumexp(logits,axis=1)[:,None]
                    assert np.max(np.ptp(logs,axis=0))<=float(allowed)+1e-12
                    checked+=1;degraded+=int(fallback)
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
        assert {p['family_id'] for p in predictions}==test_ids
        assert len(predictions)==48
        mae=[];hit=[]
        for family in sorted(test_ids):
            group=[p for p in predictions if p['family_id']==family]
            assert len(group)==2
            mae.append(np.mean([np.mean(p['errors'][selection['selected']['mae']]) for p in group]))
            hit.append(np.mean([np.mean(np.array(p['errors'][selection['selected']['hit100']])<=100) for p in group]))
        assert abs(row['MAE_m']-np.mean(mae))<1e-10 and abs(row['Hit100']-np.mean(hit))<1e-14
    gate=next(u['utility_gate'] for u in results['utility'] if u['method']=='segment_g1' and u['cache']=='current')
    assert results['promotion']['utility_gate']==gate and results['promotion']['promote']==gate
    assert results['promotion']['default_engine_changed'] is False
    # Complete inventory includes the model bytes, receipts, predictions and
    # public source snapshots. The report itself is deliberately not a source.
    names=[p for folder in ('attacker_models','attacker_selection','attacker_predictions') for p in (output/folder).iterdir()]
    receipt=dict(status='pass',protocol_sha256=study.sha(output/'protocol.json'),results_sha256=study.sha(output/'results.json'),
        verifier_sha256=study.sha(Path(__file__)),native_validation_sha256=study.sha(study.DATA.parent/'validation.json'),
        public_support_decisions_checked=checked,floor_degraded_decisions=degraded,
        numerical_likelihood_diagnostic='public point masses only; mathematical bound covers arbitrary beliefs',
        public_support_uses_raw_GPS=False,models_fit_on_test=False,numerical_sampler_certified=False,
        artifact_files_sha256={str(p.relative_to(output)):study.sha(p) for p in sorted(names)})
    target=output/'independent_validation.json'
    if write:
        if target.exists():assert study.read(target)==receipt
        else:study.save(target,receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='artifact_files_sha256'},indent=2))
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=study.OUT)
    args=parser.parse_args();verify(args.output)
