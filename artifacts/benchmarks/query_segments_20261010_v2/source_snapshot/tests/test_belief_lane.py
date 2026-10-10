"""Exact emission oracle, postprocessing boundary and new dataset mutation QA."""
from copy import deepcopy
import json

import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.stats import laplace

from benchmark.anchor_belief import PublicAnchorModel, AnchorBelief
from benchmark.engines.belief_lane import BeliefLaneDummy
from benchmark.engines.contextual_lane import ContextualLaneDummy
from core.demo_protocol import TrajectoryPoint
from experiments.verify_scenario_suite_v2 import DEFAULT, verify_new_gates
from tests.test_lane_comparison import road
from tests.test_contextual_lane import context


def fixture(cache=None):
    rn=road(); c,_=context(rn)
    m=PublicAnchorModel(rn,c,np.linspace(1,2,len(rn)),spacing_m=2.,cache_path=cache)
    return rn,c,m


def test_emission_exact_full_support_duplicate_coordinate_and_reuse():
    rn,c,m=fixture()
    logits=np.exp(-.005*cdist(m.xy,rn.xy))
    logits/=logits.sum(axis=1,keepdims=True)
    coords=sorted(set(rn.latlon(i) for i in range(len(rn))))
    for prev in (None,rn.latlon(0),rn.latlon(15)):
        likelihood=[]
        for a in coords:
            ids=[i for i in range(len(rn)) if rn.latlon(i)==a]
            fresh=logits[:,ids].sum(axis=1)
            if prev is not None:
                q=laplace.cdf(200-np.linalg.norm(m.xy-rn.point_xy(*prev),axis=1),scale=100)
                expected=(1-q)*fresh+(q if a==prev else 0.)
            else: expected=fresh
            assert np.allclose(m.emission(a,prev),expected,rtol=1e-12,atol=1e-15)
            likelihood.append(m.emission(a,prev))
        assert np.allclose(np.sum(likelihood,axis=0),1.)
    # Same coordinate includes a positive fresh-draw term, not just reuse.
    q=laplace.cdf(200-np.linalg.norm(m.xy-rn.xy[0],axis=1),scale=100)
    assert np.all(m.emission(rn.latlon(0),rn.latlon(0))>q)


def test_filter_prediction_expected_poi_and_cache(tmp_path):
    rn,c,m=fixture(tmp_path/'model.npz')
    again=PublicAnchorModel(rn,c,np.linspace(1,2,len(rn)),spacing_m=2.,cache_path=tmp_path/'model.npz')
    assert again.sha256==m.sha256
    with pytest.raises(ValueError,match='Stale'):
        PublicAnchorModel(rn,c,np.ones(len(rn)),spacing_m=2.,cache_path=tmp_path/'model.npz')
    assert np.allclose(np.asarray(m.transition(5).sum(axis=1)).ravel(),1)
    b=AnchorBelief(m); a=rn.latlon(15)
    expected=m.prior*m.emission(a); expected/=expected.sum()
    assert np.allclose(b.update(a,0),expected)
    expected=m.predict(expected,5)
    assert np.allclose(b.update(a,5,observed=False),expected)
    assert np.allclose(m.predict(expected,2000),m.prior)
    oracle=sum(w*c.reference_weights(rn.latlon(int(s))) for w,s in zip(b.weights,m.state_ids))
    assert np.allclose(b.poi_weights(),oracle)
    with pytest.raises(ValueError): b.update(a,5)


@pytest.mark.parametrize('center',['anchor','belief_mean'])
def test_causal_prefix_anchor_pairing_and_no_extra_gps(center):
    rn,c,m=fixture()
    points=tuple(TrajectoryPoint(i*5.,0.,.0001+i*.0001) for i in range(15))
    def make():
        return BeliefLaneDummy(rn,belief_model=m,coverage_weight=6,center_mode=center,
                              budget=.24,horizon=12,rng=np.random.default_rng(19))
    model=make(); full=model.protect_run(points).to_attacker_dict()
    assert make().protect_run(points[:5]).to_attacker_dict()['events']==full['events'][:5]
    control=ContextualLaneDummy(rn,budget=.24,horizon=12,rng=np.random.default_rng(19))
    control.protect_run(points)
    assert control.evaluator_anchors==model.evaluator_anchors
    assert model.spent_bound==pytest.approx(.23)
    assert not any(key in json.dumps(full) for key in ('evaluator','mean_xy','effective_states'))
    a,b=make(),make()
    for x in (a,b): x.anchor.perturb=lambda *args,**kwargs:rn.latlon(15)
    for i in range(15):
        assert a.protect_step(0.,0.,i*5.)==b.protect_step(20.,30.,i*5.)
    assert a.protect_step(float('nan'),float('nan'),75)==b.protect_step(80,90,75)


def test_mechanism_mismatch_rejected():
    rn,c,m=fixture()
    with pytest.raises(ValueError,match='match the actual'):
        BeliefLaneDummy(rn,belief_model=m,budget=.25,horizon=12)


@pytest.mark.parametrize('mutation',['rare_category','future_day','target_slot','history_frequency','duplicate_id'])
def test_v2_gate_mutations_rejected(mutation):
    d=json.loads(DEFAULT.read_text())
    verify_new_gates(d)
    if mutation=='rare_category':
        r=next(r for r in d['records'] if r['case_id']=='S1.C'); r['evidence']['category']='restaurant'
    elif mutation=='duplicate_id': d['records'][1]['record_id']=d['records'][0]['record_id']
    else:
        r=next(r for r in d['records'] if r['case_id']=='S6.C')
        if mutation=='target_slot': r['labels']['target_slot']=0
        elif mutation=='history_frequency': r['labels']['historical_destination_counts']['routine']=6
        else: r['evidence']['query_day']=2
    with pytest.raises(AssertionError): verify_new_gates(d)
