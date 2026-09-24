"""Composition with the real switching engine on the existing tiny lane fixture."""
import numpy as np
import pytest
from tests.test_belief_lane import fixture
from benchmark.engines.switching_cover import SwitchingCoverLaneDummy
from core.boundary_release import BoundaryPolicy,BoundaryProtectedStream


def test_stream_composes_with_switching_core_without_refunds():
    rn,_,belief=fixture()
    def model():return SwitchingCoverLaneDummy(rn,belief_model=belief,budget=.24,horizon=12,k=5,rng=np.random.default_rng(29))
    engine=model();gate=BoundaryProtectedStream(engine,BoundaryPolicy(10,10));out=[]
    control=model();control.reset();expected=[]
    for i in range(20):
        t=i*5.;lat,lon=0.,.0001+i*.0001
        out.extend(gate.ingest(t,lat,lon))
        if t>=10:expected.append(control.protect_step(lat,lon,t))
    gate.close(95)
    assert [[(c.lat,c.lon) for c in e.candidates] for e in out]==[list(x) for x in expected[:-2]]
    assert engine.spent_bound==pytest.approx(control.spent_bound)
    assert engine.spent_bound<=.24
    assert gate.evaluator_summary()==dict(input_events=20,head_suppressed=2,protected_events=18,released_events=16,tail_cancelled=2,pending_events=0)
