from dataclasses import FrozenInstanceError
import numpy as np
import pytest
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from benchmark.public_poi_context import PublicPoiContext
from benchmark.engines.fair_cover import CoverageObjective
from benchmark.engines.paced_guard import PacedProgressLaneDummy
from evaluation.lane_travel import LanePoiService
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


def test_expected_top_k_recall_uses_top_L_reply_union():
    rn, reference, base = fixture()
    deeper = PublicPoiContext(LanePoiService(rn, list(reference.pois), k=reference.k+2))
    model = ResponseAwareAnchorModel(base, deeper)
    assert model.poi_weights is base.poi_weights
    selected = [3, 17, 29]
    weights = np.asarray(base.prior @ model.poi_weights).ravel()
    objective = CoverageObjective(deeper.signatures, deeper.access, weights)
    expected = 0.
    for mass, state in zip(base.prior, base.state_ids):
        values = []
        for ci, ref in enumerate(reference.query_indices(state)):
            target = set(ref)-{-1}
            if target:
                got = set(deeper.signatures[deeper.access[selected], ci].ravel())-{-1}
                values.append(len(got & target)/len(target))
        expected += mass*(np.mean(values) if values else 0.)
    assert objective.value(selected) == pytest.approx(expected, abs=1e-12)
    anchor = rn.latlon(0)
    assert np.array_equal(model.emission(anchor), base.emission(anchor))
    with pytest.raises(FrozenInstanceError):
        model.context = reference


def test_equal_depth_identity_and_changed_depth_causal_physical_budget():
    rn, reference, base = fixture()
    deeper = PublicPoiContext(LanePoiService(rn, list(reference.pois), k=reference.k+2))
    points = tuple(TrajectoryPoint(i*5., 0., .0001+i*.0001) for i in range(20))
    def make(model):
        return PacedProgressLaneDummy(rn, belief_model=model, k=3, budget=.24,
                                       horizon=12, rng=np.random.default_rng(34))
    plain = make(base).protect_run(points).to_attacker_dict()
    same = make(ResponseAwareAnchorModel(base, reference)).protect_run(points).to_attacker_dict()
    assert plain['events'] == same['events']
    model = ResponseAwareAnchorModel(base, deeper)
    engine = make(model); full = engine.protect_run(points).to_attacker_dict()
    prefix = make(model).protect_run(points[:7]).to_attacker_dict()
    assert full['events'][:7] == prefix['events']
    assert engine.spent_bound <= .23+1e-12
    for a, b in zip(engine.evaluator_states, engine.evaluator_states[1:]):
        assert all(v in engine.travel.reachable(u, 5.) for u, v in zip(a, b))
    assert not any(k in str(full) for k in ('evaluator_ledger', 'progress_goals'))
