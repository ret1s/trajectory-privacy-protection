from copy import deepcopy
import numpy as np
import pytest
from benchmark.engines.public_supplement import PublicServiceSupplement, append_public_view, remove_public_view
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from core.demo_protocol import TrajectoryPoint
from experiments.research_loop_cases import recall_pair
from tests.test_belief_lane import fixture


def test_embedding_exact_inverse_duplicates_and_tamper_detection():
    original = {'events': [{'timestamp_s': 0., 'candidates': [
        {'candidate_id': 'candidate_0000', 'lat': 1., 'lon': 2.}]}]}
    saved = deepcopy(original); fixed = ((1., 2.), (1., 2.))
    hybrid = append_public_view(original, fixed)
    assert remove_public_view(hybrid, fixed) == original == saved
    assert len(hybrid['events'][0]['candidates']) == 3  # No hidden query deduplication.
    hybrid['events'][0]['candidates'][-1]['lat'] = 3.
    with pytest.raises(ValueError, match='suffix'):
        remove_public_view(hybrid, fixed)
    with pytest.raises(ValueError, match='collide'):
        append_public_view(append_public_view(original, fixed), fixed)
    assert append_public_view(original, ()) == original


@pytest.mark.parametrize('count', [0, 1, 3])
def test_online_prefix_parent_ledger_and_lifted_outputs_are_identical(count):
    rn, context, belief = fixture()
    points = tuple(TrajectoryPoint(i*20., 0., .0001+(i%14)*.0001) for i in range(20))
    def parent():
        return PacedSlackProgressLaneDummy(rn, belief_model=belief, k=5, budget=.24,
                                           horizon=12, utility_slack=.03, rng=np.random.default_rng(2727))
    control = parent(); plain = control.protect_run(points).to_attacker_dict()
    wrapped = PublicServiceSupplement(parent(), public_queries=count)
    full = wrapped.protect_run(points).to_attacker_dict()
    assert full['events'] == append_public_view({'events': plain['events']}, wrapped.coordinates)['events']
    assert remove_public_view({'events': full['events']}, wrapped.coordinates)['events'] == plain['events']
    assert wrapped.parent.evaluator_anchors == control.evaluator_anchors
    assert wrapped.parent.evaluator_ledger == control.evaluator_ledger
    assert wrapped.spent_bound == control.spent_bound
    prefix = PublicServiceSupplement(parent(), public_queries=count).protect_run(points[:8]).to_attacker_dict()
    assert prefix['events'] == full['events'][:8]
    online = PublicServiceSupplement(parent(), public_queries=count)
    coordinates = [online.protect_step(p.lat, p.lon, p.timestamp_s) for p in points]
    assert coordinates == [tuple((c['lat'], c['lon']) for c in e['candidates']) for e in full['events']]
    if count == 0:
        assert full == plain
    else:
        assert full['public_parameters']['k'] == 5+count
        assert all(state in wrapped.parent.travel.reachable(state, 20) for state in wrapped.plan['states'])
    # Directly evaluate the service union at every test-network reference state.
    for a, b in zip(plain['events'], full['events']):
        old = [rn.nearest(c['lat'], c['lon'])[0] for c in a['candidates']]
        new = [rn.nearest(c['lat'], c['lon'])[0] for c in b['candidates']]
        for state in range(len(rn)):
            before, after = recall_pair(context, context, old, state), recall_pair(context, context, new, state)
            for L in ('5', '10'):
                assert (before[L] is None and after[L] is None) or after[L] >= before[L]
