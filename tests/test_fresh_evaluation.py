"""Selection gates, worker isolation and postprocessing input boundary."""
import numpy as np
import pytest

from core.demo_protocol import TrajectoryPoint
from experiments.run_fresh_switching import METHODS, CASES, choose, generate, make_model
from tests.test_belief_lane import fixture


def summaries(recall=.89):
    return [{'method': m, 'case_id': c, 'mae_by_attack': {'a': 10., 'b': 20.},
        'hit_by_attack': {'a': .1, 'b': .2}, 'envelope_hit100': .2,
        'utility': {d: {'poi_recall_at_5': recall} for d in ('5', '10')}} for m in METHODS for c in CASES]


def test_no_silent_relaxation_and_separate_attacker_losses():
    rows = summaries(); selected = choose(rows)
    assert all(v['chosen'] is None for v in selected['method_selection_by_depth'].values())
    assert all(v == {'mae': 'a', 'hit': 'b'} for v in selected['attackers'].values())
    for r in rows:
        if r['method'] == 'switching_exchange': r['utility']['10']['poi_recall_at_5'] = .90
    choice = choose(rows)['method_selection_by_depth']
    assert choice['5']['chosen'] is None
    assert choice['10']['chosen']['method'] == 'switching_exchange'
    rows[0]['utility']['5']['poi_recall_at_5'] = 1.  # one good case cannot qualify a method
    assert choose(rows)['method_selection_by_depth']['5']['chosen'] is None
    with pytest.raises(AssertionError): choose(rows[:-1])


def test_switching_worker_reset_equals_new_instance_and_no_private_leak():
    rn, _, belief = fixture(); m = 'switching_exchange'
    pool = {m: make_model(m, rn, belief, 0)}
    for seed, n in ((17, 5), (99, 15), (17, 5)):
        points = tuple(TrajectoryPoint(i*5., 0., .0001+i*.0001) for i in range(n))
        reused, fresh = generate(m, points, rn, belief, seed, pool), generate(m, points, rn, belief, seed)
        for key in ('public', 'evaluator_states', 'evaluator_anchors', 'evaluator_objective', 'spent_bound'):
            assert reused[key] == fresh[key]
    a, b = make_model(m, rn, belief, 71), make_model(m, rn, belief, 71)
    for model in (a, b): model.anchor.perturb = lambda *args, **kwargs: rn.latlon(15)
    for i in range(15): assert a.protect_step(0., 0., i*5.) == b.protect_step(20., 30., i*5.)
    assert a.protect_step(float('nan'), float('nan'), 75) == b.protect_step(80, 90, 75)
