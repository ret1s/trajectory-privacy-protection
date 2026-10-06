import itertools
import math

import pytest

from benchmark.risk_aware_cover import (
    TailAwareCoverObjective, improve_tail_cover, weighted_lower_tail_cvar,
)


def tradeoff(risk_weight=.5):
    profiles = [tuple(range(100*j, 100*(j+1))) for j in range(4)]
    signatures = {
        0: set().union(*profiles[:3], profiles[3][:60]),
        1: set().union(*(p[:85] for p in profiles)),
    }
    return TailAwareCoverObjective(profiles, [1, 1, 1, 1], signatures,
                                   tail_mass=.25, risk_weight=risk_weight)


def test_fractional_discrete_tail_and_variational_formula():
    values, weights, a = [0., .5, 1.], [.1, .3, .6], .25
    result = weighted_lower_tail_cvar(values, weights, a)
    assert result == pytest.approx(.3)  # All lowest10% plus15% at utility.5.
    candidates = [eta - math.fsum(w*max(0., eta-v) for v, w in zip(values, weights))/a
                  for eta in values]
    assert result == pytest.approx(max(candidates))
    assert weighted_lower_tail_cvar(values, weights, 1.) == pytest.approx(.75)
    assert weighted_lower_tail_cvar([100., 0., 1.], [0., 1., 3.], .5) == pytest.approx(.5)


def test_weight_normalization_handles_large_finite_weights():
    assert weighted_lower_tail_cvar([0., 1.], [1e308, 1e308], .75) == pytest.approx(1/3)
    with pytest.raises(ValueError, match='Finite'):
        weighted_lower_tail_cvar([0., 1.], [10**1000, 1.], .75)
    with pytest.raises(ValueError, match='Matched'):
        weighted_lower_tail_cvar([0.], [1., 2.], .75)


def test_tail_tradeoff_compared_with_exact_tiny_oracle():
    objective = tradeoff()
    baseline, balanced = objective.score((0,)), objective.score((1,))
    assert baseline.mean == pytest.approx(.9)
    assert baseline.lower_tail_cvar == pytest.approx(.6)
    assert balanced.mean == balanced.lower_tail_cvar == pytest.approx(.85)
    result = improve_tail_cover(objective, [[0, 1]], [0], mean_slack=.05)
    exact = max(itertools.product([0, 1]), key=lambda q: objective.score(q).objective)
    assert result.action == exact == (1,)
    assert result.score.objective > result.baseline_score.objective
    assert result.score.mean >= result.mean_floor
    mean_only = improve_tail_cover(tradeoff(0.), [[0, 1]], [0], mean_slack=.05)
    assert mean_only.action == (0,)


def test_mean_floor_rejects_tail_gain_using_full_action_score():
    objective = tradeoff()
    assert objective.score((1,)).objective > objective.score((0,)).objective
    result = improve_tail_cover(objective, [[0, 1]], [0], mean_slack=.01)
    assert result.action == (0,)
    assert result.iterations == 0
    assert result.termination == 'no_improving_exchange'


def test_empty_references_are_na_before_renormalization_and_copies_are_causal():
    profiles = [[], [1, 1, 2], [3]]
    signatures = {0: [1, 1], 1: [3]}
    objective = TailAwareCoverObjective(profiles, [8., 2., 0.], signatures)
    assert objective.empty_profile_count == 1
    assert objective.empty_profile_mass == pytest.approx(.8)
    assert objective.zero_weight_profile_count == 1
    assert objective.probabilities == (1.,)
    assert objective.score((0,)).mean == .5  # Not.1 from the omitted80% mass.
    profiles[1].append(3); signatures[0].append(2)
    assert objective.score((0,)).mean == .5


@pytest.mark.parametrize('profiles,weights', [([[]], [1]), ([[1]], [0]),
    ([[], [1]], [1, 0]), ([[1]], [-1]), ([[1]], [math.nan]), ([[1]], [math.inf])])
def test_undefined_or_invalid_profile_mass_is_rejected(profiles, weights):
    with pytest.raises(ValueError):
        TailAwareCoverObjective(profiles, weights, {0: [1]})


def test_track_feasibility_and_ties_do_not_depend_on_bucket_insertion_order():
    objective = TailAwareCoverObjective([[1], [2]], [1, 1],
        {0: [1], 1: [1, 2], 2: [1, 2], 3: []})
    a = improve_tail_cover(objective, [[2, 0, 1, 1], [3]], [0, 3])
    b = improve_tail_cover(objective, [[1, 2, 0], [3]], [0, 3])
    assert a.action == b.action == (1, 3)
    assert a.history == b.history
    assert all(step.action[0] in {0, 1, 2} and step.action[1] == 3 for step in a.history)
    assert all(step.score.mean >= a.mean_floor for step in a.history)
    with pytest.raises(ValueError, match='Baseline'):
        improve_tail_cover(objective, [[0], [3]], [1, 3])
    with pytest.raises(ValueError, match='known'):
        improve_tail_cover(objective, [[0, 99], [3]], [0, 3])


def test_duplicate_states_across_tracks_are_permitted_and_union_counts_once():
    objective = TailAwareCoverObjective([[1, 2]], [1], {0: [1], 1: [2]})
    assert objective.score((0, 0)).mean == .5
    result = improve_tail_cover(objective, [[0, 1], [0, 1]], [0, 0])
    assert result.action == (0, 1)
    assert result.score.mean == 1.


def test_local_exchange_is_not_mislabeled_as_global_optimum():
    objective = TailAwareCoverObjective([list(range(6))], [1],
        {0: [0, 1, 2], 1: [3], 2: [3, 4], 3: [0, 1, 5]})
    result = improve_tail_cover(objective, [[0, 2], [1, 3]], [0, 1])
    exact = max(itertools.product([0, 2], [1, 3]), key=lambda q: objective.score(q).objective)
    assert result.action == (0, 1)
    assert result.termination == 'no_improving_exchange'
    assert exact == (2, 3)
    assert result.score.objective < objective.score(exact).objective


def test_iteration_bound_and_zero_iteration_fallback_are_explicit():
    objective = TailAwareCoverObjective([[1, 2]], [1], {0: [], 1: [1], 2: [], 3: [2]})
    result = improve_tail_cover(objective, [[0, 1], [2, 3]], [0, 2], max_iterations=1)
    assert result.iterations == 1 and result.termination == 'iteration_limit'
    assert result.score.mean == .5
    stopped = improve_tail_cover(objective, [[0, 1], [2, 3]], [0, 2], max_iterations=0)
    assert stopped.action == (0, 2) and stopped.iterations == 0
    assert stopped.score.mean == 0.


@pytest.mark.parametrize('kwargs', [{'tail_mass': 0}, {'tail_mass': 1.01},
    {'risk_weight': -1}, {'risk_weight': math.nan}, {'risk_weight': True}])
def test_public_risk_parameters_are_validated(kwargs):
    with pytest.raises(ValueError):
        TailAwareCoverObjective([[1]], [1], {0: [1]}, **kwargs)


def test_invalid_signatures_and_optimizer_parameters_are_rejected():
    with pytest.raises(ValueError, match='POI'):
        TailAwareCoverObjective([[1]], [1], {0: [-1]})
    objective = TailAwareCoverObjective([[1]], [1], {0: [1]})
    for kwargs in ({'mean_slack': -1}, {'max_iterations': True}, {'max_iterations': -1},
                   {'tolerance': math.inf}, {'tolerance': -1}):
        with pytest.raises(ValueError):
            improve_tail_cover(objective, [[0]], [0], **kwargs)
