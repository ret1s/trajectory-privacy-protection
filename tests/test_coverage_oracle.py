from itertools import product
import numpy as np
import pytest
from evaluation.coverage_oracle import solve


@pytest.mark.parametrize('seed', [3, 19, 71])
def test_reachable_oracle_matches_exhaustive_small_problem(seed):
    rng = np.random.default_rng(seed)
    signatures = np.array([rng.choice(np.arange(6), 3, replace=False) for _ in range(12)])[:, None, :]
    weights = np.r_[rng.random(6), 0.]; weights /= weights.sum()
    groups = [list(range(0, 5)), list(range(3, 9)), list(range(8, 12))]
    def value(states):
        return weights[np.unique(signatures[list(states)])].sum()
    best = max(value(states) for states in product(*groups))
    r = solve(groups, signatures, np.arange(12), weights)
    assert r['optimal']
    assert r['feasible_value'] == pytest.approx(best, abs=1e-9)
    assert r['upper_bound'] == pytest.approx(best, abs=1e-9)
    assert all(s in g for s, g in zip(r['selected_states'], groups))


def test_union_upper_bound_and_empty_poi_sentinel_do_not_invent_coverage():
    signatures = np.array([[[0, -1]], [[1, -1]], [[2, -1]]])
    r = solve([[0], [0]], signatures, np.arange(3), np.array([.3, .3, .4, 0.]))
    assert r['optimal'] and r['upper_bound'] == pytest.approx(.3)
    assert r['selected_states'] == [0, 0]
