from fractions import Fraction

import numpy as np

from benchmark.targeted_cache_exploration import TargetedExploreLibrary, TargetedCachePolicy
from benchmark.finite_query_kernel import FiniteCenteredKernel
from evaluation.lane_travel import SparseTravel
from tests.test_public_pool_portfolio import objects


def test_targeted_support_directed_and_positive_public_base():
    rn, reply, table, portfolio, old = objects()
    library = TargetedExploreLibrary(rn, old.goals, portfolio)
    policy = TargetedCachePolicy(library, table, portfolio, use_simulator=True)
    previous, known = old.goals[0], portfolio.pool(old.goals[0])
    actions, scores, _, masses = policy.public_tables(previous, known)
    travel = SparseTravel(rn)
    for action in actions:
        prior = previous
        for frame in action.frames:
            assert all(y in travel.reachable(x, 20.) for x, y in zip(prior, frame))
            prior = frame
    assert min(masses) > 0 and scores.shape == (3, len(actions))
    kernel = FiniteCenteredKernel(masses, Fraction(1, 6), resolution=1)
    p = kernel.probabilities([0]*len(actions))
    q = kernel.probabilities([a % 2 for a in range(len(actions))])
    assert all(x/y <= 1+kernel.epsilon and y/x <= 1+kernel.epsilon for x,y in zip(p,q))
    a = policy.select(previous, Fraction(1, 6), np.random.default_rng(7), belief=[.2,.3,.5], known=known)
    b = policy.select(previous, Fraction(1, 6), np.random.default_rng(7), belief=[.2,.3,.5], known=known)
    assert a == b


def test_full_cache_does_not_create_private_fallback_or_claim_novelty():
    rn, _, table, portfolio, old = objects()
    lib = TargetedExploreLibrary(rn, old.goals, portfolio)
    actions = lib.targeted(old.goals[0], portfolio.attainable)
    assert actions and all(potential == 0 for _, potential in actions)
    policy = TargetedCachePolicy(lib, table, portfolio)
    assert all(m == 1 for m in policy.public_tables(old.goals[0], portfolio.attainable)[3])


def test_bad_private_proxy_retains_support_budget_and_public_packet_shape():
    rn, _, table, portfolio, old = objects()
    policy = TargetedCachePolicy(TargetedExploreLibrary(rn, old.goals, portfolio), table, portfolio, use_simulator=True)
    for belief in ([np.nan]*3, [1.,2.], ['bad'], [-1.,1.,1.]):
        result = policy.select(None, Fraction(1, 2), np.random.default_rng(1), belief=belief, known=0)
        assert len(result['frames']) == 3 and all(len(frame) == 5 for frame in result['frames'])
        assert result['certificate']['private_proxy_fallback_local_only']
        assert Fraction(result['certificate']['exact_likelihood_ratio_upper']) <= Fraction(3, 2)
