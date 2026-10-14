"""Exact counterexamples and adaptive hidden-state/projection proof checks.

Negative examples intentionally violate a named assumption. They prevent the
theorem from being generalized silently to private stop/support, conditioned
subgroups, marginal scores, or incomplete provider replies.
"""
from fractions import Fraction as F
from collections import defaultdict
import itertools

import pytest

from benchmark.finite_query_kernel import FiniteCenteredKernel
from benchmark.finite_session_bounds import prefix_ratio
from benchmark.whole_path_chi_square import whole_path_bound


@pytest.mark.parametrize('upstream', [((F(1, 4), F(3, 4)), (F(4, 5), F(1, 5))),
                                     ((F(1), F(0)), (F(0), F(1)))])
def test_full_hidden_state_posterior_with_adaptive_support_provider_and_wire_alias(upstream):
    # In the second case the upstream leaks X perfectly if released. Only Q
    # is released, so its global cap must still hold without upstream Geo-I.
    joint = [{(z, ()): p for z, p in enumerate(prior) if p} for prior in upstream]
    base = {(): F(1)}
    for j in range(1, 4):
        new_joint, new_base = [defaultdict(F), defaultdict(F)], defaultdict(F)
        for h, w in base.items():
            masses = (1, 2, 1) if not h else (1+h[-1][0], 2+h[-1][1])
            kernel = FiniteCenteredKernel(masses, F(1, j*(j+1)), resolution=2)
            for a, mass in enumerate(masses):
                # Replies depend only on transmitted coordinate a%2, so
                # coincident coordinates have the SAME provider law.
                reply_one = F(3, 4) if a % 2 else F(1, 3)
                for reply, rp in ((0, 1-reply_one), (1, reply_one)):
                    child = h+((a, reply),)
                    new_base[child] += w*F(mass, sum(masses))*rp
                    for x, law in enumerate(joint):
                        for z in range(2):
                            state_mass = law.get((z, h), F(0))
                            scores = tuple((x+z+j+i+sum(r for _, r in h)) % 3 for i in range(len(masses)))
                            new_joint[x][z, child] += state_mass*kernel.probabilities(scores)[a]*rp
        joint, base = new_joint, new_base
        assert sum(base.values()) == 1
        bound = whole_path_bound(1., j)
        wire_base = defaultdict(F)
        for h, p in base.items():
            wire_base[tuple((a % 2, r) for a, r in h)] += p
        wire_laws, augmented = [], []
        for law in joint:
            p = {h: sum(law.get((z, h), F(0)) for z in range(2)) for h in base}
            assert sum(p.values()) == 1
            augmented.append(p)
            projected = defaultdict(F)
            for h, probability in p.items():
                projected[tuple((a % 2, r) for a, r in h)] += probability
            wire_laws.append(projected)
            chi_aug = sum((p[h]-base[h])**2/base[h] for h in base)
            chi_wire = sum((projected[o]-wire_base[o])**2/wire_base[o] for o in wire_base)
            assert chi_wire <= chi_aug <= F(bound['exact_chi_square_upper'])
            for o in wire_base:
                # Conditional-expectation identity for the projected LR.
                fiber = [h for h in base if tuple((a % 2, r) for a, r in h) == o]
                expected_lr = sum(base[h]*p[h]/base[h] for h in fiber)/wire_base[o]
                assert projected[o]/wire_base[o] == expected_lr
        ratio = prefix_ratio(1, j)
        for left, right in itertools.product(wire_laws, repeat=2):
            assert all(left[o] <= ratio*right[o] for o in wire_base)
        for prior in (F(1, 2), F(3, 4)):
            success = sum(max(prior*wire_laws[0][o], (1-prior)*wire_laws[1][o]) for o in wire_base)
            assert success-max(prior, 1-prior) <= F(bound['exact_rational_utility_shift_upper'])


def test_marginalizing_random_scores_before_covariance_is_not_valid():
    kernel = FiniteCenteredKernel((1, 1), F(1, 2), resolution=1)
    conditional_scores = ((1, 0), (0, 1))
    actual_gain = F(0)
    for scores in conditional_scores:
        p = kernel.probabilities(scores)
        actual_gain += sum((q-F(1, 2))*v for q, v in zip(p, scores))/2
    assert actual_gain == kernel.alpha/4 > 0
    # Both marginal score vectors are (.5,.5); their covariance is zero.
    marginal_scores = (F(1, 2), F(1, 2))
    assert sum((z-F(1, 2))*v/2 for z, v in zip(marginal_scores, marginal_scores)) == 0


def test_output_conditioning_can_increase_tv_unlike_public_stopping_projection():
    p, w = (F(1, 1000), F(9, 1000), F(99, 100)), (F(1, 200), F(1, 200), F(99, 100))
    tv = sum(abs(a-b) for a, b in zip(p, w))/2
    accepted_p = tuple(v/sum(p[:2]) for v in p[:2])
    accepted_w = tuple(v/sum(w[:2]) for v in w[:2])
    conditional_tv = sum(abs(a-b) for a, b in zip(accepted_p, accepted_w))/2
    assert tv == F(1, 250) and conditional_tv == F(2, 5)
    assert conditional_tv > tv


def test_private_clock_or_private_support_can_disclose_a_secret_with_constant_scores():
    # One output determined by X has disjoint support, despite zero scores.
    private_stop = ({(): F(1)}, {('send',): F(1)})
    private_support = ({('lane0',): F(1)}, {('lane1',): F(1)})
    for left, right in (private_stop, private_support):
        assert any(p > 0 and right.get(o, F(0)) == 0 for o, p in left.items())


def test_predicted_reply_floor_does_not_survive_incomplete_provider_replies():
    # Two disjoint single-item service references with equal public mass.
    reference_sets = ({0}, {1})
    def mean_recall(cache):
        return sum(F(len(r & cache), len(r)) for r in reference_sets)/2
    predicted, received = {0, 1}, {0}
    assert mean_recall(predicted) == 1
    assert mean_recall(received) == F(1, 2)
    # Accumulation remains monotone even when the predicted floor fails.
    assert mean_recall(received | {1}) >= mean_recall(received)


def test_real_session_accepts_empty_common_replies_without_establishing_predicted_floor(tmp_path, monkeypatch):
    from benchmark.cache_exploration import mask_bits
    from tests.test_bounded_query_session import setup
    monkeypatch.setattr('benchmark.finite_query_kernel.secrets.randbelow', lambda n: 0)
    _, _, session, _, _ = setup(tmp_path)
    def empty_server(request):
        return [[] for _ in request['categories']]
    wire = session.step(0., protected_belief=[.2, .3, .5], server=empty_server)
    frame, local_certificate = session.ledger.frame(0., lambda *args: pytest.fail('Retry resampled'))
    predicted = session.policy.portfolio.quality_exact(session.policy.portfolio.pool(frame))
    received = session.policy.portfolio.quality_exact(mask_bits(wire['session']))
    assert predicted > 0 and received == 0
    assert F(local_certificate['exact_selected_public_reference_mass']) == predicted
    assert session.ledger.reserved == F(1, 2)
    session.close()


def test_upstream_restore_is_a_trust_assumption_not_authenticated_by_matching_time(tmp_path, monkeypatch):
    # This is a documented residual assumption, NOT a claimed safe restore.
    # Update this counterexample if authenticated full upstream checkpoints
    # become part of a future release; the frozen release is unchanged here.
    import numpy as np
    from scipy.spatial import cKDTree
    from benchmark.anchor_belief import PublicAnchorModel
    from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
    from core.bounded_query_session import GeoIBoundedQPipeline
    from tests.test_bounded_query_session import setup
    monkeypatch.setattr('benchmark.finite_query_kernel.secrets.randbelow', lambda n: n//2)
    rn, reply, session, server, _ = setup(tmp_path)
    model = PublicAnchorModel(rn, reply, np.ones(len(rn)), spacing_m=2.,
                              epsilon_release=.01, epsilon_test=.01)
    projection = cKDTree(rn.xy[list(session.policy.table.states)]).query(model.xy)[1]
    def make():
        return PacedSlackProgressLaneDummy(rn, belief_model=model, k=5, horizon=12,
            budget=.24, read_interval_s=60., utility_slack=.03, rng=np.random.default_rng(47))
    old = make()
    pipeline = GeoIBoundedQPipeline(old, session, projection)
    for t in range(0, 121, 20):
        pipeline.step(*rn.latlon(0), float(t), server=server)
    assert old.spent_units >= 3
    reset = make()
    reset.protect_step(*rn.latlon(0), 120.)
    assert reset.spent_units == 1 and reset.last_t == old.last_t
    # Time compatibility alone cannot establish that history/ledger was restored.
    admitted = GeoIBoundedQPipeline(reset, session, projection)
    assert admitted.engine.spent_units < old.spent_units
    assert admitted.session.ledger.reserved == F(3, 4)  # Q cap stayed persistent
    session.close()
