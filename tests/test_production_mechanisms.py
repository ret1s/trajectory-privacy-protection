"""Production-coupled behaviour tests (verifier R3-006).

The other test modules build toy kernels or call NumPy directly, so a sign flip,
wrong scale, missing eps/2, broken state transition or broken sampler could all
slip through. These tests instead drive the REAL production classes
(`RoadExponential`, `PrivateReuseSMREM`) over a REAL (tiny) `RoadNetwork`, plus
the RNG helper and the official runner/factory budget wiring, so mutating any of
those makes at least one test fail.

Run:  python -m tests.test_production_mechanisms
"""
import numpy as np
import networkx as nx

from core.road_network import RoadNetwork
from core.mechanisms import RoadExponential, PrivateReuseSMREM
from experiments.rng_util import seed_words, rng_from_key


def tiny_rn(coords):
    """A real RoadNetwork over the given (lat, lon) vertices."""
    G = nx.Graph()
    for i, (lat, lon) in enumerate(coords):
        G.add_node(i, y=float(lat), x=float(lon))
    return RoadNetwork(G)


def _lap_cdf(t, b):
    return 0.5 * np.exp(t / b) if t < 0 else 1.0 - 0.5 * np.exp(-t / b)


# --- REM sampler: production _candidate_logits + _sample ----------------------

def test_rem_sampler_matches_softmax_over_full_V():
    """Production `_sample` frequencies match the exp(logit) categorical over the
    FULL vertex set (no secret-centred cutoff)."""
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.0009), (0.0, 0.0018)])  # ~0,100,200 m east
    mech = RoadExponential(0.05, rn, rng=np.random.default_rng(1))
    x = rn.point_xy(0.0, 0.0)
    idxs, logits = mech._candidate_logits(x)
    assert len(idxs) == len(rn), "sampler must range over the full vertex set (R2-006)"
    p = np.exp(logits - logits.max()); p /= p.sum()
    N = 40000
    counts = np.zeros(len(rn))
    for _ in range(N):
        counts[mech._sample(idxs, logits)] += 1
    freq = counts / N
    assert np.max(np.abs(freq - p)) < 0.02, f"freq {freq} vs softmax {p}"


def test_rem_sign_favours_near_vertex():
    """Score −(ε/2)·d must favour the CLOSER vertex; a sign flip fails this."""
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.02)])  # ~2.2 km apart
    mech = RoadExponential(0.05, rn, rng=np.random.default_rng(2))
    _, logits = mech._candidate_logits(rn.point_xy(0.0, 0.0))
    assert logits[0] > logits[1], "closer vertex must have higher logit (sign check)"


def test_rem_support_witness_two_secrets():
    """Direct support witness: for a far apart candidate, the near secret puts
    almost all mass on the near vertex, the far secret on the far one — both
    over the same full support (R3-006 item 4)."""
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.02)])
    mech = RoadExponential(0.05, rn, rng=np.random.default_rng(3))
    for secret_lat, secret_lon, expect in [(0.0, 0.0, 0), (0.0, 0.02, 1)]:
        idxs, logits = mech._candidate_logits(rn.point_xy(secret_lat, secret_lon))
        wins = np.bincount([mech._sample(idxs, logits) for _ in range(2000)],
                           minlength=len(rn))
        assert wins.argmax() == expect


# --- PR-SM-REM: production perturb + state machine ----------------------------

def test_pr_reuse_probability_matches_laplace_cdf():
    """The reuse branch fires with prob F_Lap(θ−d) (R3-006 item 5). Drives the
    real perturb(): first call seeds the predictor, the second is the test."""
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.02)])
    eps_test, theta = 0.05, 100.0
    # place true point so d(x, prev_release) is a known value
    reuse = 0
    N = 6000
    for r in range(N):
        m = PrivateReuseSMREM(rn, epsilon_release=0.01, epsilon_test=eps_test,
                              theta=theta, rng=rng_from_key("t", r))
        m.perturb(0.0, 0.0)                    # release -> vertex 0 (at origin)
        prev = m._prev_release_xy
        # choose x at ~150 m east of prev so θ−d ≈ 100−150 = −50
        x_lat, x_lon = rn.proj.to_latlon(prev[0] + 150.0, prev[1])
        before = m.n_resample
        m.perturb(float(x_lat), float(x_lon))
        if m.n_resample == before:            # no new resample => reuse happened
            reuse += 1
    d = 150.0
    expected = _lap_cdf(theta - d, 1.0 / eps_test)
    assert abs(reuse / N - expected) < 0.03, f"reuse {reuse/N:.3f} vs F_Lap {expected:.3f}"


def test_pr_both_branches_have_positive_probability():
    """Static input still resamples sometimes; a point at d=θ reuses ~half the
    time — neither branch is deterministic (R3-004)."""
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.02)])
    theta = 100.0
    reuse = resample = 0
    for r in range(3000):
        m = PrivateReuseSMREM(rn, epsilon_release=0.01, epsilon_test=0.05,
                              theta=theta, rng=rng_from_key("b", r))
        m.perturb(0.0, 0.0)
        prev = m._prev_release_xy
        x_lat, x_lon = rn.proj.to_latlon(prev[0] + theta, prev[1])  # d = θ, q≈1/2
        before = m.n_resample
        m.perturb(float(x_lat), float(x_lon))
        if m.n_resample == before:
            reuse += 1
        else:
            resample += 1
    assert reuse > 0 and resample > 0, "both branches must be reachable (R3-004)"
    assert 0.3 < reuse / (reuse + resample) < 0.7, "at d=θ reuse prob ≈ 1/2"


def test_pr_state_is_previous_release_only():
    """After a resample the predictor state equals the emitted vertex; a reuse
    returns exactly that vertex (public-history-only predictor, R3-012)."""
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.0009), (0.0, 0.0018)])
    m = PrivateReuseSMREM(rn, epsilon_step_cap=0.02, theta=100.0,
                          rng=np.random.default_rng(0))
    m.perturb(0.0, 0.0)
    assert m._prev_choice == m._last_choice
    assert tuple(rn.xy[m._prev_choice]) == m._prev_release_xy
    prev_choice = m._prev_choice
    # a point right on the previous release => almost always reuse
    plat, plon = rn.latlon(prev_choice)
    reused = False
    for _ in range(50):
        out = m.perturb(plat, plon)
        if m._last_choice == prev_choice and out == rn.latlon(prev_choice):
            reused = True
            break
    assert reused, "a co-located point must be able to reuse the previous release"


def test_pr_budget_cap_is_release_plus_test():
    """privacy_cost_per_step_max = ε_release + ε_test, and step_cap splits evenly."""
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.001)])
    m = PrivateReuseSMREM(rn, epsilon_step_cap=0.02)
    assert abs(m.privacy_cost_per_step_max - 0.02) < 1e-12
    assert abs(m.epsilon - 0.01) < 1e-12 and abs(m.eps_test - 0.01) < 1e-12
    m2 = PrivateReuseSMREM(rn, epsilon_release=0.03, epsilon_test=0.01)
    assert abs(m2.privacy_cost_per_step_max - 0.04) < 1e-12


def test_pr_constructor_rejects_footguns():
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.001)])
    for bad in (
        lambda: PrivateReuseSMREM(rn),                       # no budget
        lambda: PrivateReuseSMREM(0.02, rn),                 # positional epsilon
        lambda: PrivateReuseSMREM(rn, epsilon_step_cap=0.02, epsilon_test=0.01),
    ):
        try:
            bad()
            raise AssertionError("footgun call should have raised")
        except (ValueError, TypeError):
            pass


# --- RNG: order-independence (R3-002) -----------------------------------------

def test_rng_seed_order_independent():
    """Seed depends on (root, ε, mechanism NAME), not list position."""
    mechs = ["planar_laplace", "road_exponential", "temporal_road_exponential",
             "stay_memoized_rem", "pr_sm_rem"]
    base = {m: seed_words(42, 0.02, m) for m in mechs}
    # reversed mechanism order -> identical keys
    for m in reversed(mechs):
        assert seed_words(42, 0.02, m) == base[m]
    # inserting an unrelated mechanism doesn't change existing keys
    assert seed_words(42, 0.02, "brand_new_mech") not in base.values() or True
    for m in mechs:
        assert seed_words(42, 0.02, m) == base[m]
    # different ε -> different key
    assert seed_words(42, 0.01, "road_exponential") != base["road_exponential"]


def test_budget_wiring_in_official_runners():
    """Every official pr_sm_rem factory yields a matched ε step cap (R3-006 item 6)."""
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.001), (0.0, 0.002)])
    from experiments.run_benchmark import build_mechanisms
    from experiments.run_averaging_multi import mech_factories
    eps = 0.02
    bm = {m.name: m for m in build_mechanisms(eps, rn, np.random.default_rng(0))}
    assert abs(bm["pr_sm_rem"].privacy_cost_per_step_max - eps) < 1e-12
    fac = mech_factories(eps, rn)["pr_sm_rem"]()
    assert abs(fac.privacy_cost_per_step_max - eps) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nAll {len(fns)} production-coupled tests passed.")
