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
from core.mechanisms import RoadExponential, PrivateReuseSMREM, StayMemoizedREM
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

def test_rem_sampler_matches_ANALYTIC_softmax_coeff_half():
    """Production `_sample` frequencies match the categorical whose expected
    probabilities are computed ANALYTICALLY from p ∝ exp(−(ε/2)·d) — NOT read
    back from production logits. Because the ε/2 coefficient is hard-coded in
    the expected distribution, a mutation of the score coefficient (ε/2 → ε)
    makes production frequencies diverge and this test fails (verifier R4-006)."""
    eps = 0.05
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.0009), (0.0, 0.0018)])  # ~0,100,200 m east
    mech = RoadExponential(eps, rn, rng=np.random.default_rng(1))
    real = np.asarray(rn.point_xy(0.0, 0.0))
    # analytic distances + expected softmax with the KNOWN coefficient ε/2
    dists = np.linalg.norm(rn.xy - real, axis=1)
    expected = np.exp(-0.5 * eps * (dists - dists.min()))
    expected /= expected.sum()
    idxs, logits = mech._candidate_logits(real)
    assert len(idxs) == len(rn), "sampler must range over the full vertex set (R2-006)"
    N = 40000
    counts = np.zeros(len(rn))
    for _ in range(N):
        counts[mech._sample(idxs, logits)] += 1
    freq = counts / N
    assert np.max(np.abs(freq - expected)) < 0.02, f"freq {freq} vs analytic {expected}"
    # sanity: the WRONG coefficient ε would give a materially different target,
    # so the tolerance above genuinely separates them.
    wrong = np.exp(-eps * (dists - dists.min())); wrong /= wrong.sum()
    assert np.max(np.abs(expected - wrong)) > 0.05, "ε/2 vs ε must be distinguishable"


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
    returns exactly that vertex AND does not increment n_resample (branch is read
    from the counter, not inferred from output equality — verifier R4-006)."""
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.0009), (0.0, 0.0018)])
    m = PrivateReuseSMREM(rn, epsilon_step_cap=0.02, theta=100.0,
                          rng=np.random.default_rng(0))
    m.perturb(0.0, 0.0)
    assert m._prev_choice == m._last_choice
    assert tuple(rn.xy[m._prev_choice]) == m._prev_release_xy
    prev_choice = m._prev_choice
    plat, plon = rn.latlon(prev_choice)
    reused = False
    for _ in range(50):
        before = m.n_resample
        m.perturb(plat, plon)
        if m.n_resample == before:            # counter says: reuse branch taken
            assert m._last_choice == prev_choice
            reused = True
            break
    assert reused, "a co-located point must be able to reuse via the reuse branch"


def test_pr_one_step_kernel_matches_analytic_mixture():
    """Exhaustive one-step production kernel vs the ANALYTIC two-branch mixture
    (verifier R4-006): with prediction h fixed, P(output=h) = q + (1−q)R(h) and
    P(output=v≠h) = (1−q)R(v), with q = F_Lap(θ−d), R ∝ exp(−ε_rel/2·d)."""
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.0009), (0.0, 0.0018)])
    eps_rel, eps_test, theta = 0.02, 0.05, 120.0
    # true point at vertex 1; predictor h will be vertex 0 (seed step 0 at v0)
    counts = np.zeros(len(rn))
    N = 8000
    x_lat, x_lon = rn.latlon(1)
    for r in range(N):
        m = PrivateReuseSMREM(rn, epsilon_release=eps_rel, epsilon_test=eps_test,
                              theta=theta, rng=rng_from_key("onestep", r))
        m._prev_choice = 0
        m._prev_release_xy = tuple(rn.xy[0])   # public predictor = vertex 0
        out = m.perturb(x_lat, x_lon)
        counts[[rn.latlon(i) for i in range(len(rn))].index(out)] += 1
    freq = counts / N
    real = np.asarray(rn.point_xy(x_lat, x_lon))
    d_xh = float(np.hypot(*(real - np.asarray(rn.xy[0]))))
    q = _lap_cdf(theta - d_xh, 1.0 / eps_test)
    dists = np.linalg.norm(rn.xy - real, axis=1)
    R = np.exp(-0.5 * eps_rel * (dists - dists.min())); R /= R.sum()
    expected = (1 - q) * R
    expected[0] += q                            # reuse mass lands on h=vertex 0
    assert np.max(np.abs(freq - expected)) < 0.03, f"freq {freq} vs mixture {expected}"


def test_sm_static_repeat_and_lifecycle():
    """SM-REM: same-cell repeat returns the SAME release (exact cache); an A-B-A
    trace returns the cached A on the revisit (memoized — the revisit channel is
    deterministic, i.e. the leak the theorem excludes); reset() clears the cache
    (lifecycle) — verifier R4-004."""
    # three widely separated vertices so A and B fall in different 60m cells
    rn = tiny_rn([(0.0, 0.0), (0.02, 0.02), (0.04, 0.0)])
    m = StayMemoizedREM(0.05, rn, grid_m=60.0, rng=np.random.default_rng(0))
    a_lat, a_lon = rn.latlon(0)
    b_lat, b_lon = rn.latlon(1)
    ra1 = m.perturb(a_lat, a_lon)
    ra2 = m.perturb(a_lat, a_lon)           # same cell -> cache hit
    assert ra1 == ra2, "static same-cell repeat must return the identical release"
    m.perturb(b_lat, b_lon)                 # different cell
    ra3 = m.perturb(a_lat, a_lon)           # A again -> still the cached A
    assert ra3 == ra1, "revisit returns cached release (deterministic revisit channel)"
    m.reset()
    assert m._cache == {}, "reset() must clear the cache (lifecycle)"


def test_sm_corner_cells_are_distinct():
    """Two points on opposite sides of a cell CORNER fall in diagonal cells whose
    representatives are √2·g apart — the basis of the exp(ε·√2·g) boundary bound
    (verifier R4-004/R4-005)."""
    rn = tiny_rn([(0.0, 0.0), (0.02, 0.02)])
    m = StayMemoizedREM(0.05, rn, grid_m=60.0)
    g = m.grid_m
    # a point just below-left of a grid corner, and just above-right of it
    import numpy as _np
    cx, cy = 5 * g, 5 * g  # a corner in projected metres
    p_lo = rn.proj.to_latlon(cx - 1.0, cy - 1.0)
    p_hi = rn.proj.to_latlon(cx + 1.0, cy + 1.0)
    c_lo = m._cell(rn.point_xy(float(p_lo[0]), float(p_lo[1])))
    c_hi = m._cell(rn.point_xy(float(p_hi[0]), float(p_hi[1])))
    assert c_lo != c_hi, "points across a corner must land in different cells"
    r_lo = _np.asarray(rn.point_xy(*m._cell_rep_latlon(c_lo)))
    r_hi = _np.asarray(rn.point_xy(*m._cell_rep_latlon(c_hi)))
    rep_dist = float(_np.hypot(*(r_lo - r_hi)))
    assert abs(rep_dist - g * (2 ** 0.5)) < 1.0, f"corner reps ≈ √2·g, got {rep_dist:.2f}"


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
    # inserting an unrelated mechanism does not change existing keys
    _ = seed_words(42, 0.02, "brand_new_mech")
    for m in mechs:
        assert seed_words(42, 0.02, m) == base[m]
    # the new mechanism's key differs from every existing one
    assert seed_words(42, 0.02, "brand_new_mech") not in base.values()
    # different ε -> different key
    assert seed_words(42, 0.01, "road_exponential") != base["road_exponential"]


def test_hmm_online_uses_logZ_like_offline():
    """The online (simulator) HMM path must apply the SAME input-dependent
    normaliser logZ as the offline benchmark path — otherwise the simulator
    evaluates a weaker attacker (verifier R4-007). Toggling logZ must change the
    online estimate; with logZ=None it must not."""
    from datetime import datetime
    from evaluation.attacks import HMMTrackingAttack, precompute_lognorm
    rn = tiny_rn([(0.0, 0.0), (0.0, 0.0009), (0.0, 0.0018), (0.0, 0.0027)])
    eps, scale = 0.05, 0.5
    ln = precompute_lognorm(rn, eps, scale=scale)
    released = [rn.latlon(1)]                      # single-report trajectory
    times = [datetime(2020, 1, 1)]
    with_ln = HMMTrackingAttack(rn, eps, emission_scale=scale, lognorm=ln)
    no_ln = HMMTrackingAttack(rn, eps, emission_scale=scale, lognorm=None)
    est_with = with_ln.online_estimates(released, times)[0]
    est_no = no_ln.online_estimates(released, times)[0]
    # logZ is non-constant across candidates, so it must shift the estimate.
    assert abs(est_with[1] - est_no[1]) > 1e-9 or abs(est_with[0] - est_no[0]) > 1e-9, \
        "online HMM ignores logZ — not at parity with offline (R4-007)"


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
