"""Property test for the PR-SM-REM per-step privacy bound (verifier R2-001).

Enumerates the exact two-branch kernel on a two-vertex toy and checks that the
per-step likelihood ratio obeys the CORRECTED bound exp((ε_test+ε_release)·d)
and VIOLATES the old (wrong) bound exp(ε_test·d). No road graph needed — the
kernel is written out analytically:

    q_x(h) = P[d(x,h) + Lap(1/ε_test) ≤ θ]              (reuse probability)
    R_x(z) = exp(-ε_r/2·d(x,z)) / Σ_v exp(-ε_r/2·d(x,v))  (REM release, ε_r=ε_release)
    K_x(z|h) = q_x(h) + (1-q_x(h))R_x(h)   if z == h
             = (1-q_x(h))R_x(z)            if z != h

Run: python -m tests.test_pr_privacy_bound   (or via pytest)
"""
import math


def _laplace_cdf(t, scale):
    """P[Lap(0, scale) ≤ t]."""
    if t < 0:
        return 0.5 * math.exp(t / scale)
    return 1.0 - 0.5 * math.exp(-t / scale)


def _kernel(x, h, verts, eps_test, eps_release, theta):
    """Exact PR two-branch kernel P(z|x,h) over a finite vertex set `verts`."""
    a = eps_release / 2.0
    w = {v: math.exp(-a * abs(x - v)) for v in verts}
    Z = sum(w.values())
    R = {v: w[v] / Z for v in verts}
    q = _laplace_cdf(theta - abs(x - h), 1.0 / eps_test)  # reuse prob
    K = {}
    for z in verts:
        if z == h:
            K[z] = q + (1 - q) * R[z]
        else:
            K[z] = (1 - q) * R[z]
    return K


def test_two_vertex_bound():
    a_pt, b_pt = 0.0, 100.0
    verts = [a_pt, b_pt]
    h = a_pt
    theta = 200.0
    eps_test = eps_release = 0.01
    d = abs(a_pt - b_pt)  # 100 m

    Ka = _kernel(a_pt, h, verts, eps_test, eps_release, theta)
    Kb = _kernel(b_pt, h, verts, eps_test, eps_release, theta)

    ratio = Kb[b_pt] / Ka[b_pt]  # event z=b, inputs b vs a
    old_bound = math.exp(eps_test * d)                 # 2.718...  (WRONG)
    new_bound = math.exp((eps_test + eps_release) * d)  # 7.389...  (CORRECT)

    assert ratio > old_bound + 1e-9, f"ratio {ratio} should exceed old bound {old_bound}"
    assert ratio <= new_bound + 1e-9, f"ratio {ratio} should be within new bound {new_bound}"
    # matches the verifier's hand-computed value exp(1.5) = 4.4817
    assert abs(ratio - math.exp(1.5)) < 1e-3, f"ratio {ratio} != exp(1.5)"
    return ratio, old_bound, new_bound


def test_random_finite_domain_bound(n=2000, seed=0):
    """Random finite one-step checks: the corrected bound is never violated."""
    import random

    rng = random.Random(seed)
    worst = 0.0
    for _ in range(n):
        verts = sorted({round(rng.uniform(-300, 300), 1) for _ in range(rng.randint(2, 6))})
        if len(verts) < 2:
            continue
        eps_test = rng.choice([0.005, 0.01, 0.02, 0.05])
        eps_release = rng.choice([0.005, 0.01, 0.02, 0.05])
        theta = rng.choice([100.0, 150.0, 200.0, 300.0])
        h = rng.choice(verts)
        x, xp = rng.choice(verts), rng.choice(verts)
        if x == xp:
            continue
        d = abs(x - xp)
        bound = math.exp((eps_test + eps_release) * d)
        Kx = _kernel(x, h, verts, eps_test, eps_release, theta)
        Kxp = _kernel(xp, h, verts, eps_test, eps_release, theta)
        for z in verts:
            if Kxp[z] > 0:
                r = Kx[z] / Kxp[z]
                worst = max(worst, r / bound)
                assert r <= bound + 1e-9, f"violated: r={r} bound={bound}"
    return worst  # should be <= 1.0


if __name__ == "__main__":
    ratio, ob, nb = test_two_vertex_bound()
    print(f"[PASS] two-vertex: ratio={ratio:.4f}  old={ob:.4f} (violated) new={nb:.4f} (ok)")
    worst = test_random_finite_domain_bound()
    print(f"[PASS] 2000 random finite-domain checks; worst ratio/bound = {worst:.4f} (<=1)")
    print("All PR privacy-bound tests passed.")
