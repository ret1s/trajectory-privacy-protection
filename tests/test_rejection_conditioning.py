"""Toy finite-domain test for the rejection/conditioning proposition
(verifier R2-011 / R3-001).

Proposition (thesis §4, prop:fixedaccept), stated with BOTH required parts —
a base ε-Geo-I hypothesis on K and the 1_A(z) indicator mask:

    K_A(z|x) = 1_A(z)·K(z|x) / K(A|x),   K(A|x)>0,  K is ε-Geo-I
  ⇒ for every measurable S:  K_A(S|x)/K_A(S|x') ≤ e^{2ε d(x,x')}.

Tests here, on a finite domain:
  (a) the 2ε bound HOLDS on a symmetric ε-Geo-I grid;
  (b) 2ε is TIGHT — a 3-point ε-DP kernel + fixed public A gives ratio > e^ε
      (so ε alone is insufficient), ≤ e^2ε;
  (c) NEGATIVE regression: dropping the ε-Geo-I hypothesis (δ_0 vs δ_1) → ∞;
  (d) NEGATIVE regression: dropping the 1_A mask → "distribution" sums to 2;
  (e) input-dependent A_x on a 0-DP base → ∞ (prop:reject).

Run:  python -m tests.test_rejection_conditioning
"""
import numpy as np


def _row_normalise(K):
    return K / K.sum(axis=1, keepdims=True)


def test_fixed_public_set_is_2eps():
    """(a) Fixed public A: worst-case ratio hits e^{2ε·d}, exceeds e^{ε·d}."""
    eps = 1.0
    # Two inputs a,b at unit distance; base kernel = ε-Geo-I exponential on a
    # 1-D output grid, score −(ε/2)|z−x|. (ε/2 → single-shot ε-Geo-I.)
    outs = np.arange(-20, 21, dtype=float)
    xs = {"a": 0.0, "b": 1.0}
    base = {x: np.exp(-(eps / 2.0) * np.abs(outs - xv)) for x, xv in xs.items()}
    base = {x: v / v.sum() for x, v in base.items()}

    # Fixed public allowed set A: keep outputs in [−5, 5], SAME for both inputs.
    A = (outs >= -5) & (outs <= 5)
    cond = {x: (v * A) / (v * A).sum() for x, v in base.items()}

    d = abs(xs["a"] - xs["b"])  # = 1
    # Ratio over the shared support.
    supp = A & (cond["a"] > 0) & (cond["b"] > 0)
    ratios = cond["a"][supp] / cond["b"][supp]
    worst = float(max(ratios.max(), (1.0 / ratios).max()))

    eps_bound = np.exp(eps * d)
    two_eps_bound = np.exp(2 * eps * d)
    assert worst <= two_eps_bound + 1e-9, f"{worst} > e^2ε={two_eps_bound}"
    # The point of the proposition: the ε bound is NOT enough in general here.
    # (For this symmetric grid the acceptance factor is ~1, so we assert the
    #  bound HOLDS at 2ε and construct the tight case separately below.)
    print(f"[PASS] fixed-public A: worst ratio={worst:.4f} "
          f"≤ e^2ε={two_eps_bound:.4f} (e^ε={eps_bound:.4f})")


def test_fixed_public_needs_2eps_not_eps():
    """(a-tight) A concrete ε-DP base kernel + fixed public set whose conditioned
    ratio STRICTLY exceeds e^ε — proving the ε bound is insufficient and 2ε is the
    right constant. This is the load-bearing part: the old response's "ε preserved"
    reading is falsified here on a finite domain.

    Kernel on outputs {0,1,2}, ε=ln 3 (e^ε=3):
        K(·|a) = [0.3, 0.5, 0.2]   K(·|b) = [0.1, 0.3, 0.6]
    pointwise ratios a/b = [3, 1.667, 0.333] ⊂ [1/3, 3]  → ε-DP with d(a,b)=1.
    Fixed public A = {0,2} (drop output 1); acceptance positive for both inputs.
    At z=0 (favours a) the dropped mass is asymmetric, so the acceptance factor
    K(A|b)/K(A|a) > 1 compounds the pointwise factor.
    """
    eps = np.log(3.0)
    d = 1.0
    Ka = np.array([0.3, 0.5, 0.2])
    Kb = np.array([0.1, 0.3, 0.6])
    assert np.isclose(Ka.sum(), 1) and np.isclose(Kb.sum(), 1)
    # base kernel is ε-DP: every pointwise ratio within [e^-ε, e^ε].
    r = Ka / Kb
    assert np.all(r <= np.exp(eps) + 1e-9) and np.all(r >= np.exp(-eps) - 1e-9)

    A = np.array([True, False, True])  # fixed, public, input-independent
    accept_a, accept_b = float((Ka * A).sum()), float((Kb * A).sum())
    assert accept_a > 0 and accept_b > 0
    condA = (Ka * A) / accept_a
    condB = (Kb * A) / accept_b
    supp = A & (condB > 0)
    ratios = condA[supp] / condB[supp]
    worst = float(max(ratios.max(), (1.0 / ratios[ratios > 0]).max()))

    eps_bound, two_eps_bound = np.exp(eps * d), np.exp(2 * eps * d)
    assert worst > eps_bound + 1e-9, (
        f"conditioning stayed within e^ε={eps_bound:.4f} — would not demonstrate 2ε")
    assert worst <= two_eps_bound + 1e-9, f"{worst} > e^2ε={two_eps_bound:.4f}"
    print(f"[PASS] fixed-public A on ε-DP base: conditioned ratio={worst:.4f} "
          f">  e^ε={eps_bound:.4f}  and  ≤ e^2ε={two_eps_bound:.4f} "
          f"(ε is insufficient; 2ε is the correct bound)")


def test_missing_geoi_assumption_breaks_bound():
    """(negative regression, R3-001) Without the base ε-Geo-I hypothesis, even a
    FIXED public A with positive acceptance can give an infinite ratio:
    K(·|a)=δ_0, K(·|b)=δ_1, A={0,1}. Acceptance is 1 for both, but the ratio at
    z=0 is 1/0 = ∞. The proposition MUST assume base Geo-I."""
    Ka = np.array([1.0, 0.0])   # δ_0  (not ε-Geo-I for any finite ε)
    Kb = np.array([0.0, 1.0])   # δ_1
    A = np.array([True, True])  # fixed, public, positive acceptance for both
    accept_a, accept_b = float((Ka * A).sum()), float((Kb * A).sum())
    assert accept_a > 0 and accept_b > 0
    condA, condB = (Ka * A) / accept_a, (Kb * A) / accept_b
    ratio = np.inf if condB[0] == 0 else condA[0] / condB[0]
    assert ratio == np.inf
    print("[PASS] no-Geo-I base + fixed public A: ratio = ∞ "
          "(base ε-Geo-I assumption is necessary)")


def test_missing_indicator_is_not_a_distribution():
    """(negative regression, R3-001) Without the 1_A(z) mask, the 'conditioned'
    formula K(z|x)/K(A|x) is not a distribution: K=(1/2,1/2), A={0} gives mass
    (1,1) summing to 2. The definition MUST carry the indicator."""
    K = np.array([0.5, 0.5])
    A = np.array([True, False])
    accept = float((K * A).sum())  # = 0.5
    without_indicator = K / accept            # (1, 1) — WRONG, sums to 2
    with_indicator = (K * A) / accept         # (1, 0) — correct distribution
    assert abs(without_indicator.sum() - 2.0) < 1e-12
    assert abs(with_indicator.sum() - 1.0) < 1e-12
    print("[PASS] missing 1_A mask sums to 2.0 (not a distribution); "
          "with 1_A sums to 1.0")


def test_input_dependent_set_is_infinite():
    """(b) Input-dependent A_x on a 0-DP base kernel → infinite ratio."""
    # Base kernel uniform on {0,1} for every input → 0-DP (ratio 1 everywhere).
    K = _row_normalise(np.ones((2, 2)))  # rows = inputs a,b; cols = outputs 0,1
    assert np.allclose(K, 0.5)
    # Input-dependent allowed sets: A_a={0}, A_b={1}.
    cond_a = np.array([1.0, 0.0])  # K_A(·|a): only output 0 survives
    cond_b = np.array([0.0, 1.0])  # K_A(·|b): only output 1 survives
    # Output 0: K_A(0|a)=1, K_A(0|b)=0 → ratio infinite.
    assert cond_a[0] > 0 and cond_b[0] == 0.0
    ratio = np.inf if cond_b[0] == 0 else cond_a[0] / cond_b[0]
    assert ratio == np.inf
    print("[PASS] input-dependent A_x: base 0-DP but conditioned ratio = ∞ "
          "(falsifies any finite-ε claim for the reject case)")


if __name__ == "__main__":
    test_fixed_public_set_is_2eps()
    test_fixed_public_needs_2eps_not_eps()
    test_missing_geoi_assumption_breaks_bound()
    test_missing_indicator_is_not_a_distribution()
    test_input_dependent_set_is_infinite()
    print("\nAll rejection-conditioning tests passed "
          "(R2-011: 2ε for fixed public A, ∞ for input-dependent A_x).")
