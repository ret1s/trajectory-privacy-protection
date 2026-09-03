"""Honest property test for the finite-precision Gumbel-max sampler (R2-006, G0).

This test does NOT assert the sampler is correct — it DOCUMENTS the exact
finite-precision defect the verifier found, so the limitation is version-checked
and cannot silently be re-claimed as "pure Geo-I". Two facts are asserted:

 1. NumPy's Gumbel draw is bounded (built from a 53-bit uniform), so the span
    between two draws cannot exceed ≈40.34. Hence a candidate whose logit is
    more than that below the leader can NEVER win argmax(logit+Gumbel).
 2. Concretely: with a logit gap of 45 (> span), the low-logit candidate is
    unreachable over many draws, even though its ideal softmax probability is
    positive — an input-dependent zero-support, i.e. the executable is only a
    numerical approximation of the ideal ε-Geo-I kernel.

Run: python -m tests.test_sampler_support   (or via pytest)
"""
import math
import numpy as np

# Documented finite bounds for numpy's Gumbel from a 53-bit uniform in (0,1).
GUMBEL_MIN = -math.log(-math.log(2.0 ** -53))   # ≈ -3.60
GUMBEL_MAX = -math.log(-math.log(1.0 - 2.0 ** -53))  # ≈ 36.74
MAX_SPAN = GUMBEL_MAX - GUMBEL_MIN               # ≈ 40.34


def test_gumbel_span_is_bounded(n=2_000_000, seed=0):
    g = np.random.default_rng(seed).gumbel(size=n)
    span = float(g.max() - g.min())
    assert span <= MAX_SPAN + 1e-6, f"observed span {span} exceeds theoretical {MAX_SPAN}"
    print(f"[PASS] Gumbel span bounded: observed {span:.4f} <= theoretical {MAX_SPAN:.4f}")


def test_far_candidate_is_unreachable(gap=45.0, draws=200_000, seed=0):
    """Leader at logit 0, one candidate at logit -gap (gap > MAX_SPAN).
    The far candidate must never win — zero executable support though its
    ideal softmax probability exp(-gap) > 0."""
    rng = np.random.default_rng(seed)
    logits = np.array([0.0, -gap])
    wins_far = 0
    for _ in range(draws):
        g = rng.gumbel(size=2)
        if int(np.argmax(logits + g)) == 1:
            wins_far += 1
    ideal_p = math.exp(-gap) / (1.0 + math.exp(-gap))
    assert gap > MAX_SPAN, "test only meaningful when gap exceeds the Gumbel span"
    assert wins_far == 0, f"far candidate won {wins_far} times (should be 0 in finite precision)"
    assert ideal_p > 0
    print(f"[PASS] far candidate (gap {gap:g} > span {MAX_SPAN:.2f}): executable wins="
          f"{wins_far}, ideal prob={ideal_p:.3e} > 0  → input-dependent zero support "
          "(documented, not fixed)")


if __name__ == "__main__":
    test_gumbel_span_is_bounded()
    test_far_candidate_is_unreachable()
    print("Sampler-support limitation is version-checked. Executable ≠ ideal kernel.")
