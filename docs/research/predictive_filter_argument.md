# Fixed-cap predictive filter: precise claim and limits

This is a new implementation of an established predictive-budget idea, not a
new definition of privacy. Primary prior work: Chatzikokolakis, Palamidessi and
Stronati, *A Predictive Differentially-Private Mechanism for Mobility Traces*,
PETS 2014: https://arxiv.org/abs/1311.4008 . The original BR fixed-horizon ledger
and all its historical evidence remain unchanged.

Let u = B/(2H). Internally consider the **extended** transcript containing each
private-test branch and every protected anchor. Each first fresh anchor costs
one unit; a later reuse costs one unit for the private test; a refresh costs two
units for test plus fresh anchor. Before accessing the next GPS value the filter
reserves the worst possible cost (one initially, two thereafter). If insufficient,
it permanently continues with postprocessing and never reads new GPS. Costs are
integer units, so no floating-point budget threshold can overspend.

For fixed public timestamps and length, compare trajectories x,x' under
D_infinity=max_t ||x_t-x'_t||. The Laplace-threshold pass probability q_x and its
complement each have likelihood ratio at most exp(u||x-x'||). The full-support
ideal REM fresh kernel has the same u-Geo-I bound. At a fixed extended transcript,
a reuse factor therefore costs u and a refresh factor costs 2u (first: u).
Future spending/stopping decisions are deterministic functions of that same
extended history, hence agree when evaluating its likelihood under x and x'.
Multiplying ratios gives exp(sum_t c_t*u*||x_t-x'_t||), at most
exp(B*D_infinity), because **every feasible path** spends at most 2H units.
Marginalizing away branches and applying public-context dummy selection preserves
this bound. It is not valid to take arbitrary old runs and retrospectively call
only their refresh count a privacy guarantee: the prospective filter is essential.

Assumptions/limits:

- This is an ideal-kernel argument, not a floating-point secure sampler theorem.
- Input domain is valid finite WGS84 coordinates, fixed public clock/schedule and
  public map/support. Secret start/stop timing, identity and cross-session linking
  are outside this single-session statement.
- Multiple linked sessions compose; two B=.24 sessions have bound .48 unless a
  separate subject-level accountant allocates a common cap.
- The approximate service belief observes protected anchors but does not fully
  condition on internal branch history. Utility calibration is empirical; this
  does not grant the belief a calibrated adversarial posterior interpretation.
- Empirical Hit/MAE can worsen despite the same formal cap. The acceptance loop
  must measure this, whole-session utility and attacks on stationary/repeated
  targets. No asymptotic anti-averaging claim follows.
