# Candidate declaration after fixed-history diagnosis — 2026-09-08

The diagnostic found mean recall 84.98% (learned/reachable), 89.74%
(uniform/reachable), 89.98% (learned/free), 95.00% (uniform/free).
The one-step truth-reference greedy diagnostic is 91.88% reachable / 100% free.
Location-mean diagnostic errors use the nearest native road-state coordinate,
not unsnapped GPS: learned 637.0 m, uniform 267.1 m, anchor 281.0 m.
Uniform changes both initial prior and the prior factors in approximate motion.
This is not proof that the initial prior alone caused the failure.

## Candidate and controls (declared before full-run scoring)

K=5 only in this bounded development cycle; B=.24/m, H=12, theta=200 m and
three original paired RNG seeds remain unchanged. Use the full first-up-to-12
allowed observations, not frozen predecessor states, in complete method runs.

Six methods:

- frozen geometric baseline, belief24 and learned service_cover;
- `uniform_cover`: same global greedy selector with a uniform-public-cell
  initial/motion prior (same grid, emission, approximate transition rule);
- `learned_corridor`: learned prior with directed-distance corridor;
- `uniform_corridor`: uniform prior with the same corridor.

Corridor rule: choose a public viable goal nearest the protected anchor.
For each track, compute its actual directed reachable candidates and their
directed road distance to that goal. Retain candidates no farther than the
minimum attainable distance PLUS 200 m. At least one candidate survives. Apply
the same greedy weighted union and original tie rules on these fixed groups.
This is a progress preference, not a traffic-aware future-route predictor.
The 200 m slack is a single prespecified value, tied to the existing spatial
scale theta; no claim that it is optimal. No grid of slack values this cycle.

All protected-anchor draws are paired with preceding runs. Uniform weights
assign equal mass to each occupied 120 m cell; do not make dense lane-state
sampling count as a more likely user location. Model hashes identify priors.
No extra private location read, budget charge or hidden future input is added.
The standard per-step 1/2 coverage bound concerns the restricted groups only;
it does not bound loss versus the unconstrained problem or real-user utility.

## Data, attacks, metrics and decision

Use pinned SQLite v3; do not enlarge/rewrite the DB just for this ablation.
101/102 train per-method shadow kNN {1,5,15}; 103/104 choose attacks/candidate;
201–204 are reused development diagnostics, explicitly NOT new confirmation.
Historical original baselines are re-used byte-identically at K=5, including
their shadow training data and seeds; train the three new methods on identical
records/seeds. Shadow labels never enter the protection method.

The declared primary metric stays six-category run Recall@5, then equal-weight
case/family/RNG aggregation. Report every category and valid/empty denominator,
completion, conditional extra distance, fixed/selected attack errors and Hit100,
distinct coordinates, initialization/step latency. Keep nominal anchor budget
and logical request counts equal. No replacement of recall by a favourable
category mix, no pseudo-independent event confidence interval.

Select among anchor-dependent methods using the existing min-case recall>=.90
then minimum strongest-validation Hit rule (fallback explicitly infeasible).
Freeze new source hashes and validation attack choices before reading the new
methods' 201–204 scores. This prevents within-cycle tuning but does NOT restore
held-out status to those already-inspected families. Publish every method.
No default promotion or submission-ready claim without fresh confirmation and
stronger adversaries. A negative corridor result must remain visible.
