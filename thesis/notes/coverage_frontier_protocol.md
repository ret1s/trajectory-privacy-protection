# Coverage frontier development cycle — 2026-09-08

## Questions and status

1. Does refining a greedy reachable query set improve the objective and actual
   service results, or merely optimize a misspecified belief?
2. Does capping already-covered categories reduce weak-category failures?
3. How much recall comes from the protector versus a larger server reply?

This is a new **development** cycle on already inspected SUMO+OSM data. It is
not fresh confirmation, not a SOTA reproduction, and not an all-scenario claim.
Existing releases, protectors, experiments and selection records stay sealed.
No auxiliary holdout (1065–1080) is used in this cycle, even for selection.

## Fixed grid, recorded before evaluation

- Four methods: `geometric` (existing BR-lane); `mean_greedy` (existing uniform
  U/U coverage); `mean_exchange` (same objective, up to three strictly improving
  single-track exchanges); `capped_exchange` (same exchanges, category-saturated
  objective, fixed cap 0.9). All are online dummy-only K=5, H=12, theta=200 m.
- B in {0.12, 0.24, 0.48}/m. Anchor release/test each receive B/(2H).
  Same raw inputs, query times, map, seed and anchor draws within each B.
- Both new methods start from a globally greedy assignment and examine ALL
  feasible alternatives for each single-track exchange. At most three exchanges;
  improvement must exceed 1e-12. No claim of convergence or global optimality.
- Capping uses each category's contribution under the existing approximate
  belief, normalized by its total weight; empty categories are excluded from
  that proxy only. This differs from population-calibrated per-category recall.
  Mean exchange retains the original unnormalized macro objective exactly.
- Primary service: client top-5, server top-5 per dummy/category. Secondary
  top-10 replies preserve the **same coordinates**; all four methods get this
  extension. Top-10 does not retune the protector or change reference top-5.
  It is a resource sensitivity, not free utility or algorithmic novelty.
- Six public categories are queried at every event (K*6 requests, duplicates
  counted; no cache). Report reply-item counts and serialized ID-list bytes,
  not network bytes or production latency. Local filtering/ranking is not sent.

## Splits, attacks and decision rules

Training: core families 101/102, three replicas per record, plus all 256 auxiliary
training windows from families 1001–1064, one replica each. No score-based data
filtering. Validation: 103/104, three replicas. Reused development: 201–204,
three replicas. Nine cases S1.A–S3.C, up to 12 allowed events. Existing RNG keys
are retained to audit B=.24 baseline parity. All runs and all candidates saved.

Each method/B gets its own shadow model trained on the same 3,233 observations:
old-core-only kNN/loss-aware rules AND expanded-data kNN/loss-aware rules and
two ExtraTrees regressors, with the expanded-shadow-v1 pinned parameters.
Keep centroid, prior, continuity, track and applicable full-window attacks.
Raw GPS targets never enter inference. Use causal 21-dimensional features;
existing full-window heuristics remain explicitly retrospective. Select MAE and
Hit@100m attackers separately per case on validation. Also show exploratory
per-case extrema on development; these are not bounds on all attackers.

No single scalar "privacy score" and no unconditional winner. For each B and
reply depth, require minimum validation case recall >=.90 (12-decimal rule),
then minimize validation macro strongest Hit100; if none meets the gate, record
no feasible candidate. Publish the chosen method AND every other method on
reused development, without reselection. Budget/depth remain user-facing axes,
not knobs secretly selected using development. Empirical non-dominance uses
recall, selected Hit100, response bytes and generation latency; it is exploratory,
not a significance claim. Report family-paired deltas, not step-level p-values.

## Claims and falsification

The category-saturated objective is a sum of truncated nonnegative weighted
coverage functions. It is monotone submodular on track-state pairs; fixed
one-state-per-track constraints form a partition matroid. Ordinary greedy's
classical per-step 1/2 guarantee survives strictly improving exchanges in ideal
arithmetic. It is NOT a guarantee of 90% true recall, worst-case category recall,
future quality, or new privacy. Each category score <=1 follows normalization.
The cap is a surrogate for reducing shortfall, not a max-min optimizer.

Postprocessing uses only protected-anchor history, public timing/map/catalogue
and past published states. The ideal inherited ledger is unchanged. Enlarging
a deterministic reply from the public catalogue at fixed public depth adds no
information beyond public query coordinates under this threat model. It does
cost communication; private adaptive depth or uploading locally ranked results
would invalidate this argument. Floating sampling/composition caveats remain.

Novelty is a hypothesis about **causal, reachable service-set optimization with
explicit resource accounting**, not invention of Bayesian remapping, greedy,
saturation, local search, Geo-I or top-L retrieval. Compare to closest sources
and retain failed ablations. No promotion if protection/utility/cost evidence
does not support it. A narrower contribution is preferable to unsupported
coverage of all ten threats.

## Verification and figure contract

Tiny exhaustive objective/gain oracles; nonnegative diminishing gains; exchange
monotonicity; category empty/normalization; exact uncapped parity; strict prefixes;
ignored coordinates after H; anchors absent from public view. Independent
recomputation of feature matrices, attack decisions/errors, service reference,
replies/ranking/cost, selection, family aggregation and directed motion. Preserve
hashes and source manifests; save negative findings in the verifier.

Figure: thesis-native Matplotlib scatter, 12 method/B points per reply-depth
panel, same development population and axes. X=macro Recall@5, Y=exploratory
Hit@100m (lower better); label B, distinguish methods with shapes and restrained
blue/orange/neutral colours. Show the 90% mean line only with a note that the
actual gate is **minimum case**, not mean. Exact tables retain minima, selected
attacks, MAE, costs and latency; no uncertainty implied by connecting points.
QA the static figure and compiled thesis PDF. No separate web report.
