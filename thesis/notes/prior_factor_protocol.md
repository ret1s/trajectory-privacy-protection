# Prior-factor and loss-aware attacker development protocol

Declared 2026-09-08, parent `50d15250603070e9f2f861a40eba2d5df52b5fea`.
Written before computing the new variants or new attack scores.

## Questions and fixed scope

The uniform-cell improvement changes initial prior, transition/reset prior and
lane-state multiplicity simultaneously. Separate these mechanisms rather than
attribute the gain to initial prior alone. Also test whether point averaging is
an adequate attack decision for Euclidean error or a 100 m hit objective.

SUMO + OSM SQLite v3 is unchanged. Training families 101/102, validation 103/104,
already-inspected development 201--204; NO fresh confirmation. Nine S1--S3 cases,
first at most 12 allowed events, K=5, B=.24/m, theta=200 m, three paired replicates.
No new city, K, budget, traffic, S4--S10 protection or faithful SOTA claim.

## Seven configurations, declared before scores

Reuse baseline BR-lane, learned service-cover (L,L), and uniform cover (U,U)
from service_recovery byte-identical in generation fields and old attack errors.
Four new complete-run variants:

1. (L,U): learned initial prior, uniform-prior motion and long-gap reset.
2. (U,L): uniform initial prior, learned-prior motion and long-gap reset.
3. (C,C): learned lane density divided by the number of states in each occupied
   120 m public cell, then normalized. Cell mass is proportional to the mean
   learned density, not its sum. This is not uniform or real population truth.
4. (M,M): fixed 50:50 mixture of normalized C and U lane-state distributions.
   No mixture/slack sweep after results.

Use the same grid/emission normalizers and greedy reachable POI union. The 2x2
comparison isolates initialization (including its fixed tie-break center) from
motion-prior weighting/reset. It does not isolate the transition from reset or
eliminate every lane multiplicity effect; the anchor kernel stays unchanged.
Do not add raw GPS access, rejection sampling, corridor or trajectory lookahead.

## Expanded attacker bank

Keep all preceding attacks, including full-window track attacks. Add causal,
mechanism-specific shadow decisions using the SAME training features/labels.
For 15 and 45 nearest standardized training examples, use uniform empirical
mass and choose (a) minimum mean Euclidean distance among neighbor coordinates
and their mean, or (b) maximum mass in a 100 m disk among neighbor coordinates,
their mean, and both radius-100 circle centers through each feasible point pair.
Include the mean-45 control. Stable neighbor/action ordering breaks ties.

This is a loss-aware nearest-neighbor empirical conditional distribution, NOT
the mechanism's exact likelihood, NOT a calibrated posterior, and NOT a global
optimal attacker. Finite-action MAE optimum is not the continuous geometric
median. Disk construction has only floating-point tolerance guarantees.
Compare its empirical objective against the mean, but do not assume it will
dominate mean prediction on true development labels. Do not expose evaluator
anchors, true positions, case labels or future events to these new predictors.
Train all seven shadow models on 101/102. Select each attack by method and case
using only 103/104, independently for MAE and Hit. Old choices remain preserved
as historical comparisons; report expanded selections and exploratory envelopes.

## Selection, aggregation and interpretation

Same selection rule: all nine validation case means must reach Recall .90;
then lowest macro strongest-validation Hit. Otherwise explicitly infeasible
fallback maximizing minimum case recall, then Hit, then stable method name.
Lock source hashes and selection before new outputs for 201--204.
Equal case/family/replicate averaging with per-run eligible category-event recall;
retain empty references as N/A. Show cafe/restaurant and worst-case recalls.
Report all seven configurations, not just a winner. No statistical significance
claim based on four already-inspected families or RNG repetitions.

## Verification gates and review

Unit tests: normalized density balancing, exact identical-factor control,
transition/long-gap separation, causal prefixes, paired anchors, post-H GPS
independence, loss-aware decisions, stable ties and invalid input rejection.
Independent verifier: source/data lineage, old-field equality, full raw-GPS
attack errors, all POI metrics, selected attacks/candidate, directed transitions,
shadow training, empirical-loss optimality on declared actions, replicate-1
full/prefix replay for new methods. Freeze results; no overwrite of predecessors.

Primary literature motivating this cycle: Chatzikokolakis et al. PoPETs 2017
Sections 3/3.3 (prior-dependent remapping); Shokri, Privacy Games, arXiv
1402.3426v3 Sections 3.3/3.4/5 (attack decisions depend on the loss).
Neither prior mixing nor Bayes decision theory is claimed as novelty.
