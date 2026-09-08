# Expanded SUMO shadow routes and inference audit

Declared 2026-09-08, parent `07634a0ed3daa2025c76a19a85e6344e26dfef89`,
before generating the new routes or scoring the expanded attack bank.

## Question and fixed boundary

Does the apparent privacy of the seven frozen prior-factor configurations
survive broader auxiliary mobility and nonlinear public-output inference?
No protector, prior, output, budget, original score, or case eligibility is
changed. Keep K=5, B=.24/m, theta=200 m, horizon=12. No GeoLife.
Existing 103/104 select attacks only; 201--204 remain reused development.
Uniform U/U remains the previously selected defense, with utility infeasible.
Do not promote M/M based on these results or call this fresh confirmation.

## Auxiliary dataset, before any attack score

- Existing pinned SUMO passenger network and OSM catalogue only. Public-map
  route planning, not selection against validation/development true locations.
- 80 independently seeded route families (1001--1080). Seeds 1001--1064 are
  auxiliary training; 1065--1080 are an auxiliary holdout diagnostic, never fit.
- Sample departure edges by occupied 120 m midpoint cell, without replacement
  across families; destinations from eligible passenger edges. This balances
  departure cells, NOT a calibrated city population or uniform road occupancy.
- Connected shortest routes, 2.5--4 km, >=12 external edges; a reachable return
  to a middle stop edge. Reject infeasible plans before simulation, keep counts.
  Full route identities must be unique. Shared road segments are allowed.
- Each family has a normal trip and a return trip with two 180-second stops
  at the same lane, separated by a route-enforcing one-second waypoint. Real
  SUMO runs at 1 Hz, maxSpeed=8 m/s, teleport disabled. Preserve raw FCD.
- Four auxiliary schedules: cruise20 (<=12 events), cruise60 (<=12), stop5
  (12 events), return5 (6 events per visit, >120 s separating visits).
  Windows use only realized FCD samples. These are AUX profiles, NOT new
  S1--S10 protection claims or substitutes for scenario eligibility gates.
- One independent paired protection RNG per auxiliary window; all seven
  methods receive the same window and private-anchor draw. No extra replicas
  to inflate the number of independent routes.
- Validate completion, geometry, temporal continuity, native route/stop XML,
  uniqueness, profile eligibility and source hashes before importing as
  `urban-shadow-v1`, parent `urban-scenarios-v3`, into the existing registry.
  Schema remains the existing normalized v2 data contract. Training and holdout
  retain distinct family splits; release purpose/role is explicitly auxiliary.
- Audit exact route/observation overlap with all old v3 families AFTER the
  fixed design; report shared-edge and spatial coverage, not fictitious zero
  overlap of public roads. Any exact old route or full observation-window
  duplicate is a blocking leakage finding, not grounds for silent replacement.

## Attack bank and separation of effects

1. Retain every old attack/error. Frozen protected outputs and utility remain
   byte-for-byte equal at their scientific fields.
2. Expand each method's original 360-observation training bank with ONLY the
   64 auxiliary training families; retain deterministic row ordering. Same
   causal 21 features: ordered current candidate XY, prefix means, elapsed time.
   Fit feature normalization on training only. No anchors, route plans, actual
   speed, stop labels, identity, case ID, or future points enter predictors.
3. Refit kNN mean 1/5/15 and the previous finite-action MAE/disk Hit rules at
   15/45 neighbors plus mean45. These matched rules isolate the data expansion.
4. Add ExtraTrees regression (128 trees, min_samples_leaf=5, max_depth=18,
   max_features=1, bootstrap=False, fixed seed, n_jobs=1). Test direct XY and
   residual XY relative to the current candidate centroid. The latter adds the
   centroid back at inference. Both use the SAME 21 causal features. These
   regressors optimize squared error, not an exact posterior or Bayes-optimal
   MAE/Hit action. Keep all outcomes, including worse-than-kNN results.
5. No hyperparameter search after validation/development results. First lock
   training artifacts, then select one attacker per method/case/metric using
   103/104 only. Report unchanged old selection alongside expanded selection.
   Keep the exploratory per-case attack envelope separate from selected scores.
6. Auxiliary holdout reports each attack on its four profiles and a globally
   validation-selected attacker; it is not a confirmation sample drawn from the
   nine core scenario gates. Do not fit or tune using it.

## Measurements and verification gates

- Raw-GPS Euclidean MAE (m), strict Hit<=100 m; equal case/family/replicate
  aggregation on original benchmark. Utility is reused exactly, not recomputed
  under different queries. Report paired family changes, not step-level p-values.
- Report training observations vs unique XY vs independent route families;
  occupied 120 m cells; nearest-training-label distance for evaluation points
  (evaluator-only coverage diagnostic, never used to make predictions).
- Check source/DB content manifests, immutable v3 parity, no new fit labels from
  validation/development/holdout, causal prefix invariance, all raw-GPS scores,
  selected-attacker provenance, monotone expanded exploratory envelopes and
  deterministic replay of sampled auxiliary protection and model fitting.
- If results show stronger leakage, report the increase; do not weaken attacks,
  remove hard cases or optimize the defender during this audit.
- A richer synthetic attack is still not an exact mechanism likelihood or an
  upper bound on adversarial power. Same-map synthetic trips, frozen schedules,
  small target validation and reused development remain limitations.

Primary motivation: SUMO Trip documentation (validated connectivity and
nonuniform departure sampling); Shokri, PoPETs 2015 Privacy Games (loss-aware
inference); Guan et al., PoPETs 2024 Zero Auxiliary Knowledge MIA (synthetic
auxiliary traces are a plausible attack resource, but on a different target).
Detailed source reading and distinctions belong in the final literature note.
