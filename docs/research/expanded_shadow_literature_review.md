# Expanded auxiliary routes and inference: literature and critique

Reviewed 2026-09-08. A targeted primary-source review, not an exhaustive survey.
Question: does the low measured location-inference success survive more diverse
auxiliary mobility and nonlinear attacks, with the defender held fixed?

## Sources and boundaries

| Primary source / reading | Consequence for this experiment | What is not claimed |
|---|---|---|
| Eclipse SUMO, [Trip tools](https://sumo.dlr.de/docs/Tools/Trip.html), randomTrips edge sampling, seed and validation documentation | Select routes on the public network, check connectivity and actual completion; retain deterministic seeds and rejected plans. Our custom planner adds repeat-stop and query-window requirements. | Uniform departure-cell sampling is not calibrated urban demand. We did not reproduce randomTrips sampling or obtain observed human mobility. |
| Reza Shokri, [Privacy Games](https://petsymposium.org/popets/2015/popets-2015-0024.pdf), PoPETs 2015(2):299–315, Sections 6.1, 8–9 | Evaluate attacker decisions for their actual loss: Euclidean error and radius-hit require different actions. Preserve old attacks when adding more, and separate validation-selected performance from an exploratory attack envelope. | kNN neighborhoods are not an exact posterior; forest regression does not solve the paper's optimal-attack problem. A formal relative privacy bound is not an absolute error guarantee. |
| Guan, Guépin, Cretu, de Montjoye, [A Zero Auxiliary Knowledge Membership Inference Attack on Aggregate Location Data](https://www.petsymposium.org/popets/2024/popets-2024-0108.pdf), PoPETs 2024(4):80–101, Sections 2, 4.1–4.2, 7.3 | Synthetic mobility can be an attacker resource, so an audit need not rely on a tiny set of real labeled routes. Explicitly document which public data and simulator access the adversary assumes. | Their target is membership in released **aggregates**, not reconstruction of our dummy-only location stream. We neither reproduce ZK MIA nor call this attack zero-auxiliary-knowledge: it has synthetic shadow labels, mechanism knowledge and the old permitted training set. |
| scikit-learn, [ExtraTreesRegressor 1.7 API](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html), parameters and prediction | Add reproducible nonlinear direct/residual XY regressors; freeze 128 trees, depth 18, leaf size 5 and **integer** max_features=1 (one split feature), with installed version 1.7.1 recorded. Independently refit and compare serialized arrays and predictions. | Trees optimize squared error, not MAE or Hit. The one-feature setting may underfit; it is retained without score-driven tuning. API docs are implementation documentation, not evidence of attack optimality. |

The 2024 paper is recent supporting evidence for the attack-resource assumption,
not an additional protection comparator. This cycle does not establish novelty
by inventing the idea of synthetic shadow training or by adding ExtraTrees.

## Falsifiable design and data-quality counterexamples

The frozen protocol and pre-attack amendment are in
`thesis/notes/expanded_shadow_protocol.md` and
`thesis/notes/shadow_route_quality_amendment.md`.

1. **Native parking geometry failed.** At family 1048 a parked FCD point was
   6.793 m away from the lane position. The first rejected dataset SHA256 is
   `974eebdbf9ed62d71d9338e2f758ecd35754a56c8026bc60d99e216985e509bc`.
   On-lane stopping (`parking=false`) was adopted; the verifier gate was not
   relaxed and no point was moved after simulation.
2. **Different routes did not imply different windows.** The second dataset
   passed all 175,620 native-point checks, but had five duplicate stationary
   windows, including train 1036 / holdout 1069. Its SHA256 is
   `b252a0f988144c25844f8c0200e1f3e4e70bd75e2b64f8fd92ea95bc3c2cc819`.
   Public stop-lane and 1 m planned-stop-cell sampling without replacement was
   added before any attack fitting or scores. The rejected snapshots and
   generator versions remain under ignored `cache/urban_shadow_v1_rejected_*`.
3. **The accepted dataset is not geographically disjoint.** Exact route/window
   duplicates are absent, but common roads and nearby positions are allowed.
   The native verification reports 3,032 external edges, 901 shared with v3;
   holdout-to-training label distances have median 30.47 m and p95 159.59 m.
   This evaluates independent synthetic route families **on the same map**,
   not unseen-city or geographically separated generalization.
4. **Observation count is not independent sample size.** Auxiliary training
   contributes 2,873 observations from 64 families, with 1,345 unique XY labels.
   Stop/return windows share locations within a family. Old 360 observations
   remain in training; this is an observation-weighted empirical distribution,
   not family-balanced or population-calibrated sampling. The expanded core
   evaluation still has only four previously inspected families.

## Why the stronger bank may still fail

- Ordered dummy coordinates may encode the mechanism's tie-breaking and road
  behavior. This is legitimate public information, but this feature set does
  not exhaust temporal, likelihood, road-topology or POI-semantic inference.
- Broader raw coverage need not improve nearest neighbors in the 21-dimensional
  normalized **output-feature** space. Nearest true-label distance is a coverage
  diagnostic only, never an input to the attacker.
- Newly simulated cruising and long stops differ from the nine core case gates.
  The 16-family auxiliary holdout is not a new confirmation set for S1–S3.
  Each family runs two separated vehicle sessions without background traffic;
  repeated-stop schedules are controlled tests, not observed demand behavior.
- Only two original validation families choose attacks. A larger candidate bank
  increases selection variance; report per-family changes and both old and new
  validation-locked choices, even when the new choice generalizes worse.
- The expanded exploratory envelope cannot improve the defender's apparent
  privacy: maximum Hit cannot decrease and minimum MAE cannot increase. That is
  a set-inclusion fact, not evidence that the chosen new attack always wins.
- Do not treat measured attacks as a bound on all adversaries, or low attack
  success as a new Geo-I theorem. Existing composition/sampling caveats remain.

## Next publication gates

First validate the frozen bank numerically and report all unfavorable findings.
Then declare an independent next design: richer temporal/topological features
or mechanism-likelihood inference; auxiliary profile and spatial-shift ablations;
more independent validation families; fresh core-scenario confirmation only
after locking any new defender. Keep real-world demand calibration, multiple
maps, budget/length curves, utility failures and faithful comparator reproduction
as open gates. This audit improves evidence quality; it does not certify that a
conference will accept the paper.

## Figure contract (thesis-native static export)

- Question: how did geographic support of shadow labels change?
- Family: spatial scatter over native SUMO road geometry, three equal-scale
  panels (old training, added training, auxiliary holdout).
- Grain: unique projected XY coordinate within each split; 36 / 1,345 / 322
  points expected; raw manifest retains families, schedules and source hashes.
- Takeaway: broader same-map support, not 1,345 independent users or geographic
  separation. Empty areas and shared roads remain visible.
- Surface: PNG embedded in canonical LaTeX/PDF; Matplotlib static renderer;
  no web tiles or a separate HTML report. This is the user's thesis, not a
  branded analytics report.
- Palette: two roots (blue/orange) plus charcoal roads; panel titles, circles,
  crosses and open triangles distinguish roles without relying on color.
- QA: equal XY scaling, kilometre axes, readable legend/panel counts, native
  network attribution and inspection inside the final thesis PDF.
