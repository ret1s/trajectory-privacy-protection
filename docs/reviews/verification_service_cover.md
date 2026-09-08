# Verification: joint service-cover cycle — 2026-09-08

Parent: `232adf38b410273f1a925cc9610d568c8c837edb` on `main`.
Scope: proposed method, new SUMO confirmation data, SQLite consumption,
method-aware attacks, critical literature review and canonical thesis/PDF.

## Research decision

The new method replaces a geometric-plus-coverage score with **global greedy
selection of the whole reachable query set**. It retains the frozen private
anchor kernel, bounded horizon and approximate protected-history filter.
The mathematical objective is weighted POI union coverage with one state per
persistent dummy track. Its classical per-step greedy guarantee is 1/2 under
exact arithmetic/oracle; this is not a new privacy theorem or horizon guarantee.

Four methods: baseline, belief24, service_cover, prior_cover. K=3/5, B=.24/m,
H=12, theta=200 m; three paired mechanism seeds per record. Prior-only is a
negative control, excluded from proposed-method selection.

The prespecified protocol is `thesis/notes/service_cover_protocol.md`. It was
written before new protected-output scores. No candidate grid or mechanism
source was changed after training/validation. Confirmation data uses fresh
201–204 seeds; old inspected 105/106 are excluded. Development 101–104 content
is unchanged. No scope claim for protection of S4–S10 is added.

## Validation lock

Validation SHA256:
`3a677cfa3d3ef6f1dab5995529310dc29adf004e7850abcabd6a09a43cc787a1`.
Selection SHA256:
`248cd93ba3b45cd9b3b21091cc7abf94a48950b25886a0e6ef26c31b2b4a438a`.
Selection pins 40 scientific/consumer source files and the training artifact.

| K | Baseline minimum case recall | Belief24 | Service cover | Validation choice |
|---|---:|---:|---:|---|
| 3 | 78.33% | 80.30% | 71.11% | belief24, infeasible fallback |
| 5 | 79.76% | 79.82% | 84.44% | service_cover, infeasible fallback |

**Neither K meets the prespecified 90% all-case condition.** Do not promote
either fallback as a successful constrained solution. Prior-only has validation
recall 64.90%/70.96% and dense-category recall 14.50%/24.84%, showing why its
zero observed Hit is not an adequate utility-preserving result.

## Confirmation decision: reject promotion of service-cover

Confirmation SHA256:
`c579a87125b5fc8daeedf327c5e247d98948710d27c8eeeff3a67da151e74faf`.
All 864 confirmation rows are retained, including unselected methods.

| K | Method | Mean recall | Min-case recall | Selected Hit@100 | Exploratory Hit envelope |
|---|---|---:|---:|---:|---:|
| 3 | baseline | 85.57% | 78.83% | 11.43% | 19.18% |
| 3 | belief24 (locked fallback) | 86.83% | 80.31% | 13.33% | 18.17% |
| 3 | service_cover | 81.84% | 76.24% | 0.00% | 6.30% |
| 3 | prior_cover | 65.60% | 60.83% | 0.00% | 0.46% |
| 5 | baseline | 87.48% | 84.01% | 8.46% | 16.81% |
| 5 | belief24 (diagnostic, not reselected) | 89.77% | 85.59% | 7.98% | 15.24% |
| 5 | service_cover (locked fallback) | 84.12% | 78.89% | 0.00% | 5.06% |
| 5 | prior_cover | 71.59% | 67.50% | 0.00% | 0.46% |

The chosen K=5 variant **does not generalize its validation utility** (94.64%
mean becomes 84.12%). Versus baseline, paired family recall changes are
+5.67, −2.20, −12.48, −4.42 percentage points for 201/202/203/204. These are
descriptive effects, not confidence bounds. Keep the original default unchanged.
Do not reselect belief24 on confirmation and call that a successful held-out
selection. It is a useful diagnostic comparator, with positive recall effects
in all four families at K=5, but this is not a new confirmatory winner.

Dense-category recall at K=5: baseline 65.48%, belief24 71.75%, service_cover
55.36%, prior_cover 20.99%. Conditional extra distance: 106.4, 79.5, 184.2,
517.2 m respectively. Completion is 100% for evaluable queries; that does not
imply correct results. Each method/K has 4,017 evaluable category-events and
69 empty references (N/A, not 100% recall). Clinic/fuel recall is 100% for all
methods, exposing the easy-category contribution to the aggregate.

Selected Hit=0 and large MAE are **not success evidence** in the presence of
utility loss and poor transfer of a two-family shadow model. The exploratory
attack envelope remains nonzero. These observations justify calibration and
prior-mismatch experiments, but do not establish their causal explanation.

## Data and storage evidence

- v3: 8 families, 176 sessions, 124,852 FCD samples, 269 scenario records,
  7,778 allowed observations; 30/30 subcases in union, 24–30 per family.
- 4 retained development families, 4 new confirmation families. S5.C still
  has 0 training / 0 validation / 1 confirmation record.
- Native verification: 72 daily runs, all 124,852 raw FCD points, 9,080 external
  transitions. Maximum observed one-second displacement is 9.55395 m under the
  existing gate; no coordinate corrections or post-score case relabelling.
- DB head `urban-scenarios-v3`; semantic SHA
  `213886fc2722bbecf5e55f61e8978d2c2842bb6018b7f65f5021a261bd79358d`.
- v3 JSON SHA
  `6ba4a1caa978540ede91b59f23d097e4a9b5e6bad17243890f17c9c49cf9a94f`.
- Native receipt: `artifacts/datasets/urban_scenarios_v3/verification.json`.
- Current registry receipt: `artifacts/datasets/urban_scenarios_v3/registry_verification.json`.
- Original migration receipt remains historical at parent 232adf3. The DB
  verifier now accepts the immutable v1/v2 prefix under a newer head; the
  integration test checks every current release. Schema/storage implementation
  itself is unchanged. Original scientific artifacts and 32 sources stay frozen.

The new runner reads allowed input windows and training-only prior coordinates
from the DB with an explicit release/hash. It does not substitute the current
head. Integrity checking can reconstruct evaluator snapshots; those full data
are not passed to a protection model or attacker.

## Attacker and utility checks

Eight shadow models (method × K), each trained on 360 observations from TWO
synthetic training families. Features are persistent-track public XY, causal
running means and elapsed time. Only training data determine normalization.
Nearest-neighbour counts 1/5/15 are separate validation-selected attacks, not
parameters chosen using confirmation. Model feature arrays/targets/provenance
are committed. This is mechanism-aware inference, not an optimal adversary.

Keep original primary recall and completion metrics; add all six category
values/denominators, labelled cafe/restaurant diagnostic and paired per-family
recall effects. No event/RNG pseudo-sample confidence interval. Public output
does not contain internal anchors, belief or labels. Evaluator artifacts DO:
never expose whole result JSON or the scenario DB to LSP.

## Limitations and protocol clarifications

1. This is a new proposed-method ablation, not a faithful neural comparator
   study. Full TransProtect/semantic training and matching original evaluations
   remain open. AnotherMe keeps its distinct release contract.
2. Basic unit tests plus the inherited ideal postprocessing argument do not
   certify floating-point differential privacy. At B=.24/m the exp(24) bound at
   100 m is weak. No identity/query/relationship claim follows.
3. Reachable dummy motion is directed free-flow lane feasibility, not SUMO
   traffic-light, acceleration, congestion or lane-change compliance.
4. Execution latency is recorded on a shared development machine, including
   periods with test/document QA activity; it is not a controlled mobile timing
   result. Initialization and LSP/network cost are excluded from step time.
5. Query overhead is the count of logical coordinate/category lookups across
   six evaluation workloads, not measured HTTP requests or network bytes.
6. New selectors retain inactive inherited public fields (offset, temperature,
   center_mode, coverage_weight) for compatibility. `selector`, `coverage_target`
   and `prior_only_control` describe the actual new decision rule; the inactive
   fields do not introduce geometric scoring. Clean up this metadata in a new
   revision, not by rewriting hash-frozen study outputs.
7. There is no claim that all 30 case definitions have enough samples to train
   attacks, nor that this method solves all ten targets/scenarios.
8. The selector objective is a per-step, belief-weighted macro-category recall
   surrogate. An entirely unavailable latent location contributes zero. It is
   not identical to whole-run benchmark aggregation, which excludes empty
   references (N/A) and averages over time. The thesis states this distinction
   explicitly; no frozen objective or metric implementation was changed.

## Literature review and next-cycle work

See `docs/research/service_cover_literature_review.md` for primary-source links,
reading scope and objections. Important new close-work lead: Atmaca et al.
(IEEE OJVT 2024) already combine approximate Geo-I, dummy data and vehicle
charging queries. Full-PDF retrieval timed out; detailed reproduction review
remains required. LR-Geo (PoPETs 2025) is related optimization literature, not
automatically a same-contract dummy baseline. Liu/Hu/Zhou's fake-query paper is
published January 2026 despite a DOI containing 2025.

Next decisions must follow the frozen readout, including failures. Priorities:
calibrate stronger inference; distinguish sparse-catalogue effects; test budget
and category/mobility mismatch; introduce public/protected lookahead only if
per-step greediness is the bottleneck; reproduce close comparator methods and
evaluate other urban maps. If 201–204 guide the next change, they become
development evidence and need new confirmation families afterward.

No conference acceptance, submission readiness, statistical superiority or
all-scenario success is claimed by this review.

## Regression and thesis artifact QA

- Independent verifier: `venv/bin/python -m experiments.verify_service_cover --replay`
  → **verified: true**. Receipt: `artifacts/benchmarks/service_cover/verification.json`.
  It checked all **1,728 rows / 11,160 events**, recomputed scores for **1,296
  rows / 49,680 category queries**, audited **37,728 directed motion transitions**
  and **41,016 native connections**, checked **432 paired-anchor groups** and
  independently reconstructed all **8 shadow-model** training feature arrays.
  One of three RNG replicates per configuration was replayed: **576 complete
  row replays plus 576 causal-prefix checks**. The other two replicates retain
  full input, score, motion and budget checks but were not replayed.
- The verifier also checks v3 JSON/SQLite allowed-view parity, retained
  development content, source hashes, validation-only selection, frozen attack
  choices, prior-only independence and POI distances against NetworkX samples.
- Full regression: `venv/bin/python -m pytest --override-ini addopts='' -q`
  → **238 passed in 22.10 s**.
- New selector tests cover global vs track-order greedy, duplicate slots,
  exhaustive tiny-instance optimum checks, paired anchors, causal prefix,
  no GPS read after H, directed motion and prior-only input independence.
- Four v3 mutation tests reject old-confirmation seed reuse, rare-POI label
  corruption, a future/history label mismatch and duplicate record IDs.
- LaTeX compiled with XeLaTeX/latexmk: no overfull boxes or undefined references.
  Remaining underfull bibliography-line warnings are typographic, not clipping.
- Reviewed whole-document contact sheets and full-size new formula/result pages.
  Fixed the near-empty fourth TOC page by locally reducing TOC line spacing,
  without changing the thesis body font. New result paragraphs prevent lone
  first/last lines across page breaks, as does the new method section.
- Final PDF: `artifacts/reports/graduation_thesis.pdf`, **81 pages**, one title
  page, no empty pages, no out-of-page word bounding boxes, no unresolved `??`.
  SHA256: `3874223e8be4a1bd51cec93ff1319014deb3d043781a3d5ccc20beb56de8f408`.
- Main review locations: §3.9.4 (v3 dataset), §5.9 (joint-cover method), §6.9
  (protocol and negative confirmation), §7.2 (publication limitations).
- Final table aggregates were rechecked against the confirmation summaries;
  values and denominators are also in `readout.json`. Previous result tables
  and their source hashes were not rewritten.
- Staged whitespace check reports only one extra blank line at EOF in each of
  `build_scenario_suite_v3.py` and `verify_scenario_suite_v3.py`. These were present
  when their evidence/source hashes were frozen and are intentionally retained;
  cosmetic cleanup belongs to a later revision, not a rewritten study receipt.
