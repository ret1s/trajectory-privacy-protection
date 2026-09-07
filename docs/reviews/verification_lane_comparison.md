# Verification: comparator diversity and the BR lane-state extension

Date: 2026-09-07. Baseline: `e9b909cad3813d3584e3c008a277e9d0af786ff9`.

## Verdict and research constraint

**Suitable for review as a reproducible development experiment, not a final
SOTA ranking or an all-scenario protection claim.** The proposed model remains
unrestricted with respect to neural/non-neural components. Non-DL work expands
the comparator set; it does not define the proposed method's architecture.

This cycle adds one comparator algorithm (enhanced-DLS) and an actual proposed
method extension (BR-lane), rather than only changing the thesis wording. The
three previously surveyed modern comparator papers are still not fully trained,
paper-equivalent reproductions. DLS and enhanced-DLS are two algorithms from
**one foundational 2014 paper**, not two additional recent SOTA sources.

## What changed

1. `benchmark/engines/enhanced_dls.py`: probability-neighbor redundancy, entropy
   screening, then sequential **distance-product sampling**. Not greedy farthest
   selection. Uses the existing DLS implementation without editing old sources.
2. `data/lane_states.py`: a public, trip-independent lane-progress graph, internal
   SUMO lanes, and permission-checked native connections; no junction merging.
3. `benchmark/engines/lane_budgeted.py`: BR with the existing noisy-reuse anchor
   and ledger, richer motion support, raw continuous input, and bounded sparse
   travel caches. No new neural component or new privacy theorem is claimed.
4. `evaluation/lane_travel.py`: sparse directed path queries with explicit zero
   edges and bounded caches; shared lane-based POI service for all six controls.
5. New run / summarize / verify commands and eight regression tests. No old
   scientific generator or evaluation source was edited.
6. Canonical thesis §4.3.1 (DLS variants), §5.6 (proposed extension), §6.6 (separate
   experiment, negative results, attacker-selection instability). The earlier
   tables remain intact. The PDF is still the canonical graduation thesis.

## Literature check and design decisions

Primary source: Niu, Li, Zhu, Cao and Li, *Achieving k-anonymity in
privacy-aware location-based services*, INFOCOM 2014, DOI
10.1109/INFOCOM.2014.6848002. Author-hosted PDF:
<https://mcn.cse.psu.edu/paper/other/infocom-ben-niu14.pdf>.
The cached original `cache/research_2026_09_07/dls2014.pdf` was inspected in text
and visually at page 6, Algorithm 2, including the surrounding explanation.

- DLS: 2K probability neighbors, 50 random subsets containing truth, maximize
  normalized entropy. Enhanced-DLS: first 4K probability-neighbor dummies,
  screen a 2K-dummy redundant set, then K−1 weighted draws alongside truth.
- Algorithm 2's intermediate stage points back to Algorithm 1 but does not fully
  specify whether truth is counted in the entropy score. This implementation
  scores the **2K dummies union truth** (2K+1 entries). This disclosed convention,
  50 trials, tie policy and road-domain mapping prevent a paper-equivalence claim.
- The first development run exposed a representation problem: distinct lane
  states may be the same location. The shipped selection view deduplicates exact
  coordinates for DLS/enhanced-DLS/uniform sets. The motion graph keeps all states.
  No private-data filtering or geometric rounding is introduced by this view.
- Spatial priors use only simulated training-family FCD, not the paper's query
  counts. Entropy is a selection objective, **not** a proof of posterior 1/K.
- The proposed raw-input anchor uses the same exponential kernel over a fixed
  public support. The triangle inequality applies directly to continuous planar
  inputs; pre-snapping is no longer required. This inherits the existing ideal
  Geo-I argument, rather than extending a snapped-input theorem by assertion.
- Float RNG / numerical sampling still prevents a pure-DP executable guarantee.
  Fixed context, public schedule, horizon accounting and query leakage caveats
  remain. Learning is allowed, but access to extra private data must be analyzed;
  a learned component is not automatically private merely because the anchor is.

## Data, denominators and fairness

Frozen input `artifacts/datasets/urban_scenarios_v1/dataset.json`, SHA-256:
`826bf236588448e5e59b5e4e0304ddef3e3cf85751df8dddf901b72ae88aabb9`.
This source remains pre-protection data; its frozen evaluation flags are not
rewritten. The new artifact is the authority for this cycle's measured coverage.

- Only S1.A / S2.B / S3.A. Family91 estimates the prior; family92 selects attacks;
  family93 reports. All are development data, including the field named
  `development_test`; family93 is not an untouched final test cohort.
- Six records, 40 retained events: validation 1/9/8 and reporting 1/9/12 by case.
  Reporting S3.A is explicitly truncated from 20 to 12 visible events; no other
  record is truncated. No whole-trip or long-session claim follows.
- K=3/5, two random replicates, six methods => 144 rows. One related route family
  per role. Replicates and multiple cases are not independent user samples.
- 102,123 motion states, 110,643 arcs, 32,496 passenger lanes, 59,973 unique
  coordinates. Public catalogue fingerprint:
  `3906d30345094bd1d5aff040c2731b4f80a84b726105e535f33ae41484342229`.
- State spacing <=20 m along lane progress, speed <=8 m/s and lane limit.
  41,016 native lane connections checked, including internal continuations.
- No lane changes, traffic-light state, congestion or acceleration model. This
  proves only free-flow graph reachability, **not** exact SUMO replay feasibility
  of every protected trajectory. Nearest-state POI access can alias adjacent
  lanes/directions; this is still a service approximation.
- Same public map, prior source, POIs, output count and service for current
  comparisons. BR projected/raw pairs share RNG seeds and all parameters; only
  input preprojection differs. Both use the lane graph, so the pair does **not**
  isolate the effect of junction-versus-lane output support.
- Gaussian-mixture prior: every 20th training FCD sample, bandwidth 250 m,
  positive 1e−9 floor, normalized on the public catalogue. DLS renormalizes the
  same values on unique coordinates; its ordering is not trained on test data.
- Service: 418 OSM POIs, 8 excluded, six categories, top-5 directed lane distance,
  nearest-state access <=250 m, union/dedup/local reranking at raw truth. The
  real-containing controls have Recall=1 by this service construction; this is
  not proof of perfect real-world LBS utility. Query type remains public.
- New scores are **not comparable numerically** to the older junction-service
  leaderboard. The extension is not yet integrated as a web-dashboard track.

## Verified checks

`venv/bin/python -m pytest -o addopts= -q tests`: **171 passed** (8 new).

`venv/bin/python -m experiments.verify_lane_comparison --replay`:

| Check | Result |
|---|---:|
| Scientific source hashes | 15 |
| Replayed rows / public-prefix checks | 144 / 144 |
| Public events | 960 |
| Recomputed event-category POI queries | 5,760 |
| Independent NetworkX native BR transitions | 1,088 |
| Native SUMO connections audited | 41,016 |
| Independent full POI-distance oracle source points | 7 |

The verifier checks actual device-view input and truncation, source hashes,
public allowlists, RNG pairing, budget bounds, every candidate's evaluator-state
coordinate, uniqueness of set mechanisms, per-event errors, all query response
details on replay, validation-only attacker selection and summary arithmetic.
The additional audit file's envelope and projection numbers are checked
independently from its summarizer. SciPy paths retain real zero-length arcs and
match NetworkX; tests also cover bounded caches and raw ideal-kernel ratios.
Seven oracle sources are a sample; they are not a full independent reimplementation
of every POI query. Full replay exercises the production POI service for all rows.

Earlier evidence rechecked unchanged:

- `verify_scenario_suite --raw`: 36 sessions, 21,818 raw FCD points, 96 records,
  28/30 subcases; all ten data groups present, no all-ten protection conclusion.
- `verify_paper_benchmark --raw`: 1,440 rows, 11,316 events, 67,896 category
  queries, 10,704 BR transitions; **96 N/A and one known failure remain**, not
  silently removed. 264/270 prefixes pass (six known offline AnotherMe cases).
- The older benchmark JSON remains SHA-256
  `2df043bfb73c3b2f630046381daf5b4bc9f6941fb5be925b0967d6de98f27538`.

## Results and counter-evidence

The validation-selected attacker scores are preserved in `results.json`.
`audit.json` and thesis Tables 6.8/6.9 instead show an **exploratory reporting-family
attack envelope**: minimum mean error and maximum Hit100 across predefined
attackers, one attacker per group/metric, never a per-event truth oracle. This
uses reporting outcomes, so must NOT be relabeled held-out attacker selection.

Why this distinction matters: DLS K=3/S1.A has validation-selected MAE 9.21 m
but validation-selected Hit100=0 because **different** attackers were selected
on a tiny validation set. Another already-defined attacker achieves Hit100=1.
Reporting zero as robust privacy would be misleading. The envelope exposes this
failure but does not fix the tiny sample or calibrate the true optimal adversary.

- Projection diagnostic over 40 input events: mean nearest-state distance
  **29.86 -> 3.82 m**; p95 **48.22 -> 7.77 m**. This is geometric resolution,
  not adversarial inference error and not a privacy gain.
- S2.B: both DLS variants have observed Hit100=100% at K=3 and K=5. Enough
  distinct spatial dummies does not prevent repeated-query inference here.
- Enhanced-DLS does not dominate DLS: at K=5/S3.A, minimum observed MAE is
  97.14 m versus DLS's 129.18 m. Do not rank all settings by one favorable row.
- BR raw, K=3: Recall 85.0/85.0/76.5% at S1.A/S2.B/S3.A. K=5:
  98.3/96.9/77.2%. The moving case still fails the development 90% utility goal.
- Raw input is not uniformly better than projected input: at K=3/S2.B Recall
  drops 86.5 -> 85.0%; reported quality/privacy metrics match at K=5. The benefit
  currently justified is the input metric/formulation, not universal performance.
- Lower observed BR attack success is paid for by lower utility. The comparison
  has not matched utility thresholds, so **no fair superiority claim** yet.
- Generator timings exclude initialization and are machine/load dependent.
  Dividing whole-run time by event count is not a p95 per-event latency claim.

## Next iteration priorities for Claude Code

1. **Evaluation first:** add independently generated families with sufficient
   samples per case and role. Keep related users/devices/companions grouped;
   freeze a fresh confirmation set only after method/attacker selection is stable.
2. Strengthen attacks using calibrated mechanism likelihood and native-road
   sequence inference; verify candidate linkage and per-metric selection on a
   sufficiently large validation split. Keep the exploratory envelope separately.
3. **Proposed method remains open:** improve moving-case POI utility through
   contextual scoring and progress modeling; a learned scorer is allowed. Begin
   with public context/protected history, or account explicitly for added access
   to private history. Compare utility-matched frontiers, not only fixed K/B.
4. Isolate lane-support changes with a matched service and controlled graph
   ablation; the current raw-vs-projected comparison does not do this. Consider
   one-way lane access ambiguity and independent SUMO replay with traffic rules.
5. Diversify further with a sequence-aware non-DL comparator (e.g. a faithful RDG
   implementation), while restoring genuine neural training/reproduction of
   modern comparators. Do not call Markov substitutes Transformer/LSTM models.
6. Extend protection evaluation to the other generated cases, especially S4–S8;
   finish absent S1.C/S6.C data recipes and sparse single-family cases. Preserving
   all ten research targets is compatible with reporting today's smaller scope.
7. Only integrate this experiment into the web app after its public-only contract
   and model labels are wired explicitly; never expose root evaluator JSON.

## Report QA and evidence lineage

The technical report workflow is mapped onto the existing user-selected LaTeX
thesis: title/summary retained; definitions in Chapters 2–4; specification in
Chapter 5; evidence, uncertainty and negative results in §6.6; next steps and
open questions in Chapter 7 and this verifier. No unrelated section was removed.
The exact-value tables are used for paired, case-level audit rather than a
ranking chart, since the sample is too small and utility is unmatched. Tables
are monochrome, with units and exploratory status adjacent to the data.

PDF build and visual QA: 60 A4 pages, one cover; full-document contact sheets
reviewed, changed comparator/method sections and result tables checked at readable
resolution. No overfull boxes or undefined references; bibliography retains
non-clipping underfull-line warnings. No new dependency was installed.

Reproduce with the commands in `artifacts/benchmarks/lane_comparison/README.md`.
Use a fresh output directory; do not overwrite this development evidence to
make results look better. Do not change old hashed scientific sources without
regenerating their associated experiment under a new version.

Release checksums:

- New results JSON: `20f93757ba2e2dde28a3b63db01363c8689aa7692b0703c5212412f9e3b7b65c`.
- Audit JSON: `fe6698b2c4e5c3e2abaa5541b127a0d53f2a35098967496e7e0196daf2f5b39a`.
- Generated tables: `72058fe2509fa9466461c61c09510d9c809de942d0f960ef7f2f98cc4449f9ae`.
- Canonical `artifacts/reports/graduation_thesis.pdf` (60 pages):
  `9cbcc364819e2674ea8110e8c9cee05e09f0bf5c5b6df0ea99d7dfe5c1473668`.
  Proposed-method chapter begins at PDF page 42 / printed page 35;
  the lane extension is §5.6, PDF page 45 / printed page 38;
  the new experiment is §6.6, PDF page 52 / printed page 45.
