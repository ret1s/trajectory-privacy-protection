# Verification: uniform-prior / corridor development — 2026-09-08

Parent: `28ef61e79df6246e99de5187be4c0e87e56bbb91` on `main`.
Question: can we recover POI utility after the learned service-cover failure,
without relaxing motion, budget, information boundaries or metric definitions?

## Overall assessment: share with caveats

**Uniform-cell coverage is a promising development candidate. The hard corridor
is rejected. No new independent confirmation or default promotion is claimed.**
This cycle improves understanding and measured development utility, not the
strength of the numeric Geo-I bound or faithful SOTA-comparator evidence.

Two design documents were written before their respective computations:
`thesis/notes/service_recovery_protocol.md` and
`thesis/notes/service_recovery_candidate_protocol.md`. The first freezes the
fixed-history diagnosis; the second records what was learned and declares the
three new complete-run variants. Neither was changed after its results.

## Data and information boundaries

- SQLite release `urban-scenarios-v3`, semantic SHA
  `213886fc2722bbecf5e55f61e8978d2c2842bb6018b7f65f5021a261bd79358d`.
- DB file SHA remains
  `fb28dc526fe7825ac025ff076470d3da3ada32cbb06c15791b28490423081499`.
  Dataset contents and update log are unchanged: no new data were generated.
- Data is SUMO + OSM only. No GeoLife inputs were used.
- Train 101/102, validation 103/104, **reused development 201–204**. The old
  immutable `confirmation` split label is historical; explicit experiment role
  overrides its interpretation for this cycle. It is NOT a fresh test set.
- K=5, B=.24/m, H=12, theta=200 m, nine S1–S3 cases, three paired RNG draws.
  No new K=3, endpoint, identity, relationship or query-content protection claims.
- All methods consume identical allowed record windows, not hidden future data.
  The new models only receive raw GPS through the inherited anchor mechanism.
  Their postprocessors use protected anchors/public context only.
- The three original methods' rows and shadow models are reused exactly from
  service_cover artifacts. This avoids rewriting original scientific evidence.
  The three new methods are generated on the same records/anchor seeds.

## Fixed-history diagnostic

All 36 old confirmation records, K=5, replicate 1, 227 events. Hold preceding
service-cover states fixed for each intervention. These are one-step contrasts,
not full trajectories from alternative histories.

| POI reference weights | Reachable | Unrestricted |
|---|---:|---:|
| Learned belief | 84.98% | 89.98% |
| Uniform-cell belief | 89.74% | 95.00% |
| Protected anchor only | 84.53% | 87.42% |
| True location (evaluator only) | 91.88% | 100.00% |

The last row is deliberately privileged, not a protection mechanism or a
proven global optimum. Unrestricted states may violate movement constraints.
Diagnostic mean-position errors use **nearest native road-state truth**:
637.04 m learned, 267.11 m uniform, 281.02 m protected anchor. These are NOT
attacker MAE; benchmark attack errors continue to use unsnapped raw GPS.

Uniform changes initial prior, transition prior weights and cell-vs-state
mass. Thus “the initial prior alone caused the problem” is unsupported.
Fixed-history effects do not determine the whole-trajectory counterfactual.

## Implemented change

`RecoveryCoverLaneDummy` supports a uniform-public-cell model and an optional
directed candidate corridor. Uniform means 1/M mass per occupied public 120 m
cell, regardless of the number of sampled lane states in that cell.

The corridor retains each track's reachable candidates whose directed distance
to the protected-anchor goal is at most the reachable minimum +200 m. It is
always nonempty, but can exclude useful service states and collapse diversity.
This is not a raw-GPS radius constraint, a realistic traffic simulation or
multi-step planning. The ordinary greedy 1/2 bound applies only to its restricted
per-step partition constraints, not the unpruned or true-user utility problem.

New public metadata omits inactive inherited offset/temperature/route-weight/
coverage-weight/center-mode fields. It explicitly names prior and corridor.
Preceding frozen sources/transcripts are not modified for cosmetic cleanup.

## Selection and complete-run results

Selection SHA:
`5c16a7d606476d262193ee47931646a8f1220e09ff59a8f3e27c62edd15be9c9`.
Frozen before generating new-method outputs for 201–204. Selection uses the
existing min-case recall>=.90 then lowest strongest-validation Hit rule.

Uniform_cover is the fallback: mean validation Recall **95.23%**, minimum case
**88.13%**, `utility_feasible=false`. Learned service-cover minimum is 84.44%.
Both corridor variants fall to 70.17% minimum. No post-score slack sweep.

Results below are **development-only**, four already-inspected families:

| Method | Recall | Min-case | Selected Hit100 | Exploratory Hit envelope | Dense-category recall |
|---|---:|---:|---:|---:|---:|
| BR-lane | 87.48% | 84.01% | 8.46% | 16.81% | 65.48% |
| Belief24 | 89.77% | 85.59% | 7.98% | 15.24% | 71.75% |
| Learned cover | 84.12% | 78.89% | 0.00% | 5.06% | 55.36% |
| Uniform cover | 92.35% | 86.01% | 2.78% | 14.30% | 78.00% |
| Learned + corridor | 82.50% | 76.07% | 4.76% | 10.21% | 54.34% |
| Uniform + corridor | 82.09% | 76.60% | 7.59% | 13.58% | 53.21% |

Uniform versus baseline: mean recall **+4.87 percentage points**; per-family
+3.83, +6.60, +4.03, +5.02 points. This is not a confidence interval or claim
of statistical superiority. Versus learned cover, utility improves while Hit
also increases: not universal dominance across privacy/utility controls.

Uniform S3.A/S3.C recall is **86.69% / 86.01%**. These are the remaining
below-90% cases. The nominal average exceeding 90% does not satisfy an all-case
constraint. On some S2 cases an alternative attack reaches 25%, despite low
selected-attack macro Hit; do not summarize privacy by 2.78% alone.

Every method has 4,017 valid category-events and 69 N/A references. Completion
is 100% for evaluable queries across methods. Uniform conditional extra road
distance falls to 49.96 m from baseline 106.39 m, while cafe/restaurant recalls
are 88.51%/67.48% (not the same as their 78.00% mean). Clinic/fuel remain 100%.
Corridor methods sometimes have only ONE distinct coordinate across five slots.
Uniform cover has five throughout these runs, not a k-anonymity guarantee.

## Reproducibility and QA

Commands (repo root, pinned venv/environment):

```bash
venv/bin/python -m experiments.diagnose_service_cover
venv/bin/python -m experiments.run_recovery_cover --phase training
venv/bin/python -m experiments.run_recovery_cover --phase validation
venv/bin/python -m experiments.run_recovery_cover --phase development
venv/bin/python -m experiments.verify_recovery_cover --replay
venv/bin/python -m experiments.export_recovery_cover
venv/bin/python -m pytest --override-ini addopts='' -q
```

Stage writers refuse to overwrite evidence. See the artifact README for rerun
paths. Whole JSON bytes include nondeterministic timing; scientific outputs,
not file hashes across different timed runs, are the reproducibility target.

- Unit/regression suite: **244 passed in 21.67 s**.
- New tests cover uniform cell mass, same emission normalizers, corridor
  boundary/nonempty/invalid inputs, exact no-corridor equivalence, causal prefix,
  directed moves, paired anchors, unchanged ledger, post-H GPS independence,
  no evaluator fields in public output and truthful uniform-prior metadata.
- Full independent verifier: **PASS**, `verification.json` has `verified=true`.
  It checks 1,296 rows (972 scored), 8,370 events, 37,260 category queries,
  35,370 directed motion transitions, 2,790 corridor events and six shadow
  models. All 648 reused rows match the preceding artifacts exactly.
- Replay: 216 new-method rows (replicate 1 in every phase) match, with 216
  prefix checks. Only 144 are strict shorter prefixes; the other 72 involve
  single-event records and are not evidence of multi-step prefix causality.
- Diagnostic verification separately checks all 227 events and 10,896
  category queries. No greedy optimality claim is made.
- Verifier source SHA:
  `28acc3fa46805a03ed3b34c49f43810747e92e8ae3bb64e9335cdefc00eff0ee`.
  A final export audit rechecks all frozen source/artifact hashes, independently
  aggregates the 648 development rows into case means and minimum-case recall,
  and matches all six formatted LaTeX rows to the readout.

### Final thesis artifact

- Added Sections 5.10 and 6.10, with the prior/corridor definitions, diagnostic
  and complete-run tables, and explicit development-only conclusions. Updated
  the abstract and final discussion. Previous negative results remain intact.
- `artifacts/reports/graduation_thesis.pdf`: **85 pages**, 425,644 bytes.
  SHA-256:
  `da6bd6debdd2f31e11de9583e49c69fa0abe2a7b09cec3eede136364287616e9`.
- Built with XeLaTeX/latexmk. No overfull boxes, undefined references or LaTeX
  warnings. The table of contents now fits three pages without the orphaned
  fourth page; body typography is unchanged.
- Rendered every final page for a full layout contact-sheet review; inspected
  the changed equations, result tables and contents at enlarged resolution.
  Automated word-bound checks find zero out-of-page words or empty pages,
  and the title-page attribution occurs once. No unresolved `??` references.
- Chose exact comparison tables, not a trend chart: the evidence consists of
  paired configuration contrasts with different metric meanings, not a time
  series. These tables do not imply statistical significance.

## Issues, remaining work and publication judgment

1. **High — generalization.** New methods were motivated by 201–204. Their gains
   are development evidence even with within-cycle source/selection locking.
   Fix a useful candidate, then declare and generate fresh confirmation.
2. **High — attacker strength.** Six mechanism-aware shadow kNN models are
   trained on only two synthetic families. Add calibrated likelihood-aware
   inference and broader auxiliary routes; do not transfer heuristic Hit to
   a worst-case privacy claim.
3. **High — privacy budget.** Ideal composition remains exp(24) at 100 m,
   with floating-point approximations. This cycle does not strengthen it.
4. **Medium — prior attribution.** Add a cell-balanced learned prior and vary
   the transition independently. Uniform's gain may reflect removal of lane
   sampling density bias as well as improved transfer beyond training routes.
5. **Medium — corridor failure.** The tested hard corridor harms recall and
   collapses distinct coordinates. Do not promote it as successful lookahead.
   Future planning should preserve distributed service coverage, not just seek
   one noisy goal, and must remain causal/public/protected-history based.
6. **Medium — measurement scope.** Nine cases at K=5, same urban map/catalogue,
   synthetic traffic and a short horizon. Six category tasks are counterfactual
   workloads, not a measured production query distribution.
7. **Medium — timing.** New timings are shared-machine measurements. Baseline
   timings come from the old artifacts; no controlled cross-method speed claim.
   Uniform p50/p95 is 3.06/93.19 ms, excluding initialization and network. Each
   method makes 20,430 logical category/location lookups, not measured packets.
8. **High for submission — novelty/comparators.** Full paper retrieval now
   confirms Atmaca 2024 is close related work with a different Edge/truncation
   contract. Prior safeguards and localized remapping were already discussed
   in PoPETs 2017. Faithful comparators and additional protected targets remain
   open. See `docs/research/service_recovery_literature_review.md`.

The suitable claim is **an empirically improved development candidate and a
falsified corridor hypothesis**, not a new privacy theorem, independent test
win, faithful-SOTA win, all-scenario protection or conference readiness.
