# Verification: reachable coverage and privacy–utility–cost frontier

Date: 2026-09-08. Parent: `cf122aec9d7fafac40aa0f9029fe9f558ba22708`.

## Questions and frozen boundary

This cycle addresses contribution specificity and the privacy–utility trade-off.
The candidate contribution is online, reachable **query-set** optimization under
a protected history, with explicit service-resource accounting. It is not the
invention of Geo-I, Bayesian remapping, submodular coverage, local search or
top-L retrieval. See `docs/research/coverage_frontier_literature_review.md` for
the closest primary sources and falsification criteria.

The protocol fixes four selectors, B=.12/.24/.48 per metre, K=5, H=12,
theta=200 m, and LSP response depth 5/10. Client/reference top-5 stays fixed.
Geometric BR-lane and uniform mean-greedy are the controls. Mean-exchange
adds at most three strictly improving single-track exchanges. Capped-exchange
also normalizes each category's coverage and caps its reward at .9. The latter
does not isolate normalization from capping, nor directly optimize the worst
scenario. All feasible single-track replacements are examined at each exchange.

The capped objective is monotone submodular on track-state pairs. Ordinary
greedy retains its classical per-step 1/2 bound under fixed partition constraints
in ideal arithmetic; exchanges cannot lower that objective. This is neither a
1−1/e guarantee, a trajectory-optimality guarantee, nor 90% actual recall.
The private-anchor ledger is inherited unchanged, with the old finite-precision
and public-schedule assumptions. B gives exp(12), exp(24), exp(48) at 100 m;
these bounds remain loose. No new privacy theorem is claimed.

The same v3 core data and 64 designated SUMO auxiliary training families are
used. No GeoLife. No new dataset release, no DB mutation and no use of the
16-family auxiliary holdout in this cycle. Core groups 201–204 were already
reviewed, so all new results are **development**, not fresh confirmation.
Measured coverage here remains nine S1–S3 cases; S4–S8 are not protected by
assertion or by having rows in a dataset.

## Comparison, selection and resource contracts

For each of 12 method/B configurations, the attacker sees the same number of
permitted training observations: 360 core + 2,873 auxiliary = 3,233. Keep the
old core-only bank, the expanded kNN/loss-aware bank, two 128-tree regressors,
and applicable public-transcript direct/temporal rules. Causal learned features
exclude private anchors, raw GPS, lane/speed labels, scenario IDs and future
events. Existing whole-window rules remain retrospective and explicitly scoped.

Per B: 1,240 training, 216 validation and 432 development protection runs.
Total: 5,664 runs. Same private anchor draws across selectors at fixed B.
The B=.24 controls must exactly retain historical core scientific outputs.
MAE and Hit attacker choices are locked separately per method/case on validation.
Development envelopes are exploratory extrema only within the tested bank,
not optimal-adversary bounds. A large selected/envelope gap is a selection
transfer weakness, not evidence of strong protection.

The utility gate is minimum validation-case Recall>=.90, a research requirement
not a standard prescribed by papers or conferences. Then choose the eligible
method with the smallest validation mean strongest Hit. No eligible method
means `chosen=null`; there is no silent fallback or softened threshold.
All methods and budgets remain in the readout regardless of selection.

L=10 uses exactly the same published coordinates as L=5 and is available to
all methods. Six categories and five dummies mean 30 requests per event,
without caching; response items and serialized ID-list JSON bytes are counted
before deduplication. These are not full POI response/HTTP bytes. Local ranking
is not uploaded. Fixed-depth public-catalogue replies add no information beyond
the published queries in this model; adaptive private depth or revealed local
choices would require a new analysis. Recall improvement from L alone is a
retrieval allowance, not algorithmic novelty.

Macro scores weight the nine cases equally, then family/replicate runs, then
eligible category-events within each run. Report minimum case, category scores,
completion and paired deltas for four reused families. No step-level p-values.
Empirical non-dominance compares recall, selected Hit and response bytes only
**within the same B**. A second frontier adds separately profiled runtime; no
claim of statistical or cross-budget superiority follows.

## Negative engineering QA retained

1. A small exact-parity test caught a floating-point summation-order difference
   in uncapped marginal gains. A one-ULP change can alter exact ties. Restored
   the original two-axis summation before any final experiment output; the
   unchanged-objective/no-exchange path must exactly match the old selector.
2. The initial B=.24 belief provenance check rejected integer `theta_m=200`
   metadata versus historical floating `200.0`, despite the same physical value.
   Fixed the constructor input, not the SHA check. The old cache is preserved
   as `cache/coverage_frontier/b0.24_belief_integer_metadata.npz`.
3. The first worker set was interrupted before sealing any training or scoring
   artifact. Empty output directories were retained under
   `build/coverage_frontier_interrupted_20260908`. A declared execution amendment
   adds hashed per-record checkpoints and reuse of public map/SCC/travel
   resources; every run resets all private session state and exact RNG streams.
   Fresh-instance/reused-worker output parity passes over repeated seeds and
   different session lengths for all four methods.
4. Concurrent generation timings are not comparative performance evidence.
   A prespecified timing amendment requires a separate 36-run serial profile
   after every generation/verification job completes. Three fixed validation
   records from one family, fresh instances; initialization is separate. This
   is a local-host diagnostic, not a mobile benchmark or confidence interval.
5. Figure review found overlapping budget labels. The revised plot separates
   labels by method/budget and checks annotation bounding-box intersections.
   The rejected first figure remains in `build/coverage_frontier_labels_overlap.png`.
   No metric or selection was changed by this layout correction.

## Verification coverage

`experiments/verify_coverage_frontier.py` reconstructs causal feature matrices,
training labels/scaling and split provenance; refits sklearn forests and checks
every saved tree array; independently recomputes empirical inference decisions
with the geometric loss oracle; checks all persisted errors with the strict
100 m threshold; reconstructs POI references, replies, client ranking and costs
from the public service rather than the new signature-based metric evaluator;
recomputes case summaries and selection; checks paired anchors and historical
control parity; verifies directed travel-time transitions; and replays a fixed
subset with fresh models and strictly shorter prefixes. Inherited non-learned
attack definitions and the public road-distance engine are shared, not claimed
to be independently reimplemented here.

`experiments/verify_coverage_frontier_readout.py` separately aggregates raw rows,
including category/case/family metrics and serial timings, and recomputes the
finite-grid pairwise dominance relation without the exporting helper. It pins
the exact readout, tables and profiling receipt. Test coverage includes tiny
exhaustive objective/marginal/approximation oracles, prefix causality, hidden
anchors, post-horizon GPS independence, old-method parity, retrieval monotonicity,
fixed references, missing metrics, ties and infeasible selection.

## Interpretation and final receipts

All three independent experiment verifiers **passed**. Combined counts:

- 5,664 protection runs / 51,216 events, including 1,944 scored runs.
- 405,324 point predictions and 149,040 category-depth queries checked.
- 24 forests refit; all 15,360 saved arrays match.
- 227,760 directed motion transitions valid; 25,608 objective histories checked.
- 216 fresh full replays and 216 strict-prefix replays match.
- All 432 historical B=.24 control rows preserve anchors, states, budget and
  public events; scored top-5 utility also matches.
- Independent readout audit passes 24 configuration-depth rows, 216 case rows,
  72 paired family-delta rows and both within-budget frontier calculations.
- Full regression suite: **266 passed**, final explicit run 20.45 s.

At B=.24, the fixed-resource comparison is:

| Selector | L | Mean Recall % | Minimum-case Recall % | Selected Hit % | Exploratory Hit envelope % | ID bytes/event |
|---|---:|---:|---:|---:|---:|---:|
| Mean greedy | 5 | 92.3465 | 86.0060 | 3.7603 | 16.2280 | 2738.03 |
| Mean exchange | 5 | 92.9280 | 86.3636 | 5.9862 | 16.1739 | 2738.06 |
| Mean greedy | 10 | 96.1247 | 92.3095 | 3.7603 | 16.2280 | 4512.84 |
| Mean exchange | 10 | 96.3810 | 92.4411 | 5.9862 | 16.1739 | 4513.10 |

Interpretation:

1. **A modest optimizer gain, not a privacy win.** At L=5, mean exchange adds
   0.5815 percentage points of recall. Paired family gains are +1.1197, +0.0446,
   +0.4826 and +0.6790 points. Gains are positive for all four families at each
   B at L=5, but not at every depth: family 202 falls 0.1412 points at B=.24,
   L=10. Selected Hit worsens at B=.24; the exploratory envelope barely changes.
2. **Most total utility gain is retrieval depth.** Keeping mean greedy and
   increasing L adds 3.7783 points; replacing its selector at the same L=10
   adds only 0.2562 points. Do not attribute all +4.0345 points to the algorithm.
   For mean exchange, 120→200 reply items and 2738.06→4513.10 ID bytes/event
   accompany top-10 (about 65% more bytes), with 30 requests unchanged. Some
   categories have fewer POIs than the requested depth, so actual items are
   below the nominal 6KL maximum.
3. **Category capping is not promoted.** At B=.24/L=5, capped recall is 91.9903%,
   below mean greedy, and three of four families lose utility. Its minimum case
   improves, but that does not rescue average quality or the validation gate.
   At B=.48 average recall also decreases versus mean greedy.
4. **Selection failure remains visible.** At B=.12 capped selected Hit is 0%,
   while its exploratory envelope is 9.5910%. This is not perfect privacy. At
   B=.48 exchange reduces selected Hit but raises the exploratory envelope.
   Neither result supports uniform privacy improvement.
5. **No silent fallback.** No candidate meets validation's minimum-case gate
   at B=.12 (either L) or B=.24/L=5. At B=.24/L=10, mean exchange is selected
   (validation minimum 91.3704%, development 92.4411%). Mean greedy validation
   is 89.9630%, which must not be rounded into eligibility. At B=.48, mean
   exchange is selected for both depths; its L=5 development minimum falls
   to 87.4154%, despite 91.6111% on validation. L=10 retains the gate in the
   reused data. The B=.24/L=10 configuration is a candidate for **new**
   confirmation, not a confirmed/default deployment decision.
6. **The optimizer also costs time.** At B=.24, fresh serial profiling gives
   mean greedy 18.82 ms mean / 85.77 ms p95, versus exchange 23.97 / 150.72 ms.
   There are only 20 events per configuration from three selected records on
   one local host. Do not infer mobile latency or statistical superiority.
7. **Tiny byte differences are not substantive cost wins.** At fixed depth,
   several selectors return the same number of items but different ID strings.
   Fractions of a byte in macro means can preserve a configuration on the exact
   finite-grid frontier. This is correct accounting for the declared encoding,
   not an operational advantage; use reply counts and the large depth-induced
   differences for interpretation. A practical tolerance/full-payload cost
   model must be declared in a future protocol, not tuned after these results.

The readout retains every B/method/depth, category, case and family delta, plus
MAE and response completeness; the compact table above is not the entire grid.
The formal ledger and high-level threat coverage have not strengthened. No
neural/non-neural paper comparator was newly reproduced in this cycle.

Machine receipts:

- Readout SHA256:
  `6dc0a6732d7724e7152dedda38a2c48e4051a9e6467d04a18e6605e10c5441cc`.
- Tables SHA256:
  `826e1fb8fa3cec594a1486a00c3665bbec3d4b6f1658e2cc89064a68ddb9c53a`.
- Timing SHA256:
  `71864a54b074bfc7cc6a5b5f0c4bb53dab473c8e650609c0b368ab3ad15c0f2d`.
- Phase/source hashes and per-B counts are in each `verification.json`;
  aggregate export and verifier hashes are in `readout_verification.json`.

## Remaining paper-level gates

- Benefit from an optimizer must survive a correctly specified surrogate,
  later reachability constraints, broader validation and fresh confirmation.
- Matching an ideal B does not ensure equal empirical privacy. Retain any
  category, case or attack regression alongside mean service improvements.
- Separate the inherited optimization tools from the system/formulation
  contribution; targeted literature review is not exhaustive novelty clearance.
- Validate beyond one public map and a few reused core families. Stronger
  likelihood/topology attacks and a broader attacker-selection distribution
  remain necessary; a failed attacker cannot certify protection.
- Correctly reproduce/adapt neural and non-neural comparators under their
  original contracts before claiming SOTA superiority. This cycle is an
  internal ablation, not a new comparator leaderboard.

Commands and immutable artifact layout:
`artifacts/benchmarks/coverage_frontier/README.md`.

## Final thesis artifact QA

- Canonical PDF: **102 A4 pages**, SHA256
  `fd6d9862bd1cd7e01ce1b62f3096b45e51f8a99406ae959d3e2a8fe8c90b7599`.
- New material: Section 4.6.6 (physical page 48); Section 5.12 (62–64);
  Section 6.13 (91–96), Tables 6.26–6.29 and Figure 6.2. Abstract (10),
  conclusion (97–98) and bibliography (102) also updated. Printed page numbers
  differ from physical PDF pages.
- All 102 pages rendered and inspected on contact sheets; new/changed content,
  formulas, tables, figure and bibliography inspected at full size. A one-line
  abstract spill and a detached float page in the first build were corrected.
  Earlier QA remains in `build/coverage_frontier_pdf_qa_v1`; final QA in
  `build/coverage_frontier_pdf_qa`. No second title page was introduced.
- No off-paper words, replacement glyphs, unresolved references/citations,
  missing glyphs or overfull boxes. Bibliographic underfull spacing warnings
  remain, with no clipping found visually.
- Final figure SHA256:
  `5054336c5827f85049c1aef2e207c89c5056298830a6d3362910a2846570a31e`.
- New evidence is 51 files / 139.25 MiB total; largest file is under 19 MiB.
  All individual new files are below 100 MiB. The near-limit existing SQLite
  database is unchanged; cache/checkpoint/render intermediates are not committed.
