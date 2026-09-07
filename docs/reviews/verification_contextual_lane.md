# Verification: contextual BR-lane proposed-method ablation

Date: 2026-09-07. Starting commit:
`d8a8af8611b566f3e3f621ff19c63f92750621d9` (pushed before this development cycle).

## Verdict

**Reproducible development evidence, not a promoted default or a final paper
result.** Two implemented components improve some local utility scores, but
neither validation-selected configuration meets the all-five-case Recall >=0.90
requirement. Empirical privacy sometimes worsens substantially. Preserve these
counterexamples when developing the next version.

The proposed method is unrestricted regarding neural/non-neural components.
Non-DL comparators diversify the comparison set; they do not constrain the main
model. This cycle is an internal ablation of BR-lane, not a new comparator paper
or a claim that existing neural comparators are now fully reproduced.

## Actual implementation changes

1. `benchmark/engines/contextual_lane.py`: a directed-road potential mixes
   Euclidean distance with remaining directed route length to a public goal
   derived from the protected anchor and the dummy's offset. Reachability from
   the preceding dummy is still mandatory. The goal is **not the true future
   destination**. Route-distance caches are bounded to 32 goals.
2. `benchmark/public_poi_context.py`: precompute top-five POI identities by
   category on the fixed public lane graph. A sequential marginal-coverage term
   rewards POIs not already covered by earlier dummies, relative to the protected
   anchor's reference answers. It does not optimize against current true GPS.
3. The inherited anchor, noisy reuse, horizon and budget remain unchanged.
   Separate anchor/dummy RNG streams ensure identical anchor sequences across
   the four variants. Zero/zero weights exactly reproduce the preceding model's
   public events and evaluator motion states.
4. `experiments/run_contextual_lane.py`: fixed 2x2 paired ablation, two additional
   moving cases, stronger track-based attackers, validation-only configuration
   selection and explicitly exploratory reporting envelopes.
5. New verifier, generated-table renderer and 16 added test cases (including
   parametrizations). The canonical thesis adds §5.7 / §6.7, counter-evidence,
   conclusion updates and a relevant remapping reference. Earlier scientific
   sources and frozen results are unchanged.

The public-context component assumes a static, public OSM catalogue available to
the device before the session. It does not assume access to proprietary/live
LSP responses. Position-dependent context fetching is outside this protocol.
The web app and its defaults are unchanged; these new research variants are not
silently registered as a production/demo default.

## Literature review and critical interpretation

Primary source checked: Chatzikokolakis, ElSalamouny and Palamidessi, *Efficient
Utility Improvement for Location Privacy*, PoPETs 2017(4), 308–328,
[authoritative proceedings PDF](https://petsymposium.org/popets/2017/popets-2017-0051.pdf),
DOI 10.1515/popets-2017-0051. The relevant distinction is postprocessing that
preserves the formal bound versus Bayesian remapping optimized for a specified
prior and loss.

Our score is a heuristic, **not that paper's optimal Bayesian remapping**. It
does not currently infer a calibrated posterior over the user's location.
Using a protected anchor as the single POI reference may optimize the wrong
neighborhood. The experiment tests that hypothesis and exposes counterexamples;
it does not establish novelty merely by combining existing principles.

For fixed public context and parameters, new scores only postprocess protected
anchors and dummy history. The inherited ideal Geo-I composition argument is
therefore retained. This does **not** mean empirical MAE/Hit are retained: a new
postprocessor may reveal the anchor more clearly than a previous one. Also:

- B=0.24/m still gives the loose upper bound exp(24) at 100 m under D-infinity.
- Finite-precision Gumbel/RNG behavior is not a proof of executable pure Geo-I.
- Event schedule and query categories remain public; there is no new S7 query
  protection or direct S4/S5/S6/S8/S9/S10 mechanism in this cycle.
- Learning from public context/protected history is allowed. Additional raw
  private input access requires new mechanism analysis, not a free extra score.

## Data, comparison controls and denominators

All inputs are SUMO/OSM; no GeoLife trajectories, fitting or parameters.

- Frozen dataset: `artifacts/datasets/urban_scenarios_v1/dataset.json`.
- Family91 estimates the public spatial prior; family92 selects mechanisms and
  attackers; family93 reports development results. The field `development_test`
  must not be described as a fresh final holdout.
- Cases S1.A / S2.B / S3.A / S3.B / S3.C. S3.B is a low-branching corridor
  (>=half the sampled external-edge observations have one legal successor),
  **not** a route with no junctions. S3.C means sparse 60-second observations,
  **not** a route-deviation case.
- Retained events, in the above order: family92 = 1/9/8/7/3;
  family93 = 1/9/12/12/7. Reporting S3.A and S3.B are truncated from 20 to 12.
- Ten input records / 69 events counted within records; views can overlap.
  There is only one related simulation family per role. Replicates, events and
  subcases must not be counted as independent people.
- Four weight pairs (route, coverage): baseline=(0,0), route=(0.5,0),
  coverage=(0,6), combined=(0.5,6). K=3/5; two RNG repetitions; 160 rows.
- Same B=.24/m, H=12, theta=200 m, offset=80 m, temperature=60 m. Same fixed
  catalogue, POI service and public prior. No scenario label enters protection.
- 102,123 lane states, 110,643 directed arcs, 32,496 lanes, 59,973 coordinates.
  Catalogue fingerprint remains
  `3906d30345094bd1d5aff040c2731b4f80a84b726105e535f33ae41484342229`.
- 418 retained POIs, eight excluded, six categories. Directed nearest-five
  responses are merged, deduplicated and reranked locally at true GPS. Recall
  uses event/category queries with nonempty reference answers, then averages
  within a run and across the two repetitions. Raw response IDs and denominators
  are retained, not merely aggregate percentages.
- Public-state signatures follow the same nearest-coordinate access convention
  as the service. Coincident lane directions remain ambiguous in this service;
  do not infer a unique true lane from the emitted coordinates.
- The minimum number of distinct emitted coordinates is 2 at K=3 and 4 at K=5.
  Sending K coordinates does not establish K distinct locations or K-anonymity.

Native connectivity checks establish free-flow directed reachability under the
model, not a traffic-light/acceleration/congestion-aware SUMO replay of dummies.

## Attacker and selection protocol

Individual public dummy tracks and their whole-window means are added to the
previous heuristics. Stable track identifiers are genuinely visible to the LSP.
The same expanded attack set is applied to all four variants. Thus even the
unchanged baseline can have different privacy scores from the old lane table;
that is an attacker-strength change, not altered baseline public events.

Two views of privacy are retained:

1. MAE and Hit attackers selected separately on family92, evaluated on family93.
2. An explicitly **exploratory reporting-family envelope**: smallest mean error
   and largest Hit100 over the predefined attackers, one attack per group/metric,
   not an event-by-event truth oracle. The two metrics need not use one attacker.

The thesis displays the second view to expose observed weaknesses rather than
interpret one poorly selected attacker as evidence of protection. This is not
an estimate of an optimal attack or a held-out paper score.

One mechanism configuration per K, shared across all five cases, is chosen only
on family92. Require every mean case Recall >=.90, then minimize macro envelope
Hit100. If infeasible, maximize minimum case Recall, then minimize macro Hit.

| K | Selected fallback | Lowest validation case Recall | Feasible? |
|---|---|---:|---|
| 3 | combined | 78.33% | No |
| 5 | route | 72.29% | No |

**Neither should be promoted as meeting the utility requirement.** The fallback
rule prioritizes utility even when privacy worsens; it is not a certification.
Weights were not retuned against family93. More powerful/calibrated attacks and
new validation/confirmation families are still necessary.

## Results and counter-evidence

All values below are family93 exploratory scores, averaged over two RNG runs.
Recall is better when larger; Hit is better for privacy when smaller.

| Comparison | Case/K | Recall change | Hit100 change |
|---|---|---|---|
| baseline → route | S3.C / 5 | 70.00% → 76.19% | 7.14% → 14.29% |
| baseline → combined | S3.C / 5 | 70.00% → 77.38% | 7.14% → 14.29% |
| baseline → coverage | S2.B / 3 | 85.00% → 89.44% | 5.56% → 50.00% |
| baseline → combined | S3.A / 3 | 76.53% → 67.78% | 16.67% → 12.50% |

The better local S3.C score for `combined` is not a reason to override the
family92 choice of `route`. The S2.B result is important negative evidence:
unchanged ideal accounting does not stop empirical privacy degradation.
There is no all-case winner or confidence interval justified by this sample.

## Verified checks

`venv/bin/python -m pytest -o addopts= -q tests`: **187 passed** (16 new).

`venv/bin/python -m experiments.verify_contextual_lane --replay` completed:

| Check | Result |
|---|---:|
| Scientific source hashes | 19 |
| Replayed rows / prefix checks | 160 / 160 |
| Public events | 1,104 |
| Recomputed event-category POI queries | 6,624 |
| Independent NetworkX motion transitions | 3,776 |
| Native SUMO connections audited | 41,016 |
| Old BR-lane rows with exact event/state/utility parity | 24 |
| Four-way paired anchor groups | 40 |
| POI membership checks / disagreements | 13,092 / 0 |
| Uncached context rebuild | Exact fingerprint match |

Index membership is checked at 32 prespecified public states plus every emitted
state (2,182 unique states total), across six categories. This is not exhaustive
verification at all 102,123 states. Final utility replay exercises the production
query service; the preceding lane verifier independently checks seven full POI
distance source points against NetworkX.

Unit tests cover zero-weight parity; prefix invariance; same anchors across all
variants; a fixed-anchor canary showing raw GPS cannot affect postprocessing;
ignoring private input beyond H; directed paths; 32-entry cache eviction/reset;
stale-cache rejection; POI signature/service agreement; marginal gain removing
already-covered answers; invalid parameters; numerical tie handling; and all
40 displayed result rows. Tests do not prove the ideal privacy theorem.

Additional checks in this cycle:

- `verify_scenario_suite --raw`: 36 sessions / 21,818 raw FCD samples / 96
  scenario records / 28 of 30 subcases / all ten data groups; unchanged.
- `verify_lane_comparison` (without full replay this cycle): 144 rows / 960
  events / 5,760 POI query records / 1,088 BR transitions / seven independent
  POI sources passed. Its 144 full replays were already recorded in the previous
  verifier; do not count them as newly rerun here.
- Frozen dataset, preceding lane and older paper-benchmark hashes are unchanged.
  The older paper raw verifier was not rerun this cycle. Its known 96 N/A, one
  failed row and six offline AnotherMe prefix failures remain disclosed by the
  preceding verifier; no files were rewritten to conceal them.
- The table renderer's `--check` validates all 40 displayed rows from JSON.

## Issues found and corrected during verification

1. The first run's protocol file was clarified from an incorrect short label
   “route deviation” to “sparse queries” for S3.C after that run had completed.
   Its source-hash mismatch was correctly rejected. The shipped artifact was
   regenerated, **not patched with a replacement checksum**.
2. Selection compared .7833333333333334 against .7833333333333333 as if the first
   were meaningful evidence of better Recall. Decision rates now round to 12
   decimals before applying the next tie-breaker, with a regression test. This
   changes the K=3 fallback selection, not the mechanism grid or raw scores.
3. All 160 initial-versus-final public transcripts, anchor/state streams,
   utility responses, attacker error arrays and budget fields were compared:
   **zero differences**. Only the protocol and selection-source hashes changed;
   runtime measurements naturally changed. The initial uncommitted run remains
   recoverable under ignored `tmp/contextual_lane_initial_20260907/`, and is not
   shipped as final evidence.
4. An added cache test initially omitted the required lane-schema metadata from
   its synthetic graph. The fixture was corrected; the mechanism's schema guard
   was preserved. Final full tests pass.

## Costs and deployment limits

Environment: Python 3.11.12, Darwin arm64, NumPy 2.3.2, SciPy 1.16.1,
NetworkX 3.5. Timings are local measurements, sensitive to cache/order/load.

The public context arrays occupy 12,663,252 bytes, excluding Python objects,
graphs, intermediate distances and caches. Its fingerprint is
`23965cecde168fdfdcb9ca686a9c5839126dab496471f9777ac741da14cd23a0`.
The shipped run took 19.62 ms to **load a warm context cache**, not to build it
from scratch. The verifier rebuilt without the cache, but does not claim that
19.62 ms as cold construction latency.

| Variant | Mean initialization ms (40 runs) | Macro mean step ms | Mean per-run p95 ms |
|---|---:|---:|---:|
| baseline | 528.89 | 13.32 | 25.79 |
| route | 663.18 | 40.59 | 61.28 |
| coverage | 613.65 | 29.29 | 58.85 |
| combined | 727.92 | 56.64 | 95.21 |

Step means/p95 above average the 20 case/K/role cells (each already averages two
runs). These are **not pooled event percentiles**, confidence intervals or phone
benchmarks. A one-event S1 prefix has no meaningful tail-latency distribution.
Score generation is measured; service evaluation is not included in step time.

## Next iteration: acceptance criteria for Claude Code

1. **Uncertainty-aware utility target.** Replace the single protected-anchor POI
   target with expected utility under a distribution inferred only from public
   context and protected history. Derive its likelihood/normalizer (including
   noisy reuse), or label a learned approximation and measure calibration. Do
   not secretly rank proposals using raw GPS or a future route. Keep the current
   four variants as controls; pair the anchor streams again.
2. **Attacker adaptation.** Include mechanism-aware inference on stable tracks,
   known offsets, directed routes and POI preferences. The defender's knowledge
   of public context must not be hidden from the attacker. Evaluate each
   proposed utility gain for privacy regression, especially S2.B.
3. **Move past this development family.** Declare new SUMO demand families and
   splitting rules before selecting parameters. Keep all windows from one
   underlying route/family together. Expand the moving/stopped duration mix and
   budget horizon. Do not call family93 an untouched test after another update.
4. **Utility-matched comparisons.** Sweep budget and overhead on validation,
   report infeasible settings, and compare at matched Recall/latency/output
   costs. Finish trained neural comparator reproductions as a separate tracked
   task; preserve exact adaptation labels until then. Non-DL remains a fair
   comparator choice, not a restriction on the proposed mechanism.
5. **Scenario-specific attacks.** Add actual S4–S10 inference/evaluation modules
   on the new labelled suite and generate missing S1.C / S6.C data. Five
   S1–S3 subcases do not equal five of the ten distinct threat models.
6. **Operational checks.** Measure cold/warm memory and event latency, test
   coincident-lane POI access, and replay movement with SUMO traffic constraints.
   Expose only `public` transcripts if integrating these variants into the app;
   keep evaluator anchors and synthetic truth behind the evaluator boundary.

## Artifact provenance

New result JSON SHA-256:
`61274220fd89b2ec06ebda6be67cc7f01f305b4e610307fe990486258ef85d0a`.

Generated table SHA-256:
`35cc06b63d595b79c6278201df042713711b3b1a3accff969a03d4eca509869e`.

Canonical PDF: `artifacts/reports/graduation_thesis.pdf`, 64 A4 pages, SHA-256:
`d2f501bd62fa8ebb996c09372bf75a1c71b8d513118786ea716da3320fffda84`.
Built with XeLaTeX/latexmk; no overfull boxes, undefined references/citations,
missing glyphs or warnings in the final log. Page renders were reviewed in
contact sheets, with detailed checks of the changed architecture diagram,
equations, tables and counter-evidence. The extra near-empty trailing page was
removed by condensing repeated experiment discussion. Local QA images are in
ignored `tmp/pdfs/contextual_lane/`, not new public deliverables.

Reading guide: §5.7 starts PDF page 46 (printed 39); §6.7 starts PDF page 58
(printed 51); the two new result tables are PDF pages 59–60 (printed 52–53).

Unchanged dataset SHA-256:
`826bf236588448e5e59b5e4e0304ddef3e3cf85751df8dddf901b72ae88aabb9`.

Unchanged preceding lane results SHA-256:
`20f93757ba2e2dde28a3b63db01363c8689aa7692b0703c5212412f9e3b7b65c`.

Unchanged older paper benchmark SHA-256:
`2df043bfb73c3b2f630046381daf5b4bc9f6941fb5be925b0967d6de98f27538`.
