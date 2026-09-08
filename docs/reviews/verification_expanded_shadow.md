# Verification: expanded SUMO auxiliary routes and inference

Date: 2026-09-08. Parent: `07634a0ed3daa2025c76a19a85e6344e26dfef89`.
This is an inference-only audit. Do not retune or promote the defender from
its findings without a new declared development cycle and fresh confirmation.

## Question, authority and frozen boundary

Expand auxiliary SUMO routes and test stronger inference. The seven existing
K=5 protectors, B=.24/m, theta=200 m and horizon=12 are unchanged. U/U remains
the earlier validation-selected fallback; the utility gate is still unmet.
Original core outputs, utility, RNG draws and attack errors are retained.
The three historical SQLite releases remain sealed and semantically unchanged.
Only SUMO+OSM is used; no GeoLife.

Training/selection order is in the source-pinned protocol and pre-attack quality
amendment. Train 64 new families 1001–1064, select attackers only on old 103/104,
then score reused 201–204 and 16 new auxiliary holdout families 1065–1080.
The auxiliary holdout is not drawn from the nine original core-case gates and
must not be called fresh confirmation of those cases.

## Accepted dataset and registry checks

- 80 groups, 160 sessions, **175,211** native 1 Hz FCD samples, 320 AUX windows.
- Train: 256 windows / 2,873 observations / 1,345 unique XY / 587 occupied
  120 m cells. Holdout: 64 windows / 714 observations / 322 unique XY / 208 cells.
- Four schedules: cruise every 20/every 60 seconds, on-lane stop every 5 seconds,
  and two six-point visits separated by movement and a gap exceeding 120 seconds.
- All 80 native SUMO families completed. Independently parsed every FCD field,
  route, arrival and stop duration; max observed 1 s displacement 9.207 m,
  lane-position geometry error 1.933 m (existing 4.1 m gate not relaxed).
- Public planning recorded 311 length/connectivity rejections, two unavailable
  returns and four duplicate-stop plans. All 80 final families were retained;
  no route was removed after observing an attack score.
- No exact route or full-window overlap with v3; no duplicate window across
  auxiliary families. All 160 full routes are distinct. Different family
  identities and split boundaries were checked. Shared roads are legitimate:
  3,032 distinct external edges, 901 shared with v3.
- Holdout-to-aux-training true-label distance: median 30.47 m, p95 159.59 m.
  This diagnoses same-map coverage, not geographic separation or an attacker
  input. Sampling without replacement and shared windows mean family counts
  are experimental blocks, not a claim of IID population sampling.
- `urban-shadow-v1` is appended as the fourth release. All three predecessor
  logs remain exactly equal; lossless export and all 320 DB device views pass.
- Content SHA256:
  `7563a904421ed569425c893eda36221146729a32a3eff50366faec0fb93901b0`.
  Source JSON SHA256:
  `0cf7e18da1107879ad7f1822f51c4e5e2d66cdd875cab70504096a854d02a11f`.

Native receipt: `artifacts/datasets/urban_shadow_v1/verification.json`.
Transaction receipt: `artifacts/datasets/urban_shadow_v1/registry.json`.
The registry's whole-file hash changes when a release is appended; old
experiments depend on semantic release hashes, not the current binary DB hash.

## Negative QA findings retained

1. The first SUMO run used inherited parking=true. Family 1048 had a 6.793 m
   parking displacement from its declared lane position. Rejected JSON SHA:
   `974eebdbf9ed62d71d9338e2f758ecd35754a56c8026bc60d99e216985e509bc`.
   Changed to on-lane stops, not a looser gate or edited FCD.
2. The second run passed 175,620 native samples but stationary windows repeated
   within training (1030/1001, 1034/1023, 1059/1047), across train/holdout
   (1069/1036), and within holdout (1079/1076). Rejected JSON SHA:
   `b252a0f988144c25844f8c0200e1f3e4e70bd75e2b64f8fd92ea95bc3c2cc819`.
   The public planner now reserves distinct stop lanes and 1 m planned-stop
   cells before simulation. Same 80 seeds/roles; no attack score was seen.
3. Full-suite testing exposed a registry integration assumption: fourth log
   entry was treated as core v4. The parity test now resolves the actual release
   identity, checking the auxiliary release too. Production schema/store and
   source-pinned historical experiment code were not changed.
4. Figure QA found incompatible coordinate frames: old model-local XY was
   initially drawn against SUMO-offset roads. The figure now projects raw GPS
   through SUMO for **every** panel and asserts all plotted points fit the map.
   This affected the draft figure only, not any attack, dataset or metric.

Rejected simulation/generator snapshots and the rejected projection figure
remain in ignored `cache/urban_shadow_v1_rejected_*` and
`cache/shadow_support_rejected_projection*`. Nothing material was deleted.

## Attack implementation and audit strategy

Eight matched empirical rules distinguish expansion of data from model-class
changes: kNN means 1/5/15, finite-action MAE and disk Hit with 15/45 neighbors,
and mean45. Two additional ExtraTrees regressors predict direct XY/residual XY
relative to the current public centroid. Each uses 128 trees, leaf 5, depth 18,
integer max_features=1, no bootstrap, random_state=20260908, n_jobs=1; sklearn 1.7.1.
This setting means **one** feature per split, not the floating-point fraction 1.0.

Every new rule uses the same 21 causal public features. Auxiliary labels and
the old permitted training labels can be used for fitting. Private anchors,
actual speed/lanes, identity, scenario/profile IDs and future candidates cannot
enter prediction. Normalization and forests fit training only; model arrays
are saved without pickle. Forests optimize squared error, not MAE/Hit.

`experiments/verify_expanded_shadow.py` independently reconstructs the feature
matrices, normalization and labels; refits sklearn forests and compares each
array; recomputes empirical decisions with the separate geometric oracle;
scores raw GPS with strict distance<=100 m; checks old-field parity and selected
attacker provenance; verifies directed travel-time feasibility; and replays a
fixed subset of auxiliary full/prefix runs. Executable verification scripts
serve as the reproducible analytical companion, rather than duplicating large
private evaluator data in a notebook.

## Interpretation and next cycle

On reused 201–204 (nine cases, three paired RNG draws), the same U/U outputs
score **2.7778% → 3.7603%** selected Hit100 and **1283.56 → 1110.12 m** selected
MAE. The exploratory per-case Hit envelope rises **14.2989% → 16.2280%**.
This is increased measured leakage, not improved defense. U/U Recall remains
92.3465%, worst case 86.0060%; no utility gate or defender selection changes.
M/M is a counterexample to selection transfer: selected Hit falls 2.0833% →
1.4660% while its exploratory envelope rises 8.7213% → 9.9317%. Do not report
that fall as stronger privacy; the output did not change.

Important counterweight: U/U's exploratory MAE envelope changes only
508.1664 → 503.7622 m. The larger improvement in validation-selected MAE must
not be presented as a comparably large improvement over the strongest old
per-case bank. Attack selection was already weak relative to that envelope.

U/U paired family changes (new minus old):

| Reused family | Selected Hit change (percentage points) | Selected MAE change (m) |
|---|---:|---:|
| 201 | +0.7202 | +7.6823 |
| 202 | +1.5432 | -16.7560 |
| 203 | +0.9259 | -512.7253 |
| 204 | +0.7407 | -171.9491 |

Hit increases in all four blocks; MAE does not improve in family 201. Four reused
families do not justify a population-significance claim or step-level p-values.
The fixed 1-NN rule on U/U changes from 1486.66 to 802.56 m MAE with added data;
the new residual forest scores 550.43 m across the nine cases but was not the
global validation-selected MAE attack. Keep this individual-attack diagnostic
distinct from selection-locked performance. The exact readout retains every
shared attack, not only the favorable examples.

The **auxiliary holdout is a negative selection-transfer result**, not a blanket
success. U/U's old global Hit choice is `continuity`; the new one is
`expanded_shadow_loss_hit_action_45`, both chosen on original validation.
On cruise20, selected Hit changes 4.1667% → 2.6042%; on return5 it changes
9.3750% → 3.1250%; cruise60 changes 2.8499% → 3.1351%; stop5 remains 0%.
The global MAE choice remains old `shadow_knn_1`, so its holdout MAE is identical.
Do not call the four profiles fresh S1–S3 confirmation or use their labels to
retune this audit. A separately declared richer validation design is needed.

Exact final scores and verification counts are in the linked machine receipts
and `artifacts/benchmarks/expanded_shadow/readout.json`; selected scores and
exploratory envelopes are intentionally separate. Never infer stronger defense
from a worse new attacker or from a validation choice that fails to transfer.
Utility is exactly the earlier result; this audit cannot fix its weak cases.

See `docs/research/expanded_shadow_literature_review.md` for primary readings,
the distinction from aggregate membership inference, and publication gates.
Richer synthetic coverage does not establish likelihood-optimal inference,
calibrated population realism, stronger Geo-I, S4–S8 protection or conference
readiness. New validation families, temporal/topological attacks, profile-shift
tests and fresh locked core confirmation remain required before stronger claims.

## Reproduction

Expanded independent verification **passed**: 3,374 runs / 32,354 events;
1,582 scored runs; all 1,134 old rows retain their scientific fields exactly;
122,430 new point predictions independently checked; 14 sklearn forests refit
with all 8,960 arrays equal; 144,900 directed transitions valid; 280 full
protection replays and 280 strictly shorter prefix replays match. Holdout
old-bank predictions were independently recomputed for all 448 runs.
Receipt: `artifacts/benchmarks/expanded_shadow/verification.json`, with phase
hashes and verifier source SHA256. No new defense selection was performed.

Regression suite: **253 passed** (`venv/bin/python -m pytest
--override-ini addopts='' -q`, final repeat 28.01 s). The old prior-factor verifier also
passed unchanged: 1,512 rows, 1,134 scored rows, 288 replays, 192 strict-prefix
checks, 41,265 directed transitions and 43,470 category queries. SQLite
verification passes all four releases. The database is 100,065,280 bytes;
future growth will need a separate storage-distribution decision rather than
silently dropping historical releases to stay within Git hosting limits.

Dataset commands and cached-input limitations:
`artifacts/datasets/urban_shadow_v1/README.md`.
Ordered experiment/reverification commands:
`artifacts/benchmarks/expanded_shadow/README.md`.
Canonical thesis sources: `thesis/dataset_registry.tex` and
`thesis/expanded_shadow_comparison.tex`; PDF: `artifacts/reports/graduation_thesis.pdf`.

## Final artifact QA

- Readout SHA256:
  `f8b1f68feda22049d38455c8a51f7622c89f075f7e46fbfe8a56ed4523fddc8b`.
- Expanded verification receipt SHA256:
  `a537dcc1bf22e5b8f602113a1867e9d9c4cfc5b103a075704a0c0cb8ed870833`.
- Canonical PDF: **94 A4 pages**, SHA256
  `51ebde109874047d3397a5dd80550c94af693ec6d1620351245a309394edde89`.
- Main additions: Section 3.9.5 (physical pages 34–35), Section 6.12
  (physical pages 85–88), Figure 6.1 and Tables 6.22–6.25; abstract/conclusion
  and source bibliography updated. Printed page numbers differ from physical.
- All 94 pages rendered and inspected on contact sheets; the new dataset,
  figure, methods, tables and interpretation pages inspected at full size.
  The last rounding correction (3.125% displayed as 3.12%, matching the table's
  round-to-even formatting) was rebuilt and visually checked on physical88.
- Final LaTeX log: no overfull boxes, undefined references/citations or missing
  characters. Eleven underfull warnings remain in historical bibliography
  formatting; none comes from the new results. No duplicate title page, no
  blank pages, no off-paper word bounding boxes, no replacement/`??` markers.
  Poppler emits a control code for one legacy math glyph in bbox XML; it was
  stripped only for XML parsing and the rendered formula checked, not edited.
- PDF/visual QA scratch files are in ignored `build/shadow_pdf_qa/`; report
  source and numerical evidence remain in their canonical tracked locations.
