# Verification — graduation thesis latest review baseline

Date: 2026-09-02

Branch: `verifier`

Canonical benchmark source commit: `e68aae7c034f0920b6d7a3d8d87169c838022566`

Canonical artifact schema: `msc-dummy-benchmark-v4`

## Verdict

The graduation-thesis source has been updated to the latest implemented research
state and is ready for detailed supervisor/author review. The document now has a
single canonical LaTeX source, compiles to 50 A4 pages, states the threat and
output contracts explicitly, reconciles all reported numbers with the current
benchmark JSON, and separates evaluator-only truth from the attacker-visible
transcript.

The quantitative chapter is **integration/smoke evidence only**. It proves that
the four pipelines execute on one shared SUMO road graph and that their output,
metric, provenance, and web boundaries are wired correctly. It does **not** yet
prove privacy superiority, faithful reproduction of published SOTA results, or
statistical generality.

## Canonical review artifacts

- Thesis source: `docs/supervisor_meeting/2026-09-05/report.tex`
- Canonical built PDF: `docs/supervisor_meeting/2026-09-05/report.pdf`
- Thin build wrapper: `thesis/main.tex`
- Reviewer PDF: `output/pdf/graduation_thesis.pdf`
- Benchmark JSON: `outputs/dummy_benchmark_results.json`
- Interactive evaluator map: `outputs/dummy_benchmark_map.html`
- Static preview: `outputs/dummy_benchmark_preview.png`
- Comparator method cards:
  - `docs/reproduction/transprotect.md`
  - `docs/reproduction/anotherme.md`
  - `docs/reproduction/semantic_correlation.md`

The files under `thesis/chapters/` are the preserved Internship 2 draft and are
not imported by `thesis/main.tex`.

## What the latest thesis version contains

1. The final working title is *Bảo vệ tính riêng tư về quỹ đạo cho người dùng
   dịch vụ dựa trên vị trí*.
2. The research questions distinguish protected targets, context-aware attacks,
   utility/operational cost, and evidence needed for comparison.
3. S1--S7 are defined as **cases in which trajectories are threatened**, not as
   intrinsic trajectory types. The thesis currently focuses on S3: a continuous
   urban LBS session observed by an honest-but-curious LSP.
4. The protected output is fixed as `DUMMY_ONLY`: the client publishes `K`
   stable dummy tracks and does not intentionally insert or label the real
   trajectory as a batch member. Accidental coordinate equality is not removed,
   because secret-dependent rejection would alter the claimed post-processing
   boundary.
5. The method uses an ideal finite-domain Euclidean REM/Geo-I anchor kernel,
   followed by trajectory-aware post-processing for persistence, road
   reachability, speed, spacing, and utility. The report clearly limits the
   formal statement to one event with fixed public history and explicitly states
   the independent downstream-randomness assumption.
6. Three published-method comparators are executable as source-mapped clean-room
   adaptations: TransProtect, AnotherMe, and the 2026 semantic-correlation
   candidate method. Their missing paper assets and parity conditions are listed.
7. The benchmark architecture uses SUMO mobility and candidates derived from the
   same passenger-only `.net.xml`, separates `attacker_view`, `evaluator_truth`,
   and SUMO `mobility_evaluator_only`, and keeps cross-contract ranking disabled.
8. The result, limitation, final experiment matrix, conclusion, and reproducible
   manifest chapters are synchronized with the latest implementation.

## Canonical benchmark identity

- SUMO version: `1.27.1`
- Dataset role: controlled Beijing passenger **smoke scenario**, not a calibrated
  urban population model
- Evaluation split: one held-out vehicle, 8 events from 30 s through 570 s
- Background training split: 19 disjoint vehicles, 133 transitions
- Candidate graph: 4,892 vertices and 9,138 directed edges
- Semantic graph SHA-256:
  `41c40b6e2ad63b57ccecfa8ebb89b97475cd12df54e9a340a11dd8d94b21f960`
- Semantic-hash schema: `sumo-passenger-road-graph-v1`
- Required external OSM gzip SHA-256:
  `ed0af579fb32c7164de85bca945a10432f818dd27eb80d6d5a7afa62213b0407`
- Root seed: 42
- Thesis configuration: `K=4`, `epsilon=0.02 m^-1`
- TransProtect configuration: top-10 candidates, `epsilon=5 km^-1`
- QoS diagnostic radius: 200 m

The two epsilon values are method-local parameters and are not treated as an
equal privacy-budget comparison.

## Reconciled smoke results

These values are diagnostic within each output contract.

### Track A — replacement trajectory

| Method | Mean displacement | P95 | QoS@200 m | Hausdorff | DTW | On-road | Total runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| TransProtect adaptation | 147.2399 m | 314.0562 m | 0.625 | 318.6379 m | 73.6199 m | 1.0 | 57.733 ms |
| AnotherMe adaptation | 2,088.5532 m | 2,864.2541 m | 0.0 | 2,836.6898 m | 1,044.2766 m | 1.0 | 20.810 ms |

Additional diagnostics:

- TransProtect expected travel-cost loss: 62.584 m.
- Its unreachable fraction is 3.1479967% over the complete
  `4,892 graph vertices x 8 targets` matrix; penalty: 17,382.5 m.
- VehiTrack expected inference error remains `null`; the actual attack pipeline
  has not yet been reproduced.
- AnotherMe generated 146 raw samples on its source-style 3-second driving grid;
  benchmark metrics use its explicitly aligned replacement trajectory.

### Track B — real plus `K-1` event-local candidates

Semantic-correlation clean-room adaptation, `K=4`:

- mean dummy distance: 385.3551 m;
- P95 dummy distance: 614.3752 m;
- candidate-set centroid distance: 273.8688 m;
- candidate spread: 161.2730 m;
- real-member inclusion: 1.0 by contract;
- representation-fingerprint attack success: 0.25, equal to uniform `1/K`;
- stable track count: 0; ASR and DER remain `null`;
- total runtime: 9.794 ms.

### Track C — thesis dummy-only stable tracks

Current thesis candidate, `K=4`:

- nearest dummy-output distance: 82.1697 m;
- candidate-set centroid distance: 123.1845 m;
- candidate spread: 188.7351 m;
- mean dummy distance: 221.2101 m;
- P95 dummy distance: 426.7735 m;
- 4 stable dummy tracks, on-road rate 1.0, speed-violation rate 0;
- track DTW diagnostic: 110.6051 m;
- total runtime: 1.838 ms.

## Verification performed

```text
venv/bin/python -m tests.run_all
=> 119 passed, 0 failed

venv/bin/python -m compileall -q benchmark core data experiments tests web
=> passed

node --check web/static/benchmark/dashboard.js
=> passed

JSON schema/inventory/provenance/hash invariants
=> passed

git diff --check
=> passed

latexmk -xelatex -interaction=nonstopmode -halt-on-error report.tex
=> passed; 50 pages

latexmk ... thesis/main.tex (temporary output directory)
=> passed; 50 pages

latexmk ... docs/supervisor_meeting/2026-09-05/report.tex from repository root
=> passed; 50 pages
```

All 50 PDF pages were rendered with Poppler and visually inspected. The previous
abstract and configuration-note orphan pages are gone; tables remain inside the
page margins; the benchmark pipeline and road-overlay preview are legible. The
remaining LaTeX messages are non-blocking underfull-box warnings only.

Artifact SHA-256 values:

- `report.tex`:
  `69b16b50fb85dcf223899e905125c433aca19284958f67931c407484df01a2c4`
- the three synchronized thesis PDFs:
  `5dc3b4a1eec53263fbe8a7e60dce0b8e54fe801801e03d32e212fc8c8f7607b2`
- benchmark JSON:
  `18557e4a93982e676fb5edab7e7ba59efcf4bedb7a96cc4daa393c026cf646d9`
- evaluator map HTML:
  `d076911d4d3b7e6f7845f30aa6bb0fa5de7a60db2d633bd86b3831aee24b07c9`
- static preview PNG:
  `7615b6f64886cba68955064c5cf5e467ea66e217be975ea96f057efa31c826ce`

## Claim boundaries that must remain explicit

1. The three comparators are `paper_adaptation`, not faithful reproduced SOTA.
   No thesis claim may call the current numerical gap an improvement over SOTA.
2. One trajectory, eight events, and one seed provide no uncertainty estimate or
   population-level conclusion.
3. Geometry, QoS@200 m, DTW, on-road rate, and runtime are diagnostics; they do
   not prove privacy against a specified attacker.
4. A single cross-contract scalar leaderboard is invalid. Methods may be compared
   only inside a compatible output contract, or after defining a common decision
   problem and utility/overhead budget.
5. `DUMMY_ONLY` means truth is neither intentionally inserted nor designated; it
   does not guarantee that a sampled dummy coordinate can never equal truth.
6. The current float64 Gumbel implementation can create input-dependent numerical
   zero support. Only the ideal real-arithmetic REM kernel has the stated pure
   per-event Geo-I argument.
7. Randomized post-processing additionally assumes downstream randomness is
   independent of the secret and anchor-sampling coins. The API test establishes
   that truth is not passed to stage 2, but continuing the same PRNG stream does
   not by itself prove this independence.
8. A per-event conditional bound does not automatically give the same bound for
   a `T`-event transcript. Sequential composition or a window budget manager is
   still required.
9. The kernel uses projected Euclidean distance, not graph shortest-path distance
   and not a GG-I theorem.
10. The evaluator JSON and map contain ground truth and must not be exposed as the
   public LSP response. Only `attacker_view` belongs to the attacker transcript.
11. The Beijing OSM gzip is outside Git and the documented BBBike URL is mutable;
    exact reruns require the stated input hash. JSON/map/PNG bytes also include
    runtime or rendering metadata and are not promised to be byte-identical.

## Highest-priority work after review

1. Implement the context-aware filtering attacker for each output contract and
   report attack advantage over a prior/random baseline.
2. Add a real POI/query workload and application-facing metrics: top-`k`
   recall/overlap, route/travel-time error, request count, bytes, and latency.
3. Extend the simulator from the smoke route to S1--S7, then run multiple users,
   routes, seeds, `K`, privacy parameters, and utility budgets with confidence
   intervals.
4. Reproduce paper assets/checkpoints or at least one reported table within a
   declared tolerance before using the term faithful SOTA reproduction.
5. Compare mechanisms at matched utility or matched operational cost; preserve
   the three output tracks until a common scientific decision problem exists.
6. Perform ablations for REM anchor, persistent offset, speed penalty, population
   prior, and semantic term; add a trajectory-level budget manager if a transcript
   privacy claim is retained.
7. Vendor map assets and add a strict CSP before any evaluator demo uses real
   trajectories.

## Review baseline

This verifier, the canonical thesis source, the synchronized PDF, and the current
benchmark artifacts form the baseline for the next detailed review. Future edits
should update this file or create a new numbered verifier and must not silently
promote smoke diagnostics into privacy or SOTA conclusions.
