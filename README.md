# Trajectory Privacy Protection

Working design decision (10/10): retain the **0.23 m⁻¹ per-session cap** and
assess the existing Geo-I model for separate **person** and **physical-vehicle**
identity targets before selecting additional protection.
[Policy and S4–S6 assessment](docs/research/2026-10-10_session_cap_identity.md)
and [six-page method report](docs/supervisor_meeting/2026-10-09_update/report_explained.pdf).
Saved benchmarks retain their original configurations; no new per-session
L30 result or additional identity layer is claimed.

Research code and thesis artifacts for **“Bảo vệ tính riêng tư về quỹ đạo cho
người dùng dịch vụ dựa trên vị trí.”** The active system studies road-aware,
Geo-I/REM protection and road-feasible query sets for continuous location-based services and evaluates
it against explicit attacker, utility, and output-contract assumptions.

## Start here

- Final thesis source: [`thesis/main.tex`](thesis/main.tex)
- Scenario database and update history: [`storage guide`](data/scenario_store/README.md)
- Current thesis PDF: [`artifacts/reports/graduation_thesis.pdf`](artifacts/reports/graduation_thesis.pdf)
- Current thesis revision (07/10): [`scope and rebuild guide`](thesis/README.md)
  — Geo-I/REM, evidence by scenario/protocol, fresh L30 utility/cost confirmation,
  local-GPS sensitivity, historical companion diagnostic and resource accounting.
- Thesis handoff: [`formal proofs and verification`](docs/reviews/2026-10-07_thesis_formal_completion_v2.md)
- Problem formulation: [`docs/research/problem_formulation.md`](docs/research/problem_formulation.md)
- Foundations study guide: [`artifacts/reports/location_trajectory_privacy_foundations.pdf`](artifacts/reports/location_trajectory_privacy_foundations.pdf)
- Comparator status and evidence: [`benchmark/README.md`](benchmark/README.md)
- Documentation index: [`docs/README.md`](docs/README.md)
- JISA publication direction (06/10): [`assessment and improvement programme`](docs/publication/jisa_20261006/README.md)
  — venue fit, proposed Q-planner improvement, matched controls and fresh-confirmation gates;
  Q-planner trials remain development; fixed-Q response depth has new same-map
  synthetic utility confirmation, with the Geo-I backbone preserved.
- Latest supervisor brief (prepared 06/10): [`report PDF`](docs/supervisor_meeting/2026-10-06_brief/report_explained.pdf),
  [`HTML`](docs/supervisor_meeting/2026-10-06_brief/report_explained.html), and
  [`speaking script`](docs/supervisor_meeting/2026-10-06_brief/preparation_guide.pdf).
  Layered Geo-I architecture, recent paper coverage/native metrics, separate
  benchmark protocols, static POI cache and robust endpoint-selection results.
- Latest method development (06/10): [`refinement and limits`](docs/research/2026-10-06_method_refinement.md).
  The Geo-I backbone stays unchanged; new runs are development evidence.
- Follow-up experiments (06/10): [`public POI Q planner`](docs/research/2026-10-06_geo_i_public_service_planner.md),
  [`Geo-I / REM response depth`](docs/research/2026-10-06_geo_i_response_depth.md),
  [`planner proxy diagnostic`](docs/research/2026-10-06_qplanner_proxy_diagnostic.md),
  [`retained trial chronology`](docs/reviews/2026-10-06_qplanner_iteration_log.md), and
  [`static catalogue control`](docs/research/2026-10-06_jisa_static_catalogue_gate.md).
  Retained planner failures, matched Geo-I realizations, response-depth utility/cost
  confirmation on24 fresh families and application limits.
- Controlled dynamic POI status (06/10): [`workload and audited readout`](artifacts/benchmarks/dynamic_provider_status_20261006_v1/README.md).
  Frozen Geo-I Q streams, causal status expiry, four local purposes and retained
  current-bulk controls; secondary evidence on the already-inspected cohort.
- Local GPS robustness (07/10): [`fixed-Q diagnostic`](artifacts/benchmarks/local_gps_robustness_20261007_v1/README.md).
  L30 gains persist with 60 s local fixes; two-fix extrapolation reduces mean
  position error but lowers POI Recall. No protection sampler or Q regeneration.
- Companion inference (07/10): [`historical S8 diagnostic`](artifacts/benchmarks/s8_companion_inference_20261007_v1/README.md).
  Actual simultaneous SUMO pairs; finite-bank results with explicit historical
  public-seed limitations, without a current S8/group-privacy claim.
- Context storage (07/10): [`array and cache accounting`](artifacts/benchmarks/public_resource_footprint_20261007_v1/README.md).
  Compressed disk size, uncompressed payload and analytic cache capacity are
  distinguished; peak RAM and device cost remain unmeasured.
- Previous method development (05/10): [`Geo-I method refinement`](docs/research/2026-10-05_method_refinement.md)
  — persistent budget across linked trips, local POI purposes, native S5/S6,
  28-family endpoint checks and the query-order attack diagnostic. The Geo-I backbone
  remains unchanged; rejected candidates and limits stay in the evidence.
- Meeting follow-up: [`native metrics and initial scenario audits`](docs/research/2026-10-05_supervisor_followup.md).
  New runs use a reconstructed public map; they do not replace frozen benchmarks.
- Historical paper cycle: [`thesis/notes/paper_cycle_v2_protocol.md`](thesis/notes/paper_cycle_v2_protocol.md)
- Preceding evaluation: [`fresh-family switching study`](artifacts/benchmarks/fresh_switching/),
  [`12 new SUMO families`](artifacts/datasets/urban_fresh_v2/), and
  [`frozen protocol`](thesis/notes/fresh_switching_protocol.md).
  Six selection and six confirmation families; no training on these routes.
  Internal ablations are not faithful external SOTA reproductions.
- Preceding method development: [`reachable coverage and resource frontier`](artifacts/benchmarks/coverage_frontier/)
  and [`prespecified protocol`](thesis/notes/coverage_frontier_protocol.md).
  Reused 201–204 families are development evidence, not new confirmation.
- Preceding method development: [`prior factors and loss-aware attacks`](artifacts/benchmarks/prior_factors/).
- Preceding method development: [`prior/corridor ablation`](artifacts/benchmarks/service_recovery/).
- Preceding independent confirmation: [`service-cover study`](artifacts/benchmarks/service_cover/)
  on [`scenario data v3`](artifacts/datasets/urban_scenarios_v3/)
- Latest inference audit: [`expanded SUMO shadow routes and attacks`](artifacts/benchmarks/expanded_shadow/),
  with [`80 auxiliary route groups`](artifacts/datasets/urban_shadow_v1/).
  Protection outputs and prior-factor utility stay frozen; auxiliary holdout
  does not replace fresh core-scenario confirmation.
- Publication-oriented critique: [`latest targeted literature review`](docs/research/fresh_switching_literature_review.md)
  and [`independent verification / next research gates`](docs/reviews/verification_fresh_switching.md).
- Preceding supervisor handoff: [`verified findings and speaking outline`](docs/supervisor_meeting/2026-09-10_verified_update.md).
- Preceding frozen study: [`belief-suite results`](artifacts/benchmarks/belief_suite/)

## Repository layout

```text
benchmark/     Comparator contracts, evidence cards, adapters, and engines
core/          Active mechanisms, road-network model, and public protocol
data/          GeoLife/SUMO loaders, graph manifest, and setup instructions
evaluation/    Privacy attacks and utility/realism metrics
experiments/   Reproducible benchmark entry points and provenance helpers
web/           Active read-only dashboard and thesis simulator
tests/         Canonical regression suite
thesis/        Canonical graduation-thesis LaTeX source
docs/          Research notes, meeting records, reproductions, and reviews
artifacts/     Current benchmark evidence and reviewed report releases
archive/       Internship 2 code/documents and superseded prototypes
```

`benchmark/engines/` and `benchmark/methods/` are intentionally separate:
engines implement algorithms; methods expose adapters plus evidence/limitation
cards. Matching filenames there are not duplicate implementations.

## Environment

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements-dev.txt
```

Install the optional SUMO stack when running the controlled mobility simulation:

```bash
venv/bin/python -m pip install -r requirements-sumo.txt
```

Legacy Internship 2 applications have additional dependencies listed in
`requirements-legacy.txt`; they are not needed by the active thesis pipeline.

## Canonical commands

Run from the repository root unless a command says otherwise.

```bash
# Canonical regression suite (supports fixtures and temporary paths)
venv/bin/python -m pytest -q tests

# Formal mechanisms on GeoLife
venv/bin/python -m experiments.run_benchmark --quick

# Repeated-report / averaging study
venv/bin/python -m experiments.run_averaging_multi

# Dummy-generation benchmark; SUMO is the default source
venv/bin/python -m experiments.run_dummy_benchmark --quick

# Read-only benchmark dashboard: http://127.0.0.1:5000/
venv/bin/python -m web.benchmark_app

# Latest S1--S3 + S9/S10 replay: http://127.0.0.1:5050/report-demo
venv/bin/python -m web.benchmark_app --port 5050

# Current controlled study using SUMO + OSM only
venv/bin/python -m experiments.run_paper_benchmark
venv/bin/python -m experiments.verify_paper_benchmark --raw
venv/bin/python -m experiments.export_paper_benchmark

# Thesis mechanism simulator: http://127.0.0.1:5003/
venv/bin/python -m web.simulator

# Build the canonical thesis without creating artifacts beside the source
mkdir -p build/thesis
cd thesis
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=../build/thesis main.tex
```

Raw mobility data and OSM/SUMO caches are intentionally not tracked. Follow
[`data/README.md`](data/README.md) before running full experiments.

## Research contract

- REM and T-REM have per-release Geo-I claims for the ideal real-arithmetic
  kernel over a fixed public road-vertex support.
- SM-REM addresses repeated static releases but leaks its exact reuse pattern;
  it is not a trajectory-level privacy theorem.
- PR-SM-REM randomizes reuse and has a scoped per-step bound. BR-Dummy adds a
  fixed-horizon session ledger and stops reading new private positions after
  exhaustion; rolling-window budgeting with sustained utility remains future work.
- Float samplers are numerical approximations. Tests make finite-support issues
  visible instead of silently upgrading them into proofs.
- The three paper comparators are executable clean-room adaptations. They are
  not claimed to reproduce the authors' published numbers or full private
  assets.
- Replacement, real-plus-dummies, and dummy-only outputs have different
  attacker tasks. Raw metrics must not be combined into one cross-contract
  leaderboard.

The detailed claim registry, missing components, and source mapping live in
[`benchmark/README.md`](benchmark/README.md) and [`docs/reviews/`](docs/reviews/).

## Artifact and archive policy

Current results and explicitly referenced predecessor evidence stay under `artifacts/benchmarks/`. Curated
human-facing PDFs stay under `artifacts/reports/`. Old timestamped maps, the
Internship 2 pipeline, and superseded SOTA prototypes are retained under
`archive/` for provenance and must not be imported by active code.

Historical verification files preserve the paths and line numbers that were
true at their source commits. Use [`docs/reviews/README.md`](docs/reviews/README.md)
for the current path map instead of rewriting old audit records.
