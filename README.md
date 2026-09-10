# Trajectory Privacy Protection

Research code and thesis artifacts for **“Bảo vệ tính riêng tư về quỹ đạo cho
người dùng dịch vụ dựa trên vị trí.”** The active system studies road-aware,
dummy-generation protection for continuous location-based services and evaluates
it against explicit attacker, utility, and output-contract assumptions.

## Start here

- Final thesis source: [`thesis/main.tex`](thesis/main.tex)
- Scenario database and update history: [`storage guide`](data/scenario_store/README.md)
- Current thesis PDF: [`artifacts/reports/graduation_thesis.pdf`](artifacts/reports/graduation_thesis.pdf)
- Problem formulation: [`docs/research/problem_formulation.md`](docs/research/problem_formulation.md)
- Foundations study guide: [`artifacts/reports/location_trajectory_privacy_foundations.pdf`](artifacts/reports/location_trajectory_privacy_foundations.pdf)
- Comparator status and evidence: [`benchmark/README.md`](benchmark/README.md)
- Documentation index: [`docs/README.md`](docs/README.md)
- Current research cycle: [`thesis/notes/paper_cycle_v2_protocol.md`](thesis/notes/paper_cycle_v2_protocol.md)
- Latest evaluation: [`fresh-family switching study`](artifacts/benchmarks/fresh_switching/),
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
- This week's supervisor handoff: [`verified findings and speaking outline`](docs/supervisor_meeting/2026-09-10_verified_update.md).
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
