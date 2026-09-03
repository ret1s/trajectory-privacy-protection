# Trajectory Privacy Protection

Research code and thesis artifacts for **“Bảo vệ tính riêng tư về quỹ đạo cho
người dùng dịch vụ dựa trên vị trí.”** The active system studies road-aware,
dummy-generation protection for continuous location-based services and evaluates
it against explicit attacker, utility, and output-contract assumptions.

## Start here

- Final thesis source: [`thesis/main.tex`](thesis/main.tex)
- Current thesis PDF: [`output/pdf/graduation_thesis.pdf`](output/pdf/graduation_thesis.pdf)
- Problem formulation: [`docs/research/problem_formulation.md`](docs/research/problem_formulation.md)
- Foundations study guide: [`output/pdf/location_trajectory_privacy_foundations.pdf`](output/pdf/location_trajectory_privacy_foundations.pdf)
- Comparator status and evidence: [`benchmark/README.md`](benchmark/README.md)
- Documentation index: [`docs/README.md`](docs/README.md)

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
output/pdf/    Three curated PDF deliverables
outputs/       Current machine-generated experiment artifacts only
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
# Fast, dependency-light regression suite
venv/bin/python -m tests.run_all

# Same suite through pytest
venv/bin/python -m pytest -q tests

# Formal mechanisms on GeoLife
venv/bin/python -m experiments.run_benchmark --quick

# Repeated-report / averaging study
venv/bin/python -m experiments.run_averaging_multi

# Dummy-generation benchmark; SUMO is the default source
venv/bin/python -m experiments.run_dummy_benchmark --quick

# Read-only benchmark dashboard: http://127.0.0.1:5000/
venv/bin/python -m web.benchmark_app

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
- PR-SM-REM randomizes reuse and has a scoped per-step bound; a complete
  window/event budget manager is still future work.
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

## Outputs and archive policy

Only current reproducible results stay under `outputs/`. Curated human-facing
PDFs stay under `output/pdf/`. Old timestamped maps, the Internship 2 pipeline,
and superseded SOTA prototypes are retained under `archive/` for provenance and
must not be imported by active code.

Historical verification files preserve the paths and line numbers that were
true at their source commits. Use [`docs/reviews/README.md`](docs/reviews/README.md)
for the current path map instead of rewriting old audit records.
