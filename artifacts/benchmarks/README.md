# Current benchmark artifacts

This directory contains tracked machine-generated evidence. The newest audit is
[`coverage_frontier/`](coverage_frontier/): reachable coverage refinements at
three fixed budgets, matched inference and explicit service-resource costs.
[`expanded_shadow/`](expanded_shadow/) retains the preceding expanded SUMO
auxiliary training and inference against unchanged prior-factor outputs. Frozen cycles are
retained for provenance; use each subdirectory's protocol and source hashes.

The original top-level artifacts are:

- `benchmark_results.json` — GeoLife REM-family moving benchmark.
- `averaging_multi_results.json` — repeated-report study.
- `dummy_benchmark_results.json` — SUMO dummy-generation benchmark.
- `dummy_benchmark_map.html` and `dummy_benchmark_preview.png` — integrity-linked
  evaluator visuals declared by `dummy_benchmark_results.json`.

Regenerate through the corresponding module under `experiments/`; do not edit
these files manually. Old timestamped maps and superseded `sota_demo_*` results
are under `archive/`.
