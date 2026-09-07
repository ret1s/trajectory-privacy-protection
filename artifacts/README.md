# Generated artifacts

This is the only top-level location for tracked generated deliverables.

- [`benchmarks/`](benchmarks/) contains current machine-generated experiment
  evidence consumed by the benchmark dashboard and thesis.
- [`reports/`](reports/) contains reviewed human-facing PDF releases.
- [`datasets/urban_scenarios_v1/`](datasets/urban_scenarios_v1/) contains versioned
  pre-protection scenario data, separate from measured benchmark results.

`benchmark/` at the repository root is source code; `artifacts/benchmarks/` is
generated evidence. Unreferenced superseded artifacts belong under `archive/`;
an explicitly compared predecessor can remain here with its original audit.

The active controlled S1--S3 and S9/S10 release is
[`benchmarks/paper_benchmark/`](benchmarks/paper_benchmark/): a checksummed result JSON
and generated LaTeX tables. It contains synthetic evaluation truth and must not
be treated as an attacker-only export. The web app exposes a separate public
transcript and an opt-in evaluator endpoint. See the
[`protocol and research decisions`](../thesis/notes/paper_cycle_v2_protocol.md).
The v1 predecessor remains unchanged in `benchmarks/report_demo/` for lineage.

[`benchmarks/lane_comparison/`](benchmarks/lane_comparison/) is a separate,
development-only S1.A/S2.B/S3.A experiment: enhanced-DLS and matched BR lane-state
variants. Its directed lane POI service differs from the earlier junction
service, so its numeric scores must not be merged into the older leaderboard.
`results.json` preserves validation-selected attackers; `audit.json` and the
thesis tables additionally show an explicitly exploratory report-family attack
envelope. Both files contain synthetic evaluator truth, not public-only exports.
