# Generated artifacts

This is the only top-level location for tracked generated deliverables.

- [`benchmarks/`](benchmarks/) contains current machine-generated experiment
  evidence consumed by the benchmark dashboard and thesis.
- [`reports/`](reports/) contains reviewed human-facing PDF releases.

`benchmark/` at the repository root is source code; `artifacts/benchmarks/` is
generated evidence. Historical or superseded artifacts belong under `archive/`.

The September 7 controlled S1--S3 release is
[`benchmarks/report_demo/`](benchmarks/report_demo/): a checksummed result JSON
and generated LaTeX tables. It contains synthetic evaluation truth and must not
be treated as an attacker-only export. The web app exposes a separate public
transcript and an opt-in evaluator endpoint. See the
[`release guide`](../thesis/notes/report_demo_release_2026-09-07.md).
