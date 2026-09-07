# Generated artifacts

This is the only top-level location for tracked generated deliverables.

- [`benchmarks/`](benchmarks/) contains current machine-generated experiment
  evidence consumed by the benchmark dashboard and thesis.
- [`reports/`](reports/) contains reviewed human-facing PDF releases.

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
