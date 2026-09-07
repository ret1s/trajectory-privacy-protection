# Controlled study v2 — not a SOTA reproduction

`results.json` contains the full evaluation record, including synthetic ground
truth. Its SHA-256 is in `results.sha256`. Do not give this file directly to an
attacker: use the public transcript endpoint in the Flask replay.

Current scope: S1/S2/S3 and controlled hidden endpoints S9/S10, three SUMO seeds,
12 distinct test trips, 60 scenario windows, K=3/5, 12 reported configurations.
1,440 attempted runs: 1,343 completed, 96 N/A, one AnotherMe mapping failure.
S4–S8 remain specified, not evaluated. Data are SUMO + OSM only; no GeoLife.

BR-Dummy variants include fresh anchors, private reuse and a configuration
selected on a separate split. All six seed/K grid searches FAILED the joint
Recall>=90% requirement. The selected configuration is the explicitly declared
fallback, not a feasible optimum. See the thesis for the measured tradeoffs.

Reproduce from repo root:

```bash
venv/bin/python -m experiments.run_paper_benchmark
venv/bin/python -m experiments.verify_paper_benchmark --raw
venv/bin/python -m experiments.export_paper_benchmark
venv/bin/python -m web.benchmark_app --port 5050
# http://127.0.0.1:5050/report-demo
```

`results_tables.tex` is generated only after checksum/source/arithmetic checks.
Raw SUMO XML and simulator binaries stay in ignored cache/runtime directories;
the result retains configurations, commands, source hashes and source lineage.
The raw verifier therefore requires regenerating these caches on a new machine.
The old v1 study in `../report_demo/` is intentionally preserved unchanged.
