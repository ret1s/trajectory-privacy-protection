# Reachable query coverage: privacy–utility–cost development frontier

This study tests four dummy-only selectors at three fixed session budgets.
It is **an internal component study on reused development data**, not fresh
confirmation, an official comparator reproduction or a SOTA leaderboard.
No previous result, dataset release or default protector is replaced.

## Inputs and method boundaries

- SUMO+OSM only. Existing `urban-scenarios-v3` core data and designated
  64-family `urban-shadow-v1` auxiliary training; no GeoLife and no use of
  auxiliary holdout families 1065–1080.
- Four selectors: BR-lane geometric, uniform mean-coverage greedy, same
  objective with bounded single-track exchanges, category-normalized capped
  coverage with the same exchanges. All use K=5, H=12, theta=200 m.
- B=.12/.24/.48 per metre. Same protected anchors within each B. The old
  B=.24 geometric and mean-greedy core outputs must match exactly.
- Client reference remains top-5; LSP replies top-5 or top-10 at unchanged
  published coordinates. All six public categories are queried. Extra reply
  items/ID-list bytes are costs, not free algorithmic improvement.
- For every method/B: 1,240 training runs (3,233 labeled observations),
  216 validation runs and 432 reused-development runs. Total: 5,664 runs.
  Shadow models have matched data and parameters; MAE/Hit attacker selection
  is separate, per case, validation-only. The nine cases are S1.A–S3.C.

The source-pinned design is `thesis/notes/coverage_frontier_protocol.md`.
Execution/checkpoint and separate timing amendments are in the same folder.
Only public map resources are reused within a worker; session state and RNG
are reset. Fresh-instance/prefix parity is independently checked.

## Artifact layout

- `b0.12/`, `b0.24/`, `b0.48/`: sealed training, validation, selection and
  development JSON + SHA256 receipts; data-only compressed forest arrays;
  independent verification receipts.
- `timing.json`: 36 fresh-instance serial runs on three fixed validation
  records, initialization reported separately. Local host, not mobile latency.
- `readout.json`: exact macro/case/category/family readout and finite-grid
  non-dominance **within each B**; no cross-budget privacy ranking.
- `readout_verification.json`: independent aggregation/dominance audit.
- `tables.tex`, `frontier.png`: thesis-native tables and scientific figure.

Training and evaluator rows contain private ground truth and protected anchors
for audit. Only each row's `public` projection is an LSP/adversary transcript.
These files are simulated research evidence, not production user logs.

## Reproduction

Run from repository root in the pinned venv (including sklearn 1.7.1). Inputs
and native SUMO network caches are resolved and audited by `prepare`; this is
not a promise of offline reconstruction without the project's cached maps.

First-generation sequence, only when these output files do not already exist:

```bash
venv/bin/python -m experiments.run_coverage_frontier --budget .12
venv/bin/python -m experiments.run_coverage_frontier --budget .24
venv/bin/python -m experiments.run_coverage_frontier --budget .48
venv/bin/python -m experiments.verify_coverage_frontier --budget .12
venv/bin/python -m experiments.verify_coverage_frontier --budget .24
venv/bin/python -m experiments.verify_coverage_frontier --budget .48
venv/bin/python -m experiments.profile_coverage_frontier
venv/bin/python -m experiments.export_coverage_frontier
venv/bin/python -m experiments.verify_coverage_frontier_readout
venv/bin/python -m experiments.plot_coverage_frontier
```

The three generation jobs may run concurrently; the separate profiling step
must wait for all generation/verification jobs. Timing from concurrent runs is
not used as comparative latency. Use `--resume` only for an interrupted run
with identical source/input pins; hashed per-record checkpoints avoid restart.
Incomplete unsealed JSON is preserved, not overwritten.

To recheck committed evidence without modifying receipts:

```bash
venv/bin/python -m experiments.verify_coverage_frontier --budget .12 --check-only
venv/bin/python -m experiments.verify_coverage_frontier --budget .24 --check-only
venv/bin/python -m experiments.verify_coverage_frontier --budget .48 --check-only
venv/bin/python -m experiments.verify_coverage_frontier_readout --check-only
venv/bin/python -m pytest --override-ini addopts='' -q
```

For a new scientific experiment, declare a new protocol and destination rather
than editing these sealed outputs. The detailed interpretation and remaining
publication gates are in `docs/reviews/verification_coverage_frontier.md`;
closest primary work is in `docs/research/coverage_frontier_literature_review.md`.
