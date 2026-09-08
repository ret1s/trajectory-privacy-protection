# Prior-factor and loss-aware attacker development

Parent `50d15250603070e9f2f861a40eba2d5df52b5fea`. SUMO + OSM only.
This study uses the unchanged SQLite v3 release. Families 201--204 have already
been inspected and are explicitly **reused development**, not fresh confirmation.

## What is included

- [Prespecified protocol](../../../thesis/notes/prior_factor_protocol.md).
- Seven configurations: BR-lane; learned L/L and uniform U/U covers; L/U and U/L
  initial/motion cross-controls; cell-balanced C/C; fixed half-mixture M/M.
- K=5, nine S1--S3 cases, at most 12 allowed events and three paired RNG draws.
- Training 101/102, validation and selection 103/104, development 201--204.
- Five additional shadow decisions using the same public-prefix features and
  training labels. Empirical neighbor mass is NOT an exact/calibrated posterior.
- `readout.json` and `results_tables.tex`: full development readout and thesis
  lookup tables. `verification.json`: independent reconstruction/replay receipt.

Control generation fields and all old attack errors are preserved exactly.
Additional attack predictions/scores live in new artifacts, never overwrite
historical studies. MAE/Hit attacker selections are distinct and validation-only.
Reused wall-clock timings do not support a controlled speed comparison.

Each shadow bank has 360 observations but only 36 unique XY labels and two
families. This is narrow auxiliary support, not 360 independent users. Neither
low Hit nor an unchanged expanded attack envelope proves strong privacy.

## Reproduction

Use the repo's pinned environment and prepared inputs named/hashes checked in
the manifests: `artifacts/datasets/scenarios.sqlite3`,
`cache/scenario_suite_v1/beijing_smoke.net.xml`, and `data/raw/Beijing.osm.gz`.
Some large raw/cache inputs are not vendored in Git; an arbitrary fresh OSM
download is not a byte-identical substitute. Rebuild/obtain the pinned inputs
using the existing dataset workflow before running this experiment. Public
context/belief caches can then be rebuilt if absent.

Stage writers refuse to overwrite evidence. Use a fresh output directory:

```bash
venv/bin/python -m experiments.run_prior_factors --phase training --output /private/tmp/prior-factor-replay
venv/bin/python -m experiments.run_prior_factors --phase validation --output /private/tmp/prior-factor-replay
venv/bin/python -m experiments.run_prior_factors --phase development --output /private/tmp/prior-factor-replay
```

Canonical committed evidence is checked/exported with:

```bash
venv/bin/python -m experiments.verify_prior_factors
venv/bin/python -m experiments.export_prior_factors
venv/bin/python -m pytest --override-ini addopts='' -q
```

The verifier replays all four new methods, replicate 1 in every phase, and
checks every row's scores. All timed JSON hashes differ across fresh runs;
scientific fields are the reproducibility target, not timing-contaminated bytes.
Evaluator files contain synthetic truth/anchors: only `row.public` represents
what the LSP may observe, not the whole artifact.
