# Experiment entry points

- `scenario_db.py`: local SQLite scenario registry; import immutable revisions,
  inspect update history/coverage, and read explicitly checksummed device inputs.
- `verify_scenario_db.py`: frozen JSON-to-SQL migration and all-device-view parity
  audit. This is distinct from the native SUMO scenario verifiers and attack evals.

- `run_benchmark.py`: moving GeoLife evaluation of the formal mechanism family.
- `run_averaging_multi.py`: repeated-report study over distinct stay points.
- `run_dummy_benchmark.py`: stable public entry point for the SUMO-first dummy
  benchmark; implementation is split under `dummy_benchmark/`.
- `provenance.py` and `rng_util.py`: source/data hashes, replay command, and
  order-independent semantic RNG.

`run_averaging.py` and `run_sota_demo.py` are compatibility forwarding modules.
Do not add new logic to them.

Experiments write to `artifacts/benchmarks/` by default. During verification,
pass explicit paths under `/private/tmp` to avoid overwriting committed evidence.
