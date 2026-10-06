# Commit/push boundary — 06/10/2026

This release contains the Geo-I refinements, frozen development evidence,
next-meeting report and JISA study design. Earlier source datasets and results
remain unchanged. No fresh-confirmation or publication-readiness claim is added.

## Private evaluator keys

The following local RNG master-state files are excluded from Git; public source,
seed-free evaluation copies, fitted attackers, metrics and hash receipts are included:

- `artifacts/benchmarks/endpoint_generalization_20261005/evaluator_randomness.json.gz`
- `artifacts/benchmarks/endpoint_generalization_20261005/failed_exact_common_endpoint_attempt/evaluator_randomness.json.gz`
- `artifacts/benchmarks/endpoint_order_20261005/evaluator_shuffle_randomness.json.gz`

Nine protected original `{fit,selection,test}-{scale025_L20,scale100_L10,scale100_L20}.json.gz`
bundles in the generalization directory are also local-only: their resolved
`rng_seed_evaluator_only` fields disclose session sampler state even without
the master key. [New release copies and provenance](../../artifacts/benchmarks/endpoint_generalization_20261005/public_release_20261006/README.md)
omit these fields; every other decoded value, including Q and labels, is
unchanged. Original hashes/receipts remain unmodified. These are synthetic
evaluator bundles; only `events` is attacker-visible in the experiment.

Full private randomness/order checks in `verify_endpoint_generalization`,
`verify_endpoint_order`, and `verify_endpoint_robust_selection_20261006_review`
and the original robust selector verifier/readout need original local inputs.
A fresh clone does not possess them; archived
verification receipts describe the original local check and do not claim that
the private checks were rerun by a public reader. Public output/metric replay
does not justify recreating original RNG keys from a public seed.

The JISA REM/Planar pilot's public-output verifier works without its generation
keys. Its keys and SQLite ledgers already live outside the repository in the
private work directory documented in the pilot README. New generation must use
a new artifact/work directory and keys; retained public outputs can be checked
without regeneration. Evaluator-only synthetic labels/accounting are research
evidence, distinct from RNG master keys.

## Restoring archived dataset input

The draft publication inventory also pins the expanded endpoint JSON. Its
compressed archive is tracked; uncompressed `dataset.json` is intentionally
ignored. On a fresh clone restore it before running the draft preflight:

```bash
python -m experiments.endpoint_dataset_archive unpack
```

See [dataset archive instructions](../../artifacts/datasets/endpoint_holdout_expanded_v1/README.md).
The helper verifies hashes and refuses to overwrite different input. This
restores existing evidence and does not create a new confirmation cohort.

## Release checks

Both canonical test commands passed **570 tests, with 9 integration skips**
because original network/cache inputs are unavailable. Dependency check and
whitespace checks pass under scoped generated-format attributes (CRLF CSV and
Matplotlib SVG path spaces), preserving original byte/hash receipts. No new file
exceeds GitHub's 100 MiB file limit;
largest new checkpoint is approximately 18.4 MiB. Reviewed report exports and
the final independent REM/Planar verifier receipt are retained. Code/config,
methodology and application gates remain in the
[JISA plan](../publication/jisa_20261006/README.md).
