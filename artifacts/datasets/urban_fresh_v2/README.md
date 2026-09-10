# Fresh SUMO scenario release (2026-09-10)

- Source: pinned OSM passenger network + SUMO, no GeoLife.
- Fixed seeds: 301–306 validation, 307–312 confirmation.
- 12 families, 264 completed trips, 172,443 native FCD points, 393 records.
- All ten scenarios / 30 subcases occur in both splits; individual families
  cover 24–30 subcases. See verification.json for exact denominators.
- 43 missing family/subcase combinations; these are not failed protected runs.
  The inherited rejection reason `not_implemented_requires_context_or_activity_model`
  is a legacy generator label, not proof that S6.C is unimplemented: ten S6.C records
  exist. The generated records and scenario predicates are authoritative.
- No exact old routes/windows. Two stationary S2 windows overlap validation and
  confirmation across different full trips; the benchmark reports exclusion sensitivity.
- Data include synthetic identity, relationship and query-intent labels. These are
  not observations of real people or proof of protection for all targets.

## Source of truth and history

Use `artifacts/datasets/evaluation_v1.sqlite3`, release **urban-fresh-v2**.
Pin the content hash from registry.json; the existing ScenarioStore API defaults
to read-only and rejects updates to sealed releases and their logs.

The original dataset.json uses fresh_validation/fresh_confirmation roles. The
SQLite adapter preserves those in evaluation_role, maps split names to the schema's
development_validation/confirmation values, and records the generator hash. It
changes no trajectories, observations, labels or identities. The independent
registry verification reverses this adapter and compares the full original bundle.
Consequently original JSON content hash and adapted DB content hash differ by design.
The old scenarios.sqlite3 database remains byte-for-byte unchanged.

## Reproducibility

From repository root:

```sh
venv/bin/python -m experiments.build_fresh_scenarios_v2
venv/bin/python -m experiments.verify_fresh_scenarios
venv/bin/python -m experiments.publish_fresh_scenarios
venv/bin/python -m experiments.verify_fresh_registry
```

These are write-once generation/publication commands, not overwrite commands.
For a new rebuild choose unused --output/--workdir paths. Native XML under
cache/fresh_scenarios_v2 is untracked, so the full native audit needs that local
cache. A clone still has the complete JSON, database, source hashes and hash-bound
verification receipts (not cryptographic signatures). Re-simulation produces
a new provenance manifest; compare realized records/trajectories, not machine- and
timestamp-dependent XML header hashes. Never relabel an absent native cache as a
passed local rerun. The rejected first build remains at ../urban_fresh_v1/.
