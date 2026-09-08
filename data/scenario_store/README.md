# Scenario database

The local evaluator dataset registry is
[`artifacts/datasets/scenarios.sqlite3`](../../artifacts/datasets/scenarios.sqlite3).
SQLite is embedded in Python; no server, credentials, cloud upload or extra
Python dependency. Requires SQLite >=3.37 with JSON functions (verified here
with 3.50.4). This registry contains **SUMO+OSM v1/v2/v3 only**, not GeoLife.

## Contract

- **Release is the unit of change.** A sealed release cannot be edited/deleted
  through the supported API or ordinary row updates. Correct labels, simulation
  data, split membership or scenario definitions in a *new* release.
- Every import is one `BEGIN IMMEDIATE` transaction: rows, lossless round-trip
  check, sealing and automatic `update_log` entry either commit together or
  roll back. Retrying identical bytes under the same release/parent is a no-op.
- Supply the expected parent explicitly. The current registry is a linear
  history; two concurrent imports based on the same parent cannot both win.
  Inspect the new log before retrying a stale update; do not silently rebase it.
- Readers use read-only connections and an explicit release ID. CLI consumers
  additionally require its **semantic content SHA256**. There is no implicit
  `latest` input for an experiment.
- SQLite foreign keys are enabled and checked for **every** connection opened
  by this API. SQL schema/guard fingerprints are checked on open and verification.
- The DB file, log timestamps and byte hash can differ across rebuilds; release
  content hashes must not. Do not use the binary DB hash as the dataset identity.
- This is a local, single-writer registry, not a multi-host/network-drive service.
  Do not resolve a Git binary conflict by arbitrarily choosing one side. Preserve
  both copies, inspect their logs, reconcile revisions, rebuild and verify.

## Tables and grain

| Table | One row represents | Main relationship |
|---|---|---|
| `releases` | Immutable dataset snapshot | Previous release, byte/content hashes and import metadata |
| `families` | Related simulation family | Release and fixed split |
| `sessions` | SUMO movement session | Family, separate person/device/physical-car IDs |
| `points` | Untouched FCD sample | `(release_id, session_id, point_index)` |
| `cases` | Declared subcase, even when empty | Release-local case and target/gate metadata |
| `records` | Scenario evaluation unit | Case, family and matching split |
| `record_sessions` | Session slot within a record | Both session and record must belong to the same family |
| `observations` | Allowed point in a session slot | FK to a real FCD point, ordered without duplicates |
| `source_hashes` | Generation source checksum | Release and source path |
| `update_log` | Successfully sealed release event | Actor, UTC import time, reason, predecessor, hashes, row counts |

Variable scenario labels, evidence, plans and provenance remain validated JSON
metadata, not duplicated flat FCD bundles. Ordered arrays retain ordinals;
cross-table joins always include `release_id`. `case_coverage` is a view over
catalogue × actual splits with a left join, so zero-count cells remain visible.
There are no public attacker observations in this database.

## Inspect the committed database

From the repository root:

```bash
venv/bin/python -m experiments.scenario_db log
venv/bin/python -m experiments.scenario_db verify
venv/bin/python -m experiments.scenario_db coverage --release urban-scenarios-v2
```

For visual inspection, open the SQLite file with any existing SQLite viewer and
start with `update_log`, `case_coverage`, `records`, then `points`. Treat the
viewer as read-only. There is deliberately no automatic upload or GUI dependency.

Release IDs and semantic hashes:

```text
urban-scenarios-v1  6fc977209401f8591360e35bd7ce390bfadd9cd5098d74b47a5c9d3a9cefe358
urban-scenarios-v2  27ea74e3d13deea059df663b5429f4333af97b096888cb8ee5f0669eefbbaff0
urban-scenarios-v3  213886fc2722bbecf5e55f61e8978d2c2842bb6018b7f65f5021a261bd79358d
```

Export one S1.A device input (output file must not exist):

```bash
venv/bin/python -m experiments.scenario_db device \
  --release urban-scenarios-v2 \
  --sha256 27ea74e3d13deea059df663b5429f4333af97b096888cb8ee5f0669eefbbaff0 \
  --record v2-r101-000 --slot 0 --output /private/tmp/scenario-device.json
```

This returns **private pre-protection** observations, not a safe LSP transcript.
Use `ScenarioStore.device_view(release_id, record_id, slot=...)` to supply the
mechanism; preserve session identity only inside the evaluator. S8 keeps the
common observation epoch, and S6.C requires slot 6 for the current trip. The six
historical sessions must also be protected under a declared history protocol.
The database is not an access-control server: never give an adversary its file.

`export` uses the same release/hash/output options to reconstruct a complete
evaluator JSON snapshot. Output formatting differs from the original; compare
semantic hashes, not file bytes. Existing frozen benchmark runners keep their
original JSON paths and byte hashes to avoid rewriting past experimental inputs.
**They have not silently been switched to SQLite.** New experiments can consume
the DB device API or a pinned export and must record that choice in their protocol.

`experiments.run_service_cover` now consumes the **pinned v3 DB device API**;
its prior query joins only training families. v3 preserves 101–104 and adds
201–204 as fresh confirmation; inspected 105/106 are excluded. The on-disk
record schema remains v2, independently of the release version.

## Rebuild or publish a new revision

Use a **new** database path; `init` refuses to replace an existing file:

```bash
venv/bin/python -m experiments.scenario_db --db /private/tmp/scenario-rebuild.sqlite3 init
venv/bin/python -m experiments.scenario_db --db /private/tmp/scenario-rebuild.sqlite3 import \
  artifacts/datasets/urban_scenarios_v1/dataset.json \
  --release urban-scenarios-v1 --parent NONE --actor researcher --reason 'Import frozen v1'
venv/bin/python -m experiments.scenario_db --db /private/tmp/scenario-rebuild.sqlite3 import \
  artifacts/datasets/urban_scenarios_v2/dataset.json \
  --release urban-scenarios-v2 --parent urban-scenarios-v1 \
  --actor researcher --reason 'Import frozen v2 without altering v1'
venv/bin/python -m experiments.scenario_db --db /private/tmp/scenario-rebuild.sqlite3 import \
  artifacts/datasets/urban_scenarios_v3/dataset.json \
  --release urban-scenarios-v3 --parent urban-scenarios-v2 \
  --actor researcher --reason 'Add fresh confirmation families; preserve development inputs'
venv/bin/python -m experiments.scenario_db --db /private/tmp/scenario-rebuild.sqlite3 verify
```

For a real update, generate a new JSON candidate under a new path, run its native
SUMO/scenario verifier, then import it with a new release ID and the expected
current parent. Log a concrete reason (e.g. new simulation families for S5.C),
not merely “improve results”. Import supports existing v1/v2 dataset schemas;
new field formats need an explicit adapter/schema migration and tests first.

Run `verify_scenario_db` for the **frozen two-release migration prefix**, including
its device inputs and historical scientific-source checks; later releases are
allowed and receive structural checks. The original migration receipt remains
at commit 232adf3. The service-cover verifier additionally audits v3 JSON/DB
parity and every allowed input. Do not overwrite historical receipts.

```bash
venv/bin/python -m experiments.verify_scenario_db \
  --output /private/tmp/scenario-registry-current-verification.json
venv/bin/python -m experiments.verify_scenario_suite_v2 --raw
venv/bin/python -m experiments.verify_scenario_suite_v3 --raw
venv/bin/python -m pytest -q tests/test_scenario_store.py
```

## Limits and source context

Structural import checks do **not** establish native route feasibility, scenario
label semantics, absence of train/test leakage through model fitting, traffic
representativeness, or protection quality. Original generator verifiers remain
mandatory. Logs record committed updates only; a rejected/rolled-back operation
has no dataset change to log and returns an error to the caller.

Triggers, stored hashes and Git reviews protect ordinary workflow integrity,
not an administrator who can replace the database, drop guards and forge logs
and hash manifests together. `actor` is a supplied label, not authenticated identity.
Keep backups/Git checkpoints after writers close; do not copy a DB during an
active write or commit transaction sidecars. At larger scale use SQLite's backup
API or a server database, not an unsafe live file copy.

Primary documentation consulted: [SQLite transactions](https://www.sqlite.org/lang_transaction.html),
[foreign keys](https://www.sqlite.org/foreignkeys.html), and
[appropriate uses](https://www.sqlite.org/whentouse.html). Constraints implement
our experimental protocol; SQLite does not prescribe location-privacy metrics.
