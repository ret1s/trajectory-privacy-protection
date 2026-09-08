# SUMO scenario release v3

Immutable evaluator dataset: **176 sessions, 124,852 FCD samples, 269 records**.
All mobility is SUMO+OSM; no GeoLife input. Synthetic identities, relationships,
intent and schedules remain evaluator labels, not observations sent to LSP.

- Preserve v2 development families 101/102 (training), 103/104 (validation).
- Add fresh confirmation families 201/202/203/204. Previously inspected 105/106
  do not occur in this release.
- Same v2 scenario definitions, mobility parameters and JSON record schema;
  v3 is a dataset revision, not a schema migration.
- Union coverage is 30/30 subcases; per-family coverage is 24–30. S5.C still has
  no training or validation record. Coverage is not protection evidence.
- Native verification: 72 daily runs, all 124,852 original FCD rows and 9,080
  external-edge transitions. Four retained families reuse their v2 native files.

Canonical DB release: `urban-scenarios-v3` in `../scenarios.sqlite3`.
Semantic content SHA256:
`213886fc2722bbecf5e55f61e8978d2c2842bb6018b7f65f5021a261bd79358d`.
Original JSON byte SHA256:
`6ba4a1caa978540ede91b59f23d097e4a9b5e6bad17243890f17c9c49cf9a94f`.

`verification.json` is the native-data receipt. `registry_verification.json`
audits the current three-release DB and the original migration prefix; the
service-cover verifier additionally compares every v3 device slot to JSON.
The older `../scenario_store_verification.json` is the two-release snapshot
from commit 232adf3 and is intentionally not overwritten.

Rebuild into a **new** candidate directory (SUMO, OSM and earlier raw cache
required; see repository data setup). Preserve the source commit and SUMO version:

```bash
venv/bin/python -m experiments.build_scenario_suite_v3 \
  --output /private/tmp/scenario-v3-rebuild --workdir /private/tmp/scenario-v3-rebuild-native
venv/bin/python -m experiments.verify_scenario_suite_v3 \
  --dataset /private/tmp/scenario-v3-rebuild/dataset.json --raw
```

The builder refuses an existing dataset file. The above command preserves the
canonical raw caches by using new paths. Compare trajectory/record content
rather than raw JSON hashes: the provenance paths themselves change. Exact
byte reproduction additionally requires the original paths and simulator.
Never fix an unsuccessful route by altering FCD coordinates.

The protocol is [`service_cover_protocol.md`](../../../thesis/notes/service_cover_protocol.md).
