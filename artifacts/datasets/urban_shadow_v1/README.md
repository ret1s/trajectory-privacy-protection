# SUMO auxiliary routes: urban-shadow-v1

Accepted auxiliary release for the frozen-defender inference audit, not core
scenario v4. No GeoLife input. Full raw-FCD evaluator records are private ground
truth in the experimental threat model, even though the research data is
synthetic and version-controlled; do not serve this bundle to the LSP attacker.

## Exact roles and counts

| Role | Seeds | Families | Sessions | Windows | Observations | Unique XY |
|---|---|---:|---:|---:|---:|---:|
| Auxiliary training | 1001–1064 | 64 | 128 | 256 | 2,873 | 1,345 |
| Auxiliary holdout | 1065–1080 | 16 | 32 | 64 | 714 | 322 |

There are 175,211 untouched native 1 Hz FCD samples across all 160 sessions.
`cruise20`, `cruise60`, `stop5`, `return5` are four AUX schedules, each with
80 windows. They are not new S1–S10 cases or proof that all threat targets are
covered. Repeated points/windows within a family are correlated. Departure
and stop sampling is without replacement, not an IID calibrated population.

Only the 64 training families expand the attacker bank. The 16 holdout families
are not used for fit or model selection. Original 103/104 choose attacks; old
201–204 remain reused core development. Auxiliary `development_validation`
is a store-schema split name; the explicit release purpose and family role
identify it as **auxiliary holdout**, not original selection data.

The accepted generation and both quality amendments were declared before any
expanded attack score. See `thesis/notes/expanded_shadow_protocol.md`,
`thesis/notes/shadow_route_quality_amendment.md` and the independent verifier
handoff `docs/reviews/verification_expanded_shadow.md`.

## Inspect and reproduce

- `dataset.json` / `dataset.sha256`: source snapshot, plans, raw FCD, schedules,
  exact source hashes, SUMO versions/commands, native XML hashes and rejections.
- `summary.json`: fixed data counts, not experimental protection scores.
- `verification.json`: all-native-point, motion, profile and duplication checks.
- `registry.json`: import receipt, semantic identity, unchanged predecessor logs.
- `spatial_support.png` and provenance: old/new/holdout unique coordinates on the
  same native road frame. Raw GPS is reprojected for **all** panels; model-local
  XY must not be plotted directly against SUMO offsets.
- Canonical store: `artifacts/datasets/scenarios.sqlite3`, fourth release:
  `urban-shadow-v1`, semantic SHA256
  `7563a904421ed569425c893eda36221146729a32a3eff50366faec0fb93901b0`.
- Dataset byte SHA256:
  `0cf7e18da1107879ad7f1822f51c4e5e2d66cdd875cab70504096a854d02a11f`.

From the repository root, with the pinned v3 SUMO network and native files:

```bash
venv/bin/python -m experiments.scenario_db verify
venv/bin/python -m experiments.scenario_db coverage --release urban-shadow-v1
```

Generation, verification, import and benchmark outputs are deliberately
write-once. Do not rerun into the committed evidence destinations. For a new
native replay, choose a fresh destination under the repo's ignored build/cache:

```bash
venv/bin/python -m experiments.build_shadow_routes \
  --output build/shadow-replay-data --workdir cache/shadow-replay-native
venv/bin/python -m experiments.verify_shadow_routes \
  --data build/shadow-replay-data/dataset.json
```

Both directories must be absent. Absolute command/output paths differ across
replays, so replay JSON bytes need not match: compare routes, raw traces,
windows, source versions and relative scientific content. Do not import a
replay as a replacement for the sealed release. `publish_shadow_routes` is the
original one-time transaction (requires a v3-headed DB), not a migration to run
again on the current four-release store.

Native XML lives in ignored `cache/urban_shadow_v1/1001` through `1080`;
the native network is inherited from v3. A Git-only checkout has the lossless
JSON/SQLite observations and hashes but not every external OSM/native cache
file. Recover pinned inputs or replay with the recorded SUMO toolchain before
claiming native-XML verification on another machine. Do not substitute a fresh
OSM download silently.
