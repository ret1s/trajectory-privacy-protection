# Public directed-cost diagnostic — 2026-10-05

Post-hoc TransProtect-Eq.13-family diagnostic of already selected frozen
GeoI-Endpoint20 (`scale025_L20`) Q transcripts, on the same reconstructed public
map and within-study holdout of 4 families, 8 sessions and 2 seeds. No re-selection or fit.
The old native-metric readout and original benchmark remain immutable.

Every accepted public OSM POI is a destination (418 IDs) with uniform prior.
Distances are exact directed shortest-path lengths on this reconstructed graph.
Unreachable costs remain infinity in the table; an undefined Q makes the
entire event score N/A, with coverage reported. There is no penalty,
straight-line fallback, private destination filtering or secret-nearest Q.

| Contract | Mean distortion |
|---|---:|
| Raw singleton, native Eq.13 | 0m |
| Five-Q-only extension: mean of Eq.13 across all Q | 1386.74m |

All 170 events / 850 Q reach all 418 public destinations. Individual Q median
1379.69 m, p95 2340.46 m. Aggregate events within session/seed, then seeds/sessions/families
equally. Higher distortion is a **utility cost**, not privacy evidence; this
does not measure final local POI ranking or user detour. No TransProtect
comparator was executed; do not turn this into a head-to-head paper claim.

- [`target_cost_tables.npz`](target_cost_tables.npz): exact 400 used states × 418 targets,
  prior and row/column keys; evaluator-only state membership.
- [`public_targets.json`](public_targets.json): public target IDs/coordinates.
- [`event_rows.json.gz`](event_rows.json.gz): every scored event, all Q values,
  coverage and evaluator-only truth-state labels; not an attacker interface.
- [`readout.json`](readout.json), [`protocol.json`](protocol.json),
  [`verification.json`](verification.json): results, source hashes and 3200
  independent NetworkX directed-distance checks.

The network's original turn equivalence remains unverified. This adds a real
non-MAE metric-family calculation, not an original-cache rerun or independent
confirmation. Original comparative Eq.13 remains N/A.

```sh
python -m experiments.native_cost_diagnostic --out /private/tmp/new-native-cost-diagnostic
python -m experiments.verify_native_cost_diagnostic
python -m pytest -q tests/test_native_cost_diagnostic.py
```
