# Full static catalogue local control — application gate

Development re-evaluation of the complete frozen matched REM/Planar pilot:
24 previously inspected native families, 192 sessions, 145,104 four-purpose
category queries. Six families retaining the old `test` label are inspected
development here. No new protection draw, defense/attacker selection, timing
simulation or original evidence modification.

Assume a provider permits full-catalogue bulk access, a fixed public region/map
and public version remain valid for epoch [0,12000), and all 418 POIs are static.
One coordinate-free request at public epoch start returns full
`id/category/lat/lon` records; local GPS/purpose/destination never enter it.
Local-only answers recover every nonempty nearest/fastest/radius/detour
reference exactly. This is expected full-candidate/reference equality, not a
new Geo-I improvement or confirmation of a real provider API.

| Six inspected development families | Nearest | Fastest | Radius | Detour | Total compact payload |
|---|---:|---:|---:|---:|---:|
| Full catalogue local | 100% | 100% | 100% | 100% | 217,968B / six bulk requests |
| REM current reply | 94.71% | 94.71% | 85.23% | 94.02% | 42,376,064B / 7,550 Q requests |
| Planar current reply | 96.97% | 96.96% | 90.85% | 96.13% | 42,364,548B / 7,550 Q requests |
| REM same epoch cache | 99.90% | 99.90% | 99.74% | 99.85% | same REM Q/reply payload |
| Planar same epoch cache | 99.34% | 99.34% | 98.06% | 99.17% | same Planar Q/reply payload |

Radius1000m has 6,174/9,060 empty reference queries; other purposes each have
648/9,060 empty queries, kept N/A. Recall is conditional on nonempty reference,
query-average within session then equal sessions/families; slight distance-only
differences from old pilot aggregation are not changed historical results.
Raw controls and every train/selection/test summary remain in `results.json`.

Bulk request196B + reply36,132B are exact compact UTF-8 JSON lengths from
`bulk_request.json`/`bulk_response.json`. Service costs use pinned pilot
`wire_rows.json.gz`, including repeated full records. No HTTP/TLS/map download,
CPU/GNSS, radio or latency measurement is claimed. The private detour target is
actual final session GPS only in evaluator/local utility; no Q/planner consumes it.

`protocol.json` binds the pilot/data/public-source closure before scoring.
`utility_rows.json.gz` retains each reference/answer/recall and local truth
separately from requests; `cost_rows.json` retains every family cost.
`validation.json` checks 145,104 queries and 1,015,728 exact ranked answers,
received-only/causal candidate pools, source hashes, payload and scope arithmetic.
The verifier uses existing `MultiPurposeRoadRanking.top` rather than the new
score/sort helper. No RNG master or model pickle is required.

```bash
python -m pytest tests/test_jisa_static_catalogue_control.py -q
python -m experiments.verify_jisa_static_catalogue_control_20261006
```

Fresh output only:

```bash
python -m experiments.jisa_static_catalogue_control_20261006 --out /private/tmp/jisa-static-catalogue-new
python -m experiments.verify_jisa_static_catalogue_control_20261006 --out /private/tmp/jisa-static-catalogue-new
```

See [application interpretation and limits](../../../docs/research/2026-10-06_jisa_static_catalogue_gate.md).
The result requires a local/bulk control and a realistic service contract in the
next study; warm static cache alone does not justify recurring location queries.
