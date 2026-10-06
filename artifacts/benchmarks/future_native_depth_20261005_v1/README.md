# Frozen-Q retrieval depth development extension

This extension was declared **after the primary test had been inspected**. It preserves every primary Q, the L10-calibrated belief/planner and all primary attacker choices/scores. It changes only how many static POI records the service returns: 10/20/40 per category. It is development evidence, not fresh test confirmation.

`protocol.json` seals selection of the smallest depth satisfying selection-family macro Recall ≥90%, median per-session Recall ≥90%, and mean reply bytes ≤2×L10. Only six selection families choose depth. Test/all/tail/family readouts are in `results.json`; all 27,198 event/depth metric rows are in `utility_rows.json.gz`.

Selection retains L10 for GeoI-SessionReset and selects L20 for **GeoI-Epoch8-H12**. At L20, epoch8 selection Recall is 94.16%, median session Recall is 97.31%, and the reply ratio is 1.5578×. Test conditional Recall is 94.96%, minimum family 91.75%; tail Recall is 94.39%, minimum family 84.09%, minimum session 70.30%. L40 exceeds the 2×byte gate. The primary L10 Recall of 89.95% remains in its original artifact.

Costs sum compact UTF-8 JSON responses for all Q, including duplicate POI records and full `id/category/lat/lon` metadata. They are **reply-only serialization estimates**, not measured HTTP/request/latency costs. Test L20 is approximately 27.3KB/event versus 17.5KB/event at L10. Recall remains conditional on a nonempty reachable reference; coverage of 1402/1510 overall and 462/528 in the tail does not improve by increasing L.

`public_reply40.npz` archives a public-map/POI-only table with no private GPS/Q. Its ordered L10 and reference5 prefixes match the original tables. `public_resource_archive.json` pins it. `validation.json` independently recomputes every utility/serialized-byte row from the native map, archived table, frozen Q and evaluator reference GPS, and reconstructs the selection-only gates.

```bash
/private/tmp/trajectory-research-20261005-venv/bin/python -m experiments.verify_native_future_depth
```

Passed 9,066 event/method rows and 27,198 depth rows. Default rechecks keep the existing validation record; use `--validation-output /private/tmp/NEW-depth-check.json` for a new record. The complete primary hash pins are in protocol/results. See [`2026-10-05_native_future.md`](../../../docs/research/2026-10-05_native_future.md).
