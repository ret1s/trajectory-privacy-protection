# Urban scenario suite v2 — SUMO + OSM only

Latest pre-protection challenge dataset, **not evidence of attack resistance**.
Six related route families, 132 completed sessions, 94,334 native 1 Hz samples,
207 scenario records. All 30 subcases appear in the union; not every family or
split contains every subcase. S5.C has only one record (confirmation family106),
so it cannot support train/validation/confirmation attacker comparison yet.

- `dataset.json` and `dataset.sha256`: frozen evaluator bundle and checksum.
- `summary.json`: generation counts, including all case denominators.
- `verification.json`: independent full native XML/gate audit.
- Builder: `experiments/build_scenario_suite_v2.py`.
- Generator: `data/scenario_suite_v2/`; v1 sources remain unchanged.
- Protocol: `thesis/notes/belief_suite_v2_protocol.md`.
- Latest method evidence: `artifacts/benchmarks/belief_suite/` (only S1–S3).

Seeds101/102 train,103/104 validation,105/106 within-city confirmation. Keep all
person/device/car/day/relationship views of a family in the same split. The
occupancy prior includes synthetic repeated-day trips; it is not a measured
human population prior. Never count records or timestamps as independent users.

S1.C: actual FCD within150m of a POI whose category accounts for <=5% of the
418 retained public POIs. S6.C: six prior completed trips (five routine,one rare)
and a bounded current prefix. **Slot6, not slot0, is the current target**.
History and future are evaluator-private until processed by an appropriate
protection protocol. No cross-day budget policy is supplied by this dataset.

Use `device_view(record,traces,slot=...)` for bounded device input; that is still
private GPS, not an LSP export. Full bundles contain synthetic identity labels,
planned routes, future endpoints and raw history. Do not serve them directly.

Reproduce after obtaining the pinned network/OSM dependencies:

```bash
venv/bin/python -m experiments.build_scenario_suite_v2 \
  --output tmp/scenario_suite_v2_rebuild --workdir cache/scenario_suite_v2_rebuild
venv/bin/python -m experiments.verify_scenario_suite_v2 \
  --dataset tmp/scenario_suite_v2_rebuild/dataset.json --raw
venv/bin/python -m experiments.verify_scenario_suite_v2 --raw
```

Raw cache files are not tracked. Original local provenance paths point to
`cache/scenario_suite_v2_final/seed_*/day_*/`; rebuilding changes absolute paths
and XML generation headers, so byte identity of the whole bundle is not a
portable reproducibility criterion. Check native observations, labels, gates
and recorded scientific source hashes. The complete network route can include
short terminal edges absent from the last 1 Hz FCD sample; no artificial FCD
endpoint is inserted. Two invalid immediate-next-edge pairs were rejected,
not silently relabeled. Details are in the verifier report.
