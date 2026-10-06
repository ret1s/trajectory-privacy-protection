# Fresh native cohort for Q-planner generalization

Created 06/10/2026. **No protection or attacker results are included here.**
This cohort supplies new source groups after the original native families were
used for development. Freeze method/configuration, selection rule and source
before reporting any new test performance.

| Publicly declared dimension | Cohort |
|---|---|
| Source | Native SUMO 1 Hz FCD on the same pinned reconstructed native network |
| Families / trips | 60 / 480; eight trips per family |
| Train / selection / test | 24 / 12 / 24 families; all trips of a family stay in its split |
| Public speeds | 6 and 8 m/s; train 12+12, selection 6+6, test 12+12 |
| Group separation | 60 distinct terminal junction groups, excluding the 23 groups of the original 24 families |
| Route duplicate guard | Reject a public route with edge-set Jaccard ≥0.8 to either route of any old/new retained family |
| Public observation window | Relative 0–600 s; native destination-lane stop until 650 s; departure slots 0,1500,…,10500 s |
| Public POI source | Same 419 archived source records; the existing public 250 m access rule retains 418 for native service |

Both possible routes are calibrated at **both** speeds before applying the
balanced public speed assignment and generating hidden choices. Eligibility
uses public road geometry, arrival before 550 s, an identical raw shared
prefix, and a clock when both choices traverse their native internal turn.
Of 178 screened public forks, 60 were retained and all 118 rejected attempts
are preserved in [public_eligibility.json](public_eligibility.json). No utility,
privacy or attacker score was used to accept/reject a route.

The six historical trips contain five routine and one rare destination. The
two query trips contain one of each in private randomized order. This preserves
the earlier controlled workload: its targets are deliberately balanced, not a
real population prior. The attacker may know the two candidate geometries;
this dataset does not turn that task into open-world destination forecasting.

`dataset.json.gz` retains the `families`, `traces` and `network` API consumed by
[native_resources](../../../experiments/future_sumo_eval.py). Family IDs are
`freshqp-001`…`freshqp-060`; each family adds `public_speed_m_s` and
`public_junction_group`. Use its `split` metadata rather than the old hardcoded
native-family split. Use the same public catalogue for every compared planner.
The 6 m/s stratum is a declared motion variation under the same map speed limit;
do not fit the protection configuration to its private traces.

All 312,480 saved native fixes were independently re-parsed from archived FCD
and compared exactly to the dataset. The verifier checked 240 counterfactual
calibration trips, 360 compressed native source files, split/speed counts,
junction exclusions, public route overlap, native turn clocks, parked points,
and immutable source/network/POI hashes. See [validation.json](validation.json).

```sh
/private/tmp/trajectory-research-20261005-venv/bin/python \
  -m experiments.verify_qplanner_fresh_native_20261006
```

The verifier rechecks existing evidence and retains its original receipt.
An explicit `--validation-output /private/tmp/new-receipt.json` writes a new
receipt. Generation is write-once: use new output **and** private work paths for
another cohort; never overwrite this dataset or reset a private working state.

| Evidence | SHA-256 |
|---|---|
| `dataset.json.gz` | `105083bc592ca63d5cf18951465a274cc74bab96edc07b060fab57e3a7f55ed8` |
| `protocol.json` | `bd9389137f131554584007b60bcc7472789a3477157ac0d0140197ed1a160166` |
| Final verifier | `0f111e080ddf6506f90bbbd5e064ef305c62421e6968f26e8ab08416a0b9a3d1` |

The generator and its dependencies/tests were pinned and copied under
`source_snapshot/` before hidden choice generation. Original route, FCD and
arrival XML are compressed under `native_sources/`, with original/compressed
hashes. The OS-generated HMAC choice key remains outside the repository in a
0700 private directory with a 0600 file; it is not an attacker input or export.
Dataset choices and route/FCD source files are synthetic evaluator truth, not
real personal GPS or deployed client sampler secrets.

Two implementation failures are preserved. The [first cohort attempt](../qplanner_fresh_native_20261006_v1/README.md)
used a public simulator seed outside SUMO's signed 32-bit range and accepted
zero families. This v2 uses a valid simulator seed with a unit guard, keeping
the scientific gates unchanged. The initial integrity verifier confused the
419-record source with the 418-record native service; its failure and code are
saved under `initial_verifier_failure.json` and `verifier_revisions/`. The final
verifier separately validates both counts. Neither repair inspected a defense
result, edited GPS or changed a target.

Distinct junction groups and bounded route-overlap exclusion do not make road
geometry mathematically independent: families may share road regions/edges
below the declared threshold. This is **same-map, same-generator synthetic
generalization/stability**, not real-user/cross-city or dynamic-service
confirmation. Public POIs are static, and warmed-cache utility remains a
separate generic postprocessing contribution requiring an offline-catalogue
baseline. Preserve the original earlier weak-tail results too.

No real raw dataset was downloaded. `data/raw/` is absent. GeoLife has a
[legacy loader](../../../data/geolife.py), but needs license/version checks and
a new chronological importer with genuine gaps/duplicate fixes/group splits.
Porto currently has no importer; add source acquisition/hash/attribution,
15-second native sampling and missing-data handling before using it. The
[JISA evaluation audit](../../../docs/research/2026-10-06_jisa_evaluation_audit.md)
records official sources and licensing gates. Do not call this synthetic
cohort a substitute for those real GPS checks.
