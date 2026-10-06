# Matched REM/Planar Geo-I primitive pilot — 06/10/2026

**Development on previously inspected native families.** New secret-keyed draws
do not create a fresh dataset holdout. No old GPS, Q, source artifact or score
was changed. No defense winner was selected from this test.

Both methods use the same archived native SUMO network/catalogue, K=5, public
clock, private-reuse threshold 200m, slack .03, and Epoch8-H12 cap:
`C=.23/m, slots=8, H=12, U=23, u=.00125/m, Bsession=.03/m`.
The belief emissions match their respective REM/continuous planar kernels.
The planner retains the **same L10 response context**; service returns L20 and
the local reference is category-top5. L10→L20 is a deliberate controlled
mismatch inherited for this kernel ablation, not an optimized L20 planner.
It is a next-refinement item, requiring new Q/attack evaluation when changed.

The model generates only public Q; local utility uses evaluator current GPS.
Ledger supplier counts measure privacy-protection probes, **not total hardware
GNSS reads, local-position freshness or battery consumption**. No battery
saving is inferred. Metadata/roads are static and all POIs available here;
dynamic availability, congestion and novelty of destinations are not tested.

## Test readout

Use [derived_readout_v2.json](derived_readout_v2.json), which independently
recomputes utility and corrects the cold/warm subgroup session denominators.
The original [results.json](results.json) is retained byte-for-byte.

| Conditional Recall@5, family macro | REM Epoch8-H12 | Planar Epoch8-H12 |
|---|---:|---:|
| Current replies only, all 0–600s | 94.69% | 97.00% |
| Current replies only, tail 400–600s | 96.56% | 98.86% |
| Same versioned static cache, all 0–600s | 99.91% | 99.35% |
| Same cache, cold first session | 99.23% | 94.75% |
| Same cache, later seven sessions | 100.00% | 100.00%* |
| Same cache, lowest tail session | 100.00% | 81.21% |

*Planar later-session value before rounding: 99.9975%.* Warm cache persists
causally within the declared public [0,12000) subject/version epoch, across
eight starts at 0,1500,…,10500s. Both query destinations were visited in history.
This small repeated/static catalogue is conducive to warming; it is not proof
of 99% utility in new regions, cold starts or live service. The unusually good
REM tail in this draw does not replace earlier low-tail results.

The public planning context already contains the 418 static POI records and
signatures. A full-public-catalogue prefetch/offline local-ranking control is
therefore required for a publication STATIC setting: repeated location queries
may be unnecessary once public metadata is known. This structural observation
is **not a newly measured baseline score** and does not reintroduce the old
PublicCover models. A live-provider study should separate public proxy/context
POIs from actual provider metadata/current status and measure their mismatch.

Coverage for each method/policy is **1,402/1,510** test events; tail **462/528**.
Reference-empty events remain N/A, not 0/1, and are reported in the denominator.
Undefined sessions in cold/warm scopes count only represented slots, not all
eight slots of a family.

At the public visible-turn clock:

| Method | Selected attacker | S5 exact future-edge accuracy | S6 Hit100 / destination MAE |
|---|---|---:|---:|
| Raw positive control | candidate_trees | 100% | 100% / 4.99m |
| REM Epoch8-H12 | motion | 50% | 50% / 614.93m |
| Planar Epoch8-H12 | curve_mean_0.02 | 50% | 50% / 501.65m |

S5 and S6 share the same binary branch decision under strong knowledge of
**two public candidate turns/destinations**; they are not independent open-world
tasks. S6 sees six protected historical trips plus one query prefix; it does
not use query7 when attacking query8 or enforce a joint one-routine/one-rare
constraint. Models are trained on 12 families, chosen on six selection
families, then scored on six test families/12 queries. Selection uses balanced
accuracy, then log-loss, then name; [all selections](attack_selection.json)
were saved before any test score. Checkpoints, all bank predictions and
permuted-training controls are retained. Raw shared-fork accuracy is 50%
because its two source prefixes are identical; that is intrinsic ambiguity.

The two visible-turn selected accuracies tie. A higher MAE for REM alone does
not establish a privacy advantage. Every finite-bank candidate and both
before/after-turn stages remain in the artifact.

## Paired uncertainty and cost

[paired_differences.json](paired_differences.json) is a **posthoc descriptive**
family-paired bootstrap (10,000 resamples, public statistical seed; six native
families, one secret draw per session). It was added after inspecting point
estimates, has no multiple-comparison adjustment, and is not used for selection
or confirmatory significance claims. REM minus Planar:

- Current-only overall Recall: **−2.31 percentage points**, 95% interval
  **[−3.66, −0.84]**.
- Same-cache overall Recall: **+0.55 points**, interval **[−0.08, +1.72]**.
- Same-cache cold Recall: **+4.48 points**, interval **[−0.63, +13.92]**.
- Visible-turn destination MAE: **+113m**, interval **[−253, +497]m**.
- Visible-turn attack accuracy: **0 points**, interval **[−33.33, +33.33]**.

Thus planar gives better current-reply utility in this pilot, static-cache
differences are uncertain, and no selected-bank privacy advantage is shown.
The intervals condition on this generator/map and do not include real-city
generalization or Monte Carlo variation across many private keys.

Both protected methods issue **7,550 Q requests** in test. Compact UTF8 JSON
cost estimates, including full `id/category/lat/lon` records and duplicate
replies: REM **1,140,292 request bytes + 41,235,772 reply bytes**; planar
**1,140,457 + 41,224,091 bytes**. Changing the local cache policy changes none
of these bytes/requests/Q. These are deterministic serialization estimates,
not measured HTTP/TLS/latency. Same cap/K/L is not exact actual-spend/byte
matching. Generation of both primitives/all 24 families took 478.16s plus
1.96s public-resource setup; no per-method speed claim is made.

Across six test families, actual epoch spend is REM **.18875–.20125/m** and
planar **.18250–.19750/m**, both below the same prospective .23 cap. Each uses
528 protection-supplier calls across those families (88 per eight-trip epoch).
These are mechanism probes; current-GPS local ranking/acquisition costs remain
outside that count. No unused cap is refunded or reused from private branches.

## Evidence, corrections and replay

[protocol.json](protocol.json) and [protocol.sha256](protocol.sha256) precede
generation. Static local import closure, dataset/network/cache and old-output
hashes are pinned; code snapshots are under `source_snapshot/`. Family partial
records, public Q, evaluator-only ledger/targets, wire ID rows, utility rows
and fitted models are retained. Secret keys/SQLite stay outside the repository
in `/private/tmp/jisa-native-anchor-20261006-v1/private_state/`, 0700 directory
and 0600 files; no keys/seeds are exported to attacker features or artifact.
Paired alternatives share a private family key for experimental coupling;
only one alternative transcript is exposed to each attacker. No claim is made
for releasing both correlated alternatives simultaneously.

Two corrections are explicitly preserved:

1. The borrowed v1 aggregate assumed eight session slots for the cold/warm
   subgroup's `undefined_session_count`. Means/minima/coverage/cost were
   independently checked; v2 uses actual represented sessions and records all
   36 count changes. No observation, Q, attack model or selection changed.
2. Additional source-target/raw-coordinate assertions were added while the
   first verifier process was already running. Its receipt is retained in
   [validation.json](validation.json); its end-of-run self-file hash should
   not be used as proof of exactly which assertions that first process loaded.
   The final unchanged verifier was then rerun; use the authoritative
   [validation_recheck.json](validation_recheck.json), which checks the final
   code including source choice/role/endGPS/actual future edge/raw controls.

Final-code recheck verifies **18,138 public events, 36,276 utility rows,
12,092 ledger steps, 2,016 attacker probability rows and 24 checkpoint refits**.
It rebuilds public received-ID unions and absolute times independently of the
runner/cache implementation, recalculates all request/reply bytes/coverage/
scope means/minima, checks source truth and public-only causal prefixes,
reproduces train-only checkpoint predictions and selection, and confirms old
source outputs unchanged. Read-only companion review found no material
matching or future-label leakage issue. Seven new tests plus six existing
planar tests pass.

```bash
PY=/private/tmp/trajectory-research-20261005-venv/bin/python
$PY -m pytest -q tests/test_jisa_native_anchor_ablation.py tests/test_planar_anchor.py -o addopts=''
$PY -m experiments.verify_jisa_native_anchor_ablation_20261006
# Additional receipt, never overwrite an existing one:
$PY -m experiments.verify_jisa_native_anchor_ablation_20261006 \
  --validation-output /private/tmp/new-jisa-anchor-check.json
```

The completed runner and paired diagnostic refuse overwrites. For a new
generation use both a new artifact path **and** a new private workdir; changing
the old protocol/source requires a new version. Frozen public replay and the
independent verifier work without disclosing private RNG keys. Do not rerun
`--stage all` on this completed artifact. Map data derives from
© OpenStreetMap contributors, [ODbL](https://www.openstreetmap.org/copyright);
lane/turn IDs are authoritative only for the archived reconstructed native net.
Kernel guarantees remain ideal-arithmetic claims; float/PRNG implementation,
static service, finite attacker bank and publication confirmation limits are
unchanged.
