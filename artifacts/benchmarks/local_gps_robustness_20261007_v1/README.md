# Local-GPS ranking sensitivity, 07 October 2026

**Independent verification: PASS.** This is a secondary diagnostic on the
already-inspected 24 TEST families, three nested draws and eight sessions per
family. It is not a new holdout confirmation or a calibrated real-GPS study.

Geo-I inputs, Q, K5, public network clocks, anchors, ledger and both frozen
L20/L30 current replies stay unchanged. Local ranking is the only intervention.
The exact-event GPS/destination oracle reproduces all six saved current-only
metric/denominator fields before any sensor-arm score is calculated.

| Local position | Mean pre-snap error (m) | Mean post-snap error (m) | Recall4 L20 (%) | Recall4 L30 (%) |
|---|---:|---:|---:|---:|
| Exact event oracle | 0.00 | 6.45 | 89.71 | 92.69 |
| Hold60s, sigma0 | 66.32 | 69.88 | 86.74 | 89.35 |
| Hold60s, sigma5 | 70.30 | 70.68 | 84.53 | 86.93 |
| Hold60s, sigma15 | 78.50 | 76.53 | 82.77 | 85.02 |
| Two-fix velocity60s, sigma0 | 56.78 | 59.44 | 85.78 | 88.28 |
| Two-fix velocity60s, sigma5 | 61.59 | 60.90 | 83.80 | 86.13 |
| Two-fix velocity60s, sigma15 | 72.48 | 68.75 | 82.54 | 84.77 |

Position errors pool events. Recall averages defined true-reference categories
within events, then events within family/draw, nested draws and families with
equal purpose weights. No CI or population generalization is claimed here.

Velocity extrapolation lowers average positional error but **lowers Recall4
and the destination-free Recall3 relative to hold at every matching sigma and
both depths**. It is retained as a negative result, not adopted as an improved
ranking policy. L30's additional responses improve all fixed local-sensor arms
over L20, at the existing paid response cost.

The position estimates use either the most recent noisy fix or the last two
noisy projected fixes. Velocity is Euclidean-clipped at 8 m/s; a single fix falls
back to hold. Only fixes at 0, 60, ..., 600 s are used, never true heading/route/lane
or future fixes. Nearest public graph snap has no secret lane correction or
distance-based censoring. Sigma is **per-axis** controlled Gaussian standard
deviation 0/5/15 m. Standardized offsets are fixed by the declared public synthetic
SHA256/Box-Muller tape and common across methods, depths and sigma scales.

The **local 60 s sensor clock is separate from the frozen Geo-I supplier clock**.
The 6,336 virtual local fixes (11/session) are not total device GNSS reads,
physical samples or energy measurements. The frozen privacy trace was generated
from the earlier exact synthetic GPS; this study does not perturb that input or
change the ideal/privacy-simulator scope. Logical estimator state uses one/two
fixes, while the offline evaluator loads full tapes/public resources; no actual
client RAM bound or memory saving is measured.

References always use actual event GPS. An estimated missing/unreachable/radius
answer scores 0 when the true reference exists; a truly empty reference stays
N/A. Within-radius has 42,258 defined category-events and 66,426 empty reference
category-events out of 108,684; 537 of 18,114 events have no radius reference.
At L30/sigma15, hold returns 16,430 outside-true-domain items among 106,488
returned items; velocity returns 17,359 among 107,622. These are **pooled repeated
POI counts**, not unique people, unique POIs or per-user failure rates. Invalid
true-domain items do not count as valid completion. Estimated empty-category
and zero-answer counts remain explicit.

Detour assumes the actual destination is known local input (supplied by the
evaluator in this diagnostic). It never enters the GPS estimator or Q planner.
The separate three-purpose macro excludes detour: at L30/sigma0 it is 87.97%
for hold and 86.73% for velocity, compared with 92.08% for the exact-event oracle.
Dynamic provider availability is not combined with this sensor experiment.

Unchanged network totals for every variant:

| Depth | Requests | Request JSON bytes | Reply JSON bytes |
|---|---:|---:|---:|
| L20 | 90,570 | 13,678,985 | 494,713,236 |
| L30 | 90,570 | 13,678,985 | 648,578,577 |

These are compact application JSON estimates, without HTTP/TLS, radio, latency
or battery. No secret-purpose fallback request is added.

## Evidence and replay

- [Protocol](protocol.json) and [source checksum](protocol.sha256): fixed arms,
  two-clock/dataflow contract, inspected-cohort scope and source/input pins.
- [Local sensor/noise tape](local_sensor_tape.json.gz): frozen before scoring;
  synthetic evaluator-local GPS fixes and public noise, no protection key.
- [Pre-score independent verification protocol](verification_protocol.json),
  [replay start](replay_started.json), and
  [full exact baseline recovery](baseline_recovery.json).
- [Readout](readout.json) and per-family/draw [blocks](blocks/).
- [Independent certificate](validation.json): 72 blocks, 18,114 public events,
  126,798 estimates, 253,596 utility arm rows and 6,336 virtual local fixes.
- [Runner](../../../experiments/local_gps_robustness_20261007.py),
  [independent checker](../../../experiments/verify_local_gps_robustness_20261007.py).

Protocol SHA256:
`54036441d375aef9702d03fd60e7743685c611236ac13cd1dbd24021abc566ad`.
Readout SHA256:
`89b72dc2483963d18e8e62af79744196a8aa58a0e3f6546cce5db5fdecb60ac7`.
The certificate binds these exact bytes and independently reconstructs the
Gaussian tape, causal estimates, direct road rankings, fixed responses/wire,
true-reference denominators, family arithmetic and positional quantiles.
Shared public projection/graph primitives are explicit; no privacy sampler or
private keys are required for replay/audit.

Existing artifacts are write-once. For a complete reproduction use a **new**
repository output path, retaining the frozen code and upstream datasets:

```bash
python -m experiments.local_gps_robustness_20261007 --stage declare \
  --output artifacts/benchmarks/local_gps_robustness_replay_NEW
python -m experiments.verify_local_gps_robustness_20261007 --declare \
  --output artifacts/benchmarks/local_gps_robustness_replay_NEW
python -m experiments.local_gps_robustness_20261007 --stage replay \
  --output artifacts/benchmarks/local_gps_robustness_replay_NEW
python -m experiments.verify_local_gps_robustness_20261007 \
  --output artifacts/benchmarks/local_gps_robustness_replay_NEW
```

The included public reply cache and upstream source archives reconstruct this
study. Local private key/ledger work files from the original Geo-I generation
are neither needed nor included.
