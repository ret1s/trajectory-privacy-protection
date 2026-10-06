# Robust endpoint selector — 06/10 inspected development

This artifact refines attacker selection only. It keeps the frozen GPS/Q/
clock/POI/utility/byte streams of
[endpoint_generalization_20261005](../endpoint_generalization_20261005/README.md)
and the privately shuffled publication view of
[endpoint_order_20261005](../endpoint_order_20261005/README.md). No Geo-I
sampler, defense configuration, dataset, seed or old evidence is changed.
The 28 families 1205–1232 were already inspected: **this is development,
not a new heldout confirmation**.

One rule was sealed before this runner reopened test labels or historical test
scores. Eight leave-one-family-out folds use 701–708: each learner **and
mobility history** excludes the held-out family. Each fold has 28 fit rows
from seven families. The original 709–710 selection uses the final bank fit
on all eight fit families (32 rows). All ten validation families are equally
weighted. Learner features are cached with empty, no-GPS history; verification
confirms their independence from the fold history. Final models never fit on
709–710 labels.

Choose minimum **family mean MAE + 1 sample SE**, or maximum **family mean
Hit − 1 sample SE**, with SD(ddof=1)/sqrt(10). Compare objectives rounded to
12 decimal places, then lexical candidate names; this numerical tie rule is
predeclared. The bank union remains invariant/slot/Hungarian/sequence geometry
and ExtraTrees64/kNN1/kNN5. Original two-family invariant and expanded
selectors are separately retained controls. No test-based fallback is used.

Cross-fit SE is a selection penalty, **not a confidence interval**: training
sets overlap, and validation fit sizes are seven versus eight families.

| Method, robust selector | S9 MAE m / Hit100 | S10 MAE m / Hit100 | S9 / S10 Hit500 |
|---|---:|---:|---:|
| Raw GPS | 0 / 100% | 0 / 100% | 100% / 100% |
| GeoI-Slack L10 | 619.67 / 3.57% | 672.61 / 0% | 43.75% / 14.29% |
| GeoI-Slack L20 | 789.95 / 1.79% | 757.74 / 0% | 24.11% / 12.50% |
| GeoI-Endpoint20 | 1406.73 / 0% | 1337.16 / 0% | 7.14% / 4.46% |

PlainL20 S10 now selects `centroid_ols3_120s`: validation mean 750.61 m,
SE 79.18 m, inspected-test 757.74 m. The old expanded selector chooses
`geometry_velocity_tree` with test 1690.98 m; old invariant test 901.48 m.
Robust-minus-old MAE is -933.24 m [-1190.33,-692.32] and -143.74 m
[-224.01,-58.51], paired family bootstrap 95%. These indicate more accurate
MAE attacks in this diagnostic, **not a defense change**.

Same-L20 robust Endpoint20-minus-plain S9 MAE is +616.78 m [474.36,763.60];
S10 is +579.41 m [468.24,695.85]. S10 Hit100 ties at 0%; Hit500 difference
-8.04 pp [-16.96,0] remains uncertain. Recall stays 96.64% versus 98.94%,
with 1308/1308 ticks and the same measured service traffic.

The fix is partial. Endpoint20 S10 still has a validation/test mean gap of
171.25 m; its two best risk objectives differ by only 13.38 m. PlainL20's
gap is 7.13 m and objective margin 30.91 m. Hit generalization is mixed:
PlainL10 S10 robust Hit500 14.29% is weaker than old invariant 30.36%
(delta -16.07 pp [-25.89,-6.25]); its Hit100 also drops from 3.57% to 0%.
Do not claim uniformly stronger/stable attackers or universal superiority.

`protocol.json`/`.sha256` pin one rule, all source inputs and code.
`fold_manifest.json`, `fold_predictions.json.gz` retain all folds and every
candidate error. `candidate_statistics.json.gz` contains all ten-family
means/SE/objectives; `selection.json` freezes every decoder and model hash.
`models.pkl.gz` preserves 64 fold banks + 8 final banks (360 learners), with
their correct background histories. `diagnostic_rows.json.gz` preserves
all candidate errors on all 28 groups; `diagnostic.json` retains robust and
both old controls. `readout.json` retains old-control equality checks, all
per-family validation/test gaps and paired family intervals.

The review verifier recomputes 324,672 predictions, 32,040 arithmetic values,
360 learner training arrays, 5232 event multisets/clocks and 120 metrics.
Twelve new tests pass. The initial sealed checker incorrectly compared
mutable history Counter dictionaries literally: queried unseen transitions
add count 0 cache keys. `initial_verifier_failure.json` retains this failure.
The new review checker ignores only zero-count keys and confirms all positive
training counts/prior arrays, without changing the protocol, models or scores.

`runtime.json`, `sources.tar.gz`/`source_archive.json`,
`models.pkl.gz`/`model_archive.json` preserve runtime and code/model evidence.
The source protocol pins the initial checker; the successful review checker
has its own SHA in `verification.json` and the source archive. Models are
trusted locally generated Python pickle; verify the SHA before loading.
Frozen upstream streams and evaluator-only shuffle key remain in the referenced
artifacts, and their exact hashes are pinned here.

```bash
ENDPOINT_PYTHON=/private/tmp/trajectory-research-20261005-venv/bin/python
$ENDPOINT_PYTHON -m pytest -q tests/test_robust_endpoint_selection.py tests/test_robust_endpoint_verifier.py
$ENDPOINT_PYTHON -m experiments.verify_endpoint_robust_selection_20261006_review artifacts/benchmarks/endpoint_robust_selection_20261006
$ENDPOINT_PYTHON -m experiments.endpoint_robust_readout_20261006 artifacts/benchmarks/endpoint_robust_selection_20261006
```

To rerun into a new directory, retaining all outcomes:

```bash
$ENDPOINT_PYTHON -m experiments.endpoint_robust_selection_20261006 all --out /private/tmp/endpoint-robust-new-run
$ENDPOINT_PYTHON -m experiments.verify_endpoint_robust_selection_20261006_review /private/tmp/endpoint-robust-new-run
$ENDPOINT_PYTHON -m experiments.endpoint_robust_readout_20261006 /private/tmp/endpoint-robust-new-run
```

The same synthetic SUMO families/public reconstructed road map are reused.
Original lane/turn resources are unavailable; 40 m public state spacing and
8 m/s assumptions remain. Bootstrap intervals are exploratory, without
multiplicity correction. New, unseen families are needed to confirm selection
stability. See the [research note](../../../docs/research/2026-10-06_robust_endpoint_selection.md).
