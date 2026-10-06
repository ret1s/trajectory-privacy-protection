# Native comparator metric readout — 2026-10-05

This is an additional readout of immutable archived local-adaptation results.
No defense/attack was fitted, no dataset was changed, and no original result was
overwritten. It is not a new superiority or independent-confirmation study.

- [`readout.json`](readout.json): separate protocols, scenarios, metric coverage,
  source/code hashes and limitations.
- [`summary.csv`](summary.csv): values and explicit unavailable denominators.
- [`rows.json.gz`](rows.json.gz): evaluator-only per-record accounting; contains
  attacker identities/errors and source labels, not a public attack interface.
- [`verification.json`](verification.json): 431 independent coordinate-to-EIE
  checks for paper-v2 K=5; source files unchanged.

There are 5,336 source rows and 135 cohort/method/scenario summaries. Empirical
point-estimate EIE is the same formula as existing MAE. Native travel-cost,
posterior/semantic and path-similarity inputs are missing from comparative
archives or incompatible with dummy-only outputs and remain N/A. The old
TransProtect smoke value 62.584m is retained as archived-reported only, not a
recomputed comparative value. Offline AnotherMe coverage remains explicit.

The GeoI-Slack expanded-development cohort and paper-v2/common-live comparator
cohorts are not combined. Retired public-cover plans are excluded.

See [definitions and evidence limits](../../../docs/research/2026-10-05_native_metrics.md).
Recompute into a fresh directory, for example:

```sh
python -m experiments.native_metric_readout --output /private/tmp/native-metric-review-new
```
