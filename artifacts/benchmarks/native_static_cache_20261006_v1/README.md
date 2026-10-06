# Frozen-Q TTL development

This new protocol evaluates current-only, the existing fixed public epoch60 cache, and rolling static windows of age <60/120/180s. All Geo-I Q, GPS traces, selected reply depths, priors, budgets and primary attackers are unchanged. Six selection families choose the smallest rolling TTL meeting mean Recall ≥90%, minimum family tail Recall ≥90%, and identical request/reply-byte counts. Selection chooses 60s; tests were already inspected previously, so this is development evidence.

For Epoch8-H12 L20, test current-only/epoch60/rolling60 mean Recall is 94.96%/95.24%/95.64%; tail 94.39%/94.74%/95.22%; minimum tail session 70.30%/72.12%/72.42%. The minimum-family-tail test gate is not met, even though the selection gate passed. Longer TTL results are descriptive and do not replace selected 60s. All candidates remain in `results.json`.

`weakest_tail_diagnostic.json` shows a native parked vehicle still reading GPS at 600s and spending 21/23 units. All causal earlier replies within that one trip would give 93.33% tail Recall; this is a diagnostic bound, not a selected policy. No cache/metadata inference creates unreceived POIs. Availability is not evaluated; stale/missing current statuses remain unknown.

```bash
/private/tmp/trajectory-research-20261005-venv/bin/python -m experiments.verify_native_static_cache_20261006
```

Passed 6,044 event/method rows and 30,220 policy rows. Rechecks keep existing validation; `validation_recheck.json` additionally checks diagnosis. Source hash pins protect all previous data, Q, budgets and attack results. The following versioned-static candidate is separate in `../native_versioned_static_20261006_v1/`. See [research note](../../../docs/research/2026-10-06_native_static_utility.md).
