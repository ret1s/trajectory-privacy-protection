# Static metadata within one public version/eight-session epoch

New development candidate after inspected TTL error analysis; previous TTL/native evidence is immutable. Cache only received `id/category/lat/lon` records for the public catalogue version within `[0,12000)`, with session starts 0/1500/…/10500s. Records cross earlier linked sessions causally; no future session replies enter a prior answer. Version/scope expiry invalidates the cache. Storage is capped at the 418-ID public catalogue. Dynamic availability still needs current public status, or remains unknown.

Selection accepts the static candidate for Epoch8-H12 L20: mean 99.59%, minimum family tail 99.43%, no extra GPS/request/reply bytes, cache ≤418 IDs. Test mean 99.34%, tail 99.48%, minimum family tail 98.33%; minimum tail session 86.67% remains below 90%. First-trip mean 94.73% versus later-seven approximately 99.99% demonstrates warm-up. This synthetic static repeated-route cohort does not establish performance in unseen areas or for live availability.

Test cost remains 7,550 requests and 41,233,562 full-metadata reply-only JSON bytes, matching the prior L20 artifact. Cache test size 167–416 IDs (mean 343.43), serialized unique static payload ≤35,868 bytes excluding Python overhead. Conditional reference coverage remains 1,402/1,510 overall and 462/528 tail. No privacy score, sampler, belief or Q change is claimed.

```bash
/private/tmp/trajectory-research-20261005-venv/bin/python -m experiments.verify_native_versioned_static_20261006
/private/tmp/trajectory-research-20261005-venv/bin/python -m pytest -q tests/test_static_poi_cache.py tests/test_versioned_static_poi_cache.py tests/test_versioned_static_geoi_lbs.py
```

Verifier passed 6,044 chronological event/method rows across all 8 public starts; 13 cache/real-client tests passed. Default BudgetedGeoILbsClient remains epoch60. The new opt-in VersionedStaticGeoILbsClient changes local static candidates only; live answers retain the original current-epoch guard. See [research note](../../../docs/research/2026-10-06_native_static_utility.md) for contracts and limitations. Default verification retains an existing validation record; an explicit new `--validation-output` path records another check.
