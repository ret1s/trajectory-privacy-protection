# Geo-I / REM, L=30: frozen same-map synthetic confirmation

**Current-only TEST primary passes:** macro Recall@5 increases
**89.714%→92.688%**, gain **2.974 percentage points**, saved95% paired-family CI
**[2.308,3.646] pp**. This is a **paid-bandwidth static-service configuration**,
not a new Q algorithm, privacy theorem or same-cost comparison with other methods.

L means **maximum server POIs per category per Q**, unrelated to historical
model/configuration IDs30/67. **K=5 Q** and **local top-k=5** stay fixed. Only
L20 and the development-selected L30 were scored on fresh data. Coordinates,
clocks, protected anchors, reads, ledger and attacker inputs are identical
because both reuse one completed frozen `legacy_l10` Q stream.

The choice, source/configuration and primary rule were frozen before fresh
Q generation and utility scoring. TEST has **24 independent families ×3
nested private draws ×8 sessions**. The predeclared gates all pass: mean
gain≥2 pp, paired95% lower bound>0, every draw gain>0. Draw means are
**2.867/2.929/3.126 pp**; draws/ticks are not independent subjects. Bootstrap
uses10,000 whole-family resamples and public statistical seed2026100617.

| TEST purpose, current responses | L20 Recall | L30 Recall | Gain, pp |
|---|---:|---:|---:|
|Nearest|92.449%|94.680%|2.230|
|Fastest|92.449%|94.678%|2.229|
|Within radius|81.694%|86.879%|5.185|
|Minimum detour|92.266%|94.516%|2.250|

Within-radius reference coverage is **42,258/108,684 category cases** and
**17,577/18,114 windows**; absent references stay N/A. Other purposes have
complete reference coverage. Purpose/tail/cold/cache intervals are exploratory.

| TEST sensitivity | L20 macro | L30 macro | Gain, pp |
|---|---:|---:|---:|
|Family lower25% mean|83.565%|88.046%|4.481|
|Cold: entire first session|88.333%|91.454%|3.122|
|Temporal tail400–600s|92.388%|94.880%|2.492|
|Cumulative static epoch cache|98.808%|99.136%|.328|

**Current replies are primary; cache is secondary.** Cache assumes the same
static catalogue across the epoch, not validated dynamic freshness.

**Actual TEST cost:** reply compact-JSON bytes increase
**494,713,236→648,578,577**, ratio **1.311019× /+31.102%**. Both variants have
**90,570 requests** and **13,678,985 request bytes**. These are application
payload estimates, not HTTP/TLS, latency, physical GPS energy or server prices.
Fixed L is public and observable.

The60s private-read schedule belongs to the protected Geo-I supplier/ledger.
Utility scoring uses synthetic ground-truth GPS **at each public event** and
the real destination as a local exact-ranking oracle. Replay adds no supplier
call, but does not validate local ranking from only60s sensor fixes.

Inspect [recommended configuration](recommended_configuration.json),
`protocol.json`, `depth_freeze.json`, `readout.json`, `paired_protocol.json`,
`paired_readout.json`, and `validation.json`. The independent certificate
checks all108 family/draw bundles,108,688 utility windows and54,344 wire rows,
exact original L20 scores/replies/cost, directed prefix/ranking, causal caches,
N/A arithmetic and nested-family bootstrap. No private RNG key was needed.
Protocol/readout/paired-readout hashes were checked before notes opened scores.

Reviewed scientific [PDF](figures_v2/response_depth_utility_cost.pdf) and
[PNG](figures_v2/response_depth_utility_cost.png) retain saved CIs/draw means,
all development depth candidates, actual TEST response cost and N/A coverage.
[Provenance](figures_v2/figure_provenance.json) and exact source snapshots bind
the inputs and tight export. The original export/source remains in `figures/`.

```bash
python -m experiments.export_qplanner_response_depth_20261006 \
  --development artifacts/benchmarks/qplanner_response_depth_development_20261006_v1 \
  --fresh artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1 \
  --output-dir /private/tmp/response_depth_fresh_figures_NEW
```

The conclusion is limited to new synthetic family groups on the **same public
map/generator/static bulk service**. Larger response pools improve reference
overlap at extra bandwidth. Real mobility/cross-city robustness, dynamic service
availability, online application workload, equivalent-cost SOTA comparisons and
publication-level novelty remain unresolved. No JISA-ready claim is made.
