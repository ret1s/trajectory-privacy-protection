# Fixed Geo-I Q: response-depth development

This is a **static-service utility/cost ablation**, not a new Q algorithm or a
privacy comparison. The exact `legacy_l10` Q streams from development v3 are
replayed at public service depths **20/30/40/60 POIs per category per Q**.
All variants retain the same K5 coordinates, clocks, protected anchors, GPS
read schedule and ledger. No private key, RNG rerun or new private read was
needed. Local ranking covers nearest, fastest, within-radius and detour.
The60s schedule is for the protected Geo-I supplier/ledger. Utility scoring
uses synthetic ground-truth GPS at each public event and the true destination
as a local exact-ranking oracle; it does not validate ranking from only60s
sensor fixes or measure physical GPS energy.

The rule was saved before depth scores: choose the smallest depth30/40/60 with
current equal-family/equal-purpose Recall gain at least2 percentage points,
nearest no loss, and total reply JSON bytes at most2.5 times L20.

| Public service depth | Selection macro Recall@5 | Gain vs20 | Reply JSON ratio |
|---|---:|---:|---:|
|20|92.188%|—|1.000|
|**30: selected**|**94.762%**|**+2.573 pp**|**1.311: +31.094%**|
|40|96.459%|+4.270 pp|1.622|
|60|97.799%|+5.610 pp|2.243|

Selection has6 independent families ×3 nested draws ×8 sessions. The saved
95% whole-family bootstrap interval for L30's mean gain is **[1.793,3.509] pp**,
with10,000 resamples and public analysis seed2026100617. These development
intervals are exploratory; they do not confirm fresh generalization.

Reply payload rises from123,907,067 to162,434,661 bytes at L30; both have22,680
requests and3,425,526 request bytes. These are compact application-JSON
estimates, not measured HTTP/TLS, latency, battery or server billing. Fixed
two-digit L values have equal request byte length; L remains observable.

Inspect these write-once records:

- `protocol.json` / `protocol.sha256`: pre-score criterion and source/input pins.
- `resources.json`: exact L60-prefix/L20 public catalogue/context pins.
- `families/*.json.gz`: every depth's per-window utility/reply IDs/cost, original
  source bundle hash and fixed Q/ledger hashes; partial failures are never replaced.
- `readout.json`: conditional family utility, coverage and JSON cost totals.
- `depth_selection.json`: all candidates retained; smallest eligible depth30.
- `validation.json`: independent directed service/ranking/JSON/prefix/cache
  replay passed, including exact original L20 utility and bytes.
- `paired_protocol.json` / `paired_readout.json`: independent family/draw
  arithmetic and saved uncertainty; these files do not choose a defense.

Empty category references stay N/A. In within-radius utility,10,008 of27,216
category references and4,506 of4,536 windows are defined on Selection, unchanged
across depths. Static cumulative epoch cache is a separate sensitivity;
**current responses** are the primary workload.

The larger POI pool can improve static reference overlap because it contains
the smaller pool. This does not improve coordinate privacy or demonstrate
superiority to other methods allowed the same depth/bandwidth. Real mobility,
dynamic availability, an online service workload and a fair equal-cost method
comparison remain outside this development artifact. The subsequently completed
[fresh L30−L20 confirmation](../qplanner_response_depth_generalization_20261006_v1/README.md)
used a separate before-generation freeze and cohort; it is reported separately.

Figure reproduction, into a new directory:

```bash
python -m experiments.plot_qplanner_response_depth_20261006 \
  --development artifacts/benchmarks/qplanner_response_depth_development_20261006_v1 \
  --output-dir /private/tmp/response_depth_figures_NEW
```

The scientific figure consumes completed readouts and emits PDF/PNG plus input,
source and output hashes. It does not select a configuration, compute a new
bootstrap, regenerate Q or access private keys.

The reviewed development-only figure is available as
[PDF](figures_v2/response_depth_utility_cost.pdf) and
[PNG](figures_v2/response_depth_utility_cost.png), with
[provenance](figures_v2/figure_provenance.json) and the exact figure source snapshot.
The earlier figure/source bytes are retained in `figures/`. This historical
development-only figure keeps its original pending-fresh label; no illustrative
TEST values are plotted. The completed
[fresh figure](../qplanner_response_depth_generalization_20261006_v1/figures_v2/response_depth_utility_cost.pdf)
uses actual TEST readouts.
