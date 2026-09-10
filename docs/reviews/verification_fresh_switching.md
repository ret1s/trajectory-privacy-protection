# Verified research handoff — 2026-09-10

## Verdict

**Reportable research cycle; not a validated default or a conference-readiness claim.**
Fresh data, a new causal algorithm variant, frozen selection, independent confirmation,
negative findings and a literature-grounded critique are complete. Do not tune on
families 307–312 after this report and still call them untouched confirmation.

The new two-mode filter has lower aggregate Hit@100m but lower MAE too, with essentially
unchanged Recall. No method passes the prespecified validation utility gate. It must
remain an experimental variant, not replace the default or be advertised as better
privacy under every metric.

## Dataset and consistency checks

- SUMO+OSM only; fixed seeds 301–306 selection, 307–312 confirmation.
- 264/264 trips arrived; 108 native runs; **172,443 FCD points** independently compared.
- **393 scenario records**, all 10 scenarios / 30 subcases represented in both splits.
  Not balanced: 43 family/subcase combinations absent; S5.C/S6.A/S10.B have 2+2 records.
  S3.B has five confirmation families; other core cases have six.
- First build failed the 10.1 m one-second displacement check (10.8665 m). Same-seed
  diagnostics identified instantaneous lane-changing as a fixable simulation issue.
  Rebuild uses 3-second continuous lane changes on ALL families; maximum step 9.452 m,
  lane-geometry error 1.966 m. Original rejected build retained, never benchmark input.
- No identical full routes or first-12 windows with old data. Two stopped windows
  overlap fresh selection/confirmation: S2.A/S2.B, families 304/311. Report the overlap
  and fixed exclusion sensitivity. Shared map edges remain; no cross-city claim.
- `artifacts/datasets/evaluation_v1.sqlite3`, release **urban-fresh-v2**; original
  `scenarios.sqlite3` unchanged. Sealed normalized data, constraints and update log.
- An explicit split-name adapter preserves original roles and all data. Independent
  reverse-adapter comparison is lossless; **9,412 private device observations** checked
  across all slots. These are inputs to local protection, not attacker-visible rows.
- Generator JSON SHA256: `b15d79f51ffc8ae57d348032495c0d8170e6b12642b7d144c2de833ef9808a79`.
- Adapted DB content SHA256: `ce1e633c3a1d48c58860b69e77449189e6907e1f77f881d44ba7d0787ebc92a9`.
- The inherited rejection reason `not_implemented_requires_context_or_activity_model`
  is stale wording on absent S6.C slots, not a reliable implementation-status field.
  Actual generated counts and independently checked predicates take precedence.

## Algorithm and fair comparison

`SwitchingAnchorBelief` implements a two-mode finite HMM forward recurrence on
protected anchors and public time/context only. It is standard filtering, not a novel
HMM/IMM. Mode transition constants and stopped/moving diffusion weights were fixed
before new scoring. The selector keeps mean POI coverage + three improving exchanges;
private anchor, B=.24, K=5, H=12 and threshold=200 m are unchanged.

Controls: geometric, mean_greedy, mean_exchange. All four use identical raw records,
paired anchor randomness and response depth L=5/10. These are **internal ablations**,
not four faithful external SOTA implementations. Each attacker trains on the same
360 core + 2,873 auxiliary observations; no new validation/confirmation rows fitted.

Validation (54 records, 648 runs): no candidate reaches minimum case Recall>=90%,
at either L. For L=10, mean_exchange minimum=89.2464%; switching=89.3184%, both S3.C.
Do not silently round these into passing. Selection was sealed before confirmation.

Confirmation (53 records, 636 runs), macro average over nine cases:

| Variant | Recall L5 | Recall L10 | Min-case L10 | Selected Hit100 | Envelope Hit100 | Selected MAE |
|---|---:|---:|---:|---:|---:|---:|
| Geometric | 84.81% | 90.11% | 82.72% | 5.89% | 12.34% | 481.8 m |
| Mean greedy | 92.46% | 95.99% | 90.51% | 4.19% | 9.38% | 571.5 m |
| Mean exchange | 92.78% | 96.33% | 91.17% | 3.73% | 9.59% | 577.4 m |
| Switching exchange | 92.76% | 96.31% | 91.22% | 2.92% | 8.40% | 555.1 m |

MAE uses the attacker chosen to minimize MAE; Hit uses a separately chosen attacker.
The envelope is descriptive over the tried bank, NOT an upper bound on every adversary.
L5/L10 use identical public releases: privacy values are unchanged by reply depth.
For coverage variants, responses rise from 120 to 200 POI entries and about 2,735 to
4,504 serialized ID bytes/event (+64.7%); 30 logical category-location requests/event.
Do not call these bytes total network traffic or infer device speed from benchmark
step timing gathered while other work was running.

Switching vs mean_exchange improves selected Hit in 2/6 families, worsens 2/6, ties 2/6.
Family-envelope Hit improves 3/6, worsens 1/6, ties 2/6. Top-10 Recall improves 1/6,
worsens 3/6, ties 2/6. No consistent per-family dominance or significance claim.

Predeclared overlap sensitivity excludes confirmation records v2-r311-002/003,
without reselecting anything: mean_exchange/switching L10 Recall=96.30/96.28%,
selected Hit=3.98/2.42%, envelope Hit=8.52/7.20%, MAE=577.7/555.6 m. The mixed
privacy conclusion remains. Confirmed top-10 utility does not override failed
validation selection.

## Verification performed

- **270 tests passed**, including dense independent joint-filter recursion, long-gap
  mass, causal prefix, worker-reset parity, no additional private-input access,
  missing-case rejection and no silent relaxation of the selection threshold.
- Independent replay verifies **2,524 runs / 21,056 events**, including reused training
  controls; 930 training control rows match historical rows exactly.
- Rebuilt 8 training matrices and 8 forests; compared all 5,120 tree arrays.
- Independently reconstructed **265,068 point predictions**, **97,488 category queries**,
  **92,660 directed dummy transitions** and **10,528 objective histories**.
- 72 full + 72 prefix replays: a subset by method/case/phase, not every RNG replicate.
- Independently reconstructed 32 report aggregates (full, sensitivity, family-level),
  selected/envelope metrics, paired changes, selection transfer and numeric TeX tables.
- Raw SUMO audit, SQLite round trip and historical DB hash checks passed.
- Canonical thesis rebuilt with XeLaTeX: **107 pages**, all pages rendered and
  inspected as contact sheets; new equations/results also inspected at high
  resolution. No undefined references, overfull boxes, off-paper words or missing
  replacement glyphs detected. The abstract and conclusion were consolidated.
  PDF SHA256: `ae674f9b2d8c6004658e96a47709d2e2ab079ce36286187df475217b605bb80d`.
  Layout/source receipt: `artifacts/benchmarks/fresh_switching/thesis_qa.json`.

Training SHA256: `81f19b0e0f97a8042fd569efee0170278875ec4ab53abf86ad968a3d461eee2b`.
Validation: `7f54e045bba364665035d58f839e0786b309777810efe1fc7209a4d3c3bd12c3`.
Selection: `bc92403a139186d630ee55590fd89ba411ec4454765ed06433c6f43cbfa4fd6e`.
Confirmation: `cda51660dc9f5c20dc653916472805a9758d6fb0e3d5a80b3f489a49b0e20e0e`.
Readout: `32871f9037726d29d5437003a8a4ced2171f83a895e27a48464351253b2707c2`.

## Recheck and next steps for the next agent

```sh
venv/bin/python -m pytest -q
venv/bin/python -m experiments.verify_fresh_switching --check-only
```

Generation order for an unused destination: build_fresh_scenarios_v2 →
verify_fresh_scenarios → publish_fresh_scenarios → verify_fresh_registry;
run_fresh_switching --phase training → validation (seals selection) → confirmation;
verify_fresh_switching → fresh_switching_readout → verify_fresh_readout.
Default artifact destinations are write-once. Do not delete receipts to rerun; use
the read-only check flag where supported or a separate destination/worktree. Full
native audit needs untracked XML cache; semantic DB/metric artifacts are committed.

1. Diagnose/calibrate sparse-query S3.C on development data only; compare public
   directed transitions against isotropic diffusion before adding more latent modes.
2. Improve thin S5/S6/endpoint scenario support from public route/observation gates,
   not from protection scores. Replace legacy rejection wording in a new generator
   revision; preserve this release.
3. Prespecify a new independent cohort if tuning follows these confirmation results.
4. Recheck external comparator faithfulness and matched output/attacker contracts
   before making any external SOTA claim. Current experiment did not rerun those
   comparators on this cohort and did not evaluate S4–S10 protection.
5. Read `docs/research/fresh_switching_literature_review.md`: mobility mismatch,
   standard HMM inheritance, resource confounding and finite-adversary limitations.
