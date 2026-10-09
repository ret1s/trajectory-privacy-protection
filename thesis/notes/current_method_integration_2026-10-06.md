# Current method chapter: integration and claim audit

Prepared06/10/2026. Owned source: [`../current_method.tex`](../current_method.tex),
Vietnamese, with both `ch:method` and `ch:current-method`. Existing mechanism,
runner, protocol, scores and historical thesis/report modules were not edited.

The module includes its own `\chapter` and uses only packages/macros already in
[`main.tex`](../main.tex): `amsmath`, `amssymb`, `graphicx`, `tabularx`, TikZ,
`natbib`, `hyperref`, `P`, `Y` and report colours. Its architecture is native
TikZ, sized by `\resizebox{\textwidth}{!}` to the16cm text area. It does not
require generated raster diagrams, a new theorem package or an external `.bib`.
All citation keys already exist in the canonical bibliography. Takagi/Mironov
are linked primary-source footnotes; they can receive dedicated bibitems during
editorial integration without changing the technical statements.

Replace the active inclusion of `report_demo_chapters` with the new chapter
and the independently authored current evaluation/conclusion. The legacy file
contains **three chapters**, so appending this module without removing its
active inclusion would duplicate method/evaluation/conclusion and `ch:method`.
Keep the old source as historical evidence. Suggested method insertion:

```tex
\input{current_method}
```

The chapter is246 source lines at first handoff; root owns compilation of the
integrated main using Tectonic in `/private/tmp` and visual review of the whole
PDF. No new final PDF was promoted by this method-authoring task. Source
whitespace check passed. The module deliberately does not claim that a source
lint is a successful LaTeX build or a privacy proof.

## Verified technical contract

| Claim in the chapter | Implementation / evidence | Scope |
|---|---|---|
| Continuous projected GPS input, REM over a fixed full public road support | [`core/mechanisms.py`](../../core/mechanisms.py), [`lane_budgeted.py`](../../benchmark/engines/lane_budgeted.py), [`lane_states.py`](../../data/lane_states.py) | Ideal Euclidean metric privacy; directed road costs are a different quantity. |
| First1unit; reuse1; refresh2; prospective reserve1/2 before GPS | [`filtered_cover.py`](../../benchmark/engines/filtered_cover.py), [`matched_filter.py`](../../benchmark/engines/matched_filter.py), [`session_budget.py`](../../core/session_budget.py) | Extended-path ideal proof supports realized branch charging; not retrospective resample accounting. |
| H12→23units, not hard12GPSreads | [`matched_filter.py`](../../benchmark/engines/matched_filter.py), [formal audit](../../docs/research/2026-10-06_jisa_formal_audit.md) | Up to22reads on a sufficiently long reuse-heavy path;600s/60s schedule has at most11. |
| Eight slots, C.23/m, u.00125/m, sessioncap.02875/m, nominalB.03/m | [recommended configuration](../../artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1/recommended_configuration.json), [`session_budget.py`](../../core/session_budget.py) | One persistent subject/epoch ledger, public admission; multiple epochs/devices compose. |
| Protected belief is an approximate utility model | [`anchor_belief.py`](../../benchmark/anchor_belief.py), [`_FilterBelief`](../../benchmark/engines/filtered_cover.py) | It does not condition on all private branch information or certify a calibrated attacker posterior. |
| Current Q planner retains nearestL10 signatures; serverL30 | [`public_service_planner.py`](../../benchmark/engines/public_service_planner.py), [frozen configuration](../../artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1/recommended_configuration.json) | Failed aligned/multi/risk arms remain development; no adopted all-purpose/risk Q algorithm. |
| Ordered public streams are retained for L30 attacker audit | [`qplanner_study_20261006_v2.py`](../../experiments/qplanner_study_20261006_v2.py), [base-Q contract](../../artifacts/benchmarks/qplanner_depth_base_q_generalization_20261006_v1/protocol.json) | Optional [`PrivateOrderCoverClient`](../../benchmark/query_order.py) is separate; geometry can still link shuffled Q. |
| Fixed all-category retrieval, four private local rankings | [`query_purpose.py`](../../benchmark/query_purpose.py), [`geoi_lbs.py`](../../benchmark/geoi_lbs.py), [`budgeted_geoi_lbs.py`](../../benchmark/budgeted_geoi_lbs.py) | All-purpose means local use of a common pool; server does not execute four private purposes. |
| Protected GPS supplier60s, utility GPS/destination at each public event | [`UtilityEvaluator` and `run_family`](../../experiments/qplanner_study_20261006_v2.py), [fresh scope](../../artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1/README.md) | No sensor-energy claim or validation with only60s local fixes. |
| Endpoint no-delay evidence belongs to Endpoint20, separately from L30 | [endpoint noise](../../docs/research/2026-10-05_endpoint_noise.md), [endpoint generalization](../../docs/research/2026-10-05_endpoint_generalization.md), [ordered attack audit](../../docs/research/2026-10-05_endpoint_order.md) | No future endpoint detector, hidden close time, theorem for home/work or cross-cohort transfer. |

## Statements to replace in the old active chapters

These were present in the old source at review start; root may already have
revised the active `main.tex`. They remain in the preserved legacy module.

- **``H private accesses then no reads``:** correct for the old fixed-read
  controller, incorrect for the current branch-charged matched filter. Use the
  unit/cap derivation and distinguish H, number of reads and public clock.
- **Session B.24 as the current linked-history cap:** replace with effective
  epochC.23 and eight preallocated sessioncaps. Do not re-label separate resets
  as the same total budget.
- **Random ring/offset/Gumbel dummy scoring as the active Q selector:** retain
  only as historical ablation. Current Q uses protected belief, ordered service
  signatures, global greedy, bounded exchanges and progress/slack.
- **Every request reveals the user's true category/purpose:** current network
  retrieval requests all public categories; actual category/purpose/radius/
  destination are local. Keep the conditional noninterference limits.
- **Persistent Q IDs as an invariant of every production adapter:** L30 audit
  retains ordered slots conservatively, while an optional client shuffles and
  omits stable IDs. Neither case makes geometric linkage impossible.
- **S9/S10 only observe the old220s cropped window or always omit60s endpoints:**
  preserve that old protocol as history; distinguish full-window no-delay
  endpoint evidence and current native Q study, each with its own targets/bank.
- **No implementation of S7 or linked-session accounting exists:** no longer
  accurate. Implemented conditional content boundary/allocator exist, but
  unconditional intent secrecy and identity anonymity remain unproved.
- **Old top5/top10 retrieval numbers as the current method's confirmation:**
  the L20/L30 fresh study is a separately frozen static-service comparison.
  Its utility gain is paid bandwidth, not superiority over paper baselines at
  equal depth or a new privacy mechanism.
- **The float simulator satisfies pure Geo-I:** every ideal statement must
  retain finite-precision/PRNG/public-clock caveats. Support tests are concrete
  counterexamples; they are not a sampler certificate.

## Primary sources checked

- [Andrés et al., Geo-I, CCS2013](https://arxiv.org/abs/1212.1984): inherited metric
  definition; chapter REM proof is the explicit fixed-support triangle argument.
- [Chatzikokolakis et al., predictive mechanism, PETS2014](https://petsymposium.org/2014/papers/Chatzikokolakis.pdf):
  inherited private test/prediction/bounded composition; current integer filter
  instantiates the extended-path argument under ideal randomness.
- [Takagi et al., GEM/GG-I](https://arxiv.org/html/2010.13449v1): graph metric and
  exponential mechanism; Euclidean REM on road support is not the same metric.
- [Mironov, CCS2012](https://www.microsoft.com/en-us/research/publication/on-significance-of-the-least-significant-bits-for-differential-privacy/):
  finite-precision hazards; the exact bounded-span defect is documented in the
  current source and support tests.
- [NumPy random sampling documentation](https://numpy.org/doc/stable/reference/random/index.html):
  simulator PRNG is distinct from a cryptographic implementation claim.

The chapter's formal statements are **specification-level conditional results**.
Operational tests verify cap/supplier/API/source invariants. Fresh artifacts
verify finite attacker/utility calculations on a synthetic same-map service.
Neither is converted into a production privacy certificate or a claim that
S1–S10 are fully solved.
