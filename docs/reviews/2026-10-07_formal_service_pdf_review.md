# Formal service and mathematics PDF review — 7 October 2026

**Status: PASS within the scope below. No remaining mathematical or rendering blocker was found.** This is a targeted independent review of the proof text and its rendered integration, not a certification of executable pure Geo-I, a full 83-page visual review, or a new benchmark result.

## Reviewed release and source pins

| File | SHA-256 |
|---|---|
| `build/thesis/main.pdf` — 83 pages | `e5f1981f2788ed7cac851c44ceca0528b2a38cac1ab0b9ded267303225662a4f` |
| `thesis/current_formal_service.tex` | `db15e58820f63ae35fdcb3aaf73132cc9fc328250c9becb6351336d8d4fa58a1` |
| `thesis/current_formal_privacy.tex` | `b3a6f09db962ea0211addc5166ca6495aaef0004e8a4f80ed9b3b0dae3b6dd3a` |
| `thesis/current_method.tex` | `cdd38f8bcf9cc507a282d4422623d2cee6936adf742b0de799f361aa5f4b6f98` |
| `thesis/main.tex` | `919716d2bde99f913fcf7df253c6751455e2bbc033f889160a10f7732ef0c3b8` |

The reviewed PDF is the build candidate. A canonical copy with identical bytes inherits this review; a rebuilt PDF with a different hash needs an appropriate follow-up review. No canonical PDF, theorem source, executable, or historical artifact was edited by this review.

## Visual and integration checks

- Scope: physical PDF pages **47–59**, printed pages **40–52**, sections 5.7–5.8 and the transition into 5.9. Inspected complete page renders at their original size, including formula bodies, equation numbers, footnotes and page transitions.
- Reinspected final affected pages **52–59** from `build/thesis/qa_formal_20261007_final`. Final renders for pages 47–51 are byte-identical to the already visually reviewed candidate renders; the latter review therefore applies directly to those final pages.
- Equations, subscripts, inequality signs, union/set differences, fractional terms and proof-ending squares are legible. No clipped formula, overlapping equation number, missing glyph, isolated subsection heading or unreadable footnote was found. Ordinary sentences continue across page breaks; proofs remain understandable.
- An independent text-span check on all 13 target pages found no text outside physical page bounds, replacement glyph or `??` reference. The integrated legacy purpose and transcript labels occur once each. The main compile log has no Overfull, undefined-reference, multiply-defined-label or missing-character diagnostic; conventional hyperref removal of math delimiters from a bookmark is not a rendered-content defect.
- The final service ending and S9–S10 scope/table fit on page59. Page60 begins Chapter6, rather than a sparse page containing only the previous chapter's ending lines.
- Root-generated layout metadata agrees with the authenticated PDF: 83 pages, all fonts embedded, 330 links, no invalid internal links and no text outside the page. Those metadata checks supplement the targeted visual inspection; they do not substitute for it.

## Substantive mathematical checks

1. **Directed feasibility.** The previous internal state belongs to its next reachable bucket because its directed travel time to itself is zero. Greedy, quotient reduction, exchange, progress and slack all preserve the relevant track bucket. Concatenated paths with waiting establish feasibility in the public free-flow graph. The text correctly limits this to an existing internal lane-state assignment; nearest coordinate access can choose another coincident lane. It does not claim traffic realism, collision avoidance, distinct Q locations or untrackability after shuffling.
2. **Weighted coverage and the half bound.** Nonnegative weighted union is normalized, monotone and submodular. The final text correctly defines matroid independence as *at most one* labeled state per track, then completes a base to all K nonempty groups. In the direct proof, the optimal element belonging to greedy's newly selected track is still feasible at that greedy step. Diminishing marginal gains and summing greedy gains give `F(opt) ≤ 2 F(greedy)`. Quotient representatives preserve the current-step coverage objective; accepted exchanges increase it; progress preserves the ordered signature; a fixed full-action slack floor gives `F(final) ≥ F(opt)/2 − delta_F` in the stated ideal arithmetic. This is an inherited current-step surrogate bound, not a trajectory, calibrated posterior, executable float-optimality, four-purpose Recall or privacy guarantee.
3. **Reference5 versus signature10.** Checked the actual factory/resource wrapper: `native_resources` builds a reference-top5 base, while `ResponseAwareAnchorModel.poi_weights` preserves that base's weights and exposes the planner's top10 signatures. The final text correctly distinguishes those two depths from public retrieval L30. The earlier reviewer's tentative top10-weight interpretation was corrected before integration.
4. **Purpose request noninterference.** Coupling the same mechanism/planner/order/server randomness establishes pathwise identical logical network transcripts when location trace, public admission/clock/schema and server strategy are held fixed. Private `QuerySpec` affects local ranking only. The conclusion is conditional request-content noninterference, not independence of intent from a route, account, activation, click or physical latency; secret-driven retries would violate its assumptions.
5. **Local exactness and denominator.** With the same true ranking, eligibility domain and public ID tie-break, each received reference element still ranks within local top-k. Thus `R ∩ P(A) = R ∩ A`; complete reference retrieval is necessary and sufficient for the exact answer. The denominator is `min(k, eligible_count)`, not always5. Empty true reference is N/A; an empty answer for a defined true reference receives zero. Availability needs current known status and detour needs a supplied local destination. Completion alone need not imply correct Recall.
6. **Depth monotonicity.** On exactly fixed Q, static catalogue, coordinate-access mapping, service ordering and correct local ranking/reference, response prefixes imply nested unions, hence nondecreasing pointwise Recall and fixed nonnegative-weight averages. It is not a strict gain, privacy improvement or equal-cost superiority. Matched causal cache/expiry rules can preserve inclusion; different cache policies, clocks, versions or status validity can break the premise.
7. **Ideal privacy cross-review.** REM includes the normalizer ratio; the Laplace test pays for both outcomes; the proof bounds the *joint* branch/anchor probability under the same prefix. Worst-case prospective reservation before GPS gives a pathwise cap and common-support chain rule, followed by a shared postprocessing kernel. Branch-dependent stopping is included, not assumed input-independent. Bayesian odds and mixture conditions are appropriately limited; float/PRNG/rollback and unpublished-local-answer assumptions remain explicit.
8. **New single-coordinate corollary.** For traces differing only at event `i0`, every other input distance in the path ratio is zero and `c_i0 ≤ 2`. Therefore the complete later transcript satisfies the ideal `exp(2 u r)` bound even though later histories and decisions may differ between realized runs. Marginalization/postprocessing preserves it. The final text correctly excludes multi-event route changes and semantic home/work claims. At100m, the displayed factor `exp(.25) ≈ 1.284` is correct.

## Executed source-linked checks

These bounded deterministic checks were run independently as inline Python programs against the existing functions. They support the derivations; they are not new defense-performance evidence or formal numerical certificates.

- Enumerated **4,096** weighted-union instances: four candidate state signatures over three POIs, every signature in `{0,…,7}`, groups `[0,1]` and `[2,3]`, weights `[.2,.3,.5]`, zero public tie costs. Actual `greedy_cover` was compared with exhaustive feasible actions. Every instance met the half bound at numerical tolerance `1e-12`; the minimum observed ratio was exactly0.5.
- Ran the position-error counterexample through actual `MultiPurposeRoadRanking`, including exact service-depth prefixes on a directed strongly connected graph. There are three source states (true, estimated, Q), five reference POIs `a1…a5`, fifteen filler POIs and POI `b`. Direct costs from the true source to `a_i` are1…5, to fillers100 and to `b`10; from the wrong estimated source they are6…2,100 and1; from Q they are1…5,6…20 and21. Return edges of cost1000 preserve those shortest paths while making the graph strongly connected. The service returns20 and21 IDs for L20/L30. Fixed wrong-position local top5 Recall falls **1.0 → 0.8**, despite exact prefix inclusion; this verifies the stated caveat using the actual ranker rather than changing reference denominators. This synthetic witness is not a run on the thesis map or Geo-I mechanism.
- Independently recomputed the displayed accounting/probability examples: `U=23`, `u=.00125/m`, effective session cap `.02875/m`, nominal budget `.03/m`; `q(50)=.5854854409`, `q(1200)=.1432523984`; standalone two-hypothesis REM posterior upper bound `.5312093734`; epoch likelihood-ratio bound at100m `exp(23)≈9.7448×10^9`. The latter is correctly described as loose rather than low inference risk.

## Source pins used for the service checks

| Source | SHA-256 |
|---|---|
| `benchmark/query_purpose.py` | `110ce445bb1a89283728b263c1994eda2cf601b3c6ae1eab49e31b2b6bad4443` |
| `benchmark/engines/service_cover.py` | `56894618d61135c016f11609ee379b7f0867de46ddfc0699d7f174fd49ba6e48` |
| `benchmark/engines/fair_cover.py` | `c739b71374a6682f6468588720660152bf73ad11c825237d90ed9a213cb8d8a6` |
| `benchmark/engines/quotient_cover.py` | `93db80c1acce17b50e6dcdf4c711189604b003735b41a087c4d8d9ccaada8ed2` |
| `benchmark/engines/progress_cover.py` | `9e1121793209e86c825dfc785a6b6859219443fd0ec86557eebdf5035a3cb4d0` |
| `benchmark/engines/slack_progress.py` | `655e2a730323d1e365fef4ac8a14cbd67e7049b27816cf41fcae850651a97963` |
| `benchmark/response_aware_belief.py` | `2f75b486472430260fa5a112cf5f49cb2bfd18d67da378d8c62954c72a43f070` |
| `experiments/future_sumo_eval.py` | `8705492563025bb6c4207c3b644d7db9f5366f466b7963bee4e824b2c6395e7c` |
| `evaluation/lane_travel.py` | `111e1bc109dd53e29922d44bfbc16bded8f1409e467e43faba3723026dc45bde` |
| `benchmark/public_poi_context.py` | `98aa1301ce921630dbb77ef1bb7a19c6b70241e1689cccaa132192019b88ae86` |

No costly benchmark, sampler, attacker search, privacy-key audit or full regression suite was rerun for this read-only PDF review. The mathematical result remains conditional on ideal kernels and stated protocol assumptions; its presentation does not enlarge the empirical claim scope.
