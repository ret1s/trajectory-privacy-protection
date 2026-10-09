# Continued formal analysis — 07 October 2026, V2

The canonical manuscript is [graduation_thesis.pdf](../../artifacts/reports/graduation_thesis.pdf).
This revision continues the ideal proofs before the empirical chapter. The
Geo-I/REM backbone, algorithms, configuration, datasets, scoring protocols and
retained benchmark results are unchanged. This is a supervisor-review
manuscript, not a human peer-review decision or an executable privacy certificate.

## New results and their conditions

| Thesis section | Result | Required scope |
| --- | --- | --- |
| 5.8 | Mutual likelihood bound implies TV ≤ tanh(α/2) and balanced binary Bayes success ≤ exp(α)/(1+exp(α)); explicit arbitrary-prior and multi-hypothesis bounds | Same full protected observation and public context; all-pair bounds for multiple candidate traces |
| 5.8 | Finite epochs compose with exponent Σ CₑDₑ; a sensor coupling transfers a measured-input bound to a physical-position bound | Conditional epoch kernels valid for every full protected prefix; valid sensor samples and a separately established coupling |
| 5.9 | Exact finite-road REM CDF/quantile, geometry-mass tail bound and exact reuse mixture tail | Ideal REM/Laplace randomness, fixed public support; bounds describe internal Z and measured inputs |
| 5.9 | Simultaneous read bounds propagate to no-read events through a displacement/age assumption | Uniform pre-read failure bound and finite read count; 60 s minimum pacing does not bound maximum anchor age |
| 5.11 | Directed-road score error, top-k set stability, partial stable-reference Recall and radius membership band | Same catalogue/category/destination/ties; finite two-way road distances and sufficient score separation |
| 5.12 | The reference5 weighting equals expected nearest-reference coverage on the public grid; TV belief mismatch ζ costs at most 3ζ/2 in the greedy–slack lower bound | Same grid, coordinate-access map, reference/category weighting and feasible action domains; actual ζ is not established |

The binary example with α = 0.25 gives 56.22% only for two balanced
hypotheses differing in one protected measured input by 100 m under the
stated u. It is not a bound on every endpoint/identity attacker. The whole
epoch exponent α = 23 at 100 m remains weak. Reset does not erase prior
observations, and generating five Q does not establish five-candidate anonymity.

The accuracy illustration is an explicitly mathematical three-state map, not
a modified dataset or measured benchmark. Its fresh REM tail at 1,000 m is
0.0908169; a held reference at 3,000 m makes the mixed tail 0.1045444. A
5% failure budget split into 0.025 for each component needs the actual REM
97.5% quantile as well as the test radius 2,596.6 m. These examples prevent
confusing privacy with accuracy or a joint far-hold event with a conditional
accuracy guarantee after holding.

The ranking argument distinguishes Euclidean sensor error from directed-road
error. Detour's common destination term cancels in a POI-pair comparison,
yielding the same distance-gap bound as nearest. Within-radius additionally
needs membership stability. Gaussian σ = 15 m per axis is not a hard bound
on coordinates or roads. The belief argument is nearest/grid-specific and
does not claim calibration, continuous-GPS accuracy or the four-purpose
benchmark mean.

## Review and verification

- [Four-module mathematical cross-review](2026-10-07_thesis_formal_v2_math_review.md)
  records authorship, independent reviews, exact source pins and numerical
  sanity checks. Finite enumerations/LP checks support the review; they do not
  replace proofs.
- [Accuracy PDF review](2026-10-07_thesis_formal_v2_accuracy_pdf_review.md)
  covers physical pages 57–61.
- [Inference/ranking/bridge PDF review](2026-10-07_thesis_formal_v2_inference_ranking_pdf_review.md)
  covers physical pages 54–57 and 65–70.
- Tectonic builds the 95-page final PDF without overfull boxes, undefined
  references or errors. All pages were rendered; all 12 contact sheets and
  the changed theorem pages were visually reviewed. Embedded fonts, 383
  links, internal destinations and text bounds were checked. A split
  theorem heading on pages 69–70 was repaired and those final pages reviewed.
- Both read-only exporters pass: 29 source JSON/8 deterministic TeX and
  10 extension readouts/3 deterministic TeX. No scores were regenerated.
  All nine evidence pins and both export-manifest pins from V1 remain
  identical. The full executable test suite was not rerun for this prose-only
  revision; its earlier receipt remains in the preceding review release.
- The PDF-skill operation marker was attempted once before root authoring;
  it returned exit 127 because `node` is unavailable. This was not a
  successful marker. Rendering used the established Tectonic/PyMuPDF path.

Final PDF SHA-256:
`d3d51ad7f97995cc098b3ba7a0546e679214a36501c137e885b8d0ca14e3f920`.
The [V2 release manifest](../../artifacts/reports/thesis_formal_review_20261007_v2/manifest.json)
pins the retained reviewed PDF, active TeX, source/assets snapshot, build and
layout checks, review notes and unchanged evidence.

The [83-page formal V1 release](../../artifacts/reports/thesis_formal_review_20261007_v1/manifest.json)
and the earlier 74-page review remain intact with their original PDF/source
bytes and receipts. This new review does not rewrite their claims.

Primary Geo-I and predictive-mechanism full texts were read in the preceding
formal analysis, and the derived lemmas are scoped to the current ideal
kernel. AnotherMe's exact theorem/proof remains unverified because its full
text was unavailable; the [source audit](../research/2026-10-07_anotherme_theory_comparison.md)
is preserved. No theorem, absence claim or executable guarantee is attributed
to unread text.
