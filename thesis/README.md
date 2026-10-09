# Graduation thesis

[`main.tex`](main.tex) is the single canonical source. The reviewed PDF is
[`../artifacts/reports/graduation_thesis.pdf`](../artifacts/reports/graduation_thesis.pdf).
The 07/10/2026 revision brings the active thesis up to the current Geo-I research
state; it is a supervisor-review manuscript, not a certificate of graduation
approval or journal acceptance.

The seven chapters cover:

1. Problem, trust boundary, research questions and scoped contributions.
2. Urban vehicle motion and causal online processing.
3. S1–S10, dataset design and the source/role of each cohort.
4. Comparator output contracts, original metrics and common task measurements.
5. The current Geo-I/REM pipeline: persistent cap, noisy reuse, protected belief,
   road-feasible queries, fixed multi-purpose service and private local ranking,
   followed by explicit ideal privacy and service proofs before the benchmarks.
6. Evidence by protocol: historical baselines, matched development, retained
   planner failures, fresh L20/L30 confirmation, application controls and a
   controlled dynamic-status/cache diagnostic, local-GPS sensitivity,
   historical companion inference and resource accounting.
7. Answers to the research questions, limitations and further work.

Chapters 5–7 are `current_method.tex`, `current_evaluation.tex` and
`current_conclusions.tex`. Chapter 3 uses `current_dataset.tex`. Numerical tables
in the current evaluation are exported from pinned artifacts rather than
invented or copied from a differently configured model. Follow the source and
rebuild commands recorded in the evaluation module and its generated tables.

The principal confirmed result is a **service-depth utility/cost tradeoff**:
L30 retains the legacy Geo-I Q stream, raises current-only conditional
Recall@5 from 89.714% to 92.688% on 24 new same-map synthetic test families,
and costs 31.102% more reply JSON bytes. Three private draws remain nested
within each family. This is not a new privacy theorem or equally costed SOTA
victory. S5/S6 are a controlled two-choice future task; S7 is conditional
payload noninterference; S4 and S8 have no established real-user protection.

The thesis separates ideal mathematical guarantees from the floating-point
simulator. `current_formal_privacy.tex` proves REM normalization, the private
reuse test, joint branch costs, prospective epoch composition, server
postprocessing and Bayesian odds. `current_formal_service.tex` proves road
feasibility, the inherited greedy/slack bound, conditional purpose
noninterference, local top-k exactness and the conditions for L-depth
monotonicity. The continued formal revision adds four modules:
`current_formal_inference.tex` derives TV/Bayes discrimination bounds, finite
epoch composition and a conditional sensor-channel transfer;
`current_formal_accuracy.tex` derives finite-road REM quantiles, the exact
reuse mixture tail and simultaneous read/no-read displacement bounds;
`current_formal_ranking_robustness.tex` establishes directed-road score and
top-k stability conditions; and `current_formal_belief_bridge.tex` identifies
the nearest reference objective and its conditional calibration penalty.
These establish the system contract; they do not prove benchmark superiority,
certify the executable sampler or establish actual sensor/belief calibration.
It also reports the full-static-catalogue local-ranking control:
the small public catalogue can remove the need for periodic coordinate
queries when bulk retrieval is permitted. Primary local utility uses exact
evaluator GPS at each event; a secondary diagnostic now changes only the local
ranking position to 60 s fixes. Neither clock measures total GNSS reads or energy.

The local-GPS diagnostic independently checks all 72 retained streams and
14 fixed arms: exact-event control and two estimators at per-axis Gaussian
noise 0/5/15 m, each with L20/L30. L30 gains remain 2.24–2.61 percentage points
in the six non-oracle variants. Two-fix extrapolation lowers mean position
error but also lowers Recall compared with holding the last fix, so it has
not been adopted. The test uses inspected same-map synthetic data and a known
local destination; the three-purpose result excludes that destination oracle.

A separate historical S8 diagnostic compares actual simultaneous SUMO pairs.
The protected-partner bank ties target-only; raw-partner information decreases
MAE but also decreases Hit100. Three test families and publicly reproducible
historical RNG constrain this to a finite-bank diagnostic, without a privacy
claim for current Epoch8/L30 or linked groups.

Context accounting distinguishes 1.71 MB compressed L60 archive from its
95.31 MB signature payload. Response prefix views retain the L60 allocation;
the planner has a separate L10 context. Cache capacities are analytic storage
estimates, without a measured peak-RAM or phone feasibility claim.

A secondary dynamic-status replay keeps all 72 frozen test streams, makes
availability expire at public 60 s epoch boundaries and separates unreceived
records from unknown current status. L30 obtains 91.44% Recall; causal
within-epoch accumulation obtains 91.78% with unchanged requests/bytes. It is
one synthetic status world on the already-inspected cohort. Current bulk
status still dominates, and the public seed/world can reconstruct availability
when given to the client. This is a freshness/dataflow diagnostic under an
assumed provider contract, not a new independent privacy or application proof.

Build from `thesis/` into ignored scratch output:

From the repository root, first authenticate or rebuild the retained-statistic
exports (these commands do not score a model):

```sh
python -m experiments.plot_thesis_depth_20261006
python -m experiments.export_thesis_evaluation_20261006
python -m experiments.export_thesis_evaluation_20261006 --check
python -m experiments.audit_public_resource_footprint_20261007 --check
python -m experiments.export_thesis_extensions_20261007
python -m experiments.export_thesis_extensions_20261007 --check
```

The exporter checks the independently audited dynamic readout as well. To
verify that workload separately without re-scoring or creating private draws:

```sh
python -m experiments.dynamic_provider_status_20261006 contract
python -m experiments.verify_dynamic_provider_status_20261006_v2
```

The V1 checker failure and explicitly post-score V2 row-order correction are
retained in the dynamic artifact; the frozen workload and numbers did not
change. Do not run `replay` to overwrite the saved results.

Independent local-GPS and S8 verifiers reconstruct the existing public evidence
without protection sampling or private keys (the full GPS replay takes longer):

```sh
python -m experiments.verify_local_gps_robustness_20261007
python -m experiments.verify_s8_companion_inference_20261007
```

The existing protocols/checkers were fixed before their scores. Never edit their
self-pinned sources or overwrite the saved study; a changed design needs a new
version/output. These diagnostics reuse inspected data and are separate from
the original fresh L30 confirmation.

```sh
mkdir -p ../build/thesis
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=../build/thesis main.tex
# Alternative, when Tectonic is installed:
tectonic -X compile --outdir ../build/thesis main.tex
```

After checking references, numbers and rendered pages, replace the one
canonical PDF under `artifacts/reports/`. `thesis/main.pdf` and LaTeX
intermediates are ignored. Do not create another competing final thesis.

Previous method/evaluation chapters remain in `report_demo_chapters.tex` and
the associated modules; they are no longer imported by the active thesis.
Their frozen benchmark results and negative findings are retained. Older
dataset registry/specification modules describe the corresponding historical
cohorts and remain available for provenance. The previous teaching examples
in `threat_records.tex` use the old S1–S7 taxonomy and are not benchmark data.

The full 05/09 source remains at
[`notes/snapshots/graduation_thesis_full_2026-09-05.tex`](notes/snapshots/graduation_thesis_full_2026-09-05.tex).
Internship 2 modules remain in `archive/internship_2/thesis/`. These are
historical material and are not the current canonical source.

Current evidence and reviewer handoff:

- [`Geo-I response-depth result`](../docs/research/2026-10-06_geo_i_response_depth.md).
- [`Planner chronology`](../docs/reviews/2026-10-06_qplanner_iteration_log.md).
- [`Formal audit`](../docs/research/2026-10-06_jisa_formal_audit.md).
- [`Evaluation audit`](../docs/research/2026-10-06_jisa_evaluation_audit.md).
- [`Publication plan`](../docs/publication/jisa_20261006/README.md).
- [`Dynamic status diagnostic`](../artifacts/benchmarks/dynamic_provider_status_20261006_v1/README.md).
- [`Local-GPS sensitivity`](../artifacts/benchmarks/local_gps_robustness_20261007_v1/README.md).
- [`Historical companion diagnostic`](../artifacts/benchmarks/s8_companion_inference_20261007_v1/README.md).
- [`Context storage audit`](../artifacts/benchmarks/public_resource_footprint_20261007_v1/README.md).
- [`Formal analysis revision`](../docs/reviews/2026-10-07_thesis_formal_analysis.md).
- [`Continued proof revision`](../docs/reviews/2026-10-07_thesis_formal_completion_v2.md).
- [`AnotherMe theory source audit`](../docs/research/2026-10-07_anotherme_theory_comparison.md); its exact proof remains unverified without full text.
- [`Previous 07/10 thesis review`](../docs/reviews/2026-10-07_thesis_completion.md); its exact PDF/source bytes are retained under `artifacts/reports/thesis_review_20261007/`.
- [`Previous review`](../docs/reviews/2026-10-06_thesis_completion.md); its exact
  reviewed PDF/source bytes are retained under `artifacts/reports/thesis_review_20261006/`.
