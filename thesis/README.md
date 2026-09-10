# Graduation thesis

[`main.tex`](main.tex) is the single canonical LaTeX source for the evolving
graduation thesis. It no longer imports a dated supervisor-meeting file.

The current main document contains seven chapters:

1. System architecture and trust boundary.
2. Urban scope, motion constraints and causal online processing.
3. Protection targets, S1--S10 threat scenarios, related protective work and
   SUMO-based dataset design.
4. Neural/non-neural comparator methods, original metrics and a common
   privacy/POI-utility/cost evaluation specification.
5. BR-Dummy: private anchors, noisy reuse, fixed-horizon ledger and reachable-road
   postprocessing; protected-history service coverage, bounded exchanges and
   category-capped coverage; ideal proof, complexity and limitations.
6. Controlled SUMO S1--S3 plus S9/S10 study, stronger offline adversaries,
   five-way split, constrained parameter selection and negative findings.
7. Conclusions with explicit implemented versus specified coverage.

The 2026-09-07 research revision and evidence boundaries are documented in
[`notes/research_update_2026-09-07.md`](notes/research_update_2026-09-07.md).
It adds local map context and a functional protection diagram, separates person
and device identity, and includes protected trajectory endpoints. The common
benchmark specification is not a claim that all scenarios or metrics are implemented.

Chapters 5--7 live in `report_demo_chapters.tex`; their numeric tables are
generated from `artifacts/benchmarks/paper_benchmark/results.json`, never copied
manually. The release guide is
[`notes/paper_cycle_v2_protocol.md`](notes/paper_cycle_v2_protocol.md).
The earlier complete source before the four-chapter scope reduction is preserved in
[`notes/snapshots/graduation_thesis_full_2026-09-05.tex`](notes/snapshots/graduation_thesis_full_2026-09-05.tex).
To rebuild that reference, run LaTeX from `thesis/` using the snapshot path and
a separate output directory; it is not a second canonical thesis.

The illustrative records for the **previous S1--S7 taxonomy** remain in
[`threat_records.tex`](threat_records.tex), referenced by the full snapshot but
not included in the current main document. They are hand-constructed teaching
examples, not SUMO benchmark records or measured attack results. Their IDs must
be translated using the migration table in the research revision note; they were
not silently relabelled to match the new ten-scenario specification.

Build from this directory into an ignored scratch directory:

```bash
# From repo root first, if the result artifact changed:
# venv/bin/python -m experiments.verify_paper_benchmark
# venv/bin/python -m experiments.export_paper_benchmark
mkdir -p ../build/thesis
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=../build/thesis main.tex
```

The reviewed deliverable is committed once at
`artifacts/reports/graduation_thesis.pdf`. LaTeX intermediate files and
`thesis/main.pdf` are ignored if a local editor creates them; neither is a
canonical artifact.

The modular chapters and BibTeX database from Internship 2 are preserved at
`archive/internship_2/thesis/`; they are not imported by the current thesis.
Older benchmark artifacts remain available for provenance. Chapter 6 now covers
the controlled paper/lane/context/belief/service/prior-factor cycles and the
`artifacts/benchmarks/expanded_shadow/` inference audit. This is a controlled study, not a
publication-ready SOTA leaderboard. S4--S8 have no measured coverage yet. No BR
grid configuration met Recall >=90% simultaneously across five scenarios on
defense-validation data. The earlier chapters remain in `notes/report_demo_chapters_v1.tex`.

Development status, implementation caveats and reproducibility notes are kept
separately in [`notes/draft_clarifications.md`](notes/draft_clarifications.md)
so that `main.tex` retains the tone and structure of the final thesis.

The latest auxiliary-data specification is in `dataset_registry.tex`; the
frozen-defender attack comparison is in `expanded_shadow_comparison.tex`.
It adds 80 SUMO route groups (64 auxiliary training, 16 auxiliary holdout), not
a new core-scenario confirmation set. Exact tables are exported only after the
independent verifier passes. Literature and QA handoff:
`docs/research/expanded_shadow_literature_review.md` and
`docs/reviews/verification_expanded_shadow.md`.

The preceding matched development cycle is in `coverage_frontier_method.tex`,
`coverage_frontier_comparison.tex` and `coverage_frontier_findings.tex`.
It separates selector changes from top-L response allowances at a fixed top-5
reference, across three B values. Category balancing is a tested hypothesis,
not guaranteed robustness. The original anchors and legacy controls stay fixed;
no default defender or SOTA claim is promoted by these internal ablations.
Source-pinned protocol: `notes/coverage_frontier_protocol.md`. Exact evidence,
independent checks and rebuild order: `artifacts/benchmarks/coverage_frontier/`.

The newest fresh-family cycle adds `fresh_dataset.tex`, `switching_method.tex`
and `fresh_switching_comparison.tex`. It tests a two-mode protected-history
filter on six selection and six confirmation families. No configuration passes
the prespecified 90% minimum-case validation gate, even at top-10 server depth;
the candidate is not promoted as a new default. Full confirmation and overlap
sensitivity results are in `artifacts/benchmarks/fresh_switching/`.
The canonical PDF is still the same file under `artifacts/reports/`.
