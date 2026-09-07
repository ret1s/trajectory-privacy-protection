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
5. Executable Geo-I-anchor dummy candidate, causal API, directed-road ablation,
   ideal-kernel post-processing proof and composition limits.
6. Controlled SUMO S1--S3 pilot, exact measured tables, attack/service protocol,
   negative findings and verification.
7. Conclusions with explicit implemented versus specified coverage.

The 2026-09-07 research revision and evidence boundaries are documented in
[`notes/research_update_2026-09-07.md`](notes/research_update_2026-09-07.md).
It adds local map context and a functional protection diagram, separates person
and device identity, and includes protected trajectory endpoints. The common
benchmark specification is not a claim that all scenarios or metrics are implemented.

Chapters 5--7 live in `report_demo_chapters.tex`; their numeric tables are
generated from `artifacts/benchmarks/report_demo/results.json`, never copied
manually. The release guide is
[`notes/report_demo_release_2026-09-07.md`](notes/report_demo_release_2026-09-07.md).
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
# venv/bin/python -m experiments.verify_report_demo
# venv/bin/python -m experiments.export_report_demo
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
Older benchmark artifacts remain available for provenance, but Chapter 6 cites
only `artifacts/benchmarks/report_demo/`. This is a controlled pilot, not a
publication-ready SOTA leaderboard. S4--S10 have no measured coverage yet.

Development status, implementation caveats and reproducibility notes are kept
separately in [`notes/draft_clarifications.md`](notes/draft_clarifications.md)
so that `main.tex` retains the tone and structure of the final thesis.
