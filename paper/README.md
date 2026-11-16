# English research manuscript

This is a compact, two-column English author-review paper, distilled from the current thesis and frozen GeoI-BQ v1 evidence. It provisionally targets a regular research article in the Journal of Information Security and Applications (JISA). It has not been submitted.

The main article uses Elsevier's 5p two-column layout. It keeps the mechanism, four privacy propositions, end-to-end bound, principal utility properties and six-arm results. Detailed specifications, full proofs, confidence intervals, selected attackers and extra metrics are in the single-column supplement. Utility and acquisition costs share one main table; repeated explanations have been removed.

- Main source: [main.tex](main.tex).
- Full proofs, CIs, selected attackers and extended tasks: [supplement.tex](supplement.tex).
- Output location: [artifacts/papers/geoi_bq_20261116_v1/](../artifacts/papers/geoi_bq_20261116_v1/).
- Source receipts: [evidence_manifest.json](evidence_manifest.json).
- Submission decisions still requiring authors: [submission_notes.md](submission_notes.md).

## Content and review order

1. Introduction: transcript privacy, answer-level utility, incompatible output contracts, and three scoped contributions.
2. Related work: foundational Geo-I/prediction and recent surveyed methods; the main comparison uses three explicitly labeled adaptations.
3. Proposed method and analysis: system/model diagrams, stateful mechanism, four propositions, end-to-end and inference consequences, and conditional utility analysis.
4. Experiments and analysis: six-arm comparison; S1–S3/S9–S10 privacy; four-purpose utility; costs, native metrics, extended scenarios and adverse controls.
5. Conclusion: supported contributions, remaining conditions and follow-up evidence.

Review the central end-to-end theorem and its assumptions first, then the paper-adaptation fidelity and experimental information-access model. Full reference access, catalogue saturation, the stronger Γ0 control and the untested cap-exhausted regime remain explicit. Good observed scores do not replace the theory or prove general superiority.

The user confirmed author order: **Hanh Ngo-Minh** (first author), followed by **Phan Trong Nhan** (supervisor). Both PDFs and their metadata use this order. Affiliations, ORCIDs, contact details and the corresponding author remain to be supplied.

## Rebuild inside the research repository

The exporter requires the repository's normal scientific Python environment. Tectonic supplies the official Elsevier class and standard TeX packages.

    python experiments/export_english_paper_20261116.py --check
    python experiments/build_english_paper_20261116.py --tectonic /path/to/tectonic
    python experiments/validate_english_paper_20261116.py

The build authenticates the frozen evidence, exports tables, compiles both PDFs and creates a flat source archive. It never changes the method, benchmark, attacker or retained draws. The validation file records structural checks; visual QA is recorded only after the latest PDF pages are actually inspected.

## Build from the flat source archive

Extract the archive to an empty directory and run:

    tectonic main.tex
    tectonic supplement.tex

The archive contains every LaTeX input and figure required for these two builds. No benchmark data, Python environment or repository-relative TeX input is needed to compile the manuscript. Reproducing the scientific results is a separate repository workflow, identified by the evidence manifest; this archive is not a complete public research-data deposit.
