# English research manuscript

This is a compact, two-column English author-review paper, distilled from the current thesis and frozen GeoI-BQ v1 evidence. It provisionally targets a regular research article in the Journal of Information Security and Applications (JISA). It has not been submitted.

The main article's visual style follows the [original Geo-I paper](https://arxiv.org/pdf/1212.1984): US Letter, two columns, 9 pt Times text, a bold sans-serif title, separate author blocks, a left-column abstract, bold numbered headings and bold captions. This is a reference-inspired review layout, not an assertion of ACM publication or current JISA template compliance.

The main article is limited to **six pages, including references**. Its third figure places all five Q on the actual benchmark SUMO/OSM road map, with a GPS/Z zoom inset, and walks through protected reads/reuse, belief prediction/update, committed Q frames, budget reservations and local POI ranking. It replaces the privacy–cost bar chart whose values are already tabulated. [layout_reference.json](layout_reference.json) records the original visual reference and explicitly authorized illustration revisions; equations, benchmark values and all other protected scientific inputs are unchanged. Detailed specifications, full proofs, confidence intervals, selected attackers and extra metrics remain in the single-column supplement. Utility and acquisition costs share one main table.

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

The exporter requires the repository's normal scientific Python environment. Tectonic supplies standard TeX packages; the flat source package includes the custom visual style [geoipaper.sty](geoipaper.sty).

    python experiments/export_paper_walkthrough_20261117.py
    python experiments/export_english_paper_20261116.py --check
    python experiments/build_english_paper_20261116.py --tectonic /path/to/tectonic
    python experiments/validate_english_paper_20261116.py

The build authenticates the frozen evidence, recreates the retained-sample figure, exports tables, enforces the six-page main-article ceiling, compiles both PDFs and creates a flat source archive. It never changes the method, benchmark, attacker or retained draws. The validation file records structural checks; visual QA is recorded only after the latest PDF pages are actually inspected. The walkthrough data and its source hashes are retained in [the figure artifact](../artifacts/figures/paper_walkthrough_20261117_v1/).

## Build from the flat source archive

Extract the archive to an empty directory and run:

    tectonic main.tex
    tectonic supplement.tex

The archive contains every LaTeX input and figure required for these two builds. No benchmark data, Python environment or repository-relative TeX input is needed to compile the manuscript. Reproducing the scientific results is a separate repository workflow, identified by the evidence manifest; this archive is not a complete public research-data deposit.
