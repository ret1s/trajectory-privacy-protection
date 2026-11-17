# GeoI-BQ English paper — author-review package

- [Main manuscript](manuscript.pdf): 6 pages in a compact two-column layout inspired by the original Geo-I paper, including references.
- [Supplementary material](supplement.pdf): 15 single-column pages, including detailed specifications/proofs, full CIs, selected attackers, costs and extended scenario results.
- [Flat editable LaTeX package](latex_sources.zip): both documents and all required source figures/inputs.
- [Build receipt](build.json), [structural/evidence checks](validation.json), and [visual QA receipt](visual_qa.json).

The paper is separate from the Vietnamese thesis. Its five main sections cover motivation, recent related work, the GeoI-BQ specification and analysis, controlled experiments, and conclusions. Figure 3 illustrates the retained multi-turn sample cache-046_day7 across 0--180 s using separate Q-overview and larger GPS/Z-detail maps, with a step ledger below. Same-time GPS/Z pairs share colors and labels; dashed links show reference displacement, not travel. Both maps retain actual SUMO/OSM geometry and explicit, different scales. Selection uses geometry for visual clarity, not attack/utility scores; the supplement tabulates the same sample. It replaces the redundant privacy–cost bar chart, preserving the six-page limit and all equations/benchmark results. The original visual-reference receipt records the authorized illustration/text revisions; the other 35 protected scientific files remain unchanged. Author order is Hanh Ngo-Minh first, followed by Phan Trong Nhan; affiliations and corresponding-author details remain pending.

The article preserves GeoI-BQ v1, the six-arm paper-adaptation comparison, all 120 retained TEST draws, negative controls and explicit limitations. No defense or attacker was rerun to write the paper. Full-reference access and the stronger public-base control remain central qualifications of the application/utility claim.

The source ZIP was extracted and compiled independently: both PDFs had identical extracted text to their repository builds. All 21 final pages were rendered with Poppler and visually inspected. Mathematical certificates, source pins and evidence binding were checked separately; visual QA is not a mathematical or deployment certificate.

See [paper/submission_notes.md](../../../paper/submission_notes.md) for the provisional JISA direction and author-dependent submission items. Nothing has been submitted to a journal.
