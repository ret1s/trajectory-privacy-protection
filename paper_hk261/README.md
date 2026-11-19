# GeoI-BQ: compact HK261 template version

This separate English-paper version uses the actual LLNCS 2.24 class and splncs04 bibliography style supplied in HK261_Hạnh_Trajectory_Privacy.zip. Both files are byte-identical to the supplied archive. The original paper/ version remains unchanged.

The user-requested two-column adaptation uses A4 pages, a 9 pt body, 8 pt tables/captions, smaller headings and compact spacing. The native template supplies the title/author treatment, abstract and keywords, numbered headings, running heads, theorem style, credits and alphabetically ordered bibliography. All overrides are isolated in hk261_compact.sty. This is the requested compact adaptation, rather than the original single-column template dimensions.

The title uses two balanced semantic lines, with unchanged wording and PDF metadata. Institution names and addresses have deliberate line breaks; run-in headings have consistent spacing/punctuation; numeric result columns are centered for comparison.

The five sections follow the supplied outline: Introduction, Related Work, Proposed Method, Experiments and Analysis, Summary. Framework/Workflow and Environment Settings/Empirical Experiments follow the original subsection structure; privacy/utility analysis and extended results remain within those sections. Contributions use the template's bullet list. Scientific paragraphs, equations, result values and the three current figures are inherited from the frozen English paper. No method or benchmark is rerun.

Both authors carry affiliation markers 1,2 and the two complete institution/address entries supplied by the user. The email block is {nmhanh.sdh241, nhanpt}@hcmut.edu.vn. Hanh Ngo-Minh is the sole corresponding author, explicitly confirmed by the user and marked with the template's envelope symbol. The original institutional Acknowledgments paragraph is restored for author review. No funding grant or competing-interest statement is invented. See author_info.json for metadata and template_alignment.md for the layout audit.

At the user's request, the current review PDF prints only Acknowledgments in the credits block. The data/code availability and generative-AI assistance sections are omitted from this version. The selected venue's disclosure requirements remain a separate submission decision; source/evidence manifests remain in the research repository.

The detailed proof/benchmark supplement remains available in [the original review package](../artifacts/papers/geoi_bq_20261116_v1/). It is a separate document with its own reference numbering.

## Build

    python experiments/build_hk261_paper_20261119.py --tectonic /path/to/tectonic
    python experiments/build_hk261_paper_20261119.py --check

The new PDF, flat editable LaTeX ZIP and validation receipts are written to artifacts/papers/geoi_bq_hk261_20261119_v1/. The source ZIP compiles independently with tectonic main.tex or a LaTeX/BibTeX workflow.
