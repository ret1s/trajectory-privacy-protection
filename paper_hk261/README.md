# GeoI-BQ: compact HK261 template version

This separate English-paper version uses the actual LLNCS 2.24 class and splncs04 bibliography style supplied in HK261_Hạnh_Trajectory_Privacy.zip. Both files are byte-identical to the supplied archive. The original paper/ version remains unchanged.

The user-requested two-column adaptation uses A4 pages, a 9 pt body, 8 pt tables/captions, smaller headings and compact spacing. The native template supplies the title/author treatment, abstract and keywords, numbered headings, running heads, theorem style, credits and alphabetically ordered bibliography. All overrides are isolated in hk261_compact.sty. This is the requested compact adaptation, rather than the original single-column template dimensions.

The five sections follow the supplied outline: Introduction, Related Work, Proposed Method, Experiments and Analysis, Summary. Scientific paragraphs, equations, result tables and the three current figures are inherited from the frozen English paper. Section headings and bibliography presentation change; no method or benchmark is rerun.

Only the confirmed author names are asserted: Hanh Ngo-Minh and Phan Trong Nhan. Template institution, email, corresponding-author and support placeholders are not treated as verified author information. Those fields remain for the authors to complete.

The detailed proof/benchmark supplement remains available in [the original review package](../artifacts/papers/geoi_bq_20261116_v1/). It is a separate document with its own reference numbering.

## Build

    python experiments/build_hk261_paper_20261119.py --tectonic /path/to/tectonic
    python experiments/build_hk261_paper_20261119.py --check

The new PDF, flat editable LaTeX ZIP and validation receipts are written to artifacts/papers/geoi_bq_hk261_20261119_v1/. The source ZIP compiles independently with tectonic main.tex or a LaTeX/BibTeX workflow.
