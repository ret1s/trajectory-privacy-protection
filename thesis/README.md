# Graduation thesis

[`main.tex`](main.tex) is the single canonical LaTeX source for the evolving
graduation thesis. It no longer imports a dated supervisor-meeting file.

Build from this directory into an ignored scratch directory:

```bash
mkdir -p ../build/thesis
latexmk -xelatex -interaction=nonstopmode -halt-on-error \
  -outdir=../build/thesis main.tex
```

The reviewed deliverable is committed once at
`output/pdf/graduation_thesis.pdf`. LaTeX intermediate files and
`thesis/main.pdf` is ignored if a local editor creates it and is not a canonical
artifact.

The modular chapters and BibTeX database from Internship 2 are preserved at
`archive/internship_2/thesis/`; they are not imported by the current thesis.
Current benchmark data used by Chapter 5 remain under `outputs/`.
