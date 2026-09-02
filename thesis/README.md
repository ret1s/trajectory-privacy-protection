# Thesis source boundary

`main.tex` is the stable build entry point for the evolving final thesis. It
loads the self-contained canonical source at
`docs/supervisor_meeting/2026-09-05/report.tex` so the meeting milestone and
final document cannot drift.

Compile with:

```bash
cd thesis
latexmk -xelatex main.tex
```

The files under `chapters/` and `refs.bib` are the preserved modular source of
the Internship 2 / August draft. They are **not imported by `main.tex`** and
must not be edited as though they were current thesis chapters. They may be
migrated into the canonical source later, section by section, after their
claims are reconciled with the final threat model and benchmark.
