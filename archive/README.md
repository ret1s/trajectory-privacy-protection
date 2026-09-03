# Archive

This directory keeps provenance without presenting historical code as active.
Nothing under `archive/` may be imported by the current thesis mechanisms,
benchmarks, web applications, or tests.

```text
internship_2/
  report.pdf              Final Internship 2 report
  thesis/                 Old modular LaTeX chapters and bibliography
  supporting_documents/  Earlier rendered threat/system notes
  apps/                   Legacy CLI and Flask implementations
  demo_outputs/           Timestamped maps produced by those applications
prototypes/
  sota_demo/              Superseded pre-benchmark SOTA demo artifacts
```

The archived Python applications are unmaintained. If inspection requires
running them, install `requirements-legacy.txt` and invoke modules from the
repository root. Their results must not be mixed with the current benchmark.
