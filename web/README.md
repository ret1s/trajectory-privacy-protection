# Active web applications

- `thesis_demo.py`: current **GeoI-BQ v1** teaching replay and verified thesis
  results. Run `python3 -m web.thesis_demo --port 5011` and open
  `http://127.0.0.1:5011/`. Standard-library runtime; frozen offline bundle.
  [Presentation and verification guide](../docs/demos/2026-11-11_geoi_bq/README.md).
- `benchmark_app.py`: read-only viewer for integrity-checked benchmark
  artifacts, served at `http://127.0.0.1:5000/`.
- `simulator.py`: earlier experimental simulator, served at
  `http://127.0.0.1:5003/`.

The web layer must not launch experiments from HTTP requests. Ground truth and
evaluator-only routes remain explicitly separated from attacker-visible data.

Internship 2 Flask applications and templates are archived under
`archive/internship_2/apps/web/`.
