# Active web applications

- `benchmark_app.py`: read-only viewer for integrity-checked benchmark
  artifacts, served at `http://127.0.0.1:5000/`.
- `simulator.py`: interactive replay of active thesis mechanisms, served at
  `http://127.0.0.1:5003/`.

The web layer must not launch experiments from HTTP requests. Ground truth and
evaluator-only routes remain explicitly separated from attacker-visible data.

Internship 2 Flask applications and templates are archived under
`archive/internship_2/apps/web/`.
