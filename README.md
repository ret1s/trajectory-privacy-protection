# Trajectory Privacy Protection System

An interactive web application and research codebase for privacy protection in
location-based services (LBS) using Geo-Indistinguishability, developed as part
of the thesis **“Bảo vệ tính riêng tư về quỹ đạo cho người dùng dịch vụ dựa
trên vị trí.”** The evolving final LaTeX source is
`docs/supervisor_meeting/2026-09-05/report.tex`; `thesis/main.tex` is its stable
build entry point. `docs/internship_2.pdf` is retained as prior-stage context.

## Features

- **Interactive Map Interface**: Draw routes directly on the map
- **Real-time Processing**: See privacy-protected trajectories instantly
- **Adjustable Privacy Parameters**: Control privacy level (epsilon) and quality of service radius
- **Visual Comparison**: View real vs. fake trajectories side-by-side
- **Privacy Metrics**: See average/maximum distances and QoS satisfaction rates

## Project Layout

```
benchmark/      Benchmark contracts, method cards, algorithm engines and
                  source-mapped paper adaptations
core/           Formal mechanisms and shared public/evaluator protocol:
                  geo_indistinguishability.py, trajectory_privacy.py (internship-2 baseline)
                  road_network.py  — OSM graph → candidate lattice + KD-tree
                  mechanisms.py    — PlanarLaplace, BaselineThesis, REM, T-REM, SM-REM, PR-SM-REM (proposed)
data/           Mobility sources (SUMO demo + GeoLife validation data)
                  — see data/README.md for setup
evaluation/     metrics.py (QoS, realism, kNN-POI, Hausdorff/DTW) + attacks.py
                  (Bayesian point attack, HMM tracking attack)
experiments/    run_benchmark.py — internship-2 mechanisms × ε on GeoLife
                  run_dummy_benchmark.py — SUMO dummy-generation benchmark
web/            benchmark_app.py — read-only benchmark dashboard;
                  simulator.py — earlier REM-family experiment viewer
legacy/         Archived, unmaintained prototypes (see legacy/README.md)
thesis/         Stable LaTeX build entry point + archived internship-2 chapters
docs/           internship_2.pdf + research_notes.md (datasets/metrics/SOTA survey)
outputs/        Generated evaluator artifacts (JSON, map and preview)
demo_trajectory_privacy.py   CLI demo (no web server needed)
cache/, road_network_cache/  OSM data caches (gitignored, rebuilt automatically)
```

## Proposed mechanisms (thesis contribution)

- **REM** (Road-network Exponential Mechanism): exponential mechanism over the
  road-graph vertices, `P(v|x) ∝ exp(−ε/2·d(x,v))` — formal ε-Geo-I with every
  output on the road network by construction (Euclidean sibling of
  Geo-Graph-Indistinguishability, Takagi et al. 2019).
- **T-REM**: REM + reachability weighting w.r.t. the previously *released*
  point only (public info ⇒ per-release guarantee unchanged); reduces the
  speed-implausibility signal that velocity-linkage attacks use.
- **SM-REM**: T-REM + exact memoization keyed to a fixed public grid, sampling
  from the public cell representative — a static repeated location returns one
  cached release, so arithmetic averaging (home-inference) gains nothing.
  Guarantee is narrow (static repeat only); the revisit pattern leaks.
- **PR-SM-REM**: replaces the exact reuse decision with a noisy-threshold test
  (predictive-mechanism style) for a finite per-release
  `(ε_test+ε_release)`-Geo-I bound instead of SM-REM's ∞-ratio revisit leak.

All ε-Geo-I claims are for the **ideal real-arithmetic kernel**; the float
sampler is a numerical approximation (finite-precision zero-support — see
`tests/test_sampler_support.py`). Attacker MLE/HMM are REM-emission **proxies**
(exact for REM only). See `docs/reviews/` for the independent verification
rounds and `docs/reviews/response_*.md` for the finding-by-finding responses.

## Thesis artifact (the formally analyzed mechanisms)

These are the commands that generate the results in the thesis (Chapter 5). They
use `core/mechanisms.py` (REM / T-REM / SM-REM / PR-SM-REM), NOT the legacy demo
pipeline below.

```bash
python3 -m tests.run_all                      # canonical test suite (no pytest needed)
python3 -m pytest -q tests                    # same tests, if pytest is installed (see requirements-dev.txt)
python3 -m data.build_beijing_graph           # (re)build the pinned road graph + manifest
python3 -m experiments.run_benchmark          # 20 GeoLife trajs × 3 ε × 6 mechanisms → outputs/benchmark_results.json
python3 -m experiments.run_benchmark --quick  # smoke test
python3 -m experiments.run_averaging_multi    # multi-home averaging / home-inference study (S4) → outputs/averaging_multi_results.json
python3 -m web.simulator                      # http://localhost:5003 — replay real trajectories
                                              # (user/LBS/attacker views + a synthetic-POI k-NN use case)
cd thesis && latexmk -xelatex main.tex         # build the evolving final thesis
```

Every official experiment fails closed if the road graph does not match the
committed manifest, and writes a `msc-experiment-v1` provenance block (source
commit + dirty flag, graph SHA-256, RNG schema, selected record IDs, raw rows).

### Dummy-generation benchmark and Python dashboard

The canonical harness runs three recent dummy-generation **paper adaptations**
and the evolving thesis candidate on one truth-separated protocol. Its default
mobility source is a deterministic **SUMO** passenger simulation over the local
Beijing OpenStreetMap extract:

```bash
venv/bin/python -m pip install -r requirements-sumo.txt
venv/bin/python -m experiments.run_dummy_benchmark --quick
venv/bin/python -m web.benchmark_app
# open http://127.0.0.1:5000/
```

It writes `outputs/dummy_benchmark_results.json`, an evaluator-only map at
`outputs/dummy_benchmark_map.html`, and a static preview at
`outputs/dummy_benchmark_preview.png`. The Flask app reads these artifacts; it
does not execute experiments from an HTTP request. Its attacker endpoint omits
ground truth, while evaluator routes are explicitly labelled, SHA-256 checked,
and intended for localhost use only.

The canonical classes are `TransProtectAdaptation`, `AnotherMeAdaptation`,
`SemanticDummyAdaptation`, and `GeoIAnchoredDummyTrajectories`. Each artifact
contains a method card that maps paper components to implemented, adapted or
missing code and pins any audited upstream revision. These are stable runnable
adaptations, **not yet faithful SOTA reproductions**. In particular, learned
pipelines, original preprocessing/data artifacts and calibrated paper attackers
are still missing. The following command therefore fails closed by design:

```bash
venv/bin/python -m experiments.run_dummy_benchmark --require-faithful-sota
```

Replacement, real-plus-dummies, and dummy-only outputs remain separate tracks;
their raw metrics must not form a cross-contract leaderboard. GeoLife is only
an explicit optional validation source (`--mobility-source geolife`); SUMO never
silently falls back to it. See
[`benchmark/README.md`](benchmark/README.md) for the source audit and completion
gates, and [`docs/supervisor_meeting/2026-09-05/sota_demo.md`](docs/supervisor_meeting/2026-09-05/sota_demo.md)
for the current experiment boundary.

### Claim registry — what is proven, and where (read before quoting any result)

| Mechanism | Code | Guarantee (ideal kernel) | Executable / evaluation caveat |
|---|---|---|---|
| REM | `RoadExponential` | ε-Geo-I (Euclid) over the fixed public vertex set | float Gumbel-max is a finite-precision approximation (`tests/test_sampler_support.py`) |
| T-REM | `TemporalRoadExponential` | same ε-Geo-I per release (reachability weight is public) | same finite-precision caveat |
| SM-REM | `StayMemoizedREM` | static same-cell repeat, empty initial cache → one release's ε-Geo-I (cell level) | revisit pattern leaks (∞ ratio); NOT a trajectory theorem; corner boundary ratio ≤ e^{ε·√2·g} |
| PR-SM-REM | `PrivateReuseSMREM` | per-step (ε_test+ε_release)-Geo-I | anti-averaging is **finite-horizon only** (both branches probabilistic); no w-event manager |

All attacker columns (Bayes/HMM/averaging MLE) use a **REM-emission proxy** —
exact for REM only, an **upper bound** for the others, so they are diagnostics
and are **not** used to rank privacy between mechanisms. See `docs/reviews/` for
the full verification history.

## Installation

```bash
pip install -r requirements.txt
```

## Legacy demo apps (NOT the thesis method — no valid formal guarantee)

> ⚠️ These two Flask apps run the **internship-2 pipeline**
> (`core/trajectory_privacy.py`): planar Laplace capped at a QoS radius +
> reject-in-building + snap. That cap/reject step is conditioned on the *true*
> location, so its nominal ε is **not a valid Geo-I guarantee** (thesis §4,
> Prop. `prop:cap`/`prop:reject`). They are kept as an interactive
> **visualization/surrogate** only. For the formally analyzed mechanisms and the
> thesis results, use the **Thesis artifact** commands above (`web.simulator`,
> `experiments.*`), not these.

- **`web/app.py`** (port 5002) — reference visualization demo, live console
  logging over SocketIO.
- **`web/app_optimized.py`** (port 5001) — same demo with disk/in-memory road-network
  caching (`road_network_cache/`), via a simplified standalone class.

```bash
python -m web.app             # http://localhost:5002  (legacy demo)
python -m web.app_optimized   # http://localhost:5001  (legacy demo, cached)
```

## Usage

1. **Draw a Route**: Click on the map to create waypoints for your route (minimum 2 points)
2. **Adjust Parameters**:
   - **Privacy Parameter (ε)**: Lower values provide stronger privacy (default: 0.1)
   - **QoS Radius**: Maximum allowed distance from real location in meters (default: 175m)
3. **Process Route**: Click "Process Route" to generate the privacy-protected trajectory
4. **View Results**:
   - Real trajectory (snapped to roads) vs. fake/obfuscated trajectory
   - Metrics panel shows privacy performance

## Core Components

- `core/mechanisms.py`: **the thesis mechanisms** (REM / T-REM / SM-REM / PR-SM-REM)
  — the formally analyzed contribution; see the claim registry above.
- `core/trajectory_privacy.py`: the **legacy internship-2 pipeline** (Geo-I noise
  capped at QoS + reject-in-building + snap). This is the pipeline the thesis §4
  *critiques*: its cap/reject step is conditioned on the true location, so its
  nominal ε is not a valid guarantee. Used only by the legacy demo apps.
- `core/geo_indistinguishability.py`: the polar-Laplace noise used by that legacy
  pipeline.

## What the legacy demo does (and does not) guarantee

The legacy demo pipeline follows roads, avoids buildings/water, keeps trajectories
continuous, and satisfies a QoS radius — but the QoS cap + building-rejection are
data-dependent, so it does **not** carry a valid ε-Geo-I guarantee. For the valid
guarantees, use the thesis mechanisms (`core/mechanisms.py`) and the claim registry
above.

## CLI Demo (legacy)

```bash
python demo_trajectory_privacy.py
```

Generates a sample trajectory around San Francisco and saves the visualization
as a timestamped HTML file under `outputs/`.
