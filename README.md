# Trajectory Privacy Protection System

An interactive web application and research codebase for privacy protection in
location-based services (LBS) using Geo-Indistinguishability, developed as part
of a Master's thesis (see `docs/internship_2.pdf` for the full write-up).

## Features

- **Interactive Map Interface**: Draw routes directly on the map
- **Real-time Processing**: See privacy-protected trajectories instantly
- **Adjustable Privacy Parameters**: Control privacy level (epsilon) and quality of service radius
- **Visual Comparison**: View real vs. fake trajectories side-by-side
- **Privacy Metrics**: See average/maximum distances and QoS satisfaction rates

## Project Layout

```
core/           Core algorithms:
                  geo_indistinguishability.py, trajectory_privacy.py (internship-2 baseline)
                  road_network.py  — OSM graph → candidate lattice + KD-tree
                  mechanisms.py    — PlanarLaplace, BaselineThesis, REM, T-REM, SM-REM, PR-SM-REM (proposed)
data/           Mobility sources (SUMO demo + GeoLife validation data)
                  — see data/README.md for setup
evaluation/     metrics.py (QoS, realism, kNN-POI, Hausdorff/DTW) + attacks.py
                  (Bayesian point attack, HMM tracking attack)
experiments/    run_benchmark.py — mechanisms × ε on real GeoLife trajectories
web/            Flask apps: app.py (interactive demo), app_optimized.py,
                  simulator.py (LBS privacy simulator, user/LBS/attacker views)
legacy/         Archived, unmaintained prototypes (see legacy/README.md)
thesis/         LaTeX thesis draft (compile: cd thesis && latexmk -xelatex main.tex)
docs/           internship_2.pdf + research_notes.md (datasets/metrics/SOTA survey)
outputs/        Generated maps, benchmark_results.json
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
```

Every official experiment fails closed if the road graph does not match the
committed manifest, and writes a `msc-experiment-v1` provenance block (source
commit + dirty flag, graph SHA-256, RNG schema, selected record IDs, raw rows).

### Paper-inspired SOTA demo (prototype only)

The repository also contains an early executable demo of three recent
dummy-generation directions and the proposed thesis architecture. Its default
mobility source is a deterministic **SUMO** passenger simulation on the local
Beijing OpenStreetMap extract:

```bash
venv/bin/python -m pip install -r requirements-sumo.txt
venv/bin/python -m experiments.run_sota_demo --quick
```

It writes `outputs/sota_demo_results.json`, an interactive map at
`outputs/sota_demo_map.html`, and a four-panel static preview at
`outputs/sota_demo_preview.png`. The HTML embeds the pinned local OSM road
geometry, so geographic context remains visible when online raster tiles are
unavailable. The comparators are deliberately named `*Lite`:
they demonstrate the papers' high-level output contracts but are **not faithful
or official reproductions**, and their numbers must not be presented as SOTA
results. Replacement, real-plus-dummies, and dummy-only outputs are reported in
separate tracks. GeoLife is retained only as an explicit optional real-data
validation source (`--mobility-source geolife`); the SUMO path never silently
falls back to it. See
[`docs/supervisor_meeting/2026-09-05/sota_demo.md`](docs/supervisor_meeting/2026-09-05/sota_demo.md)
for the exact scope and limitations.

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
