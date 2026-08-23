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
data/           Dataset loaders (GeoLife) — see data/README.md for downloads
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

## Benchmark & simulator

```bash
python3 -m experiments.run_benchmark          # 20 GeoLife trajs × 3 ε × 6 mechanisms
python3 -m experiments.run_benchmark --quick  # smoke test
python3 -m experiments.run_averaging          # home-inference / averaging attack (S4)
python3 -m web.simulator                      # http://localhost:5003 — replay real
                                             # trajectories with user/LBS/attacker views
                                             # and a live k-NN POI use case
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the web application

There are two Flask apps, both built on the same core algorithm:

- **`web/app.py`** (port 5002) — the reference demo, with live console
  logging over SocketIO. Uses `core/trajectory_privacy.py` directly.
- **`web/app_optimized.py`** (port 5001) — an experimental variant that adds
  disk/in-memory caching of downloaded road networks (`road_network_cache/`)
  for faster repeated runs, using a simplified, standalone
  `web/trajectory_privacy_optimized.py` class.

Run either one from the project root:

```bash
python -m web.app             # http://localhost:5002
python -m web.app_optimized   # http://localhost:5001
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

- `core/trajectory_privacy.py`: Core privacy protection algorithm (Geo-I + road/building
  constraints + QoS + trajectory continuity — matches Algorithm 1 in the thesis)
- `core/geo_indistinguishability.py`: The ε-Geo-Indistinguishability noise mechanism
  (polar Laplace distribution)

## Privacy Algorithm

The system uses Geo-Indistinguishability to provide formal privacy guarantees while maintaining trajectory realism by:
- Following actual road networks
- Avoiding invalid locations (buildings, water bodies)
- Maintaining continuous trajectories
- Satisfying quality of service constraints

## CLI Demo

```bash
python demo_trajectory_privacy.py
```

Generates a sample trajectory around San Francisco and saves the visualization
as a timestamped HTML file under `outputs/`.
