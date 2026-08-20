# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Master's thesis research project implementing privacy protection for location-based services (LBS) using Geo-Indistinguishability. The project provides a novel algorithm that generates fake trajectories while maintaining quality of service (QoS) constraints and following real-world geographic constraints like roads and buildings. The full write-up (theory, related work, algorithm, experiments) is in `docs/internship_2.pdf`.

## Repository Layout

```
core/           geo_indistinguishability.py, trajectory_privacy.py (internship-2 baseline),
                road_network.py (OSM graph wrapper), mechanisms.py (PlanarLaplace,
                BaselineThesis, REM, T-REM — the proposed mechanisms)
data/           geolife.py loader; data/raw/ holds GeoLife + Beijing graph (gitignored,
                download instructions in data/README.md)
evaluation/     metrics.py (Q_loss, QoS, on-road, speed-violation, kNN-POI recall,
                Hausdorff, DTW) and attacks.py (BayesianPointAttack, HMMTrackingAttack)
experiments/    run_benchmark.py (mechanisms × ε on real GeoLife data →
                outputs/benchmark_results.json)
web/            app.py (interactive demo, port 5002), app_optimized.py (port 5001),
                simulator.py (LBS privacy simulator, port 5003) + templates/
legacy/         Archived prototypes — see legacy/README.md
docs/           internship_2.pdf + research_notes.md (dataset/metric/SOTA survey and
                benchmark analysis — read this first for research context)
outputs/        Generated maps, benchmark_results.json
demo_trajectory_privacy.py   CLI demo entrypoint (stays at repo root)
cache/, road_network_cache/  OSM data caches (gitignored, rebuilt on demand)
```

## Research state (updated 2026-08-20)

The thesis direction: replace the internship-2 pipeline (planar Laplace capped at QoS
+ reject-in-building + snap — whose cap/reject steps break the formal Geo-I guarantee)
with road-network-native mechanisms: REM (exponential mechanism over road vertices,
formal ε-Geo-I, outputs on-road by construction) and T-REM (adds reachability weighting
conditioned only on previously released points — guarantee unchanged, closes
velocity-linkage attacks). Benchmarked on 20 real GeoLife trajectories against
Planar Laplace and the baseline, with Bayesian + HMM tracking attacks.
Full rationale, citations, and results table: docs/research_notes.md.

## Common Development Commands

### Running the Interactive Web Application
Run from the project root (so the `core` package resolves):
```bash
python -m web.app             # reference demo, http://localhost:5002
python -m web.app_optimized   # cached-road-network variant, http://localhost:5001
```

### Running the Command-Line Demo
```bash
python demo_trajectory_privacy.py
```
This generates a privacy-protected trajectory visualization as a timestamped HTML file under `outputs/`.

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Running Tests
No test framework is currently configured. When implementing tests, check for pytest or unittest conventions first.

### Linting
No linting configuration found. If implementing linting, check Python conventions (flake8, pylint, black).

## Code Architecture

### Core Components

1. **`core/trajectory_privacy.py`** — Main implementation containing:
   - `TrajectoryPrivacy` class: Core privacy protection algorithm
   - Spatial constraint handling using OpenStreetMap data (buildings, water, rivers)
   - Alternative-road point selection and road network snapping for realistic trajectories
   - Trajectory continuity smoothing and privacy metric evaluation
   - This matches "Algorithm 1" described in `docs/internship_2.pdf` (Chapter 4)

2. **`core/geo_indistinguishability.py`** — Differential privacy mechanism:
   - Implements ε-Geo-Indistinguishability via the polar Laplace distribution
   - Noise generation based on the epsilon parameter
   - QoS radius (`delta`) enforcement by capping the noise radius

3. **`web/app.py`** — Reference Flask web application (port 5002):
   - Interactive map interface for drawing routes, with live SocketIO console logging
   - Wraps `core.trajectory_privacy.TrajectoryPrivacy` with progress reporting
   - Renders `web/templates/index.html`

4. **`web/app_optimized.py`** + **`web/trajectory_privacy_optimized.py`** — Experimental caching
   variant (port 5001):
   - Caches downloaded road networks to disk/memory (`road_network_cache/`) to avoid
     re-hitting OpenStreetMap on repeated runs over the same area
   - Uses a simplified, standalone privacy class (no alternative-road logic)
   - Renders `web/templates/index_optimized.html`

5. **`demo_trajectory_privacy.py`** — Command-line demonstration:
   - `generate_realistic_trajectory()`: Creates test trajectories following roads
   - Visualization using Folium interactive maps
   - Privacy metrics evaluation

6. **`legacy/`** — Earlier prototypes (`simple_app.py`, `test_app.py`), kept for reference
   only; not part of the active codebase. See `legacy/README.md`.

### Key Parameters

- **epsilon**: Privacy parameter (lower = stronger privacy, typical: 0.1)
- **qos_radius**: Quality of service radius in meters (typical: 150-200m)

### External Data Sources

- **OpenStreetMap (OSM)**: Used for road networks and spatial constraints
- **cache/**: Directory storing cached OSM data for performance (gitignored)
- **road_network_cache/**: Disk cache used only by `web/app_optimized.py` (gitignored)

### Output Files

- **outputs/trajectory_privacy_map_*.html**: Interactive map visualizations
- **outputs/report_samples/**: Directory containing experimental results referenced in the thesis

## Important Technical Details

1. **Coordinate System**: Uses (latitude, longitude) tuples throughout
2. **Distance Calculations**: Haversine formula for geographic distances
3. **Road Network**: OSMnx library for road network data and routing
4. **Visualization**: Folium for interactive HTML maps
5. **Privacy Guarantee**: Implements formal differential privacy through geo-indistinguishability

## Research Context

This implementation is based on academic research combining differential privacy theory with practical geographic constraints. The algorithm ensures privacy while maintaining trajectory realism by:
- Following actual road networks
- Avoiding invalid locations (buildings, water)
- Maintaining continuous trajectories
- Satisfying QoS constraints

Related work surveyed in the thesis (see `docs/internship_2.pdf`, Chapter 3) includes PPST-tree
(k-anonymity over predicted future locations), the KUR-Algorithm (context-aware movement
prediction + privacy-enhancing obfuscation), and PTPPM (Geo-I + Distortion Privacy under
temporal correlations) — useful context if extending this algorithm or picking benchmarks,
per the open TODOs in `Notes.md`.
