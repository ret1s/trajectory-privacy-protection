# Active core

- `mechanisms.py`: Planar comparator, Internship-2 surrogate, REM, T-REM,
  SM-REM, and PR-SM-REM.
- `road_network.py`: projected OSM graph and fixed candidate support.
- `demo_protocol.py`: public/evaluator data contract currently shared by the
  benchmark.
- `sota_demo.py` and `thesis_demo.py`: small compatibility shims for pre-v4
  callers; new code should import canonical classes from `benchmark.methods`.

`core/__init__.py` intentionally performs no eager imports. In particular, the
archived GIS-heavy Internship 2 pipeline must not load when an experiment only
needs the active protocol or mechanisms.
