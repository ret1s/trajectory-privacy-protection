# Dummy-generation benchmark

This package is the canonical home of the thesis comparison harness. It keeps
algorithm execution separate from claims about how closely an implementation
matches a paper.

## What can be run now

| Method ID | Executable class | Output contract | Current evidence level |
|---|---|---|---|
| `transprotect_adaptation` | `TransProtectAdaptation` | replacement trajectory | paper adaptation |
| `anotherme_adaptation` | `AnotherMeAdaptation` | replacement trajectory | paper adaptation |
| `semantic_dummy_adaptation` | `SemanticDummyAdaptation` | real + `K-1` dummies | paper adaptation |
| `geo_i_anchored_dummy` | `GeoIAnchoredDummyTrajectories` | dummy-only trajectories | thesis candidate |

The first three rows are runnable, seeded, source-mapped adaptations. They are
not official or faithful reproductions and must not be cited as the papers'
reported results. `benchmark.registry.require_faithful_sota` and the CLI flag
`--require-faithful-sota` enforce that boundary.

## Audited upstream references

- **TransProtect / VehiTrack** — paper DOI
  `10.1145/3678717.3691211`; repository
  <https://github.com/sourabhy1797/VehiTrack>, audited revision
  `035684c6c666a9af7cbd9984d92300000eb65536`. The snapshot provides attack and
  evaluation scripts/data artifacts, but the current audit did not identify a
  drop-in Python training pipeline for the paper's GCN/transformer protection
  model on the Beijing/SUMO graph.
- **AnotherMe** — paper DOI `10.1109/TDSC.2023.3314200`; repository
  <https://github.com/fang-zhiyou/AnotherMe>, audited revision
  `0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f`. The research code includes
  virtual-trajectory generation and mobile applications, but parts of the
  workflow use AMap services, local paths and unreleased/locally arranged data.
  No source is copied into this repository.
- **Semantic-correlation dummy paths** — DOI
  `10.1007/s44443-026-00899-w`. No public implementation or trained artifact
  was identified; the paper describes an LSTM/attention model, POI/time
  embeddings and a 100x100 Beijing-grid transition pipeline.

The exact component mapping is machine-readable in
`benchmark/methods/paper_adaptations.py` and is serialized into every v3
benchmark artifact.

## Completion gates

A paper method is reportable as reproduced SOTA only when all of the following
hold:

1. its `ImplementationLevel` is `official` or `faithful_reimplementation`;
2. no source-mapped component is `missing`;
3. its source is sufficiently pinned for that implementation level;
4. validation evidence is recorded in the method card; and
5. the paper-specific dataset/preprocessing, attacker and metrics have been
   reproduced or any intentional deviation is stated and tested.

The present adaptations intentionally fail this gate. A normal run is useful
for integration, output-contract and visualization work; it is not a basis for
ranking privacy performance.

## Package boundaries

- `contracts.py`: evidence levels, source/component mapping and fail-closed
  claim gate.
- `engines/`: dependency-light algorithm engines. These contain the current
  local adaptations but no claim metadata.
- `methods/`: public benchmark adapters with stable IDs and method cards.
- `registry.py`: the single inventory consumed by runners and UIs.
- `core/sota_demo.py` and `core/thesis_demo.py`: deprecated import shims only.

Run the harness and dashboard from the repository root:

```bash
venv/bin/python -m experiments.run_dummy_benchmark --quick
venv/bin/python -m web.benchmark_app
```

The Flask dashboard is read-only and localhost-oriented. Before any deployment,
disable evaluator routes with `BENCHMARK_ENABLE_EVALUATOR_VIEW=False` in the app
configuration or place them behind authentication; they intentionally expose
ground truth for offline evaluation.
