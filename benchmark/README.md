# Dummy-generation benchmark

This package contains the three paper-based comparators used by the thesis and
the proposed model. Algorithm execution and reproduction claims are kept
separate: a comparator can be complete enough for the common local benchmark
while still failing the stricter paper-equivalence gate.

## Executable methods

| Method ID | Class | Public output | Local benchmark status |
|---|---|---|---|
| `transprotect_adaptation` | `TransProtectAdaptation` | one replacement trajectory | executable clean-room core + explicit local predictor/target adapters |
| `anotherme_adaptation` | `AnotherMeAdaptation` | one virtual replacement trajectory | executable public VTGA + local virtual-user/routing adapters |
| `semantic_correlation_local_adaptation` | `SemanticCorrelationComparator` | real + `K-1` dummies per event | executable clean-room primitives + bounded local semantic adapters |
| `geo_i_anchored_dummy` | `GeoIAnchoredDummyTrajectories` | `K` dummy-only trajectories | thesis candidate |

The active API uses the explicit name `SemanticCorrelationComparator`. The
retired prototype remains only in `benchmark.methods.paper_adaptations` for
historical scripts and is not part of the benchmark registry.

### v4 API migration

The completed comparators intentionally replace the ambiguous prototype API:

- import `SemanticCorrelationComparator`, not `SemanticDummyAdaptation`;
- construct the local TransProtect comparator with
  `TransProtectAdaptation.from_road_network(...)`, or inject explicit learned
  probability and utility providers into its constructor;
- import engines by their precise names (`AnotherMeVTGAEngine`,
  `SemanticDummySelector`, `TransProtectEngine`).

Historical heuristic classes remain importable only from
`benchmark.methods.paper_adaptations` and
`benchmark.engines.paper_adaptations`; they are excluded from v4 artifacts.

### TransProtect

The implementation includes Equation 13 travel-cost loss, the
`h + alpha / loss` top-K rule, candidate-restricted Laplace sampling, the Geo-I
LP, and an optional Node2Vec-input → GCN → causal Transformer architecture. The
normal SUMO run trains a sparse Markov proxy on the other simulated vehicles
and holds out the evaluated vehicle. Utility uses an `N x M` target table, so
the common passenger graph does not require an `N x N` matrix. The proxy and
SUMO-derived target locations are labelled as adaptations in public metadata;
the target prior comes from normalized visits in the disjoint background set.
The runner exposes TransProtect's budget separately in `km^-1` (default 5,
matching the paper's reported sweep) and converts it to `m^-1` internally; it
does not reuse the thesis candidate's numerically different epsilon blindly.
Its own `K`, target count and alpha are separate CLI parameters. The JSON stores
setup/inference/end-to-end time and Equation-13 expected travel-cost loss;
VehiTrack EIE remains explicitly unavailable. Candidate selection depends on
the current secret, so the harness claims no end-to-end Geo-I theorem for the
whole TransProtect adaptation.

### AnotherMe

The implementation follows the authors' public VTGA: speed/mode extraction,
navigation filtering, roughly two-metre densification, Bezier turn smoothing,
three-second speed replay, discrete coordinate noise, and timestamps. A local
mapper relocates the origin/destination pattern and a local road router replaces
AMap. The raw variable-length VTGA output is retained before it is aligned to
the benchmark event grid.

### Semantic-correlation scheme

The implementation provides the paper's Beijing grid, transition equations,
published time weights, two-layer LSTM/attention inference shell, semantic
ranking and top-`K-1` selector. Because the authors did not publish weights or
AMap annotations, the runnable SUMO path uses bounded OSM road-context labels,
an empirical semantic predictor, a distance transition kernel, and an explicit
decay rule. Candidates have event-local IDs because the paper does not specify
cross-event track labels. A shared nearest-vertex catalog prevents a representation
fingerprint between continuous SUMO truth and graph-vertex dummies. Every
substitution is visible in the method card and public parameters.

## Reproduction boundary

All three comparators remain `paper_adaptation`, not `official` or
`faithful_reimplementation`. In particular:

- TransProtect lacks the authors' Node2Vec/GCN/Transformer trainer, checkpoint,
  complete configuration, split manifest, and table-level VehiTrack parity.
- AnotherMe lacks frozen AMap/GCJ02/POI responses, a canonical mapping between
  its Python/mobile variants, and paper-equivalent classifier/mobile tests.
- The semantic scheme lacks processed AMap semantics, trained weights, several
  model dimensions and functions, and an executable attacker posterior.

`benchmark.registry.require_faithful_sota` and the CLI flag
`--require-faithful-sota` therefore fail closed. Local benchmark numbers must
not be described as the papers' reproduced results.

Detailed primary-source audits and exact blockers are in
`docs/reproduction/{transprotect,anotherme,semantic_correlation}.md`. The
machine-readable component maps live beside each method in
`benchmark/methods/` and are serialized into the JSON artifact.

## Run

```bash
venv/bin/python -m tests.run_all
venv/bin/python -m experiments.run_dummy_benchmark --quick
venv/bin/python -m web.benchmark_app
```

The Flask dashboard is read-only and localhost-oriented. Evaluator routes
contain ground truth and are off by default in the WSGI factory; the local
`python -m web.benchmark_app` command explicitly enables them. Keep them
disabled or add authentication before any deployment.
