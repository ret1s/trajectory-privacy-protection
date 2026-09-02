# Verification — three source-mapped benchmark comparators

Date: 2026-09-02

Verified source commit: `946f587fd94b5be0e1dc004db1131c444d2c1e11`

Canonical artifact schema: `msc-dummy-benchmark-v4`

## Verdict

The benchmark now has three complete, executable local comparator pipelines:

1. `transprotect_adaptation`;
2. `anotherme_adaptation`;
3. `semantic_correlation_local_adaptation`.

All three run end to end on the same held-out SUMO trajectory and the exact same
passenger-only directed SUMO road graph. Their adapters, source mappings,
method cards, paper-specific diagnostics, tests, provenance, map layers, and web
views are implemented.

This is **ready as a local integration/ablation benchmark**, but is **not evidence
of faithful reproduction of the papers' reported SOTA results**. Each comparator
correctly remains `implementation_level = paper_adaptation`, and
`--require-faithful-sota` fails closed while paper assets/parity evidence are
missing.

## Source basis and implemented scope

### TransProtect

- Primary paper: [Yadav et al., ACM SIGSPATIAL 2024](https://doi.org/10.1145/3678717.3691211)
  and [arXiv manuscript](https://arxiv.org/abs/2409.09495).
- Attack reference implementation: [VehiTrack](https://github.com/sourabhy1797/VehiTrack),
  pinned by the method documentation to commit
  `035684c6c666a9af7cbd9984d92300000eb65536`.
- Implemented paper-facing primitives:
  - Equation (13) probability/utility candidate score;
  - deterministic top-K candidate selection;
  - restricted planar-Laplace sampler;
  - finite-domain Geo-I linear-program sampler;
  - GCN + masked-Transformer architecture/training shell;
  - expected travel-cost loss and explicit epsilon units.
- Local executable adapter:
  - causal empirical Markov predictor trained only on 19 disjoint background
    SUMO vehicles;
  - empirical visit prior for utility targets;
  - directed shortest-path travel cost on the common SUMO graph;
  - `5 km^-1` converted explicitly to `0.005 m^-1`.
- Deliberately unavailable:
  - paper-equivalent learned checkpoint/training recipe;
  - paper Rome/San Francisco split and table parity;
  - VehiTrack end-to-end inference-error parity.
- Privacy-claim boundary: only candidate-restricted sampling is represented.
  Because the top-K support depends on the current secret, the implementation
  does not claim end-to-end Geo-I.

### AnotherMe

- Primary paper: [Li et al., IEEE TDSC](https://doi.org/10.1109/TDSC.2023.3314200).
- Author repository: [fang-zhiyou/AnotherMe](https://github.com/fang-zhiyou/AnotherMe),
  pinned to commit `0eda877b7328ee1c0b3e0cf9a9bb9d48cb24323f`.
- Implemented VTGA flow:
  - WGS84 ellipsoidal speed computation and mode selection;
  - virtual endpoint generation with the source distance pattern;
  - directed route query and six-metre route-point filtering;
  - approximately two-metre route densification;
  - 70--110 degree Bezier smoothing;
  - source-style three-second speed-profile replay and discrete noise;
  - timestamped virtual trajectory generation.
- Local executable adapters replace AMap/GCJ-02/mobile services with the common
  WGS84 SUMO graph, a local endpoint mapper, and local directed routing.
- The raw virtual trajectory retains its three-second grid. Only the public
  benchmark replacement is aligned to the real event grid, so aligned metrics
  are not presented as raw VTGA temporal parity.
- Deliberately unavailable: AMap routing/POI parity, GCJ-02 conversion parity,
  Android sensing, and the original transport classifier/service environment.

### Semantic-correlation dummy method (2026)

- Primary paper: [Liu, Peng, and Zhou, 2026](https://doi.org/10.1007/s44443-026-00899-w).
- No public official implementation or learned checkpoint was identified; the
  paper states data availability on request.
- Implemented source-mapped components:
  - 100-by-100 spatial grid primitive;
  - historical location statistics, transition/time weighting, and feature
    dimensions described by the paper;
  - two-layer LSTM + attention + fully connected inference shell;
  - semantic/history/transition candidate selector;
  - ASR and DER metric primitives.
- Local executable adapter:
  - bounded empirical predictor instead of unavailable trained weights;
  - OSM/SUMO road class as the locally available semantic label;
  - distance-based transition kernel;
  - explicit K-feasible order-statistic threshold.
- Every event publishes one real plus `K-1` dummies using opaque, event-local
  candidate IDs. All candidates, including the real member, share the same
  nearest-vertex representation, eliminating the previous representation
  fingerprint.
- Deliberately unavailable: paper AMap semantic annotations, calibrated LSP
  posterior/ASR, effectiveness labels/DER, weights, and reported-table parity.

Detailed mappings are in:

- `docs/reproduction/transprotect.md`;
- `docs/reproduction/anotherme.md`;
- `docs/reproduction/semantic_correlation.md`.

## Common benchmark controls

- Mobility generator: Eclipse SUMO `1.27.1`.
- Scenario: controlled Beijing passenger smoke scenario.
- Evaluation records: one held-out vehicle, eight sampled events.
- Background training set: 19 other vehicles, 133 observed transitions.
- Candidate/mobility graph: the exact `.net.xml` used by the FCD run, converted
  without adding reverse one-way edges.
- Graph size: 4,892 vertices and 9,138 directed edges.
- Canonical semantic graph hash:
  `41c40b6e2ad63b57ccecfa8ebb89b97475cd12df54e9a340a11dd8d94b21f960`.
- Semantic-hash schema: `sumo-passenger-road-graph-v1`.
- Raw network-file hash:
  `bfb971458c62f83e8fda27a2ad5cdd24a429dfc46a01908ec125e19d41177812`.
  The raw hash may change when SUMO rewrites build-time/path comments; the
  semantic hash is the stable graph identity.
- Source provenance in the generated JSON points to commit `946f587...` and
  records `source_dirty_before_run = false`.

No separately built multimodal OSMnx graph is mixed into this canonical run.
Continuous FCD lane coordinates and graph candidates may differ geometrically,
but they originate from the same SUMO network.

## Canonical smoke result

The values below verify execution and data-flow integrity. They are not a paper
ranking and must not be compared across output contracts.

| Track | Method | Key local result | End-to-end runtime |
|---|---|---:|---:|
| Replacement trajectory | TransProtect | mean displacement 147.24 m; QoS@200 m 0.625; on-road 1.0 | 50.22 ms |
| Replacement trajectory | AnotherMe | mean displacement 2,088.55 m; QoS@200 m 0.0; on-road 1.0 | 19.94 ms |
| Real + K-1 candidates | Semantic correlation | K=4; mean dummy distance 385.36 m; on-road 1.0 | 9.09 ms |
| Thesis dummy-only reference | Geo-I anchored dummy | K=4; mean nearest output distance 82.17 m; on-road 1.0 | 1.71 ms |

Additional checks:

- TransProtect expected travel-cost loss: 62.584 m.
- TransProtect VehiTrack EIE remains `null` with an explicit unavailable status.
- Semantic candidate inclusion rate: 1.0, as required by its real-plus-dummies
  output contract.
- Semantic vertex-offset fingerprint attack success: 0.25, equal to the uniform
  K=4 guessing baseline of 0.25.
- All replacement and candidate outputs in this smoke run are on the common road
  graph; observed speed-violation rate is zero where stable tracks exist.

## Output-contract separation

The benchmark intentionally creates three result tracks:

- replacement trajectory: TransProtect and AnotherMe;
- real plus `K-1` candidates per event: semantic-correlation method;
- dummy-only stable tracks: the thesis candidate.

Metrics are computed only when meaningful for the public output. For example,
track DTW is `null` for event-local semantic candidate sets, while real-member
inclusion is `null` for the thesis dummy-only output. A single cross-contract
leaderboard is explicitly disabled.

## Privacy and web exposure verification

- Public record IDs are opaque and do not contain the SUMO vehicle ID.
- SUMO vehicle ID, route edges, lane IDs, ground-truth trajectory, paper metrics,
  and runtime diagnostics remain evaluator-only.
- AnotherMe transport mode and raw sample count remain evaluator-only.
- TransProtect's count of secret-dependent forced-real-membership events was
  removed from public parameters and retained only in evaluator paper metrics.
- Public TransProtect metadata exposes the tie policy, not whether it fired.
- Evaluator endpoints return HTTP 404 by default and become available only with
  `BENCHMARK_ENABLE_EVALUATOR_VIEW = true`.
- Canonical endpoint checks:
  - public `/`, `/api/benchmark`, and TransProtect attacker view: HTTP 200;
  - public TransProtect evaluation endpoint: HTTP 404;
  - explicitly enabled evaluation, map, and preview endpoints: HTTP 200.

Security caveat: the evaluator HTML map uses Folium/Leaflet assets loaded from a
CDN. It is acceptable for this synthetic SUMO artifact. Before displaying a
private GeoLife/evaluator trajectory in a networked browser, vendor the assets
locally and apply a strict CSP so evaluator truth is not exposed to third-party
asset hosts.

## Verification commands and results

```text
venv/bin/python -m tests.run_all
=> 118 passed, 0 failed

venv/bin/python -m compileall benchmark core data experiments tests web
=> passed

node --check web/static/benchmark/dashboard.js
=> passed

git diff --check
=> passed

venv/bin/python -m experiments.run_dummy_benchmark --quick
=> completed; JSON, interactive map, and static preview regenerated
```

The final static preview was inspected visually. It contains an embedded local
road overlay in all four panels and no longer shows trajectories on an empty
background.

## Remaining work before thesis-grade SOTA claims

1. Obtain or reconstruct paper-equivalent assets for each comparator, then
   reproduce at least one reported table/figure within a declared tolerance.
2. Implement the final thesis threat scenarios (S1--S7) rather than relying on
   the controlled single-trajectory smoke scenario.
3. Run multiple seeds, users, routes, privacy budgets, and K values; report
   confidence intervals rather than one example trajectory.
4. Implement/calibrate the selected attacks, especially VehiTrack-style
   inference and the semantic paper's LSP attack, using a leakage-safe split.
5. Decide a common scientific comparison question before comparing methods with
   different output contracts. Do not turn the current three tracks into a
   single scalar leaderboard.
6. Vendor evaluator-map assets before using private trajectories.

Until items 1 and 4 are complete, publications and thesis text must use
“source-mapped clean-room adaptation/local comparator”, never “faithful SOTA
reproduction”.
