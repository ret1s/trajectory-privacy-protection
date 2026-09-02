# Semantic-correlation 2026: clean-room reproduction status

## Source and claim boundary

Primary source: H. Liu, H. Peng, and N. Zhou, “A dummy-based location
privacy protection scheme with semantic correlation of moving paths,” *Journal
of King Saud University Computer and Information Sciences*, 38:478, 2026,
[DOI 10.1007/s44443-026-00899-w](https://doi.org/10.1007/s44443-026-00899-w)
([open-access paper](https://link.springer.com/article/10.1007/s44443-026-00899-w)).

The official article page lists no code or supplementary implementation and
states that data are available on reasonable request. Therefore this repository
does **not** claim a faithful reproduction. It provides source-mapped clean-room
primitives and a separate local adaptation that runs without unreleased assets.

## What the paper specifies and we implement

| Paper component | Local implementation |
|---|---|
| Beijing study box 39.8–40.0° N, 116.2–116.5° E, split into 100 × 100 cells represented by centroids | `BeijingGrid` |
| Historical query probability and first-order, row-normalized transition probabilities | `historical_query_probabilities`, `estimate_transition_probabilities` |
| Day-of-week embedding size 3; 48 half-hour slots with embedding size 6; semantic embedding size 10 | `FeatureEmbeddingTables`, `encode_paper_features` |
| Two stacked LSTM layers; attention score from `[h_t, q_t]`; sigmoid attention; weighted sum; two linear output layers | `StackedLSTMSemanticNetwork` with externally supplied dimensions/weights |
| Table-3 transition weights: 75%, 21%, 10%, 5%, 3%, 2%, 1% over the seven published time intervals | `paper_time_weight`, `adjusted_transition_probability` |
| Dynamic threshold form `epsilon_hat = epsilon * Gen(delta_t)` | `dynamic_threshold`; `Gen` is mandatory rather than guessed |
| Retrieve predicted semantic types; filter candidates; accumulate time-weighted transition scores; expand to semantic siblings when short; choose top `K-1` | `SemanticDummySelector` |
| ASR success condition `P(real) <= 1/K` and `N_success/N_total`; effective-dummy ratio `k'/k` | `anonymity_success_rate`, `dummy_effectiveness_rate` |
| 70/10/20 split, MSE + SGD, learning rate 0.01, batch 128, 20 epochs | Constants are recorded; training is blocked because reproduction-critical details and data are absent |

The input prose and Figure 6 say that spatial, temporal, and semantic fields are
concatenated. Equation (6), however, prints addition between vectors whose
reported dimensions are incompatible. `encode_paper_features` follows the
dimensionally valid prose/figure interpretation and documents that decision.

## Unresolved reproduction blockers

The paper does not publish enough information to reproduce its reported neural
or privacy results uniquely:

- LSTM hidden width, fully connected widths, sequence/window length, padding,
  weight initialization, SGD shuffle/seed and checkpoint rule;
- exact coordinate normalization statistics/formula and whether the split is
  by user, path or sample;
- target vector construction, normalization of `q_t`, output activation and
  the similarity function used to rank semantic embeddings;
- whether the “normalized” sigmoid attention weights are additionally
  normalized across time;
- processed GeoLife paths, Amap labels, the exact depth-three category tree and
  trained weights;
- the definition of Equation (18)'s `total` normalizer, smoothing of unseen
  transitions, the decay function `Gen(delta_t)`, aggregation of Step-1 tests
  over multiple historical locations, cutoff tie behavior and failure behavior
  when fewer than `K-1` candidates survive;
- an executable LSP posterior for ASR and the `Sim` function used by DER.

The clean-room core raises `PaperSpecificationGap` for these choices or requires
the caller to inject them explicitly. This prevents a convenient engineering
default from being mislabeled as the paper's method.

## Runnable SUMO/OSM adaptation

`SemanticCorrelationComparator.from_road_network(road_network, ...)` produces a
complete `real_plus_dummies` transcript for the common benchmark. To stay fully
local and bounded, it uses:

- at most 128 nearby road vertices per event by default;
- deterministic OSM highway-class groups as semantic surrogates, **not** Amap
  POI semantics;
- a bounded empirical semantic-frequency predictor, **not** the trained LSTM;
- a normalized distance kernel as the transition prior;
- an explicit exponential threshold-decay interpretation and “any historical
  transition” eligibility rule;
- event-local opaque candidate IDs and evaluator-only real-member labels. The
  paper selects a new location set at each query and does not specify stable
  identifiers that would link candidates into `K` public trajectories;
- one shared nearest-vertex catalog for every member of an event, including a
  nearest-network-vertex representation of the real location. This prevents a
  raw SUMO lane point from being distinguished from vertex-valued dummies by a
  trivial representation test;
- a K-feasible order-statistic threshold (rather than an unreliable median),
  explicit exponential decay, “any historical transition” aggregation, and a
  deterministic location-ID tie rule. These are serialized local choices,
  because the paper does not specify them.

These substitutions are machine-readable in the method card as `ADAPTED` or
`MISSING`. The canonical eight-event, `K=4` SUMO smoke run uses the same
passenger graph as the simulator and records setup, inference, and end-to-end
timings in the generated artifact. Those values are runtime sanity checks, not
a comparison with the paper's reported results.

## Evidence needed to promote the method

Promotion from `paper_adaptation` to `faithful_reimplementation` remains blocked
until the authors' processed data/model details are obtained or every ambiguity
above is resolved transparently, and the implementation reproduces the paper's
model accuracy, ASR, DER and delay trends on the stated setup. Until then, its
numbers belong only to the local integration benchmark.
