# TransProtect reproduction record

## Status

The repository now contains an executable clean-room implementation of the
paper-defined TransProtect protection core. It is still classified as
`paper_adaptation`, not `faithful_reimplementation`, because the authors did
not publish the learned TransProtect pipeline or its weights and the published
description omits parameters needed to recreate the same model.

Primary sources used for this audit:

- S. Yadav, C. Yu, X. Xie, Y. Huang, and C. Qiu, “Protecting Vehicle
  Location Privacy with Contextually-Driven Synthetic Location Generation,”
  ACM SIGSPATIAL 2024, DOI
  [10.1145/3678717.3691211](https://doi.org/10.1145/3678717.3691211),
  [arXiv:2409.09495](https://arxiv.org/abs/2409.09495).
- Authors’ official
  [VehiTrack repository](https://github.com/sourabhy1797/VehiTrack), audited at
  commit `035684c6c666a9af7cbd9984d92300000eb65536` (2024-06-07).

No upstream code was copied into this project.

## Algorithm reconstructed from the paper

At time (n), TransProtect performs the following operations:

1. Node2Vec embeds every road-network node. A GCN updates these embeddings
   using graph connectivity and edge weights (paper Equations 7–9).
2. Sinusoidal positional encodings are added (Equation 10).
3. A masked multi-head Transformer uses only
   (x_1,\ldots,x_{n-1}) to estimate
   (h_{j,n}=\hat p(v_j\mid x_1,\ldots,x_{n-1})) for every location
   (v_j) (Equations 11–12). Cross entropy against the true location is the
   training loss.
4. For LBS target prior (q_l), utility loss is the expected absolute
   difference in travel cost from the real and candidate locations to each
   target:

   \[
   \Delta c_{x_n,v_j}=\sum_l q_l
   \left|c_{x_n,v_l}-c_{v_j,v_l}\right|.
   \]

   The absolute difference is confirmed by the official MATLAB scripts.
5. Each location receives score
   (h_{j,n}+\alpha/\Delta c_{x_n,v_j}); the top (K) locations form the
   allowed candidate set (Section 4.4).
6. Planar Laplace or the LP Geo-I mechanism is applied only within this set;
   one pseudolocation is released (Sections 4.1 and 5.1).

## Implemented components

`benchmark/engines/transprotect.py` provides:

- exact Equation 13 utility calculation;
- utility-adjusted top-(K) ranking. The score equation is exact, while the
  paper's unspecified zero-loss tie case is handled by deterministically
  replacing the final selected item with the real node when needed. The policy
  and number of affected events are included in public run metadata;
- candidate-restricted exponential/Laplace sampling;
- a Geo-I LP solver with both directional privacy inequalities and row-sum
  constraints;
- a causal probability-provider interface, preventing access to the current
  true location during prediction;
- an optional PyTorch Node2Vec-input → GCN → masked-Transformer architecture,
  causal cross-entropy training loop, and probability-provider adapter. The
  reported batch size and learning rate are defaults; optimizer, epoch count,
  layer configuration and seed remain explicit adaptation parameters;
- a sparse fitted first-order transition provider for an end-to-end local
  benchmark. Its identifier explicitly states that it is a Markov proxy and
  **not** the TransProtect Transformer. Its additive smoothing and empirical
  global-frequency backoff are explicit and serialized;
- an (N\times M) shortest-path utility provider for (M) explicit
  POI/application targets. It avoids an infeasible (N\times N) allocation on
  the benchmark road graph. If POIs are unavailable, deterministic graph
  targets may be used, but metadata labels them as a proxy.

The default Laplace execution computes distances only from the real location
to its top-(K) candidates. The LP execution builds only (K\times K)
matrices. Thus neither path needs a city-wide pairwise distance matrix.

In the canonical SUMO run, the longest vehicle trace is held out for
evaluation. The Markov proxy is fitted only on the other resampled SUMO
vehicles after vehicle IDs, lanes and edge metadata have been discarded. The
most frequently visited training vertices act as local service-target proxies;
their normalized background visit counts form the target prior. The target
indices, prior, selection rule, distance unit, and disconnected-pair penalty
are emitted in public metadata. Both the training source and target-proxy
status are also emitted in run provenance.
If no disjoint trajectory corpus is supplied, the factory falls back to graph
edges and labels that run `graph_edges_only` rather than using the evaluation
trajectory as training data.

The runner exposes TransProtect's epsilon in `km^-1` and converts it to `m^-1`
because projected graph distances are metres. It also exposes TransProtect's
own `K`, target count, alpha, smoothing and backoff parameters instead of
silently reusing the thesis method's `K` or epsilon. Runtime is split into
setup, online inference, and end-to-end values. Equation-13 expected travel-cost
loss is reported; VehiTrack EIE remains explicitly unavailable until a
paper-equivalent attacker can be executed.

The Laplace/LP stage is private only relative to its already-selected candidate
domain. Because the top-K domain itself depends on the current secret through
utility loss, this implementation makes **no end-to-end Geo-I claim** for the
whole selection-plus-release pipeline.

## Paper settings that are known

- PyTorch 2.1 on Ubuntu 22.04/NVIDIA RTX 4090;
- embedding dimension 128;
- batch size 50;
- initial learning rate 0.001;
- Rome dataset: 367,052 trajectories, about 320 taxis, over 30 days; OSM area
  centered at `(41.9028, 12.4964)` with 20 km radius;
- San Francisco dataset: 34,564 trajectories, 536 taxis, 30 days; OSM area
  centered at `(37.7739, -122.4312)` with 10 km radius;
- 100 randomly selected evaluation trajectories, samples approximately every
  20 seconds;
- reported privacy budgets 5.0, 7.5, and 10.0 km⁻¹;
- candidate-size analysis for (K\in\{5,10,15,20,25\}), with additional
  analysis of utility weight \(\alpha\).

## Blocking evidence for a faithful claim

The audited official revision contains 36 MATLAB files, 64 MAT files, 11 CSV
files, and a README containing only the heading `# VehiTrack`. It contains no
Python/PyTorch source, Node2Vec trainer, GCN/Transformer definition, model
configuration, learned embedding, or checkpoint. In particular, the following
are not recoverable from the primary sources:

- Node2Vec walk length/count, return/in-out parameters, context window,
  negative-sampling count, and training epochs;
- GCN depth and exact nonlinearities;
- Transformer layer/head count, feed-forward size, dropout, optimizer details,
  epoch count, padding/batching policy, and stopping rule;
- trained weights or reference probability outputs;
- exact train/validation/test trajectory IDs and random seeds;
- the exact LBS target prior (q_l) used for each experiment;
- one fully documented default pair ((K,\alpha)) for the headline tables.
- the min-heap tie rule when more than (K) locations have zero utility loss;
  without an explicit rule the real node is not mathematically guaranteed to
  survive an implementation's arbitrary tie order.

The repository also refers to paths such as `Dataset/location_set_new100.mat`,
`Dataset/utility_loss.mat`, and other `Dataset/*` files that are not present
under those names in the checked tree (`Dataset_Rome` and `Dataset_SF` are
present instead). Some attack outputs/intermediate MAT files are published,
but they do not replace the missing learned TransProtect model or define a
single reproducible end-to-end command. No root license file was found, which
is another reason not to vendor the upstream MATLAB code.

## Promotion gate

Do not change the method card to `faithful_reimplementation` until all of the
following are archived and verified:

1. fixed Rome/SF data manifests, graph construction, POIs, splits, and seeds;
2. fixed Node2Vec/GCN/Transformer configuration and trained checkpoints;
3. predictions proving causal input alignment;
4. Laplace+TransProtect and LP+TransProtect results within a predeclared
   tolerance of Tables 1–5;
5. VehiTrack/VehiTrack-I parity under the same obfuscation outputs.
