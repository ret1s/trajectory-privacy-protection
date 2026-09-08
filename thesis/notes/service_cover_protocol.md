# Service-cover cycle — prespecified 2026-09-08

## Question and frozen design

Can joint selection of reachable dummy queries improve service recall over the
frozen geometric BR-lane and belief24 selectors, without hiding a stronger
attacker or sacrificing difficult categories? This is a within-city diagnostic,
not a paper-readiness certificate or an all-scenario protection experiment.

Methods: `baseline`, `belief24` (unchanged), `service_cover` (exact global greedy
weighted set coverage under the protected-anchor belief), and `prior_cover`
(same selector using only the fixed training/public prior). No parameter sweep.
K = 3, 5; B = 0.24/m; horizon = 12; theta = 200 m. Three paired RNG replicates
per record. Same anchor seeds across all methods. The prior-only control still
runs the inherited anchor code for pairing, but its PUBLIC output ignores those
anchors; its charged ledger is conservative, not a claim that it needs GPS.
Timestamp, map, catalogue, and training prior remain public assumptions.

The coverage objective is a nonnegative weighted union of POI response sets.
Each track has its own directed reachable candidate set, computed before greedy
selection. Select the globally largest marginal gain over all unassigned tracks;
assign one state to that track and repeat. No coordinate uniqueness restriction.
Exact-gain ties prefer less movement, then proximity to fixed prior mean, then
track/state order. No Gumbel term, geometric weight, or raw-location tie-break.
The classical 1/2 bound applies only to the per-step partition-matroid objective
with exact arithmetic/oracle, not to horizon utility, actual-user recall, privacy,
or an approximate Bayesian filter. Numerical roundoff is not a new theorem.

## Data separation and provenance

Immutable dataset release `urban-scenarios-v3`, parent `urban-scenarios-v2`.
Reuse v2 families 101/102 for training and 103/104 for validation, byte-identical
trajectory/record content. Previously inspected 105/106 are NOT confirmation.
Generate fresh SUMO+OSM families 201,202,203,204 for confirmation using the frozen
v2 scenario definitions and parameters; no hand-fixing FCD or case labels.
Declare every one of the 30 subcases, including zero coverage/rejected cases.
Keep native route/FCD/stop hashes. JSON schema stays v2 because record format is
unchanged; release version is v3. Verify native evidence before DB import.

The new runner reads records and bounded device views from SQLite, requiring a
pinned release ID and canonical content SHA. Historical runners stay frozen.
Run evaluation only on S1.A–S3.C, first up-to-12 allowed events. S4–S10 receive
dataset gates only, not fabricated protection/attack scores. Independent units
are simulation families, not events or RNG repetitions. Four fresh families do
not establish cross-city generalization or support strong significance claims.

## Attacker and utility controls

Keep existing public transcript heuristics, and add method-specific shadow
nearest-neighbour inference trained only on protected outputs from 101/102.
Features: persistent-track XY coordinates (km), elapsed public time (minutes),
and causal running means of those coordinates. No future events, evaluator
anchors, true query/intent, hidden states, or confirmation labels. Fit coordinate
standardization on shadow training only; use neighbours {1,5,15} as separate
attacks, with per-case MAE and Hit@100m attack choice on validation only.
Record full training features/labels and model parameters for audit. Training
routes are synthetic auxiliary knowledge, NOT the target's ground truth.

Retain primary macro recall@5 across six eligible POI categories; report each
category separately, valid/empty counts, complete-result rate and conditional
extra distance. Also report the unweighted cafe/restaurant mean as a labelled
dense-category diagnostic, NOT a replacement primary metric. K actual dummy
slots, distinct coordinates, actual request counts and step latency remain
visible. Current category mix is not a production query distribution.

Selection: one method per K over the THREE anchor-dependent candidates only;
prior_cover is a negative control, not eligible for promotion. First require
minimum validation case recall >= .90 (12-decimal decisions), then minimize
macro strongest validation Hit@100m; if none qualifies, choose highest minimum
recall and explicitly retain `utility_feasible=false`. Freeze source hashes,
validation outputs, shadow model, method and attack choices before confirmation.
Publish ALL method rows on confirmation; never reselect after observing them.
Training controls/attacks are fixed, not optimally trained adversaries.

## Falsification and publication gates

- Compare proposed versus prior-only: utility can come from catalogue coverage
  rather than private anchors. Do not equate far dummy coordinates with privacy.
- Unit tests: marginal union identity, global-vs-track-order greedy, exhaustive
  tiny-instance optimum, ties/duplicates, causal prefix, ignored GPS after H,
  prior-control input independence, directed motion, paired anchors.
- Independent verifier: DB integrity/pinned identities, split disjointness,
  recomputed POI sets and attacks/selection, native FCD, source hashes, budgets,
  and all emitted transitions. Hash-frozen preceding studies stay unchanged.
- Geo-I is inherited postprocessing under the existing ideal mechanism and
  composition assumptions; B=.24 gives exp(24) at 100 m and is not a strong
  numerical guarantee. This cycle neither fixes that nor proves identity,
  query-content, destination, or relationship protection.
- Novelty requires a closer-work comparison beyond combining known filtering,
  set coverage and private anchors. Three paper-inspired comparator adapters
  are not sufficient to claim wins against faithful neural SOTA reproductions.
- Before submission: independent map/traffic/POI regime, calibrated adversary,
  more independent families, budget/utility Pareto, all intended target attacks,
  comparator fidelity and runtime/memory measurements, proof audit, venue fit.

## Literature anchors (primary sources)

- Chatzikokolakis, ElSalamouny, Palamidessi, *Efficient Utility Improvement for
  Location Privacy*, PoPETs 2017(4), 308–328. Bayesian utility remapping is prior
  work, not our invention. https://petsymposium.org/popets/2017/popets-2017-0051.pdf
- Calinescu, Chekuri, Pal, Vondrak, *Maximizing a Monotone Submodular Function
  Subject to a Matroid Constraint*, SIAM J. Comput. 40(6), 1740–1766, 2011.
  Ordinary greedy has 1/2, not their continuous-greedy 1−1/e bound.
  https://chekuri.cs.illinois.edu/papers/submod_max_sicomp.pdf
- Buchholz et al., *SoK: Can Trajectory Generation Combine Privacy and Utility?*,
  PoPETs 2024(3), 75–93. Publishing synthetic trajectories differs from online
  LBS, but explicit privacy units, attack evaluation and utility validation apply.
  https://petsymposium.org/popets/2024/popets-2024-0068.pdf
- Liu, Peng, Zhou, *A dummy-based location privacy protection scheme with semantic
  correlation of moving paths*, J. King Saud Univ. Comput. Inf. Sci. 38, 478 (2026).
  Journal article, not a conference paper. Its semantic sequence modelling remains
  related work; this cycle does not claim to reproduce its trained neural model.
  https://link.springer.com/article/10.1007/s44443-026-00899-w

No additional plot required: the final thesis uses exact metric tables for the
four methods and category-level values, with units, denominators and limitations.
Any deviations/negative results go into the dated verifier, not an overwritten
protocol after scores are known.
