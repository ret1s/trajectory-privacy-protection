# Research cycle v2: protocol frozen before examining test results

Development evidence: report-demo v1, seeds 71–73. Observed issues: stationary averaging, causal-only adversaries, road-constrained dummies getting stuck, POI dependence on one category per seed, and no measured endpoint tasks.

## Literature-driven decisions

- Predictive Geo-I (Chatzikokolakis et al., PETS 2014, arXiv:1311.4008) already studies private prediction tests and budgeting. Reusing a protected location is not our novelty. BR-Dummy combines this existing primitive with a declared horizon ledger and public road-reachable dummy postprocessing. No novelty or superiority claim is made merely for combining them.
- Shokri et al. (IEEE S&P 2011) motivates evaluating an explicit adversary and inference loss rather than entropy alone. Point-estimate MAE here is NOT their exact posterior expected error.
- TransProtect uses inference error and destination-related utility; semantic correlation reports ASR/DER/time. These cannot be averaged into a common scalar. The new protocol uses operationally defined inference, POI-task utility and cost dimensions.
- AnotherMe's complete paper metric specification has not been independently verified; official source and metadata do not justify inventing exact original metrics. Its local adaptation is an offline reference, not a reproduced learned detector result.

## Fixed design

New SUMO seeds: 81, 82, 83. Demand 32 vehicles; 1800-second simulation; programmed 180-second stop; FCD at 1 Hz. Existing OSM map, no GeoLife input. Eligibility-only amendment before running any protection/test metrics: initial demand 24 yielded only 23 qualifying users in seed 81, so increase demand to 32 for EVERY seed. The five split roles and first 18 user assignments remain fixed; all remaining eligible vehicles go to the model prior. Rejections remain reported.

Disjoint vehicle splits per seed: test 4, shadow 6, adversary selection 4, mechanism selection 4, model prior at least 6. These seeds differ from v1 but use the same city/generator, not an external validation population.

Cases S1/S2/S3 plus S9/S10 on completed trips. Endpoint truth is the first/last FCD coordinate of a completed simulated trip, NOT a real home or a person's identity. Mask 60 seconds; visible window 12 releases separated by 20 seconds. Endpoint labels are evaluator-only. The same trip can supply several correlated scenarios, so sample count is trips, not events or independent scenarios.

BR-Dummy: total cap B=0.24 m^-1, public horizon H=12; fresh-anchor and private-reuse ablations. Private-reuse grid theta={100,200,400} m × offset={40,120} m; baseline offset=80 m, theta=200 m. Temperature=60 m. Public largest strongly connected component used for dummy viability; anchors retain full fixed graph support. Beyond H, no new private coordinate is consumed. Session reset must be accounted as a new budget, not a free refresh.

Per mechanism/scenario/seed/K, shadow data train a full-transcript 3-NN adversary; adversary-selection data independently choose minimum-MAE and maximum-Hit100 adversaries. Offline path backtracking and stationary set intersection strengthen the baseline attack suite. Adversaries use approximate observation models, not calibrated Bayes posteriors.

Defense selection uses only its own split: require minimum scenario Recall@5 >= .90, then minimize macro Hit100. If no grid setting satisfies utility, explicitly record infeasibility and select maximum minimum-case recall. One configuration across scenarios per seed/K; the defense does not receive a private scenario label. This is a small constrained search, not globally optimal protection or the same-utility comparison of all comparators.

Test fixed K={3,5}; no modification of parameters based on test results. If subsequent changes use these results, seeds 81–83 must be relabelled development and a new confirmation set run.

## Reporting rules

MAE/median/P90 (m) and Hit100 with 50/200 m sensitivity, after separately selected adversaries. S9/S10 have one endpoint error per trip. Utility over all six available POI categories: Recall@5, complete-result rate, extra directed travel distance CONDITIONAL on complete results. Costs: coordinates, JSON bytes for one six-category release batch, amortized generation; per-step p50/p95 only where actually instrumented. Graph-node feasibility and unique-coordinate ratio are diagnostics, not privacy guarantees.

Keep failed runs and N/A distinct; equal trip weights; SD across trips is not a confidence interval. Preserve the v1 artifact. Verify raw FCD, provenance, five-way split, attacker/defender selection, metric recomputation, causal prefixes, budgets and deterministic replay before exporting thesis tables.
