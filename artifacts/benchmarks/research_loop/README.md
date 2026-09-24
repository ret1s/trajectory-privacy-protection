# Algorithm improvement loop

Status and interpretation: `docs/research/algorithm_loop_status.md`.
Acceptance protocol: `docs/research/algorithm_improvement_loop.md`.

The recovered OSM/network is DIFFERENT from the old benchmark. Evidence here is
separate. New full-session experiments use 88 newly simulated SUMO trips in four
development families; iterations 3–9 cover selected roles; iteration 10 has all 15 case views of the five target scenarios, under an explicit conditional request clock and finite attacks.
Repeated control runs are not independent samples or fresh confirmation.

- `resources.json`: pinned public map/catalogue/POI provenance.
- `iteration01_quotient.json`: exact-output paired runtime comparison.
- `iteration02_static_cache.json`: public-static-POI service assumption audit.
- `iteration03_boundary.json`: 72 actual full-session BR/boundary executions.
- `iteration04_filter.json`: 48 predictive privacy-filter/control executions.
- `iteration05_progress.json`: 72 ledger × progress ablation executions.
- `iteration06_depth.json`: L5/L10 service re-evaluation on iteration 5.
- `iteration07_slack.json`: 60 bounded-slack/control executions.
- `iteration07_slack_depth.json`: matched L5/L10 re-evaluation on iteration 7.
- `iteration08_matched.json`: 60 executions matched to the fixed-H tighter .23 bound.
- `iteration08_matched_depth.json`: matched L5/L10 service results for iteration 8.
- `iteration09_switching.json` / `_depth.json`: 60 runs including the latest switching-belief core.
- `iteration10_cases_checked.json`: canonical 264 persistent-session executions / 448 case rows, all 15 A/B/C types. Empty references are null and counted, matching existing metric semantics.
- `iteration10_cases.json`: preserved original generation; contains NaN in utility aggregates from empty references. **Do not use its utility summaries.** Its defender outputs/privacy errors are preserved unchanged in the checked artifact.
- `iteration10_cases_null_repair_v1.json`: intermediate null correction, before the 1e-12 threshold-roundoff guard.
- `sources/iteration10_cases_v1.py`: exact original generation source for provenance.
- `iteration11_shadow_training.json`: 2,000 public-road simulated first-query shadows, no validation target labels.
- `iteration11_shadow_attack.json`: trained attacker predictions and train-selected readout.
- `iteration11_all_learned_attacks.json`: descriptive readout of all 16 prespecified learned estimators; exposes the kNN15 S9 counterexample.
- `iteration12_origin_guard.json` / `_depth.json`: 60 full-session executions; first/60-second guarded epsilon, matched .23 cap. Better first-query privacy does not meet the utility gate.
- `iteration12_origin_shadow_training.json` / `_attack.json`: 2,000 mechanism-matched guarded first-query shadows and attack results.
- `iteration13_paced_cases.json`: 330 executions / 560 exact case rows; 60-second public spacing of private reads. Unguarded pacing passes 14/15 utility gates; this is not a privacy certification.
- `iteration13_rare_case_diagnosis.json`: posthoc two-source S1.C planner diagnosis, not a new selection or confirmation dataset.
- `sequential_shadows/manifest.json`: hashes for 80 gzip shards, 640 defended runs on 160 completed auxiliary SUMO trips; 64 attacker-fit / 16 attacker-selection families.
- `iteration14_sequence_attacks.json` (when present): independently selected empirical sequence attacks, all estimators, raw controls, core A/B/C and whole-session endpoint probes. Core results remain exposed defender development.
- `iteration14_raw_sequence_audit.json`: same fixed learned endpoint models also fitted to raw auxiliary traces; prevents relying solely on geometric raw controls. S10.B remains weak on raw and cannot certify protection.
- `iteration15_response_cases.json` (when present): same K5/L10 and top-five target, but selection optimizes top-ten reply support. Three new variants, 198 new executions, with 264 explicitly replayed controls. Changed outputs require matched shadow retraining.
- `response_shadows/manifest.json` / `iteration16_response_attacks.json`: 320 matched auxiliary executions for the two response-aware variants; 64 fit / 16 attacker-selection families, all case windows and a separate whole-session probe.
- `iteration16_first_query_training.json` / `_attack.json` / `iteration16_first_query_paired.json`: 2,000 public-road first-query shadows for the changed selector, plus paired scoring of old/new selectors on the same 16 source sessions and two RNG repetitions. These are still only two core validation families.
- `iteration16_s1_oracle_diagnosis.json`: 24 verified prefix replays and optimal true-reference reachable-cover MILPs. Evaluator-only conditional bound, not a privacy mechanism or a global impossibility result.
- `iteration17_paced_slack_cases.json`: 66 new paced/slack executions plus 198 replayed controls. S1.C improves but S1.B falls below the fixed gate; no final winner.
- `paced_slack_shadows/manifest.json` / `iteration18_slack_attacks.json`: 160 matched auxiliary executions and attacker fitting/selection for the fixed .03 slack variant.
- `expanded_screening/` / `iteration18_expanded_screening.json` (when complete): frozen shortlist on 12 additional development families, 264 completed source trips and 173 target-case records. Not final confirmation; exact records and family eligibility remain explicit.
- `iteration18_expanded_attacks.json` (when complete): fixed auxiliary-only sequence learners and raw positive controls on the expanded case views. No fitting or attack selection on expanded labels.
- `iteration18_paired_comparisons.json`: 45 exploratory paired comparisons; 10,000 family-cluster bootstrap draws, no multiplicity adjustment or final-confirmation claim.
- `iteration19_reserve_cases.json`: 132 new executions with remaining-budget pacing, plus 198 exact replayed controls; neither new variant improves core S1.C.
- `iteration19_expanded_rare_diagnostic.json` (when complete): unchanged reserve variants on all 12 expanded S1.C source sessions, two RNGs and persistent full-session clocks; 48 new executions plus 48 replayed controls. A utility diagnosis, not a new matched learned-privacy result.
- `verification.json`: completed sessions, indices, budget/accounting and control checks.
- `iteration20_mobility_fit.json` / `_mobility_training.npz` / `_generator_tau*.npz`: public synthetic mobility generator fitted on 64 auxiliary families, selected by likelihood on 16 other families. No defender target labels used.
- `iteration20_mobility_null.json`: post-selection likelihood diagnostic for converting the old prior into continuous time without learning; no full defender ablation for this null.
- `iteration20_mobility_cases.json`: 132 new executions / 198 replayed controls, all core cases, paired anchors and ledger. Both new variants pass 14/15 utility gates; no matched learned privacy results yet.
- `iteration21_lookahead_pilot.json`: 24 new full-session executions / 16 replayed controls on all four core S1.C sources. Three fixed two-slice planning ablations; no all-case or learned privacy claim.
- `iteration22_public_cover.json` (when complete): deterministic GPS-independent remote-query control, public K frontier, all expanded target records. K5 is the matched-cost comparison; larger K costs more. Static local cache remains an exact no-query alternative under the present workload.
- `iteration22_public_cover_comparisons.json`: 30 paired-family exploratory utility comparisons against public K5, 10,000 bootstrap draws; no multiplicity adjustment or confirmation.
- `iteration22_public_cover_state_access_v1.json` / `sources/iteration22_public_cover_v1.py`: preserved original score used internal state IDs instead of coordinate-to-road server access. **Do not use its utility readout.** Corrected canonical artifact above keeps the exact same public query coordinates and matches the adaptive runners' interface.
- `iteration23_public_backbone_cases.json`: 132 new executions / 198 exact replayed controls. Two fixed public queries plus three adaptive queries, total K5, same anchors and ledger. S1.C reaches 90% on core but other cases regress; candidates pass 7/15 and 13/15, not selected. Finite geometric attackers can remove the known fixed tracks; no new learned-attack result.
- `iteration24_site_density.json`: prior-corrected repeated-site KDE attack, trained on existing mechanism-matched auxiliary arrays (64 families), selected on 16 other families and scored on all expanded S9.C/S10.C rows for raw/response-paced/slack (144 rows). The combined bank still selects old estimators.
- `iteration24_site_density_components.json`: independently auxiliary-selected prior-only/pool/product readout. Paced S9.C product mean at 500 m bandwidth hits within 100 m on 3/24 record–RNG pairs; descriptive counterexample, not population confidence or a uniformly stronger attacker.

No current artifact establishes thesis readiness or full protection of all five
scenarios. The static-cache control applies to historical static results. Iteration28 adds a separate live-availability workload and retains a bulk-status control.

## Iterations 25–26

- `iteration25_capped_profile.json` / `iteration25_capped_grouped_profile.json`: auxiliary-prefix engineering profiles, including setup time. The first profile's pre-hook engine source is preserved as `sources/iteration25_capped_service_v1.py`. Exact public reference grouping reduces 2,073 rows to 458; no posterior mass pruning.
- `iteration25_capped_service_cases.json`: per-location capped-Recall planner, cap .9, slack 0/.03, all core cases; same anchors and ledger as controls.
- `iteration26_planar_anchor_cases.json`: standard full-plane planar-Laplace/private-reuse ablation, same planner, K5/L10 and .23 cap; 132 new full-session runs. It is not a full reproduction of CCS2013/PETS2014.
- `iteration26_first_query_training.json` / `_trees.npz` / `_attack.json`: 2,000 public training locations and 32 generated auxiliary-selection first queries; matched attacker for the planar first-query probe. These 2,032 one-query executions are not full-session runs or new SUMO trips. This probe is separate from masked S9.A/B/C.
- `iteration25_26_readout.json`: all-case utility and paired family deltas, plus the first-query privacy comparison. S9.B and S10.A have only one eligible core validation family; other cases have two. No confidence or confirmation claim.

Method notes: `docs/research/capped_service_argument.md` and
`docs/research/planar_anchor_ablation.md`. New full-session job timings overlap,
so the combined readout does not claim a comparative speed advantage.

## Iterations 27–28

- `iteration27_public_supplement.json`: 1,224 deterministic public-suffix service replays, 2,076 case rows, and 204 new matched fixed-control evaluations. No new noise runs; parent transcript has an exact public inverse.
- `iteration28_protocol.json`: live-availability point API, frozen probability/epoch/world settings and explicit cache/bulk controls.
- `iteration28_live_service.json` / `live_service/`: nine paired settings, 14,688 deterministic service-session evaluations and 24,912 case rows. All protected coordinates replayed unchanged.
- `iteration28_readout.json`: working service configuration, means, communication costs and paired family bootstrap contrasts. Three status worlds are not extra independent families.
- `iteration28_verification.json`: source/status hashes, aggregate checks, cache/cost invariants and 648 independent forward-Dijkstra category queries.

See `docs/research/live_service_results.md`. New utility gains are conditional
on a point-query API; the 419-bit bulk alternative is explicitly measured.
