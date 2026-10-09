# Verification index

Verification files are append-only evidence snapshots. Do not rewrite old path
or line references after a reorganization; inspect the commit named by the
review when reproducing an old finding.

## Current entry points

- Current thesis proof completion: [`2026-10-07_thesis_formal_completion_v2.md`](2026-10-07_thesis_formal_completion_v2.md)
- Continued independent math/PDF reviews: [`four-module math`](2026-10-07_thesis_formal_v2_math_review.md), [`accuracy PDF`](2026-10-07_thesis_formal_v2_accuracy_pdf_review.md), [`inference/ranking/bridge PDF`](2026-10-07_thesis_formal_v2_inference_ranking_pdf_review.md)
- Previous formal V1 analysis and review: [`2026-10-07_thesis_formal_analysis.md`](2026-10-07_thesis_formal_analysis.md); exact PDF/source bytes retained in `artifacts/reports/thesis_formal_review_20261007_v1/`.
- Independent formal reviews: [`privacy source`](2026-10-07_formal_privacy_source_review.md), [`service/PDF`](2026-10-07_formal_service_pdf_review.md)
- Previous 07/10 thesis review: [`2026-10-07_thesis_completion.md`](2026-10-07_thesis_completion.md); reviewed PDF/source bytes retained in `artifacts/reports/thesis_review_20261007/`.
- Method/resource source review: [`2026-10-07_thesis_method_resource_review.md`](2026-10-07_thesis_method_resource_review.md)
- Final PDF method review: [`2026-10-07_thesis_pdf_method_review.md`](2026-10-07_thesis_pdf_method_review.md)
- Local-GPS PDF/readout review: [`2026-10-07_thesis_sensor_review.md`](2026-10-07_thesis_sensor_review.md)
- Public artifact boundary receipt/scope: [`2026-10-07_thesis_public_boundary.json`](2026-10-07_thesis_public_boundary.json), [`2026-10-07_thesis_public_boundary.md`](2026-10-07_thesis_public_boundary.md)
- Previous thesis and method reviews: [`2026-10-06_thesis_completion.md`](2026-10-06_thesis_completion.md), [`2026-10-06_thesis_method_review.md`](2026-10-06_thesis_method_review.md)
- Local-GPS independent check: [`../../artifacts/benchmarks/local_gps_robustness_20261007_v1/validation.json`](../../artifacts/benchmarks/local_gps_robustness_20261007_v1/validation.json)
- Historical companion inference check: [`../../artifacts/benchmarks/s8_companion_inference_20261007_v1/validation.json`](../../artifacts/benchmarks/s8_companion_inference_20261007_v1/validation.json)
- Current research chronology and retained failures: [`2026-10-06_qplanner_iteration_log.md`](2026-10-06_qplanner_iteration_log.md)
- Controlled dynamic workload verification: [`../../artifacts/benchmarks/dynamic_provider_status_20261006_v1/validation.json`](../../artifacts/benchmarks/dynamic_provider_status_20261006_v1/validation.json)
- Preceding coverage / privacy–utility–cost verification: [`verification_coverage_frontier.md`](verification_coverage_frontier.md)
- Preceding expanded SUMO shadow / inference audit: [`verification_expanded_shadow.md`](verification_expanded_shadow.md)
- Preceding prior-factor / loss-aware attack verification: [`verification_prior_factors.md`](verification_prior_factors.md)
- Preceding prior/corridor development ablation: [`verification_service_recovery.md`](verification_service_recovery.md)
- Preceding independent confirmation and literature critique: [`verification_service_cover.md`](verification_service_cover.md)

- Dataset database and thesis reproducibility: [`verification_scenario_store.md`](verification_scenario_store.md)

- Preceding proposed method and scenario data: [`verification_belief_suite.md`](verification_belief_suite.md)
- Preceding contextual ablation: [`verification_contextual_lane.md`](verification_contextual_lane.md)
- Historical BR-Dummy study: [`verification_paper_benchmark_v2.md`](verification_paper_benchmark_v2.md)
- Previous report/demo release: [`verification_report_demo_release_v1.md`](verification_report_demo_release_v1.md)
- Artifact hierarchy: [`verification_artifact_hierarchy_v1.md`](verification_artifact_hierarchy_v1.md)
- Repository structure: [`verification_codebase_reorganization_v1.md`](verification_codebase_reorganization_v1.md)
- Thesis baseline: [`verification_graduation_thesis_latest_v1.md`](verification_graduation_thesis_latest_v1.md)
- Foundations guide: [`verification_location_trajectory_privacy_foundations_v2.md`](verification_location_trajectory_privacy_foundations_v2.md)
- SOTA benchmark: [`verification_sota_benchmark_completion_v2.md`](verification_sota_benchmark_completion_v2.md)
- Benchmark/web application: [`verification_thesis_benchmark_webapp_v1.md`](verification_thesis_benchmark_webapp_v1.md)
- SUMO map overlay: [`verification_sumo_demo_map_overlay.md`](verification_sumo_demo_map_overlay.md)

Files named `verification_<commit>.md` and `response_<commit>.md` record earlier
review/repair rounds. They remain at stable paths because other agents use their
commit hashes as hand-off points.
