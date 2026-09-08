# Verification index

Verification files are append-only evidence snapshots. Do not rewrite old path
or line references after a reorganization; inspect the commit named by the
review when reproducing an old finding.

## Current entry points

- Dataset database and thesis reproducibility: [`verification_scenario_store.md`](verification_scenario_store.md)

- Latest proposed method and scenario data: [`verification_belief_suite.md`](verification_belief_suite.md)
- Preceding contextual ablation: [`verification_contextual_lane.md`](verification_contextual_lane.md)
- Current BR-Dummy study: [`verification_paper_benchmark_v2.md`](verification_paper_benchmark_v2.md)
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
