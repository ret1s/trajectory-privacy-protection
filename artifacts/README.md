# Generated artifacts

This is the only top-level location for tracked generated deliverables.

The current scenario registry is [`datasets/scenarios.sqlite3`](datasets/scenarios.sqlite3).
It contains three frozen SUMO dataset releases, normalized records and an append-only
update log. See the [storage guide](../data/scenario_store/README.md) and
[migration verification](datasets/scenario_store_verification.json).

- [`benchmarks/`](benchmarks/) contains current machine-generated experiment
  evidence consumed by the benchmark dashboard and thesis.
- [`reports/`](reports/) contains reviewed human-facing PDF releases.
- [`datasets/urban_scenarios_v1/`](datasets/urban_scenarios_v1/) contains versioned
  pre-protection scenario data, separate from measured benchmark results.
- [`datasets/urban_scenarios_v2/`](datasets/urban_scenarios_v2/) is the frozen
  six-family SUMO challenge suite, including rare-POI and multi-day controls.
- [`datasets/urban_scenarios_v3/`](datasets/urban_scenarios_v3/) retains development
  families 101–104 and adds fresh confirmation families 201–204 (176 sessions,
  269 records). Its registry verification supersedes the original two-release
  migration receipt for the current DB head; that original receipt is historical.

`benchmark/` at the repository root is source code; `artifacts/benchmarks/` is
generated evidence. Unreferenced superseded artifacts belong under `archive/`;
an explicitly compared predecessor can remain here with its original audit.

The active controlled S1--S3 and S9/S10 release is
[`benchmarks/paper_benchmark/`](benchmarks/paper_benchmark/): a checksummed result JSON
and generated LaTeX tables. It contains synthetic evaluation truth and must not
be treated as an attacker-only export. The web app exposes a separate public
transcript and an opt-in evaluator endpoint. See the
[`protocol and research decisions`](../thesis/notes/paper_cycle_v2_protocol.md).
The v1 predecessor remains unchanged in `benchmarks/report_demo/` for lineage.

[`benchmarks/lane_comparison/`](benchmarks/lane_comparison/) is a separate,
development-only S1.A/S2.B/S3.A experiment: enhanced-DLS and matched BR lane-state
variants. Its directed lane POI service differs from the earlier junction
service, so its numeric scores must not be merged into the older leaderboard.
`results.json` preserves validation-selected attackers; `audit.json` and the
thesis tables additionally show an explicitly exploratory report-family attack
envelope. Both files contain synthetic evaluator truth, not public-only exports.

[`benchmarks/contextual_lane/`](benchmarks/contextual_lane/) adds a separately
versioned 2x2 ablation of the proposed method: directed-road potential and
marginal public POI coverage, on S1.A/S2.B/S3.A/S3.B/S3.C. All four variants use
paired private anchors. Stronger track-based attackers are applied to all four,
so privacy scores are not interchangeable with the earlier lane tables. Neither
validation-selected configuration meets the all-case utility threshold; no new
default or all-ten-scenario coverage is claimed. This local research artifact
contains evaluator anchors/truth and is not exposed as a public web transcript.

[`benchmarks/belief_suite/`](benchmarks/belief_suite/) extends the proposed method
with protected-history belief-weighted POI coverage. Validation and confirmation
are separate checksummed stages on v2 data, covering nine S1--S3 cases. This is
a preceding internal diagnostic, not a replacement for the comparator benchmark
or an all-ten-scenario protection result. Frozen predecessor evidence above
remains unchanged.

[`benchmarks/service_cover/`](benchmarks/service_cover/) is the preceding proposed-
method ablation: joint POI coverage, prior-only negative control, method-specific
shadow attacks, three RNG repetitions and four new confirmation families.
It reads a pinned SQLite release; training, validation/selection and confirmation
have separate manifests. This remains an S1–S3 diagnostic, not ten solved attacks
or a new leaderboard against faithful SOTA implementations.

[`benchmarks/service_recovery/`](benchmarks/service_recovery/) diagnoses the
service-cover failure and tests a uniform-public-cell prior and a directed
candidate corridor at K=5. Its 201–204 results are **reused development**, not
fresh confirmation; the immutable dataset and its historical split labels stay
unchanged. Only experiment roles change. Six methods, including three exactly
reused controls, share the same allowed inputs, anchor draws and service metrics.

[`benchmarks/prior_factors/`](benchmarks/prior_factors/) separates initialization
from motion weighting, adds cell-balanced/fixed-mixture priors and five empirical
loss-aware/mean shadow decisions. Seven configurations share the same frozen v3
data and budget; 201–204 remain reused development. The uniform configuration
still wins validation, without meeting the all-case utility requirement. The
mixture's higher development Recall is exploratory, and each shadow bank has
only 36 distinct true-coordinate labels. See its independent replay receipt and
[publication critique](../docs/reviews/verification_prior_factors.md).

[`benchmarks/expanded_shadow/`](benchmarks/expanded_shadow/) audits those same
seven frozen protectors with expanded auxiliary data and ten additional
inference rules. [`datasets/urban_shadow_v1/`](datasets/urban_shadow_v1/) adds
80 SUMO route groups (64 train, 16 auxiliary holdout), 160 sessions and 320 query
windows, as a fourth immutable SQLite release. It is not core scenario v4.
Same-output comparisons preserve old utility and scores; reused development
and auxiliary transfer are reported separately. See the
[independent audit](../docs/reviews/verification_expanded_shadow.md).
