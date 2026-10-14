# GeoI-BQ research-reference evidence, L60

K5/L60/Gamma1:99.595% macro Recall,95% family-bootstrap interval
[99.225,99.902]%. Prespecified−2pp lower-CI gate PASSES for macro/all four
purposes and all draw means. Static session-valid metadata, current local GPS.
L60 was chosen before private defense draws from exact PUBLIC semantic
reference masses; all baselines/ablations here use the same depth/cache/cadence.

Native geometry/GPS is REUSED:20 train/8 selection/20 test groups,176 trip-draws,
120 test. Random defense draws are new; no untouched-trajectory/cross-city claim.
Gamma0 has higher Recall99.711%; no demonstrated actual-utility gain from
private belief in this static catalogue. Late coverage is near full418POIs.
Reply82.557KB is JSON indices only; all assumptions and cost controls retained.

Independent audit:5984 kernel selections,81840 cache/actual-utility recomputations,
21241 matched-initial utility checks. Immutable-source replay passes without
private keys, resampling or refitting. `semantic_floor_clarification.json`
corrects an inherited descriptor; numerical behavior/sources never changed.
Inherited EOF blank lines in copied sources remain intact to preserve hashes.

[Complete method/readout](../../../docs/research/2026-10-14_main_model.md),
[proof](../../../docs/research/2026-10-14_whole_session_utility.md),
[main manifest](../../../configs/geo_i_bq_v1.json).

Replay: `python -m experiments.replay_bounded_q_release --output artifacts/benchmarks/main_depth_20261014_v1`.
