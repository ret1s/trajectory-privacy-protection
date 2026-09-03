-- DuckDB-compatible reconstruction of the reviewed meeting-report snapshot.
-- Canonical upstream evidence:
--   docs/internship_2.pdf
--   outputs/benchmark_results.json
--   outputs/averaging_multi_results.json
--   docs/reviews/verification_1293304.md
-- Snapshot: HEAD a82d67c; source C5 1293304; outputs O5 31ab0d3.

CREATE OR REPLACE VIEW summary_metrics AS
SELECT *
FROM (VALUES
  ('tt2_points', 80),
  ('o5_points', 858),
  ('averaging_rows', 1600),
  ('tests', 23)
) AS t(metric_id, value);

CREATE OR REPLACE VIEW tt2_results AS
SELECT *
FROM (VALUES
  (1, 91.68, 160.44, 1.00),
  (2, 110.31, 203.83, 0.90),
  (3, 90.48, 222.88, 0.95),
  (4, 87.60, 170.34, 1.00)
) AS t(run, mean_m, max_m, qos);

CREATE OR REPLACE VIEW problem_map AS
SELECT *
FROM (VALUES
  (1, 'Radial law sai', 'Exponential radius không phải planar Laplace 2D chuẩn', 'Correct Planar comparator + fixed-support REM', 'Đã xử lý ở mathematical design'),
  (2, 'Hard QoS cap', 'Support phụ thuộc secret; có zero-probability event', 'Bỏ cap khỏi formal mechanisms; QoS trở thành measured usefulness', 'Đã xử lý trong REM family'),
  (3, 'Secret-dependent gates', 'Alternative road, smoothing/snap acceptance đọc vị trí thật', 'Direct sampling trên public road support; tách fixed-A, A_x và finite retry', 'Core design xử lý; narrative cần tiếp tục làm rõ'),
  (4, 'Off-road/map cue', 'Adversary có thể prune output không hợp road graph', 'REM/T/SM/PR phát road vertices by construction', 'Structural property; chưa direct RAoPT'),
  (5, 'Temporal correlation', 'Speed/reachability và HMM dùng toàn chuỗi release', 'T-REM public-history reachability weighting', 'Giảm tín hiệu; chưa trajectory guarantee'),
  (6, 'Repeated-report averaging', 'Independent noise collapse khi report cùng stay-point', 'SM exact memoization; PR randomized reuse', 'SM static scope; PR finite horizon'),
  (7, 'Revisit-pattern leak', 'Exact hit/miss có infinite-ratio event', 'Noisy threshold với bound ε_test + ε_release', 'One-step bound; chưa w-event'),
  (8, 'Displacement được gọi là privacy', 'Không mô hình hóa adversary inference', 'Bayesian/HMM/averaging proxy attacks', 'Exploratory; non-REM attacker chưa mechanism-aware'),
  (9, 'Evidence/provenance yếu', 'Synthetic data, không seed/raw/hash/CI', 'GeoLife, semantic RNG, graph hash, raw summaries, user-cluster bootstrap', 'Cải thiện mạnh; moving study vẫn one-seed/7 users')
) AS t("order", problem, why, update, status);

CREATE OR REPLACE VIEW moving_headlines AS
SELECT *
FROM (VALUES
  (1, 'Planar off-edge cue', 'On-road 0.818 / 0.876 / 0.921 tại ε=.01/.02/.05', 'Road mechanisms loại off-road cue theo cấu trúc; chưa chứng minh kháng RAoPT'),
  (2, 'REM → T-REM tại ε=.01', 'Speed violation 0.364 → 0.067; displacement 377m → 299m', 'T-REM giảm speed-implausibility trong exploratory benchmark'),
  (3, 'Planar temporal inference tại ε=.01', 'Bayes error 200.7m; HMM proxy 125.0m', 'Temporal model khai thác thêm chuỗi trong setup này; không là universal attack proof')
) AS t("order", comparison, evidence, safe_interpretation);

CREATE OR REPLACE VIEW averaging_results AS
SELECT *
FROM (VALUES
  ('Planar Laplace', 23.4, '[18.5, 30.0]', 0.944),
  ('REM', 28.4, '[23.6, 31.1]', 0.897),
  ('T-REM', 28.5, '[23.6, 33.7]', 0.897),
  ('SM-REM', 148.9, '[138.8, 171.8]', 0.091),
  ('PR-SM-REM', 47.4, '[40.8, 50.0]', 0.566)
) AS t(mechanism, median_n100, ci, success_50);

SELECT * FROM summary_metrics;
SELECT * FROM tt2_results ORDER BY run;
SELECT * FROM problem_map ORDER BY "order";
SELECT * FROM moving_headlines ORDER BY "order";
SELECT * FROM averaging_results ORDER BY median_n100 DESC;
