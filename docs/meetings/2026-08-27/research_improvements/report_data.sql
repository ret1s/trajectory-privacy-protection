-- DuckDB-compatible reconstruction of the averaging evidence used in the report.
-- Canonical source: outputs/averaging_multi_results.json
-- Source commit recorded by the experiment: 1293304828cacf567a78cf16e4a9b9e4229105e0
-- Scope: GeoLife v1.3, 40 distinct stay-points, 21 users, 8 seeds, epsilon = 0.02.
-- Important: the MLE is a REM-emission-form proxy and is not mechanism-aware for
-- T-REM, SM-REM, or PR-SM-REM. These values support threat diagnosis, not a
-- global ranking of privacy mechanisms.

CREATE OR REPLACE VIEW averaging_curve AS
SELECT *
FROM (VALUES
  ('Planar Laplace', 1,   79.7, 73.9, 94.0),
  ('Planar Laplace', 2,   70.3, 64.0, 80.4),
  ('Planar Laplace', 5,   39.2, 34.4, 43.1),
  ('Planar Laplace', 10,  34.4, 29.6, 37.0),
  ('Planar Laplace', 20,  29.4, 23.4, 34.3),
  ('Planar Laplace', 50,  24.1, 19.2, 30.3),
  ('Planar Laplace', 100, 23.4, 18.5, 30.0),
  ('REM', 1,   173.1, 157.8, 189.4),
  ('REM', 2,   148.4, 128.5, 162.7),
  ('REM', 5,    84.9,  76.8,  95.3),
  ('REM', 10,   62.5,  53.3,  70.0),
  ('REM', 20,   41.4,  37.4,  49.3),
  ('REM', 50,   31.7,  28.0,  36.2),
  ('REM', 100,  28.4,  23.6,  31.1),
  ('T-REM', 1,   159.2, 140.5, 187.2),
  ('T-REM', 2,   140.6, 125.9, 157.0),
  ('T-REM', 5,    84.7,  74.5,  96.5),
  ('T-REM', 10,   60.6,  49.9,  69.0),
  ('T-REM', 20,   39.3,  35.4,  44.4),
  ('T-REM', 50,   31.7,  25.2,  36.1),
  ('T-REM', 100,  28.5,  23.6,  33.7),
  ('SM-REM', 1,   151.8, 139.8, 179.6),
  ('SM-REM', 2,   158.8, 140.6, 175.9),
  ('SM-REM', 5,   148.8, 137.6, 182.6),
  ('SM-REM', 10,  148.1, 138.1, 176.6),
  ('SM-REM', 20,  148.5, 137.2, 177.4),
  ('SM-REM', 50,  149.6, 138.6, 182.6),
  ('SM-REM', 100, 148.9, 138.8, 171.8),
  ('PR-SM-REM', 1,   355.5, 317.6, 387.2),
  ('PR-SM-REM', 2,   283.9, 242.8, 317.3),
  ('PR-SM-REM', 5,   170.9, 156.3, 184.6),
  ('PR-SM-REM', 10,  119.1, 104.2, 134.6),
  ('PR-SM-REM', 20,   87.5,  78.3, 102.0),
  ('PR-SM-REM', 50,   63.7,  54.3,  71.6),
  ('PR-SM-REM', 100,  47.4,  40.8,  50.0)
) AS t(mechanism, n_reports, median_attacker_error_m, ci_low_m, ci_high_m);

SELECT
  mechanism,
  n_reports,
  median_attacker_error_m,
  ci_low_m,
  ci_high_m,
  40 AS n_stay_points,
  21 AS n_users,
  8 AS n_seeds,
  0.02 AS epsilon,
  'REM-emission-form MLE proxy' AS attacker
FROM averaging_curve
ORDER BY mechanism, n_reports;
