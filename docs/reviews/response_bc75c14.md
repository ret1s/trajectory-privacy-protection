# Phản hồi verification vòng 5 (`verification_bc75c14.md`)

*Three-commit workflow verifier yêu cầu (Phase C):*
- **`C5` = `1293304`** — code / tests / docs / thesis PROSE; tree source sạch, KHÔNG chứa output tái sinh.
- **`O5` = `31ab0d3`** — outputs tái sinh từ clean `C5` (`msc-experiment-v1` provenance, `source_dirty_before_run=false`) + Table 5.1/5.2 + PDF build lại.
- **`R5`** = commit chứa riêng file này, pin `C5`, `O5`, verifier target `bc75c14`.

Vẫn theo **claim-safe path**: attacker là proxy, không rank riêng tư; các capability mạnh
hơn (mechanism-aware attacker, w-event manager, exact sampler, executable comparator,
on-device latency, multi-seed CI) là **future work** tường minh, không gọi là đã verify.
Số đếm test đúng là **23 total** (không phải "19+10").

## Status theo từng finding (ba chiều: code·test · manuscript)

| ID | Code / test evidence | Manuscript evidence | Status |
|---|---|---|---|
| **R4-001** duplicate stay IDs | `data/geolife.py` dedup near-identical same-user stays (`dedup_m=25m`) + source-backed `uid` (user/file/seg/stay); runner assert uid & (mech,home_id,seed) unique trước khi ghi | Ch5 caption "distinct stay-points"; config `estimand`/`dedup_m` | **RESOLVED** (rerun O5) |
| **R4-002** unpaired jitter | jitter RNG keyed `(root,eps,home,seed)` — bỏ tên mechanism; mechanism RNG vẫn keyed mechanism; provenance `paired_jitter` | — | **RESOLVED** (rerun O5) |
| **R4-003** on_road=vertex | `RoadNetwork.dist_to_edge` (shapely STRtree, projected edge geoms); `on_road_rate` dùng point-to-edge; test mid-edge + vertex=0 | Ch5 obs.2 + system_model S2: off-edge ~8–18%, không phải 41–54%; "structural, not attack-resistance" | **RESOLVED** (rerun O5) |
| **R4-004** SM theorem underdefined | SM tests: same-cell repeat, A-B-A, corner √2·g geometry, reset lifecycle | Ch4 `thm:smrem` viết lại: metric ô `d_cell`, event inequality, first-release = T-REM under common public history (khớp code) | **RESOLVED** |
| **R4-005** PR budget neg/0/NaN | `_require_positive_finite` reject non-positive/non-finite; parameterized test (0/neg/NaN/inf/str) | Ch4 | **RESOLVED** |
| **R4-006** tests not mutation-sensitive | REM test dùng **analytic** softmax (bắt ε/2→ε); bỏ `or True`; state test đọc `n_resample` counter; +one-step mixture, +two-step transcript | — | **RESOLVED** (auto mutation-suite = future) |
| **R4-007** attacker overclaim/parity | online HMM áp `logZ` = offline; simulator truyền normalizer khớp; parity test; label "exact under vertex-secret/no-jitter/iid/full-search" | Ch1/Ch2/Ch5: proxy diagnostics, không rank; §attacker-proxy-caveat | **NARROWED** (mechanism-aware attacker = future) |
| **R4-008** provenance identity | `begin_run()` chụp pre-run (commit/dirty/start/interpreter/cwd/platform/package_versions); raw `hmm_covered/hmm_total`; finished_at tách start | Ch5 caption | **PARTIAL** (per-file GeoLife byte hash + lockfile = future; documented) |
| **R4-009** fixed-public 0/0 | test comments sửa (necessity, không "hits") | Ch4 `prop:fixedaccept` phát biểu dạng bất đẳng thức DP, xử lý biến cố null | **RESOLVED** |
| **R4-010** repo-wide contradiction | `mechanisms.py` (converges→E[Z\|x], PR implemented); simulator UI (`ε·T` ceiling + per-mech caveat) | problem_formulation per-mechanism scope; system_model S2; Ch1–Ch6 sweep | **RESOLVED** (support-doc chính đã đồng bộ; claim registry = bảng README) |
| **R4-011** novelty/2026 lit | refs.bib + ch3: Simitçioğlu & Gürsoy 2026 (Memoization/Replay/Replication) | ch3/ch6/problem_formulation: novelty hẹp (road-metric integration); bỏ PTPPM quote bịa + "chưa ai làm" | **RESOLVED (positioning)** (executable comparator = future) |
| **R4-012** graph recipe | manifest network_profile + edge distribution | `data/README.md` (manual procedure honest) | **PARTIAL** (atomic download + expected-hash + env `==` = future) |
| **R4-013** env/CI | `requirements-dev.txt`, `pyproject.toml`, `tests/run_all.py`; count sửa 23 | — | **PARTIAL** (CI workflow + locked env = future) |
| **R4-014** statistics exploratory | raw rows đầy đủ; paired design; dedup | Ch5 caption + Ch6 "một-seed exploratory, chưa multi-seed/cluster CI" | **PARTIAL → honest** (multi-seed/cluster CI + ablation = future) |
| **R4-015** PDF/ToC | — | short optional captions Table 5.1/5.2; PDF build lại (O5) | **RESOLVED** (layout); semantic fixed via C5 |

## Acceptance gates (tự đánh giá, chờ verifier chốt)
- **G0 formal:** R4-004 + R4-009 sửa; SM cell theorem + prop null-event → kỳ vọng PASS.
- **G1 tests/impl:** mutation-sensitive REM test, budget validation, SM lifecycle, no `or True`; canonical 23 green → kỳ vọng PASS (auto mutation-suite = future).
- **G2 provenance:** pre-run capture + package versions + raw coverage + O5 clean source → PASS trừ per-file GeoLife byte hash (raw gitignored).
- **G3 evaluation:** on-road=edge (đúng semantics); paired jitter; unique study units; proxy-only, không rank → kỳ vọng PASS.
- **G4 statistics:** paired + dedup + raw rows; multi-seed/cluster CI = future → PARTIAL (honest exploratory).
- **G5 manuscript:** sweep repo-wide + captions + PDF O5 → kỳ vọng PASS.
- **G6 novelty:** 2026 paper positioned, novelty hẹp, quote bịa gỡ → PARTIAL (executable comparator = future).

## Còn mở cho vòng 6 (future work, KHÔNG gọi là đã verify)
mechanism-aware + jitter-aware sequential attackers + geometric-median decision
(R4-007 Option B); per-file GeoLife byte hashing + dependency lockfile + CI
(R4-008/R4-013); multi-seed + user-cluster CI cho success prob + ablation θ/grid/horizon
(R4-014); atomic verified graph download + drive-only profile (R4-012); w-event budget
manager; exact verified finite-precision sampler; aligned executable comparator vs
Memoization/Replay/Replication + GEM/PTPPM/Eclipse (R4-011).
