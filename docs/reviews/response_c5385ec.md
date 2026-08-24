# Phản hồi verification vòng 4 (`verification_c5385ec.md`)

*Three-commit workflow verifier yêu cầu (Phase 0):*
- **`C4` = `4aea94b`** — code / tests / docs / thesis PROSE; tree source sạch, KHÔNG chứa output tái sinh.
- **`O4` = `7f3b8d2`** — outputs tái sinh từ clean `C4` (`benchmark_results.json`, `averaging_multi_results.json` với `msc-experiment-v1` provenance, `source_dirty_before_run=false`) + Table 5.1/5.2 cập nhật + PDF build lại.
- **`R4`** = commit chứa riêng file này, pin `C4`, `O4` và verifier target `c5385ec`.

Chọn **claim-safe path** (Phase 3, Option A): attacker giữ nguyên là proxy, hạ mọi
kết luận so-sánh-riêng-tư; đóng góp tập trung vào formal/structural/finite-horizon.
Vì vậy các finding attacker-mechanism-aware / w-event manager / on-device / exact
sampler / comparator được đặt **future work** một cách tường minh, không gọi là
capability đã verify. Trạng thái theo quy ước report: **RESOLVED / NARROWED /
PARTIAL / PLANNED**; NARROWED được dùng cho các capability chưa hiện thực (theo yêu
cầu Phase 6).

## Status theo từng finding (ba chiều: code·test · manuscript)

| ID | Code / test evidence | Manuscript evidence | Status |
|---|---|---|---|
| **R3-001** proposition fixed-public | `ch4:prop:fixedaccept` thêm giả thiết base ε-Geo-I + mask `1_A(z)`, phát biểu cho mọi event S; `tests/test_rejection_conditioning.py` thêm 2 negative regression (bỏ Geo-I→∞; bỏ `1_A`→tổng 2) | Ch4 §4.1 | **RESOLVED** |
| **R3-002** RNG order-independence | `experiments/rng_util.py` seed = SHA-256(`root,ε,tên cơ chế`), không ordinal; `test_production_mechanisms.test_rng_seed_order_independent` (đảo thứ tự, thêm cơ chế lạ → khóa không đổi) | caption Table 5.1 sửa "R3-002, permutation test" | **RESOLVED** |
| **R3-003** result provenance | `experiments/provenance.py` `msc-experiment-v1` (source commit + dirty flag bỏ qua `outputs/`, command, python, graph hash, rng schema, record IDs); chạy lại từ clean `C4` → `source_dirty_before_run=false`; three-commit `C4`/`O4`/`R4` | caption Table 5.1 nêu provenance | **RESOLVED** (clean-clone từ committed data vẫn cần raw ngoài Git → PARTIAL phần đó) |
| **R3-004** PR anti-averaging overclaim | `mechanisms.py` PR docstring "finite-horizon only, cả hai nhánh probabilistic, q=1/2 tại d=θ"; `test_pr_both_branches_have_positive_probability` | Ch4 nhận xét mới + Ch5 obs.3 + Ch6 | **RESOLVED** |
| **R3-005** SM guarantee/grid | `mechanisms.py` boundary → `exp(ε·√2·cellwidth)`, theorem thêm empty-cache; `problem_formulation.md` bỏ O(#distinct)·ε | Ch4 `thm:smrem`+`rmk:boundary`, `problem_formulation.md` H2 | **RESOLVED** (test edge/corner geometry = future) |
| **R3-006** tests bind production | `tests/test_production_mechanisms.py` (10 test) gọi thẳng `RoadExponential`/`PrivateReuseSMREM` trên real tiny `RoadNetwork`: sampler softmax, reuse F_Lap, state machine, budget wiring, RNG | — | **RESOLVED** (mutation-suite formal chưa tự động hóa → PARTIAL phần đó) |
| **R3-007** attacker proxy vs superiority | `attacks.py`/runner note "proxy, upper bound"; provenance note "không rank privacy" | Ch1/Ch2/Ch5 obs + §attacker-proxy-caveat: mọi so sánh đổi thành proxy diagnostics | **NARROWED** (claim-safe; mechanism-aware attacker = future) |
| **R3-008** graph download/hash | `assert_graph_matches_manifest` fail-closed (hash + node count) trong cả hai runner; `build_beijing_graph.py` báo hash | `data/README.md` | **PARTIAL** (fail-closed + runtime check xong; auto-download atomically + pin dated URL/env `==` = future) |
| **R3-009** README family split | README có mục "Thesis artifact" + claim-registry table và mục "Legacy demo (no valid guarantee)"; `run_averaging`→`run_averaging_multi`; bỏ "matches Algorithm 1" | README | **RESOLVED** |
| **R3-010** statistical evidence | tidy `raw_rows` trong cả hai JSON (bảng tái tạo được); home_id/record_id ổn định | Ch5 caption + Ch6 "một-seed, exploratory" | **PARTIAL** (multi-seed CI + user-cluster CI cho success prob = future) |
| **R3-011** stale numeric/semantic | `mechanisms.py` 78k→13,813; runner note | Ch1/Ch2/Ch4/Ch5/Ch6 + docs: bỏ "150.5 ở mọi n", "pharmacy"→POI tổng hợp, "Bayes tối ưu"→proxy, sub-millisecond bỏ, heading w-event, |V| | **RESOLVED** (claim registry = bảng README) |
| **R3-012** PR API footgun | constructor keyword-only + `epsilon_step_cap`; `privacy_cost_per_step_max`; `test_pr_constructor_rejects_footguns`, `test_pr_budget_cap_is_release_plus_test`; docstring "privacy-relevant state" | Ch4 | **RESOLVED** |
| **R3-013** road profile | manifest ghi `network_profile` + `edge_highway_distribution` (multimodal); build script báo phân bố | Ch5 nêu "đa phương thức, không phải drive-only"; `problem_formulation` | **PARTIAL** (chưa build filtered drive-only graph; documented) |
| **R3-014** canonical test env | `tests/run_all.py`; `pyproject.toml` (testpaths, norecursedirs=legacy, filterwarnings); `requirements-dev.txt` (pytest pinned); test assert-only | — | **RESOLVED** (CI job = future) |
| **R3-015** comparator/novelty | — | `problem_formulation`/Ch6 định vị novelty hẹp (integration + evaluation), nêu thiếu aligned comparator là limitation | **NARROWED / PLANNED** (executable comparator GEM/PTPPM/Eclipse = future) |

## Acceptance gates (tự đánh giá, chờ verifier chốt)
- **G0 formal:** R3-001 + R3-005 sửa, ideal/executable tách rõ → kỳ vọng PASS.
- **G1 tests/impl:** production-coupled tests + permutation + budget API + canonical green (19+10 test) → kỳ vọng PASS (mutation-suite tự động = future).
- **G2 provenance:** outputs `O4` pin clean `C4`, schema đầy đủ, raw rows → PASS trừ clean-clone-from-Git (raw data gitignored).
- **G3 evaluation:** claim-safe — mọi cột non-REM là proxy diagnostic, không rank → PASS theo Option A.
- **G4 statistics:** một-seed exploratory + raw rows; multi-seed/cluster CI = future → PARTIAL.
- **G5 manuscript:** sweep repo-wide + claim registry + PDF build lại → kỳ vọng PASS.
- **G6 industry/novelty:** profile + lifecycle + positioning nêu rõ; comparator thực thi = future → PARTIAL.

## Còn mở cho vòng 5 (future work, KHÔNG gọi là đã verify)
mechanism-aware + jitter-aware sequential attackers (R3-007 Option B); w-event budget
manager (R3-002 gốc); multi-seed + user-cluster CI + ablation θ/grid/horizon (R3-010);
exact verified finite-precision sampler (R3-006 gốc); auto-download + pinned env `==`
+ drive-only profile (R3-008/R3-013); aligned executable comparator GEM/PTPPM/Eclipse
(R3-015); on-device latency đo cách ly (R3-012 gốc).
