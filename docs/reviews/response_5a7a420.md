# Phản hồi verification vòng 3 (`verification_5a7a420.md`)

*Candidate kế tiếp trên branch `verifier`, theo đúng two-commit workflow verifier
yêu cầu (Phase 0): **commit thực thi/kết quả `C` = `c5385ec`** chứa mọi
thay đổi code/thesis/output; **commit phản hồi `R`** chỉ chứa file này và trỏ về
`C` (một file không thể pin hash của chính commit chứa nó).*

Vòng 3 verifier nhấn mạnh: **không được suy ra "overall verified"** từ việc
R2-001 và numerical rerun pass, và một số status vòng 2 bị **mark quá mạnh**. Bản
này hạ giọng tương ứng và theo dõi **ba chiều riêng cho mỗi finding**: `code` ·
`test/evidence` · `manuscript`. Nhãn tổng: **resolved** (cả ba chiều xong trong
scope đã thu hẹp) · **partial** (chính đã xong, phần sâu → future work có ghi
thẳng) · **planned** (chưa làm, chỉ định vị/limitations).

Chọn **minimum claim-safe path** verifier đề xuất: hạ mọi claim executable/optimal/
comparative-superiority, quan hệ đúng scope, còn lại là future work — thay vì cố
chứng minh các claim mạnh trong vòng này.

## Nguồn sự thật đã freeze (Phase 0)
- Candidate parent: `5a7a420`; report vòng 3: `d0a068a`.
- Graph artifact: `data/raw/beijing_graph.pkl`, `graph_sha256` trong
  `data/beijing_graph.manifest.json` (13.813 nodes / 41.040 edges, 1 WCC).
- Output JSON (`benchmark_results.json`, `averaging_multi_results.json`) mang khối
  `provenance`/`config`: git commit + dirty flag, graph hash, seed, tham số cơ chế
  và attacker. Benchmark đã chạy lại với seed **order-independent** (R2-009).

## P0 — sáu blocker

### R2-001 (bound PR một bước) — **RESOLVED cho ideal kernel** (`code` n/a · `test` ✓ · `manuscript` ✓)
Định lý 4.4 giữ chặn per-step `(ε_test+ε_release)`-Geo-I; case `z=h` (release trùng
representative) viết tường minh trong proof. **Thêm test committed**
`tests/test_pr_privacy_bound.py`: kernel hai-vertex phân tích, ratio `e^{1.5}` **vượt**
chặn cũ `e^1` và **thỏa** chặn đúng `e^2`; thêm 2000 check ngẫu nhiên finite-domain
worst-ratio ≤ bound. Không claim vượt ra ngoài ideal real-arithmetic kernel (xem R2-006).

### R2-002 (w-event accountant) — **PARTIAL, honest** (`code` — · `test` — · `manuscript` ✓)
Ch4 chỉ dùng trivial per-window composition; **Ch1 và Ch6 đã sửa** hết chỗ overclaim:
Ch1 nói "bốn cơ chế", bullet PR **không** còn nói "w-event thật"; Ch6 limitations ghi
rõ **chưa có w-event budget manager**. `problem_formulation.md` tách "ĐÃ hiện thực"
(per-release bound) vs "kiến trúc đích" (w-event). Budget manager theo cửa sổ = future
work có ghi thẳng.

### R2-003 (matched budget) — **PARTIAL → cải thiện** (`code` ✓ · `test` — · `manuscript` ✓)
Runner + simulator + averaging đều construct `PrivateReuseSMREM(ε/2, eps_test=ε/2)`
→ per-step worst-case = ε, khớp REM. **Config/provenance JSON nay ghi**
`{eps_test, eps_release, theta}`. Sweep θ×split = future work.

### R2-004 (attacker misspecified) — **PARTIAL** (`code` ✓ · `test` — · `manuscript` ✓)
Đã làm, repo-wide: planar/baseline dùng zero-normalizer (không nhận road logZ);
attacker đổi tên thành **"REM-emission proxy (chính xác cho REM; xấp xỉ cho cơ chế
khác)"** trong `evaluation/attacks.py`, `run_benchmark.py`, `run_averaging_multi.py`,
`README.md`, `research_notes.md`. **Không dùng bảng attacker để kết luận superiority**;
số attacker nêu là *cận trên* sai số adversary cho cơ chế non-REM. Attacker
per-mechanism, jitter-aware, optimal = future work (đây là điều kiện verifier đặt cho
mọi optimal/comparative claim — nên các claim đó đã bị rút).

### R2-005 (graph recipe) — **PARTIAL → cải thiện** (`code` ✓ · `test` — · `manuscript` ✓)
Local graph đúng bbox + largest-WCC. **`data/build_beijing_graph.py`** deterministic,
tracked: tải BBBike, hash, truncate đúng thứ tự `(left,bottom,right,top)`, giữ largest
WCC, ghi manifest từ chính artifact. `data/README.md` recipe sửa: thêm bước largest-WCC
và count đúng (19.634/47.745 trước WCC → 13.813/41.040 sau). Clean-clone vẫn cần
`beijing_graph.pkl` (bị `.gitignore`) → nêu thẳng là limitation R2-013.

### R2-006 (Gumbel-max zero support) — **RESOLVED (Option 1: ideal-only claim)** (`code` ✓ · `test` ✓ · `manuscript` ✓)
Rút **mọi** claim pure-DP mức executable. Chọn **ideal real-arithmetic kernel** làm
đối tượng claim; caveat finite-precision trong `_sample` docstring, module docstring,
`README.md`, và Nhận xét `rmk:finiteprec` (Ch4). **Test committed**
`tests/test_sampler_support.py`: chứng minh Gumbel span bị chặn (~40.34), candidate
gap 45 **không đạt được** trong float64 dù ideal prob = 2.86e-20 > 0 → input-dependent
zero support (documented, not fixed). Exact verified discrete sampler = future work.
(Không đổi sang sampler khác — verifier nói việc đó *không đủ*.)

### R2-011 (rejection proposition) — **RESOLVED** (`code` n/a · `test` ✓ · `manuscript` ✓)
Mệnh đề tách đôi trong `ch4_phuongphap.tex`: `prop:fixedaccept` (fixed public `A`,
positive acceptance → **2ε** vì normalizer input-dependent) và `prop:reject`
(input-dependent `A_x` → ratio ∞, phản ví dụ `{0,1}` base 0-DP đúng như verifier
yêu cầu); case (c) retry hữu hạn + fallback tách riêng. **Test committed**
`tests/test_rejection_conditioning.py`: (a) kernel 3-điểm ε-DP + fixed public `A` cho
ratio **4.2 > e^ε=3.0** và ≤ e^2ε=9.0 (chứng minh ε *không đủ*, 2ε mới đúng); (b)
input-dependent → ∞. Đây chính là counterexample cũ bị falsify bằng test committed.

## P1

- **R2-007** (previous-release scope) — **PARTIAL** (`code` ✓ · `manuscript` ✓): docstring
  + Ch4 nêu PR-SM-REM là predictive reuse của release liền trước, không phải persistent
  memoization. Thí nghiệm A-B-A / cross-session = future work.
- **R2-008** (stay-point ≠ home) — **PARTIAL** (`manuscript` ✓ · `code` ✓): Ch5 caveat +
  `run_averaging_multi` `estimator_note` ghi "stay-points, not ground-truth homes".
  Nighttime/recurrent filter = future work.
- **R2-009** (benchmark provenance) — **PARTIAL → cải thiện** (`code` ✓): RNG
  **order-independent** `default_rng(SeedSequence([SEED, ei, mi]))` — output không phụ
  thuộc thứ tự chạy; provenance JSON đầy đủ. Multi-seed CI + raw tidy rows = future work.
- **R2-010** (cross-file consistency) — **RESOLVED (mâu thuẫn load-bearing đã gỡ)**
  (`manuscript` ✓): sweep vòng này — `README.md` (6 cơ chế + caveat proxy/ideal-kernel),
  `problem_formulation.md` (ĐÃ hiện thực vs kiến trúc đích), `system_model_and_threats.md`
  + `attack_scenarios.md` (S4: SM-REM/PR-SM-REM ĐÃ memoize, không còn "4 cơ chế chưa
  memoize → future work"), `research_notes.md` (caveat proxy + trỏ Table 5.1 authoritative).
- **R2-012** (on-device lifecycle) — **PLANNED** (`manuscript` ✓): là kiến trúc đích;
  nêu ở Ch6 (R8–R12). Prototype hiện là server-side.
- **R2-013** (provenance) — **PARTIAL** (`code` ✓): manifest + build script + provenance
  block trong cả hai JSON. Raw rows + dependency lockfile + clean-clone = future work.
- **R2-014** (novelty comparator) — **PLANNED** (`manuscript` ✓): chỉ có positioning
  (Ch4/§5.5), **chưa** có aligned comparator benchmark (GEM shortest-path, PTPPM,
  Eclipse). Ghi thẳng là future work; không claim superiority định lượng vs SOTA.
- **R2-015** (numerical/narrative) — **PARTIAL** (`manuscript` ✓): PDF build lại với số
  order-independent mới; count 13.813/41.040; caveat proxy/ideal-kernel. Overfull hbox
  minor còn lại.

## Còn mở (đề xuất verifier round 4)
w-event budget manager (R2-002); attacker per-mechanism + jitter-aware + optimal
(R2-004, điều kiện cho mọi optimal/superiority claim); θ×budget sweep (R2-003); exact
verified finite-precision sampler + proof (R2-006); A-B-A + cross-session (R2-007);
nighttime/recurrent home filter (R2-008); multi-seed CI + raw rows + clean-clone
provenance (R2-009/R2-013); aligned comparator benchmark vs GEM/PTPPM/Eclipse (R2-014).
