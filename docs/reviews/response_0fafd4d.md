# Phản hồi verification vòng 2 (verification_0fafd4d.md)

*Candidate kế tiếp trên branch `verifier`. Ánh xạ từng finding R2-001…R2-015 sang
trạng thái: **resolved** · **partial** (sửa phần chính, phần còn lại planned) ·
**planned**. Mọi thí nghiệm đã chạy lại trên graph đã sửa bbox + sampler Gumbel-max
+ ngân sách khớp.*

## P0 — sáu blocker

### R2-005 (graph build sai bbox) — **RESOLVED** (gốc, sửa trước)
Xác nhận bug: graph cũ trải lat 39.96–40.35, lon 115.33–116.36 (~98km). Rebuild
đúng thứ tự OSMnx 2.x `(min_lon,min_lat,max_lon,max_lat)` + giữ thành phần liên
thông lớn nhất → **13.813 nodes / 41.040 edges**, extent khớp bbox GeoLife
(39.96–40.02, 116.29–116.36). Sửa **mọi** call ssite bbox sai trong
`core/trajectory_privacy.py`, `web/app.py`, `web/app_optimized.py`,
`web/trajectory_privacy_optimized.py`, `data/README.md`. Thêm
`data/beijing_graph.manifest.json` (SHA-256, node/edge, extent, lệnh build,
versions). **Toàn bộ benchmark + averaging + normalizer đã chạy lại.**

### R2-001 (bound `{z2≠z1}` thiếu ε_release) — **RESOLVED**
Định lý 4.4 + docstring viết lại: chặn per-step worst-case là
`(ε_test+ε_release)`-Geo-I (bước đầu chỉ ε_release), kèm liệt kê hai-vertex xác
nhận ratio `e^{1.5}` vượt chặn cũ `e^1` nhưng thỏa chặn đúng `e^2`. Không còn
chuỗi `ratio ≤ exp(ε_test·d)` cho event resample trong repo.

### R2-003 (PR không matched budget) — **RESOLVED**
Runner + simulator construct `PrivateReuseSMREM(ε/2, eps_test=ε/2)` → worst-case
per-step = ε, khớp REM. Ghi rõ split trong caption/config. Sweep θ×split = planned.

### R2-006 (underflow/CDF rounding) — **RESOLVED (mitigated) + caveat**
`_sample` đổi sang **Gumbel-max log-domain** (argmax logit+Gumbel) — không `exp`
underflow, không CDF zero-width bin, mọi candidate reachable tới độ phân giải
float. Graph nhỏ đúng cũng giảm logit range. Nêu rõ caveat "ideal real-arithmetic
kernel vs floating-point implementation"; proof pure-DP hữu-hạn-độ-chính-xác đầy
đủ = planned.

### R2-002 (chưa có w-event accountant) — **PARTIAL (Option A honest)**
Không claim budget saving hình thức. Định lý 4.4 + Nhận xét w-event nay phát biểu
per-step worst-case + composition product-metric $D_\infty$ rồi mới derive scalar;
`n_resample` gọi rõ là **chẩn đoán hậu nghiệm, không phải guarantee ex-ante**.
Budget manager/privacy-filter theo cửa sổ (Option B) = planned.

### R2-004 (MLE misspecified) — **PARTIAL**
Sửa: planar Laplace **không** còn nhận road normalizer (dùng zero-normalizer);
đổi tên "optimal/consistent MLE" → "MLE dạng-REM (chính xác cho REM, proxy cho cơ
chế khác)"; **rút** claim "REM tệ hơn Laplace" (graph đúng cho hai bên ~25–30m
ngang nhau); nêu rõ hạn chế: không exact end-to-end dưới GPS jitter, PR sequential
kernel mạnh hơn proxy. Cài attacker per-mechanism (T-REM `Z_t(x,z_{t-1})`, PR
sequential kernel, jitter-aware) = planned.

## P1

- **R2-007** (PR đổi bài toán) — **RESOLVED (scope/naming)**: docstring + Ch4 nêu rõ
  PR-SM-REM là *predictive reuse của release liền trước*, KHÔNG phải persistent
  memoization; mẫu rời-rồi-quay-lại không được nhớ. Thí nghiệm A,B,A = planned.
- **R2-008** (không phải home thật) — **RESOLVED**: §5.5 gọi "stay-point/significant
  location, không phải ground-truth residence". Nighttime/recurrent filter = planned.
- **R2-009** (moving benchmark provenance) — **PARTIAL**: nêu trong Ch6; multi-seed
  moving benchmark = planned.
- **R2-010** (docs mâu thuẫn) — **PARTIAL**: response docs + thesis đồng bộ theo số
  mới; rà quét toàn diện tiếp tục.
- **R2-011** (proposition rejection) — **RESOLVED**: đã thêm câu phân biệt fixed-public
  conditioning (giữ $2\eps$) vs input-dependent rejection (mới phá vỡ).
- **R2-012** (on-device/identifier lifecycle) — **PLANNED**: ngoài phạm vi cơ chế
  nhiễu tọa độ; nêu ở Ch6 (R8–R12).
- **R2-013** (provenance) — **PARTIAL**: manifest graph + hash + versions; seed dẫn
  xuất per-(mech,ε,traj) và dependency lock = planned.
- **R2-014** (novelty positioning) — **RESOLVED**: PR positioned là tích hợp
  predictive-mechanism + REM on-road + w-event (bảng đối thủ Ch4/§5.5).
- **R2-015** (numerical/narrative) — **PARTIAL**: rút claim không support, sửa
  node/edge count (13.813/41.040), "năm quan sát", bib. Overfull hbox = còn lại.

## Còn mở (đề xuất verifier round 3)
w-event budget manager (R2-002 Option B); attacker mechanism-aware đầy đủ +
jitter-aware (R2-004); θ×budget sweep (R2-003); finite-precision DP proof (R2-006);
A,B,A + cross-session experiment (R2-007); nighttime/recurrent home filter (R2-008);
seed-derivation + dependency lock (R2-013).
