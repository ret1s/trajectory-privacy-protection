# Phản hồi verification 5b39226 — candidate sửa lỗi

*Đối chiếu trực tiếp từng finding V-001…V-015 của `verification_5b39226.md`.
Trạng thái: **resolved** (đã sửa đúng) · **narrowed** (thu hẹp claim cho khớp
bằng chứng) · **partial** (sửa phần chính, phần còn lại planned) · **planned**
(chưa làm, ghi rõ là future work) · **out-of-scope**.*

Số liệu mới đã chạy lại với cơ chế + attacker đã sửa (`outputs/benchmark_results.json`,
`outputs/averaging_results.json`); thesis rebuild 28 trang, không undefined ref.

## P0 — bốn lỗi nghiêm trọng

### V-001 (cutoff phá pure Geo-I) — **RESOLVED**
`core/mechanisms.py`: `RoadExponential._candidate_logits` giờ tính trên **toàn bộ
tập đỉnh $V$** (`self._all`, ~78k), bỏ hẳn `cutoff_m` và `query_ball_point`. Support
không còn phụ thuộc input → REM/T-REM/SM-REM là exponential mechanism trên tập
công khai cố định, đúng điều kiện của Định lý 4.1. Bỏ claim sai "tail < 1e-4"
(Nhận xét~\ref{rmk:cutoff} viết lại). Chi phí: 1 norm vector hóa + 1 sample
categorical mỗi release, <1ms cho 78k đỉnh (đã đo trong benchmark: ~57s cho
858 điểm × cả pipeline).

### V-002 (SM-REM không dùng rep(cell)) — **RESOLVED**
`StayMemoizedREM.perturb` khi cache-miss gọi `super().perturb(rep_lat, rep_lon)`
với `rep(cell)` = tâm ô (công khai), không còn dùng toạ độ thật. Phân phối release
của mọi điểm trong một ô giờ đồng nhất.

### V-003 (memoization lộ revisit pattern; trajectory theorem sai) — **NARROWED**
Rút lại định lý toàn-quỹ-đạo. Định lý~\ref{thm:smrem} nay **chỉ** phát biểu cho
vị trí tĩnh lặp lại (transcript $(Z,\dots,Z)$ = hậu xử lý của một mẫu). Thêm
Nhận xét~\ref{rmk:notrajectory} nêu đúng phản ví dụ $X{=}(a,a)$ vs $X'{=}(a,b)$
và kết luận: bảo vệ cả pattern revisit cần reuse-decision riêng tư (predictive
mechanism) = future work. Docstring `StayMemoizedREM` viết lại tương ứng.

### V-004 (attacker sai likelihood) — **RESOLVED (REM) / PARTIAL (T-REM,SM-REM)**
`evaluation/attacks.py`: thêm `precompute_lognorm()` tính log-normalizer $Z(x)$
đủ trên full support (per ε,scale); `BayesianPointAttack` viết lại với likelihood
REM chính xác, **prior đều cố định** (không chọn theo $z$), estimator
**geometric-median** (khớp loss khoảng cách). HMM thêm term $\log Z$. Đặt tên
trung thực: chính xác cho REM; với T-REM/SM-REM là adversary REM-emission cố định
áp dụng đồng nhất, **không** tuyên bố Bayes-optimal riêng. Likelihood exact cho
history/state của T-REM/SM-REM = future.

## P1 — High

### V-005 (averaging one-home/one-seed) — **NARROWED + PLANNED**
Ch5 §5.4 viết lại: bảng averaging là **minh hoạ cấu trúc** (SM-REM phẳng là hệ quả
tất định của Định lý~\ref{thm:smrem}), *không* phải bằng chứng thống kê "an toàn
nhà"; nêu rõ giá trị 429,6m chỉ là một mẫu đơn, trung bình cơ chế đường hội tụ về
$\mathbb{E}[Z\mid x]$ ≠ $x$. Đa-home/seed + CI = future (Ch6).

### V-006 (cache lifecycle) — **PLANNED**
Chưa định nghĩa privacy principal / TTL / cross-session / persistence. Ghi vào Ch6
hạn chế. Hiện `reset()` xoá toàn bộ; benchmark reset theo trajectory. Cần tách
"reset temporal state" khỏi "reset protected memo" — chưa làm.

### V-007 (T-REM/on-road overclaim) — **RESOLVED (wording)**
Ch1/Ch4/Ch5 hạ claim: T-REM = "soft regularizer speed-implausibility, KHÔNG đóng
velocity attack"; số liệu mới cho thấy HMM err T-REM (185,0) *thấp hơn* REM
(198,8) → nói thẳng T-REM đánh đổi chút riêng-tư-HMM lấy realism+utility. On-road
100% gọi là "structural property, chưa phải attack-resistance evidence".

### V-008 (HMM truncation bias) — **RESOLVED**
`max_candidates` 800→2000, radius 1000→2000m; thêm tracker `true_covered` báo cột
`cov` trong bảng. Coverage giờ 0,97–1,00 cho mọi cơ chế (trước 83,6% cho REM).

### V-009 (requirement table > artifact) — **PARTIAL**
Ch4 §4.1 đã có bảng R1–R7 + đoạn CAN/CANNOT scope trung thực (R8–R12 ngoài phạm vi
cơ chế). Tách target/formal/implemented một phần trong wording. Web UI vẫn hiển thị
$T\eps$ — cần đổi sang budget đã định nghĩa: planned.

### V-010 (baseline không faithful) — **RESOLVED (naming) + clarified**
Đổi tên khắp nơi thành "Baseline TT2 (surrogate)"; docstring `BaselineThesis` ghi
rõ bỏ alternative-road + reject stage. Mệnh đề~\ref{prop:reject} thêm câu phân biệt
fixed-public conditioning (giữ $2\eps$) với input-dependent rejection (mới là thứ
phá vỡ). Cài lại pipeline đầy đủ = future.

## P2 — Medium/Low

### V-011 (metric mislabels) — **PARTIAL**
Docstring `dtw` (chia $n{+}m$, không track path), `on_road_rate` (nearest-vertex,
không point-to-edge) ghi rõ; POI đổi nhãn "synthetic" trong simulator. True
point-to-edge distance + pharmacy POIs thật = future.

### V-012 (provenance) — **PLANNED**
Chưa thêm manifest (commit/seed dẫn xuất/hash/CI/lock deps). Ghi Ch6. RNG hiện vẫn
tuần tự qua mechanisms — cần seed dẫn xuất độc lập theo (mech,ε,traj).

### V-013 (citation errors) — **PARTIAL**
Sửa: "W3C 15-phút" → Google/WICG explainer (đề xuất, không phải W3C Rec) trong
`problem_formulation.md`; bỏ "60m ≈ Strava/AOSP" (thực là 200m) trong docstring;
"PTPPM tự thừa nhận" → suy luận từ thiết kế stateless của họ (Ch4). Rà thêm primary
citation cạnh từng platform claim = còn lại.

### V-014 (composition/metadata scope) — **PARTIAL**
Nhận xét w-event (Ch4) nêu product-metric $\sum_t d_t$ và điều kiện metadata công
khai. Định nghĩa adjacency/observation-window/metadata đầy đủ = future.

### V-015 (consistency/LaTeX) — **PARTIAL**
README thêm SM-REM + `python3`; Ch5 "bốn"→"năm quan sát"; `kellaris2014wevent`
đổi `@inproceedings`→`@article`. Overfull hbox ở bảng requirements = còn lại.

## Bảng kết quả mới (thay bảng cũ ở 5b39226)

Xem `outputs/benchmark_results.json`. Điểm khác biệt do sửa V-001+V-004: attacker
error giờ ≈ displacement cho cơ chế đường (adversary hiệu chỉnh đúng), và câu
chuyện headline trung thực hơn: cơ chế đường **không** thắng trên trục attacker-error
thô ở cùng utility, mà thắng ở **tính thực tế** (100% on-road, 0% speed-viol) và
**guarantee hợp lệ**; SM-REM thêm chống-averaging (S4).

## Còn mở (đề xuất cho verifier round sau)
P1: đa-home/seed averaging + CI (V-005), cache lifecycle (V-006), provenance
manifest (V-012), exact per-mechanism likelihood cho T-REM/SM-REM (V-004 phần
còn). P2: RAoPT trực tiếp, point-to-edge distance, cài lại full baseline pipeline.
