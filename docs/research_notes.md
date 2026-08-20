# Research Notes — Datasets, Metrics, Related Work & Proposed Improvements

*Tổng hợp khảo sát (2026-08-20) phục vụ nâng cấp luận văn từ baseline internship-2.
Mọi URL đã được kiểm tra sống tại thời điểm khảo sát.*

---

## 1. Datasets thực tế cho bài toán trajectory privacy

Hai nhánh văn liệu dùng dataset khác nhau:
- **Cơ chế per-point (sporadic)**: chủ yếu Gowalla / Brightkite check-ins (SNAP).
- **Trajectory-level (đúng nhánh của luận văn)**: chủ yếu **GeoLife**, sau đó T-Drive, Porto Taxi.

| Dataset | Nội dung | Sampling | Tải trực tiếp | Papers dùng |
|---|---|---|---|---|
| **GeoLife v1.3** (MSR Asia) | 182 users, 17.621 quỹ đạo, ~24.9M điểm, Bắc Kinh 2007-2012 | 91% dày 1–5s | [download.microsoft.com](https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip) — verified 2026-08, 313MB, không cần đăng ký | PIVE (NDSS'17), PIM (CCS'15), RAoPT (ACSAC'22), AdaTrace (CCS'18) |
| **T-Drive** (MSR) | 10.357 taxi Bắc Kinh, 1 tuần 2/2008, ~15M điểm | thưa ~177s/~623m | 9 zip trên trang MSR (cần User-Agent browser) | PETS'14 predictive Geo-I, nhiều paper dummy |
| **Porto Taxi** (ECML/PKDD'15) | 1.71M chuyến, 442 taxi, 1 năm | 15s, 100% trên đường | [UCI zip](https://archive.ics.uci.edu/dataset/339/) không cần đăng ký, CC-BY 4.0 | AdaTrace, Gursoy TPS'20 |
| Gowalla / Brightkite | 6.4M / 4.5M check-ins toàn cầu | check-in rời rạc | SNAP Stanford, trực tiếp | Bordenabe CCS'14, Oya CCS'17, Elastic PETS'15 |
| SF Cabspotting, Rome Taxi | taxi dày | 60s / 7-15s | IEEE DataPort (cần account) | PoPETs'20, attack papers |

**Chọn cho luận văn**: GeoLife (chính — chuẩn de-facto, dày, cùng Bắc Kinh nên 1 road graph dùng chung) + Porto Taxi (phụ — city thứ 2 cho claim tổng quát hoá). **Đã tải GeoLife về `data/raw/` và tích hợp qua `data/geolife.py`.**

---

## 2. Metrics chuẩn (kỳ vọng của reviewer)

### 2.1 Privacy
- **Expected inference error / adversarial error** (Shokri et al. S&P 2011 — chuẩn vàng):
  `ExpErr = Σ_{x'} min_{x̂} Σ_x π(x)·f(x'|x)·d(x̂,x)` — sai số kỳ vọng của kẻ tấn công Bayes tối ưu **biết cơ chế + prior** (Kerckhoffs). PIVE Eqs. 5–9 là công thức tham chiếu.
- **Tracking error theo thời gian** — adversary Markov/HMM (Xiao & Xiong CCS'15): lan truyền prior qua ma trận chuyển `p⁻_{t+1} = p⁺_t·M`, posterior hoá bằng emission. Đánh giá chỉ bằng attack per-point độc lập là **lỗi bị reviewer bắt** từ 2015.
- Phụ: MAP attack success probability, conditional entropy (Oya CCS'17), time-to-confusion (Hoh CCS'07), drift ratio & |ΔX| (PIM).
- ε luôn là **trục x**, không phải kết quả; trajectory phải báo cáo cả `ε_total = Σ ε_t` (sequential composition).

### 2.2 Utility / QoS
- **Expected quality loss** `Q_loss = Σ π(x) f(x'|x) d(x,x')` — mặc định của mọi paper. Planar Laplace: `E[d] = 2/ε`.
- **k-NN POI precision/recall** (Xiao & Xiong): truy vấn k'-NN tại điểm giả so với k-NN thật — thí nghiệm LBS-utility sạch nhất, dễ tái lập.
- (α, δ)-usefulness: `Pr[d ≤ α] ≥ δ` (Andrés CCS'13) — chính là "QoS radius" của luận văn nhưng phát biểu xác suất.
- Shape: **Hausdorff + DTW** (RAoPT, TS-TrajGen dùng), EDR.

### 2.3 Realism (bề mặt tấn công plausibility-filtering)
- % điểm trên road network / khoảng cách tới đường gần nhất.
- Speed/acceleration plausibility; **reachability tối đa tốc độ giữa 2 report liên tiếp** (velocity-linkage attack, Ghinita GIS'09).
- Với synthetic publishing: bộ AdaTrace/LDPTrace (density JSD, trip error, length error, FP F1).

### 2.4 Attack chuẩn để đánh giá
1. Optimal Bayesian inference (bắt buộc).
2. Markov/HMM tracking (bắt buộc cho trajectory).
3. Map-matching / road-constrained pruning (Takagi DBSec'19 lượng hoá thiệt hại của Geo-I phẳng khi adversary biết bản đồ).
4. ML reconstruction: **RAoPT** (ACSAC'22, code công khai — giảm >65% khoảng cách về quỹ đạo gốc với DP noise ε≤1) và **TUL/MARC**. SoK PoPETs'24 nêu đây là bar tối thiểu.

---

## 3. Survey các hướng giải quyết gần nhất

### 3.1 Geo-I và mở rộng
- **Gốc**: Andrés et al. CCS 2013 (planar Laplace, ε·r guarantee).
- **Geo-Graph-Indistinguishability (GG-I)** — Takagi, Cao, Asano, Yoshikawa (DBSec 2019, arXiv:2010.13449, IEICE 2023): thay metric Euclid bằng **shortest-path trên đồ thị đường**, output là đỉnh đồ thị. Cơ chế **GEM**: `Pr[o|v] ∝ exp(−(ε/2)·d_s(v,o))`. Định lý quan trọng: *Geo-I + snap về đỉnh gần nhất ⇒ GG-I* (post-processing). GEM thắng planar-Laplace-rồi-snap trên trade-off.
- **Adaptive/personalized ε**: Al-Dhubhani & Cazalas 2018 (correlation-aware), Mendes CODASPY 2023 (velocity-aware), SAGEO TVT 2024 (semantic-warped metric), CORGI EDBT 2023 (user-customizable). Lỗi hay gặp: chọn ε phụ thuộc vị trí thật ⇒ tự rò rỉ.
- **Elastic metrics** (Chatzikokolakis PETS 2015): metric co giãn theo mật độ/semantics từ OSM.

### 3.2 Temporal correlation
- **δ-location set + PIM** (Xiao & Xiong CCS 2015), **PIVE** (NDSS 2017 — chú ý: arXiv:2101.12602 chứng minh PIVE **có lỗi hình thức**, phải cite bản phê bình), **PTPPM** (Cao et al. arXiv:2401.11225, WCNC 2024 — δ-location set + Hilbert PLS + Permute-and-Flip), CTS-DP (correlated Laplace noise chống lọc).
- **Mới nhất, sát luận văn nhất**: arXiv:2511.21020 (11/2025) "Road Network-Aware Personalized Trajectory Protection" — road network + temporal + personalized, chưa peer-review ⇒ **khe hở đang mở, luận văn nhắm đúng chỗ này**.

### 3.3 Dummy generation & 3.4 Deep learning
- Dummy 2020+: bắt buộc dummy liên tục theo thời gian (speed-consistent) vì adversary lọc dummy rời rạc; GAN-based dummy; caching. Không có formal guarantee.
- LSTM-TrajGAN (GIScience'21), DP-TrajGAN (FGCS'23), DiffTraj (NeurIPS'23), ControlTraj (KDD'24), Diff-RNTraj (TKDE'24 — diffusion trực tiếp trên road network). Không DP thì bị memorization; có DP-SGD thì utility rơi mạnh (arXiv:2506.09312).
- **RAoPT** (ACSAC'22): LSTM khôi phục quỹ đạo từ bản DP — bằng chứng "noise per-point + điểm phi thực tế = khôi phục được".

### 3.5 Surveys nên cite
SoK PoPETs 2023(2) (DP trajectory publication), SoK PoPETs 2024(3) (trajectory generation), ACM CSUR 2021 (LPPM), JNCA 2024, arXiv:2502.08970 (metric-DP decade), Primault CS&T 2019.

---

## 4. Hai lỗi hình thức của baseline internship-2 (phải sửa)

1. **Cap bán kính nhiễu tại δ_qos quanh điểm thật phá vỡ Geo-I thuần**: hai vị trí cách nhau >2δ có support rời nhau ⇒ likelihood ratio không chặn được. (Sửa: truncate vào vùng public cố định, hoặc phát biểu lại guarantee dạng bounded.)
2. **Rejection-resampling theo building/water điều kiện hoá trên vị trí thật ⇒ vỡ guarantee** — và chính loại lọc plausibility này là thứ RAoPT khai thác. (Sửa: chỉ dùng **post-processing độc lập input** — snap về đỉnh đường gần nhất; bước reject là thừa và có hại.)
3. (Bug phụ đã phát hiện khi đọc code: `geo_indistinguishability.py` sample bán kính `r ~ Exponential(1/ε)` trong khi planar Laplace đúng là `r ~ Gamma(2, 1/ε)`; mật độ exponential-radius có tỷ số phân kỳ tại r→0 nên không thoả Geo-I.)

---

## 5. Thiết kế đề xuất (đã implement)

### REM — Road-network Exponential Mechanism (`core/mechanisms.py`)
Exponential mechanism trên tập đỉnh đồ thị đường OSM, score Euclid:
`P(v|x) ∝ exp(−(ε/2)·d(x,v))` ⇒ **ε-Geo-I** trên tập ứng viên (chuẩn EM, hệ số ε/2 hấp thụ normalizer); mọi output **nằm trên đường theo cấu trúc** — không còn bề mặt tấn công map-matching/RAoPT. Là "anh em Euclid" của GEM (Takagi).

### T-REM — Temporal REM
Nhân score với trọng số reachability **chỉ phụ thuộc điểm ĐÃ CÔNG BỐ z_{t-1}** (thông tin công khai):
`w(v) = exp(−λ·max(0, d(z_{t-1},v) − v_max·Δt − slack))`
Vì w độc lập với vị trí thật, chặn ε-Geo-I per-point **không đổi** (cả tử số lẫn normalizer), nhưng chuỗi công bố trở nên speed-consistent ⇒ đóng tấn công velocity-linkage mà smoothing heuristic của baseline không có guarantee nào.

### Benchmark (`experiments/run_benchmark.py`, 20 quỹ đạo GeoLife thật, Bắc Kinh, sampling 20s, QoS 200m)

| ε | Mechanism | Q_loss (m) | QoS | On-road | Speed-viol | Bayes err (m) | HMM err (m) |
|---|---|---|---|---|---|---|---|
| 0.01 | Planar Laplace | 205.6 | 0.59 | 0.58 | 0.08 | 200.5 | 134.8 |
| 0.01 | Baseline internship-2 | 94.9 | 0.99 | 0.97 | 0.00 | 95.6 | 80.2 |
| 0.01 | REM | 370.6 | 0.31 | **1.00** | 0.35 | 369.9 | **265.1** |
| 0.01 | **T-REM** | 307.2 | 0.35 | **1.00** | **0.06** | 309.5 | 233.6 |
| 0.02 | Planar Laplace | 102.8 | 0.89 | 0.63 | 0.01 | 98.9 | 78.0 |
| 0.02 | Baseline internship-2 | 80.0 | 1.00 | 0.99 | 0.00 | 79.8 | 74.9 |
| 0.02 | REM | 180.2 | 0.67 | **1.00** | 0.06 | 181.8 | 127.4 |
| 0.02 | **T-REM** | 175.8 | 0.66 | **1.00** | **0.01** | 174.3 | 129.2 |
| 0.05 | Planar Laplace | 41.1 | 1.00 | 0.68 | 0.00 | 41.0 | 37.9 |
| 0.05 | Baseline internship-2 | 68.8 | 1.00 | 0.99 | 0.00 | 69.4 | 68.6 |
| 0.05 | REM | 72.1 | 0.97 | **1.00** | 0.00 | 73.7 | 64.6 |
| 0.05 | **T-REM** | 75.5 | 0.96 | **1.00** | 0.00 | 76.3 | 67.6 |

**Đọc kết quả:**
- **HMM attack luôn mạnh hơn point attack** (vd PL@0.01: 200.5→134.8m, giảm 33%) — bằng chứng thực nghiệm rằng đánh giá per-point là không đủ.
- **Planar Laplace để 32–42% điểm ngoài đường** — đúng bề mặt tấn công RAoPT; REM/T-REM 100% trên đường theo cấu trúc.
- **T-REM sửa lỗi lộ liễu của REM**: speed violation 35%→6% tại ε=0.01, đồng thời *giảm* Q_loss (370→307m).
- **So ở cùng mức utility** (~PL@0.01 disp 205.6m vs T-REM@0.02 disp 175.8m): T-REM cho sai số attacker tương đương (129 vs 135m) với **utility tốt hơn, QoS cao hơn (0.66 vs 0.59), 100% on-road, không speed-leak** — trội trên mọi trục còn lại.
- Baseline có số đẹp ở cùng ε danh nghĩa **nhưng ε danh nghĩa của nó không phải guarantee hợp lệ** (mục 4) — đây là luận điểm trung tâm khi so sánh.

---

## 6. Hướng tiếp theo (ưu tiên theo impact/effort)

1. ~~Sửa lỗi hình thức baseline~~ → đã thay bằng REM/T-REM.
2. Chạy **RAoPT** (code công khai) đối kháng 4 cơ chế — bar tối thiểu của SoK 2024.
3. GEM đúng nghĩa (shortest-path metric) + so REM vs GEM (chương thí nghiệm kiểu Takagi).
4. δ-location set phiên bản road-network (candidate set = đỉnh reachable từ vùng đã công bố) — guarantee kiểu PIM.
5. Porto Taxi làm city thứ 2; elastic/semantic metric trên graph (khe hở publishable — chưa ai làm "elastic GG-I").
