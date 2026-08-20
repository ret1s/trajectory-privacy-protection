# System Model & Threat Grounding

*Đối chiếu các giả định của luận văn với hệ thống LBS thực tế được triển khai
(khảo sát 2026-08-20, mọi claim có nguồn dẫn). Tài liệu này trả lời bốn chủ đề:
(1) "bán kính QoS" có thật không (§1); (2) cấu trúc bài toán chuẩn & "privacy
budget"/composition thật sự là gì (§2); (3) cơ chế đặt ở đâu và nhận data ra sao
(§3); (4) các tấn công có thật để dựng scenario (§4).*

---

## 1. "Bán kính hiệu quả" (QoS radius) — có thật, nhưng ở dạng nào?

Giả định cũ của luận văn ("điểm giả OK nếu nằm trong δ=200m của điểm thật") một
phần đúng, một phần cần phát biểu lại.

### 1.1 Dạng tồn tại trong hệ thống thật (3 dạng)

| Dạng | Deploy ở đâu | Chi tiết |
|---|---|---|
| **Lượng tử hóa xác định** (snap-to-grid/rounding — *có* chặn sai số cứng) | Android coarse location, iOS approximate, dating apps | AOSP `LocationFudger`: snap-to-grid **2000m** + offset ngẫu nhiên (σ=grid/4, đổi mỗi giờ để chống averaging), `MINIMUM_ACCURACY_IN_METERS=200`. iOS: snap vào region ~10 dặm², cập nhật ~4 lần/giờ. Tinder round 1 dặm; Recon snap-to-grid. |
| **Bán kính tin cậy xác suất** (contract app tiêu thụ) | W3C Geolocation, Android | Trường `accuracy` = bán kính tin cậy **95%** (W3C) / **68%** (Android). **Không API nào hứa "sai số không bao giờ vượt X".** |
| **Tầng độ chính xác theo query** | iOS Core Location, Android FLP | iOS: `BestForNavigation`/`NearestTenMeters`/`HundredMeters`/`Kilometer`/`ThreeKilometers`/`Reduced`. Android FLP: block ~100m / city ~10km. |

### 1.2 Dung sai thực tế theo loại dịch vụ (trải 4 bậc độ lớn)

- Đón xe (Uber): **~10–50m** — Uber phải làm shadow-matching vì 50m đã hỏng
  pickup; fallback là nhập pin tay chứ không suy vị trí thật từ fix nhiễu.
- Geofencing: **≥100–150m** (Android geofencing doc).
- Local search "restaurants near me": **1–2km không mất utility**, 5–20km mới
  suy giảm (Micinski et al. MoST 2013, đo thật trên top 750 app).
- Thời tiết: **~10km** (lưới dự báo NWS 2.5km; nhiều km cùng một ô).

### 1.3 Số 200m của luận văn "đúng chỗ" một cách tình cờ

Trùng: (i) mức ẩn mặc định Strava (200m) + bán kính privacy-zone tối thiểu ~1/8
dặm; (ii) `MINIMUM_ACCURACY_IN_METERS=200` của AOSP; (iii) ví dụ r=0,2km trong
paper Geo-I gốc (Andrés et al. CCS 2013, ℓ=ln4, r=0,2km → ε=ln4/0,2); (iv) *dưới*
ngưỡng pháp lý "precise geolocation" của luật Mỹ (~564m/1850ft — CPRA) → điểm
nhiễu 200m vẫn *được luật coi là "precise"*, một điểm cần nói thật.

### 1.4 Phát biểu lại cho luận văn (defensible)

Thay hard cap bằng **(α, δ)-usefulness** — chính notion của Andrés et al. §5.1:
> Cơ chế là (α, δ)-useful nếu với mọi vị trí x, điểm công bố z thỏa
> P[d(x,z) ≤ α] ≥ δ (ví dụ α=200m, δ=0,95).

Lý do khớp thực tế: (a) trùng semantics trường `accuracy` (bán kính tin cậy 95%)
mà mọi app web/mobile đang tiêu thụ; (b) là **thuộc tính để ĐO**, không phải
**bước trong cơ chế**, nên không phá guarantee ε-Geo-I — khác hẳn việc cap nhiễu
tại δ quanh điểm thật (điều kiện hóa trên vị trí thật → vỡ Geo-I, đã chứng minh ở
Mệnh đề 4.1). REM trên tập đỉnh cố định công khai còn có thể đạt δ=1 nếu tập ứng
viên độc lập với điểm thật (cùng điều kiện Andrés et al. áp cho vùng truncation).

**Kết luận (1):** bán kính là *có thật*, nhưng dạng deploy hợp lệ là (a) lượng tử
hóa lên tập rời rạc công khai — mà REM snap-lên-đỉnh-đường chính là bản LBS-native
của nó — và (b) phát biểu xác suất (α,δ), *không* phải hard cap quanh điểm thật.

---

## 2. Cấu trúc bài toán chuẩn & "privacy budget" (composition)

*Trả lời trực tiếp lo ngại: "privacy budget" tự đặt KHÔNG dùng được vì Geo-I là*
***metric-DP*** *— ε là **mức riêng tư của MỘT lần công bố**, không phải một "túi ngân*
*sách" tiêu dần. Dưới đây là (a) cách bài toán được đặt chuẩn trong literature để bám,*
*(b) đối tượng "budget" đúng, (c) regime phơi lộ thật → đơn vị bảo vệ đúng cho GeoLife.*

### 2.1 Bài toán chuẩn (canonical LPPM formulation) — bộ khung để follow

Ba paper nền (Shokri S&P 2011; Andrés et al. CCS 2013; Bordenabe et al. CCS 2014)
hợp thành **một pipeline nhất quán**: `prior → cơ chế → quan sát → suy luận adversary →
hai metric đối ngẫu (privacy = sai số adversary; utility = quality loss)`. Đây là "bài
toán đã đặt ra" luận văn nên dùng đúng ký hiệu:

| Thành phần | Ký hiệu | Ý nghĩa |
|---|---|---|
| Tập vị trí (secrets) | **𝒳** (regions R) | vị trí thật khả dĩ, đã rời rạc hóa |
| Tập điểm công bố | **𝒵** | output mà LBS/adversary thấy |
| Prior / mobility profile | **π** (Markov trong Shokri) | phân bố trên 𝒳; kiến thức nền của adversary |
| Cơ chế obfuscation (LPPM) | **f(z\|x) = K(x)(z) = k_{xz}** | kernel ngẫu nhiên: điểm giả z từ điểm thật x |
| Quality / utility loss | **Q = Σ π_x k_{xz} d_Q(x,z)**, hoặc (α,δ)-usefulness (§1.4) | suy giảm chất lượng dịch vụ kỳ vọng |
| Guarantee (Geo-I) | **k_{xz} ≤ e^{ε·d(x,x′)}·k_{x′z}** | metric DP; "ε·r trong bán kính r" |
| Metric privacy | posterior Bayes Pr(x̂\|z); **expected inference error** Σ Pr(x̂\|z)·d(x̂,x) | privacy = adversary tối ưu sai bao nhiêu |

- **Shokri S&P 2011** — bộ khung đo ⟨U, A, LPPM, O, ADV, METRIC⟩: event = ⟨u,r,t⟩,
  prior = ma trận chuyển Markov, privacy metric = **expected estimation error**
  ("correctness"), *không phải* entropy/k-anonymity. Tách rõ **sporadic** vs
  **continuous** exposure. [DOI](https://dl.acm.org/doi/10.1109/SP.2011.18) ·
  [companion sporadic PETS'11](https://link.springer.com/chapter/10.1007/978-3-642-22263-4_4)
- **Andrés et al. CCS 2013** — kernel K(z\|x); `K(x)(Z) ≤ e^{ε·d(x,x′)}·K(x′)(Z)`;
  planar Laplace `D_ε(x₀)(x)=(ε²/2π)e^{−ε·d}`, radius ~ Gamma(2, 1/ε); utility =
  (α,δ)-usefulness. [arXiv 1212.1984](https://arxiv.org/abs/1212.1984)
- **Bordenabe et al. CCS 2014** — bài toán tối ưu (LP): *minimize Σ π_x k_{xz} d_Q(x,z)
  s.t. Geo-I + Σ_z k_{xz}=1*. [arXiv 1402.5029](https://arxiv.org/abs/1402.5029)

→ Design tension = privacy–utility trade-off chuẩn: **tối đa sai số adversary (Shokri)
/ thỏa ε-Geo-I (Andrés) trong khi tối thiểu Q (Bordenabe)**. Bám đúng khung này thì mọi
metric trong `evaluation/` đã có chỗ đứng lý thuyết.

### 2.2 "Privacy budget" — vì sao khái niệm tự đặt không khớp, đối tượng ĐÚNG là gì

1. **Geo-I là metric DP** (Chatzikokolakis et al. PETS 2013, *"Broadening DP Using
   Metrics"*, [DOI](https://doi.org/10.1007/978-3-642-39077-7_5)): ε là *"riêng tư trên
   mỗi đơn vị khoảng cách"*, **vô nghĩa nếu tách khỏi metric**. Luôn phát biểu
   **"εr-indistinguishable ở bán kính r"**, *không* gọi ε trần trụi là "budget".
2. **Không có "túi tiêu hao" trong định nghĩa gốc.** ε là *mức* của MỘT release. Khi công
   bố lặp T điểm với nhiễu ε-Geo-I độc lập, **composition tuần tự** cho
   **ε·T-geo-indistinguishability** trên cả quỹ đạo — Andrés et al. phát biểu thẳng
   *"n query → nε"* và **tự nhận là loose/impractical** (do các điểm tương quan). Đây là
   **trần worst-case trung thực** cần báo cáo, và là baseline mà REM/T-REM cải thiện.
   → Trực giác "budget" của bạn **không sai**, chỉ cần gắn vào đối tượng đúng dưới đây.
3. **Đối tượng "budget" đúng = w-event ε-DP** (Kellaris, Papadopoulos, Xiao, **Papadias**,
   VLDB 2014, [PDF](http://www.vldb.org/pvldb/vol7/p1155-kellaris.pdf)): quỹ đạo là một
   *stream* điểm. `w=1` → **event-level** (bảo vệ 1 điểm); `w=T` → **user-level** (bảo vệ
   cả trace). **Theorem 3 (quy tắc load-bearing):** tổng các ε_i trên **MỌI cửa sổ trượt
   dài w** phải ≤ ε: `∀i: Σ_{k=i−w+1}^{i} ε_k ≤ ε`. Đây **chính xác** là "budget trượt"
   mà trực giác của bạn đang với tới — không phải một túi toàn cục bí ẩn. Ưu điểm: sai số
   phụ thuộc w, **không phụ thuộc độ dài stream** (utility không rơi theo thời gian).

### 2.3 Ba họ "làm tốt hơn nε" (nơi trú đúng của ý tưởng budget)

Tất cả nằm dưới ràng buộc Theorem 3 và khác nhau ở **cách rải budget cửa sổ cố định**:

- **Predictive mechanism / budget manager** (Chatzikokolakis, Palamidessi, Stronati,
  PETS 2014, [arXiv 1311.4008](https://arxiv.org/abs/1311.4008)): mỗi bước có
  prediction Ω + noisy test (tốn ε_θ nhỏ) + hard step (tốn ε_N lớn) **chỉ khi dự đoán
  sai**. Tiêu kỳ vọng/bước `ρ = ε_θ + (1−PR)·ε_N`. → budget hiệu dụng **≪ n·ε** khi di
  chuyển dễ đoán. **Cùng họ tư tưởng với T-REM** ("không tiêu khi release chẳng lộ gì
  mới" / điều kiện reachability).
- **Chiến lược rải w-event** (Kellaris Uniform/Sample/**BD**/**BA**; RescueDP, Wang et al.
  INFOCOM 2016, mở rộng **IEEE TDSC** 2018): **sampling/skip** kéo giãn budget — chỉ trả
  khi release "đáng", thu hồi budget khi timestamp cũ trượt khỏi cửa sổ.
- **δ-location set DP** (Xiao & Xiong CCS 2015, [arXiv 1410.5919](https://arxiv.org/abs/1410.5919)):
  gấp tương quan thời gian vào **adjacency biến thiên theo thời gian** (Markov posterior)
  thay vì chia budget; mỗi timestamp release 1 điểm với ε cố định (*không* composition
  chéo thời gian). Rò rỉ **tích lũy** được định lượng ở follow-up Cao et al. ICDE 2017
  (Temporal Privacy Leakage). PIVE (Yu, Liu, Pu, NDSS 2017) = budget cá nhân hóa theo
  error bound — *caveat*: [arXiv 2101.12602](https://arxiv.org/abs/2101.12602) chỉ ra
  guarantee LDP của PIVE có lỗi hình thức, phải cite kèm phê bình.

### 2.4 Regime phơi lộ THẬT & đơn vị bảo vệ (sporadic / continuous / periodic)

| Regime | Đơn vị bảo vệ | Ví dụ app + mật độ tài liệu hóa |
|---|---|---|
| **Sporadic / one-shot** (điểm độc lập) | **1 query** | "restaurants near me" (Places), check-in Foursquare, tra thời tiết. **Đây đúng regime mà planar-Laplace & optimal-LP Geo-I thiết kế cho.** |
| **Continuous / dense stream** (tương quan mạnh) | **session/trip hoặc cả stream** | Nav turn-by-turn; **Strava ~1s**; **Uber webhook mặc định 4s** ([docs](https://developer.uber.com/docs/guest-rides/references/api/webhooks/driver-location)); live sharing. Per-point independence **hỏng** → nhiễu per-point rò qua tương quan. |
| **Periodic / sampled** | **per-interval → per-day** | geofence wake-up; refresh thời tiết; **RTB bidstream ~17 fix/máy/ngày** ở precision ~1m (FTC Gravy, [EPIC](https://epic.org/ftc-takes-action-against-data-brokers-for-selling-sensitive-location-data/)). Càng dày càng giống continuous. |

Ranh giới **không sắc**: Mendes et al. PoPETs 2020 cho thấy "coi độc lập được hay không"
phụ thuộc tần suất, không có mốc hình thức. Tần suất tăng → sporadic bào mòn thành
continuous.

### 2.5 GeoLife là continuous/dense — KHÔNG sporadic → structure đúng cho luận văn

- GeoLife: 182 user, 17.621 trajectory, **91% log dày 1–5s hoặc 5–10m**, mỗi trajectory
  là chuỗi điểm (lat,lon,alt) có timestamp ([user guide MSR](https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/)).
  Nhịp 1–5s nằm ở **đầu dày của regime continuous** (dày hơn Uber 4s, ngang Strava 1s).
- **Vì thế structure áp dụng là mức-quỹ-đạo, có nhận thức tương quan thời gian.** Giả định
  per-point independence chống lưng các optimal-Geo-I mechanism **KHÔNG giữ**:
  - Bordenabe CCS 2014 nguyên văn *"focus on the case of **sporadic** location disclosure
    … can be considered independent"*.
  - Andrés CCS 2013 cũng *"considers reports independent … discarding the threat from
    correlation"*.
  → Các kết quả "optimal Geo-I" (Shokri CCS'12, Bordenabe CCS'14) là **per-release /
  sporadic**: **dùng được để nói chất lượng 1 release, KHÔNG dùng được làm guarantee mức
  quỹ đạo**. Với quỹ đạo phải quay về ε·T (trần trung thực) hoặc khung stream (§2.2–2.3).

**Câu framing cho luận văn:** *GeoLife hiện thực hóa regime continuous-exposure (chuỗi GPS
dày 1–5s), nơi đơn vị bảo vệ tự nhiên là trajectory/session. Các kết quả optimal-mechanism
của Geo-I (Andrés CCS'13, Bordenabe CCS'14) được chứng minh dưới giả định **sporadic** (các
release độc lập) nên không phủ trực tiếp; mở rộng khung correctness của Shokri và guarantee
Geo-I sang quỹ đạo tương quan **trên road network** chính là khe hở mà T-REM lấp.* Phát biểu
"budget" của luận văn nên là **budget trượt w-event** (§2.2) + báo cáo ε·T làm trần, đặt
predictive/δ-location-set làm hướng beat-nε — thay cho một túi ngân sách toàn cục tự đặt.

---

## 3. Cơ chế đặt ở đâu, nhận data ra sao (system model)

### 3.1 Ba mô hình kiến trúc và deployment thực tế

| Mô hình | Cơ chế đặt ở đâu | Ai đã deploy | Ghi chú cho luận văn |
|---|---|---|---|
| **Local** (làm nhiễu trên thiết bị trước khi gửi — mô hình của Geo-I) | Trên máy người dùng | Android LocationFudger, iOS approximate, **Location Guard** (planar Laplace thật, ~9k user Chrome 2015), LP-Doctor (prototype) | **Chỗ đặt duy nhất mà guarantee đúng với MỌI bên downstream** (LBS, ad exchange, broker, trát tòa). REM/T-REM thuộc mô hình này. |
| **Trusted anonymizer / k-anonymity cloaking** | Máy chủ proxy trung gian | **Chưa từng có LBS thương mại nào deploy** (Yahoo Fire Eagle 2008–2013 là broker coarsening-by-policy, không phải k-anonymity, đã chết) | Lý do chính đáng để luận văn *bỏ* hướng k-anonymity. |
| **Server-side aggregation DP** | Máy chủ, chỉ bảo vệ aggregate | Google COVID Mobility (ε=1,76/user-day, δ=0), Meta Movement Range (ε=2/user-day), SafeGraph, Placer.ai | **Chỉ bảo vệ publication, KHÔNG bảo vệ collection** — raw trace vẫn tồn tại. Google 2024 chuyển Timeline về on-device chính là thừa nhận rủi ro này. |

### 3.2 Sự thật quan trọng: chưa ai chạy local DP trên GPS thô

- Apple có local-DP thật (emoji ε=4, QuickType ε=8...) nhưng **vị trí vắng mặt
  khỏi danh sách** — với location Apple dùng heuristic (rotating trip ID, cắt
  đoạn quỹ đạo, fuzz điểm đầu/cuối trong 24h, ngưỡng hoạt động trên đường nhỏ).
- DP hình thức cho vị trí *chỉ* ship ở central model trên aggregate.
- Local model cho vị trí *chỉ* ship dưới dạng lượng tử hóa **không có guarantee**
  (grid 2km, region 10 dặm², round 1 dặm) hoặc planar Laplace quy mô nhỏ
  (Location Guard).
- **→ Vị trí của luận văn**: cơ chế ε-Geo-I client-side với output on-road là
  *tổng hợp* của hai truyền thống đã ship — guarantee hình thức từ dòng DP,
  thiết kế output-rời-rạc-hợp-lý từ dòng OS quantizer. Đây là câu chuyện
  "academic có đảm bảo nhưng chưa deploy được / industry deploy được nhưng không
  có đảm bảo" mà luận văn lấp vào giữa.

### 3.3 Tính khả thi on-device (đã đo trên chính data của project)

- Graph Bắc Kinh của project: 77.727 nodes / 208.994 edges = **33MB pickle**;
  nếu pack mảng float32/int32 chỉ **~2,5MB**. Nhỏ hơn một tập podcast.
- Điện thoại thừa sức: offline maps (Organic Maps toàn Đức ~650MB, OsmAnd ~950MB,
  Google offline city 200–600MB *kèm routing turn-by-turn*). Một lần lấy mẫu
  exponential mechanism trên vài trăm–vài nghìn đỉnh **rẻ hơn một lần định tuyến**.
- LP-Doctor (USENIX Sec 2015) đã chạy planar Laplace trên Android với chi phí
  không cảm nhận được, **và cache output cho query lặp** — tiền lệ đúng cho việc
  T-REM điều kiện trên điểm đã công bố.

### 3.4 Payload một request LBS thật chứa gì

- Places Nearby Search: `location=lat,lng` + `radius` + `key` (+keyword/type) —
  đúng interface 1 điểm + bán kính cho cơ chế point-perturbation.
- OpenRTB `device.geo`: `lat`, `lon`, `type`, **`accuracy`(m)**, `lastfix`(s) +
  `device.ifa` (Mobile Ad ID) gửi cho *mọi* bidder → adversary thực tế là broker
  giữ **toàn bộ chuỗi release dưới một identifier** (FTC: Gravy xử lý 17 tỷ
  tín hiệu/ngày, ~17 fix/máy/ngày). Khớp đúng threat model Bayesian+HMM.

### 3.5 System model đề xuất cho luận văn (mỗi phần gắn tiền lệ đã ship)

1. **Cơ chế chạy client-side** (mô hình local), tại tầng OS location hoặc callback
   của app. Tiền lệ: LocationFudger + iOS approximate (ship cho hàng tỷ máy),
   Location Guard + LP-Doctor (planar Laplace cụ thể).
2. **Cơ chế được biết**: (a) fix thật hiện tại; (b) **road graph công khai lưu
   trên máy** (public → điều kiện trên nó không tốn privacy, post-processing);
   (c) **các output ĐÃ công bố của chính nó** (không phải vị trí thật trước đó) —
   tiền lệ LP-Doctor cache-noise + LocationFudger random-walk offset.
3. **Server/adversary thấy**: điểm nhiễu on-road + metadata transport (timestamp,
   nội dung query, app id, identifier linkable). Model adversary = LBS honest-but-
   curious hoặc broker giữ toàn chuỗi release dưới một id, có road graph + mobility
   prior → đúng bộ attack Bayesian + HMM. Identifier linkable trong 1 trip, tùy
   chọn xoay giữa các trip (thiết kế rotating-trip-id đã ship của Apple cho ranh
   giới "session" để cite).
4. **Interface tiện ích**: 1 điểm + QoS radius mỗi query; máy báo `accuracy` nhất
   quán cho điểm nhiễu.

---

## 4. Các tấn công có thật → taxonomy scenario

Mỗi scenario: (a) khả năng adversary, (b) data thấy, (c) thuộc tính phòng thủ đối
ứng → gắn với cơ chế/metric đã có trong `core/mechanisms.py`, `evaluation/`.

| # | Lớp tấn công (ví dụ THẬT) | (a) Khả năng | (b) Data thấy | (c) Phòng thủ → cơ chế/metric |
|---|---|---|---|---|
| **S1** | **Trilateration / distance oracle** (Tinder 2014, Grindr, Bumble 2021+2024 pinpoint ~2m, happn) | Query lặp, spoof vị trí mình, biết luật rounding | Distance / distance làm tròn / oracle nhị phân | ε-Geo-I bound làm mọi điểm gần bất khả phân biệt; distance suy từ **release đã lượng tử/on-road** → **REM**. Round/hide-distance ad-hoc *không* phải guarantee — lớp này là bằng chứng thực nghiệm. Cần budget cho query lặp (S4). |
| **S2** | **Map-matching / lọc off-road** (Takagi; RAoPT; Strava snap-to-street) | Biết lưới đường | Chuỗi điểm công bố | **On-road theo cấu trúc** — không có mass off-road để loại → **REM/T-REM**; đo `on_road_rate` (PL 0,58–0,68 vs REM/T-REM 1,00). Luận điểm trung tâm REM > "planar-Laplace-rồi-snap". |
| **S3** | **Reachability/velocity linkage + HMM tracking** (Ghinita GIS'09; Xiao-Xiong CCS'15; Mendes CODASPY'23) | Biết v_max + mô hình Markov + toàn chuỗi + timestamp | Quỹ đạo công bố + thời gian | **Trọng số reachability điều kiện chỉ trên điểm ĐÃ công bố** → fake khả thi động học, ε per-point không đổi → **T-REM**; đo `speed_violation_rate` (REM 0,35 → T-REM 0,06 @ε=0,01) + `HMMTrackingAttack`. |
| **S4** | **Averaging / báo lặp từ 1 chỗ tĩnh** (Andrés nε; Mendes PoPETs'20; Strava EPZ home recovery 84–95%) | Thu nhiều release cùng 1 điểm thật | Release lặp từ 1 chỗ | **Không phát nhiễu độc lập mới cho cùng điểm thật**: memoization / noise tương quan + kế toán ε·T. *Khoảng trống thật:* 4 cơ chế hiện CHƯA memoize; T-REM chỉ giúp một phần → future work, báo cáo ε_total. |
| **S5** | **Bayesian optimal inference / remap có prior** (Shokri S&P'11, CCS'12; Oya CCS'17) | Biết cơ chế (Kerckhoffs) + prior | Điểm công bố | **Giữ nguyên bound ε — không bao giờ cap/reject/re-check theo điểm thật** (chính lý do `BaselineThesis` vỡ). Support on-road có prior thu hẹp lợi ích remap; đo `expected_inference_error` với adversary có thông tin → **REM/T-REM**. |
| **S6** | **Re-id qua home/work uniqueness + linkage** (de Montjoye 4 điểm→95%; Golle-Partridge block-unique; NYC taxi; priest 2021; NYT 2019) | Có full trace pseudonym + public records | Toàn quỹ đạo dưới 1 pseudonym | Nhiễu per-point *không đủ* — bản thân quỹ đạo là identifier. Đối ứng: composition ε·T đẩy mỗi điểm đủ xa + báo cáo inference error. **Đây là section MOTIVATION mạnh nhất** (pseudonym ≠ ẩn danh). |
| **S7** | **Aggregation failure mật độ thấp** (Strava heatmap → căn cứ quân sự 2018; NC State home ID 2023) | Đọc aggregate/heatmap, mật độ thấp | Raster tổng hợp | k-anonymity / suppression theo mật độ tối thiểu trên bản phát aggregate. Ngoài scope LPPM per-user nhưng cần nêu ranh giới. |
| **S8** | **API / side-channel / supply-chain** (Polar API 2018; fake Strava segments; SDK/RTB resale; Gravy breach 2025) | API quá hớ / bidstream / corpus rò | Raw data, không obfuscation | **Ngoài scope cơ chế nhiễu** — access control, rate limit, retention/consent. Nêu 1 đoạn để định ranh giới đóng góp. |

### Cách dùng trong luận văn
- **S1, S2, S3, S5**: chứng minh được ngay với code hiện tại (`on_road_rate`,
  `speed_violation_rate`, `BayesianPointAttack`, `HMMTrackingAttack`). Bảng
  benchmark đã cho thấy lợi thế REM/T-REM ở S2/S3 và luận điểm chống baseline ở S5.
- **S4**: khoảng trống thật, chưa cơ chế nào memoize → nêu thẳng, cite Andrés (nε)
  + Mendes PoPETs'20, đặt memoization/budget làm future work (đón đầu phản biện
  "chỉ đánh giá per-point là lỗi reviewer bắt từ 2015").
- **S6**: section motivation mạnh nhất — de Montjoye, Golle-Partridge, Krumm
  (~13% homes), Citi Bike 84%, vụ priest/NYT/taxi → biện minh cho ε-Geo-I mức
  quỹ đạo thay vì pseudonymization/k-anonymity.
- **S7/S8**: định *ranh giới* đóng góp — 1 đoạn để scope defensible.

---

## 5. Việc cần đưa vào luận văn từ khảo sát này

1. **Viết chương Problem Formulation theo khung chuẩn** (Ch3/Ch4) dùng §2.1 (bộ
   khung Shokri/Andrés/Bordenabe: 𝒳, π, f(z\|x), Q, ε-Geo-I, expected inference
   error) — dùng đúng ký hiệu literature, không tự chế.
2. **Phát biểu lại "privacy budget" theo §2.2** — bỏ khái niệm túi toàn cục tự đặt;
   nêu ε là *mức* metric-DP của 1 release, ε·T là trần trung thực (composition tuần
   tự), và **budget trượt w-event ε-DP** (Kellaris VLDB'14, Theorem 3) là đối tượng
   đúng; đặt predictive-mechanism / δ-location-set làm hướng "beat-nε". Đóng khung
   GeoLife là **continuous exposure** (§2.5) → nêu rõ optimal-Geo-I là *sporadic* nên
   không phủ trace dày ⇒ đây là gap T-REM lấp.
3. **Viết lại QoS thành (α,δ)-usefulness** (Ch4/Ch5) — bỏ ngôn ngữ "hard cap",
   giữ đúng cách đo hiện có (bảng đã đo QoS = P[d≤200m]).
4. **Thêm chương/section System Model** (Ch4) dùng §3.5 — nêu rõ local model,
   cơ chế biết gì, adversary thấy gì, kèm tiền lệ đã ship + con số on-device 33MB.
5. **Viết lại chương đánh giá quanh taxonomy S1–S8** (Ch3/Ch5) — mỗi scenario có
   ví dụ tấn công thật, biến "chúng tôi nghĩ trông thực tế" thành "đối ứng vụ X
   có thật". Bổ sung S4 (averaging) làm limitation/future work đã định lượng.
