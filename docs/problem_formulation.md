# Problem Formulation — Reframe theo industry + Requirements + Chỉnh hình thức

*Khảo sát 2026-08-20 (2 research agent, nguồn primary verify). Đây là bản grounding
cho chương phương pháp: (1) phát biểu lại bài toán theo cách industry đặt; (2) bảng
requirements R1–R12 ↔ nguồn ↔ harm S1–S8; (3) ba chỉnh hình thức bắt buộc cho REM/
SM-REM; (4) positioning vs công trình sát nhất. Đi kèm `system_model_and_threats.md`
(QoS/budget/threats) và `attack_scenarios.md` (catalog tấn công).*

---

## 1. Reframed problem statement (thay cho "làm nhiễu mỗi điểm trong bán kính δ")

> LBS không cần vị trí *chính xác*; nó cần **mức granularity tối thiểu đủ cho một mục
> đích cụ thể**, và mọi nền tảng lớn đã phản ánh điều này — Apple/Android phơi bày các
> tầng độ chính xác theo query, xử lý significant location on-device, và fuzz điểm
> đầu/cuối hành trình; còn quy định (GDPR Art. 5; ngưỡng "precise geolocation" 1.750–
> 1.850 ft của luật bang Mỹ; các lệnh FTC với Kochava/X-Mode/InMarket) yêu cầu purpose
> limitation, data minimization, storage limitation và loại trừ vị trí nhạy cảm. Nhưng
> thứ ship trong thực tế chỉ là **lượng tử hóa heuristic không guarantee** (snap-to-grid,
> rounding) hoặc **DP hình thức chỉ ở server trên aggregate** — để hở khoảng trống: một
> **cơ chế client-side, on-device, cho guarantee ε-Geo-I hình thức với mọi bên downstream,
> chỉ phát ra vị trí on-road thực tế, kháng các tấn công tương quan gây hại thật
> (velocity-linkage, averaging stay-point, map-matching), và coi significant location
> (nhà/chỗ làm/phòng khám) là endpoint được bảo vệ hạng nhất**. Luận văn reframe bài toán
> từ "obfuscate mỗi điểm trong bán kính QoS δ toàn cục" → **"phát ra mức tối thiểu theo
> mục đích, on-device, trên lưới đường, có guarantee endpoint-aware và correlation-aware,
> đánh giá theo threat model rút từ tấn công thật S1–S8".**

## 2. Bảng requirements R1–R12 (nguồn → harm đối ứng)

| # | Requirement | Nguồn industry / pháp lý / framework | Harm |
|---|---|---|---|
| R1 | Perturbation client-side / on-device | Apple "process on device"; Google Timeline on-device; AOSP LocationFudger; GDPR controller model | tiền đề; hạn chế S6,S8 |
| R2 | Granularity theo mục đích (minimization), chọn từ **context công khai** | GDPR 5(1)(c); iOS `CLLocationAccuracy` tiers; Android FLP priorities; Micinski MoST'13; Contextual Integrity (Nissenbaum) | S1,S6 |
| R3 | Bất khả phân biệt hình thức các vị trí gần; distance suy từ release đã lượng tử | W3C explainer "must ensure it is not possible to infer the user's precise location"; ε-Geo-I; LINDDUN Identifying/Detecting; NIST disassociability | S1,S5 |
| R4 | Output on-road hợp lệ; không có mass off-road để prune | LocationFudger snap-to-grid; map-matching literature; REM | S2 |
| R5 | Kháng tương quan thời gian/velocity; điều kiện chỉ trên **điểm đã công bố** | Apple rotating trip-ID; Xiao–Xiong δ-location-set; T-REM | S3 |
| R6 | **Ổn định khi báo lặp (anti-averaging)** — cùng vùng trong cửa sổ trả cùng giá trị | Google/WICG *approximate-geolocation explainer* (đề xuất, KHÔNG phải W3C Recommendation) — rule "cùng site trong cửa sổ → cùng cached value"; AOSP persistent offset; LP-Doctor; RAPPOR-PRR | **S4** |
| R7 | Bảo vệ endpoint/stay-point hạng nhất (memoized nhất quán, fuzz lớn hơn, không hình học lộ tâm) | Apple "convert precise locations from the beginning of your route"; Strava hide start/end; Dhondt CCS'22 (85% recovery ⇒ hiding ngây thơ không đủ) | S4,S6,S7 |
| R8 | Disassociation định danh (rotating id; pseudonym ≠ anonymity) | Apple rotating id; NIST disassociability; Zang–Bolot | S6,S8 |
| R9 | Consent + loại trừ vị trí nhạy cảm | FTC Kochava/X-Mode/InMarket; GDPR 6(1)(a) | S6,S8 |
| R10 | Purpose limitation (không dùng lại) | GDPR 5(1)(b); Solove; CI | S6,S8 |
| R11 | Storage limitation / retention / deletion | GDPR 5(1)(e); FTC orders; Google auto-delete 3 tháng | S6,S8 |
| R12 | Minh bạch & kiểm soát người dùng (accuracy indicator, precise/approximate toggle) | Apple halo + toggle; iOS `CLAccuracyAuthorization`; GDPR 5(1)(a); LINDDUN Unawareness | cross-cutting |

**Scope trung thực — cơ chế (REM/T-REM/SM-REM/PR-SM-REM) LÀM được:** R1 (client-side), R3 (ε-Geo-I
Euclid), R4 (on-road by construction), R5 (T-REM reachability), **R6 (SM-REM/PR-SM-REM memoization
= đúng anti-averaging spec của W3C, nhưng scope hẹp — chỉ static exact-repeat; pattern thăm-lại
vẫn lộ)**, R7 một phần (memoized endpoint), R2 có điều kiện
(chọn ε/δ từ context công khai, KHÔNG từ vị trí thật). **KHÔNG tự làm được (thuộc governance/
identity/UX phía khác):** R8 (rotating identifier), R9–R11 (consent/retention/purpose — phía
server), R12 (UX). **S6 chỉ vá một phần:** nhiễu per-point không xóa tính duy nhất của cả
quỹ đạo (de Montjoye 4 điểm→95%) — cần ε·T/w-event và thừa nhận nhiễu + identifier cố định
≠ ẩn danh. **S7/S8 ngoài scope** (aggregation suppression / access control).

## 3. Ba chỉnh hình thức BẮT BUỘC (từ Agent A — tránh viết định lý sai)

- **H1 — REM là ε-Geo-I Euclid vì score dùng khoảng cách Euclid.** REM = exponential
  mechanism score `−(ε/2)·d_Euclid(x,v)` trên tập đỉnh rời rạc → single-shot **thật sự
  ε-Geo-I Euclid** (output rời rạc/on-road không phá lập luận EM). REM chính là bản Euclid
  của **GEM (Graph-Exponential Mechanism, Takagi et al. 2020)** — phải cite GEM làm tiền lệ
  trực tiếp REM tái dẫn. *Chỉ* được gọi ε-Geo-I nếu score là Euclid; nếu dùng graph distance
  thì phải gọi ε-GG-I và KHÔNG có guarantee Euclid.
- **H2 — memoization KHÔNG cho ε-Geo-I Euclid; cho ε-Geo-I ở mức cell.** SM-REM thỏa
  **ε-d̃X-privacy theo pseudometric của cell** `d̃(x,x')=d(rep(cell x),rep(cell x'))`; budget
  cộng **một lần mỗi cell riêng biệt** → O(#distinct)·ε thay vì O(#reports)·ε; stay-point
  tĩnh tốn đúng ε bất kể dwell (giết averaging). **Caveat bắt buộc nêu:** gián đoạn ở biên —
  hai điểm ε-sát hai bên biên nhận release độc lập (ratio tới exp(ε·cellwidth)); đây là lý do
  LocationFudger thêm persistent offset, và là lý do **không thể vừa có Euclid-Geo-I sạch vừa
  có memoization anti-averaging từ cùng một knob**. Không phải pan-privacy.
- **H3 — trigger memoize phải độc lập dữ liệu.** SM-REM memoize theo lưới công khai *mọi lần*
  → an toàn, không rò rỉ. (Nếu gate bằng stay-point predicate tính trên stream thật thì phải
  bọc trong Above-Threshold/PTR test — nếu không quyết định đó là leak chưa tính budget.)

## 4. Positioning (khe hở SM-REM lấp)

- **PTPPM** (Cao et al. 2024, arXiv:2401.11225) & **road-network PTPPM** (Min et al. 2025,
  arXiv:2511.21020): Permute-and-Flip **stateless**, mô hình *chỉ* tương quan liên-timestep,
  không road-native-by-construction, và **tự thừa nhận** *"repeated releases of identical true
  locations remain vulnerable if the perturbation mechanism is stateless."*
- **GEM/GG-I** (Takagi 2020): single-shot, không chống averaging.
- **Eclipse** (Niu et al. IEEE TMC 2020): chống long-term observation nhưng **off-road** qua
  anonymity set, không memoization.
- **Khe hở**: release **on-road-by-construction + memoized (stateful)** để báo lặp từ stay-point
  tĩnh không thể bị averaging — chưa ai chiếm. SM-REM (memoized, on-road, điều kiện chỉ trên
  điểm đã công bố) nằm đúng đó.

**Cách phát biểu guarantee cho luận văn — phân biệt ĐÃ hiện thực vs KIẾN TRÚC ĐÍCH
(verifier R2-002/R2-010):**

- *Đã hiện thực & test:* {ε₀-Geo-I Euclid mỗi release tươi (REM/T-REM)} + {SM-REM: static
  exact-repeat trả cache ⇒ averaging vô hiệu, nhưng revisit-channel có ratio ∞ — không phải
  per-release bound} + {PR-SM-REM: noisy-threshold test thay reuse test chính xác ⇒ per-release
  `(ε_test+ε_release)`-Geo-I hữu hạn, thay cho ∞ của SM-REM}.
- *Kiến trúc đích (CHƯA hiện thực — future work):* {Σ budget metric-DP ≤ ε_w trong mọi cửa sổ
  w timestamp — w-event, Kellaris VLDB'14 Thm 3}; hiện chỉ có per-release bound, **chưa có
  w-event budget manager thật**. Nâng ε_w lên chặn TPL-supremum (Cao et al. TKDE 2019) nếu
  muốn claim correlation-robust cũng thuộc future work.
