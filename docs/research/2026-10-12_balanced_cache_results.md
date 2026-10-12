# Bản cân bằng Geo-I: chứng minh trước, benchmark kiểm chứng

**Cấu hình nghiên cứu được đề xuất:** Geo-I/REM cap0,23m⁻¹ mỗi phiên;
K=5/L=30 mỗi category; ΓQ=1; chọn ba khung Q chung mỗi60s; gửi theo clock20s;
public exploration ưu tiên POI chưa nhận; giữ metadata hợp lệ tại cache local
trong phiên. Mã: `CacheExplorePolicy` + `SessionCoverClient` +
`PersistentQueryBudget`. [Config](../../artifacts/benchmarks/cache_balance_20261012_v2/balanced_candidate.json).

Đây là **balanced research candidate**, có bảo đảm và trade-off rõ ràng;
không tuyên bố thắng mọi metric, không tự thay canonical production engine.
Gate−2pp cũ vẫn được giữ và bản này chưa đạt. Ngưỡng riêng chọn trước cho
candidate là mean macro≥90%, mean từng purpose≥80%, kèm toàn bộ CI;
không đổi gate cũ sau khi xem số liệu.

## 1. Những gì được bảo đảm bằng toán học

- **Kênh Q toàn phiên:** ratio≤R11≈2,2343 ở600s; mọi thời lượng≤2,4374 khi
  Γ1, với bit RNG đều độc lập và logical-public-clock contract. Kernel lấy
  mẫu bằng integer intervals, không exp/softmax/floating probabilities.
- **Backbone Geo-I:** nếu protected history upstream thỏa Geo-I lý tưởng,
  cùng wire có log-ratio≤min{Cs·D∞,logR}. Bảo đảm Γ của kênh Q không cần
  belief calibrated; toàn Geo-I float/hardware stack chưa được chứng nhận.
- **Utility ở GPS thật:** cache hợp lệ chứa initial pool, nên Recall tại
  cùng GPS/request≥control giữ initial Q cùng lần lấy mẫu. Bound này không
  yêu cầu belief chính xác và không giả sử IID.
- **Thay đổi muộn:** causal selection chỉ tiêu privacy của những blocks có
  thể bị ảnh hưởng. Histories giống trước200s có ratio ở0–600s≤1,1224;
  đây là bound conditional theo cặp histories, không phải mọi future target.
- **Giới hạn downside của personalization:** cùng prefix/support, chênh lệch
  kỳ vọng của bất kỳ utility[0,1] so với public base≤α/4:5pp selection đầu,
  1,923pp selection thứ hai. Không cần belief calibrated; không áp bound này
  cho chênh lệch giữa hai policies chạy cả phiên.
- **Actual gain từ belief:** có cận theo uniform proxy error δ. Khi δ chưa
  được chứng nhận, không gán0 và không gọi planning gain là actual gain.

[Chứng minh, giả thiết và bảng nối với mã](2026-10-12_balanced_cache_proof.md).
Identity người/phương tiện và future route được giới hạn tăng posterior odds;
không nói “không thể suy ra”, không ẩn account/IP/biển số. QuerySpec/GPS thật
và local sorted answer không gửi server; tương quan content–location vẫn có
thể gây suy luận. Các benchmark arms là **views riêng**, không phải gửi cả
bảy luồng song song trong app và áp một budget Γ cho tổng luồng đó.

## 2. Utility ngoài mẫu và trade-off

Cohort mới48 groups:20 TRAIN,8 SELECTION,20 TEST;176 evaluated trip-draws,
120 TEST, ba draws/trip. Toàn384 native trips được giữ để kiểm tra nguồn.
Original24 và hai development cohorts120 junction groups bị loại; routes
near-duplicate bị loại bằng public geometry. Source/config freeze trước
private cohort generation, không fit defense trên TEST hoặc sửa GPS.

**Mọi arm dùng cùng session cache và K5/L30.** Bảng sau là service answer
sau local sorting từ metadata đã nhận; top-5 chỉ dùng cho metric. CI resample
20 TEST families, giữ siblings/draws bên trong family; không coi các GPS
points là independent samples.

| Phương án | Macro Recall@5 (%) | Gần nhất | Nhanh nhất | Bán kính1000m | Ít vòng đường |
|---|---:|---:|---:|---:|---:|
| Geo-I baseline | 99.995 | 99.996 | 99.996 | 99.995 | 99.996 |
| PQB trước sửa | 85.479 | 89.864 | 89.860 | 73.170 | 89.023 |
| Q giữ pool từng khung | 92.569 | 94.838 | 94.840 | 86.401 | 94.196 |
| Giữ initial Q | 92.569 | 94.838 | 94.840 | 86.401 | 94.196 |
| Exploration + cache, Γ0 | 96.060 | 97.481 | 97.482 | 92.069 | 97.207 |
| Exploration + public prior, Γ1 | 95.497 | 97.019 | 97.024 | 91.125 | 96.819 |
| Geo-I + exploration + cache, Γ1 | 95.548 | 97.129 | 97.132 | 91.025 | 96.907 |

Primary đạt **95,548%**, family-bootstrap95%CI **[93,329;97,444]%**.
Bán kính là mục đích yếu nhất:91,025%,CI[85,595;95,528]%. Đây là empirical
interval trên synthetic cohort, không phải mathematical absolute-QoS bound.

| So sánh trên cùng cohort/cache | Δ Macro (pp) | Paired95%CI (pp) |
|---|---:|---:|
| Primary − Geo-I baseline | -4.447 | [-6.665;-2.553] |
| Primary − PQB trước sửa | +10.069 | [+4.858;+15.631] |
| Primary − Q giữ pool từng khung | +2.980 | [+2.077;+3.904] |
| Primary − Giữ initial Q | +2.980 | [+2.077;+3.904] |
| Primary − Exploration + cache, Γ0 | -0.512 | [-1.186;+0.026] |
| Primary − Exploration + public prior, Γ1 | +0.052 | [-0.457;+0.522] |

So với bản giữ pool, gain **+2,980pp** có CI dương. Ba draws đều tăng:
**+2,316;+3,471;+3,152pp**. Gain đến từ retrieval/exploration/cache;
**không có chứng cứ protected belief làm tăng actual Recall** so với Γ0/
public-prior. Γ0 có point estimate tốt hơn primary, nên không nói Γ1 là
Pareto-optimal. Γ1 giữ khả năng adaptation theo protected Geo-I; giá trị
service của adaptation trên protocol static này vẫn chưa được xác nhận.

So với baseline mất4,447pp,CI[−6,665;−2,553]pp; chưa đạt−2pp. Đổi lại,
prototype có additional global inference cap và actual-cache invariant.
Đây là trade-off được giữ nguyên, không “sửa cho đẹp” các kết quả chưa đạt.

### Bảo đảm cache có vai trò thực tế

Cùng primary Q stream: current-frame Recall87,941%, epoch60s88,445%,
**session cache95,548%**. Session−epoch là+7,103pp,CI[+5,093;+9,370]pp.
Vì Q stream không đổi trong phép so sánh này, gain là local postprocessing,
không thêm query hay privacy budget. Current-frame performance giảm so với
old guard92,569%: provider không cho session-valid cache thì không được áp
95,548% của cấu hình này sang contract đó.

| Giai đoạn | Baseline + session cache (%) | Q giữ pool (%) | Primary (%) |
|---|---:|---:|---:|
| 0–60s | 99.993 | 90.965 | 91.066 |
| 80–300s | 99.991 | 92.634 | 95.322 |
| 320–600s | 100.000 | 92.854 | 96.852 |

Warm-up còn hạn chế: primary91,066% trong0–60s, lên96,852% trong320–600s.
Initial pool trung bình279,82 POI, cuối phiên355,75 POI: thêm75,93;
mọi TEST draw nhận thêm ít nhất45 và nhiều nhất106 POI. Old guard không
nhận thêm POI nào trên cohort này. Không biến mean cuối phiên thành một
coverage lower bound cho mọi vị trí hay mọi chuyến.

## 3. Attacker và chi phí

Mỗi ô **Hit100% / MAE mét**. Hit thấp và MAE cao có lợi cho privacy.
Attacker fit TRAIN, selection theo từng loss trên SELECTION, lưu model trước
TEST; bank gồm ExtraTrees/kNN và public geometry, ordered endpoint candidates.
Mathematical bound không phụ thuộc bank này tối ưu tới đâu.

| Scenario | Geo-I baseline | Q giữ pool | Primary |
|---|---:|---:|---:|
| S1 | 0.941% / 845.5 | 0.000% / 2104.6 | 0.054% / 2104.6 |
| S2 | 0.000% / 796.4 | 0.000% / 2124.2 | 0.000% / 2124.2 |
| S3 | 0.941% / 763.9 | 0.000% / 2104.6 | 0.054% / 2104.6 |
| S9 | 1.667% / 813.8 | 0.000% / 1986.4 | 0.000% / 2122.6 |
| S10 | 0.833% / 772.1 | 0.000% / 2171.4 | 0.000% / 2124.2 |

Không tốt hơn old guard ở mọi privacy metric: S1/S3 Hit tăng nhẹ0→0,054%;
S10 MAE giảm2171,4→2124,2m. Hit0% trên120 trip-draws không nghĩa probability0.
S2 là stopped-window coordinate proxy; không có sensitive-place semantic
classifier. S4–S6 là conditional inference analysis, không là empirical
identity benchmark.

Mọi arm155 point requests/trip, L30 mỗi category. Primary reply serialization
khoảng48,683KB, baseline50,507KB; đây là POI indices JSON, không gồm HTTP/TLS/
metadata payload thật hay latency/battery. Session mask418 bytes boolean;
không phải tổng RAM của graph/score tables/cache. Sorting trả danh sách phù
hợp đầy đủ, không ép người dùng nhận5 items.

## 4. Những giới hạn phải giữ trong bản thesis/paper

1. **Static metadata còn hợp lệ trong phiên.** Không kéo dài opening/status/
   availability TTL. Dynamic expiry có thể phá monotonicity và cần nghiên
   cứu riêng. Full-catalogue/bulk-download control có thể thắng trong contract
   khác; không gọi cache là contribution riêng của Geo-I.
2. **Same map/generator, synthetic data.** Các checks xác minh provenance/
   raw GPS, không chứng minh population representativeness. Đợt v1 thiếu
   groups49/60 nên không benchmark defense; v2 giảm48 theo public supply,
   giữ tiêu chuẩn route và freeze lại trước evaluation. Dấu vết được giữ.
3. **Mean95,55% không là guarantee cho mọi user.** CIs là descriptive paired
   family-bootstrap;20 TEST và8 SELECTION còn nhỏ. Không điều chỉnh model
   theo TEST hoặc bỏ một purpose khó để đạt gate.
4. **Bit/source assumptions.** Exact integer Q PMF cần fresh independent
   uniform bits; experiment PCG và upstream Geo-I float chưa certified.
   Logical-wire proof không là hardware/account/network-metadata anonymity.
5. **Vai trò Geo-I được giới hạn đúng.** Giữ backbone và theoretical spatial
   guarantee, nhưng current utility gain chủ yếu từ cache/exploration.
   Không chứng minh Geo-I cần thiết cho static catalogue, không giấu Γ0 control.

## 5. Bàn giao và quyết định

Đề xuất dùng **Geo-I + bounded Q Γ1 + public exploration + local static
session cache** làm bản cân bằng để thảo luận với GVHD. Cấu hình đạt hard
checks và mức utility nghiên cứu đã đặt trước. Giữ status research candidate,
không tự promote canonical production engine hoặc xóa những kết quả cũ.
Chứng minh là đóng góp chính; benchmark cho mức chi phí/utility trong scope
đã kiểm tra, không là lý do đòi mọi aspect phải hoàn hảo.

Kiểm tra độc lập: 7,920 exact finite-selection decisions; 114,576 cache masks và actual utility được dựng/tính lại; 21,241 local-purpose comparisons≥matched initial.
Toàn suite: **1075 passed,9 skipped**. Native integrity:249,9841Hz fixes,
192 native counterfactual trips,288 archives. Hai corrective records giữ lý
do đổi cỡ cohort và archive-count verifier; không sửa defense/generator sau
freeze để thay kết quả.

- [Proof](2026-10-12_balanced_cache_proof.md).
- [Frozen protocol](../../artifacts/benchmarks/cache_balance_20261012_v2/protocol.json),
  [results](../../artifacts/benchmarks/cache_balance_20261012_v2/results.json),
  [controls và shared-prefix conditional certificates](../../artifacts/benchmarks/cache_balance_20261012_v2/secondary_readout.json).
- [Independent validation](../../artifacts/benchmarks/cache_balance_20261012_v2/independent_validation.json),
  [native integrity](../../artifacts/datasets/cache_fresh_native_20261012_v2/validation.json).
- [Previous results](2026-10-12_finite_progress_results.md) giữ nguyên.
