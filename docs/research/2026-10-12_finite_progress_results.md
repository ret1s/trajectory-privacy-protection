# Cải thiện chứng minh và benchmark Geo-I — 12/10

Giữ Geo-I/REM và cap0,23/m mỗi phiên. Hai vòng phát triển được đóng băng
riêng trước khi sinh từng cohort; mọi phương án, draws và kết quả được giữ.
Đây là phần mở rộng thử nghiệm; số liệu cùng cohort mới được so sánh paired,
không thay số liệu cũ bằng kết quả từ cohort khác.

## 1. Thay đổi có căn cứ từ chứng minh

1. **Xác suất chọn Q bằng số nguyên.** Bounded centered tilt có support chung,
   PMF chuẩn hóa chính xác và likelihood ratio≤1+ε_j. Loại lỗi xác suất do
   float ở riêng kênh Q; production cần bit đều độc lập, study dùng PCG private.
2. **Privacy toàn phiên.** Chain rule trên hành động Q đã bảo vệ, gồm nhãn
   lane giả; phép chiếu xuống tọa độ giữ cận. Với Γ1,11 lần chọn có ratio≤2,2343;
   mọi thời lượng có cận hữu tỷ≤2,4374. Không khẳng định không thể suy luận
   identity/future; đây là cận tăng posterior odds, với điều kiện threat model.
3. **Pool hợp lý hơn.** Loại N/A khỏi public-purpose normalization trước khi
   trộn belief; beam public trên reply signatures cải thiện độ phủ. Public
   floor khác actual Recall; không coi sai số belief/proxy bằng0.
4. **Chuyển động không làm mất POI.** Bản hai chỉ cho Q đi tới khung có pool
   chứa toàn bộ pool trước đó. Vì vậy tại cùng GPS/request, actual Recall≥
   control giữ nguyên Q ban đầu cùng lần lấy mẫu; không cần belief đúng.
   Bảo đảm này không phải noninferiority so với baseline Geo-I.

[Chứng minh đầy đủ và bảng định lý–mã–kiểm tra](2026-10-12_finite_privacy_utility.md).
K5/L30, clock20s và chọn đoạn60s không thay; mỗi Q lấy L30 **mỗi category**.
Request thật và GPS cho sorting giữ local, ứng dụng trả danh sách đầy đủ.
Recall@5 chỉ dùng để đánh giá. Utility dùng GPS thật ở mỗi mốc20s trong
evaluator;60s là cadence của kênh **bảo vệ**, không là số lần GNSS/device
được sử dụng. Pin năng lượng và sai số sensor không được đo ở hai runs này.

## 2. Vòng một: hiệu chỉnh pool và PMF

Source/config freeze trước cohort seed2026101241. Train24/selection12/test24
junction families;8 native trips/family, chỉ ngày7/8 được đánh giá; test3draws.
Tổng216 trip-draws,144 TEST; cùng một protected Geo-I history cho mọi arm.

| Phương án | Macro Recall@5 (%) | Gần nhất | Nhanh nhất | Bán kính1000m | Ít vòng đường |
|---|---:|---:|---:|---:|---:|
| Geo-I baseline L30 | 99.670 | 99.738 | 99.738 | 99.494 | 99.712 |
| PQB theo đoạn trước sửa | 86.976 | 89.944 | 89.932 | 78.317 | 89.711 |
| Kernel integer + thư viện cũ | 86.154 | 89.227 | 89.190 | 77.308 | 88.891 |
| Portfolio Γ0 | 94.460 | 95.717 | 95.719 | 90.793 | 95.612 |
| Portfolio + public prior Γ1 | 94.382 | 95.648 | 95.650 | 90.663 | 95.567 |
| Portfolio + belief Geo-I Γ1 | 94.191 | 95.547 | 95.549 | 90.210 | 95.458 |

Primary tăng **7,215 điểm phần trăm** so với PQB trước sửa, paired95%
**[+2,360;+12,268]pp**, resample24 test families; ba draws nằm trong family.
So với baseline, giảm5,479pp,95%CI[−8,350;−2,927]pp: **không đạt gate−2pp**.
Theo mục đích, bán kính còn yếu nhất.

Γ0 có94,460%, public-prior Γ1 có94,382%, protected-belief Γ1 có94,191%.
Vì vậy **không chứng minh belief Geo-I làm tăng actual Recall**. Gain planning
score được chứng minh là một surrogate; đối chứng cho thấy phần lớn gain
thực tế đến từ pool public. Các finite attacker cũng không là chứng minh
identity anonymity. 54,92% quyết định test chỉ có một joint action hợp lệ,
động cơ để mở rộng thư viện chuyển động mà vẫn giữ QoS.

[Protocol](../../artifacts/benchmarks/finite_pool_20261012_v1/protocol.json),
[results](../../artifacts/benchmarks/finite_pool_20261012_v1/results.json),
[paired controls](../../artifacts/benchmarks/finite_pool_20261012_v1/secondary_readout.json),
[independent verification](../../artifacts/benchmarks/finite_pool_20261012_v1/independent_validation.json).

## 3. Vòng hai: motion giữ POI, cohort riêng

Seed2026101243; source/config freeze trước generation. Loại original24 và
vòng một60 junction groups cùng các routes gần trùng. Cohort mới vẫn
24/12/24,6/8m/s,216 evaluated trip-draws và144 TEST. Không đọc metric để
chọn sample. Thêm static matched-initial, Γ0 và public-prior controls;
portfolio/static/dominance Γ1 dùng cùng initial random action để kiểm tra
actual-utility theorem pointwise. Các attacker được fit TRAIN, chọn SELECTION
và lưu trước TEST, theo MAE/Hit100 loss riêng.

| Phương án | Macro Recall@5 (%) | Gần nhất | Nhanh nhất | Bán kính1000m | Ít vòng đường |
|---|---:|---:|---:|---:|---:|
| Geo-I baseline L30 | 99.534 | 99.685 | 99.685 | 99.259 | 99.506 |
| PQB theo đoạn trước sửa | 87.171 | 90.116 | 90.115 | 78.839 | 89.615 |
| Portfolio Γ1 | 89.941 | 92.196 | 92.229 | 83.902 | 91.439 |
| Giữ nguyên initial Q Γ1 | 90.068 | 92.331 | 92.365 | 84.003 | 91.572 |
| Motion giữ POI Γ0 | 90.023 | 92.284 | 92.317 | 83.941 | 91.548 |
| Motion + public prior Γ1 | 90.094 | 92.354 | 92.388 | 83.986 | 91.649 |
| Motion + belief Geo-I Γ1 | 90.068 | 92.331 | 92.365 | 84.003 | 91.572 |

- So với portfolio trên **cùng cohort**: tăng **0,126pp**,95%CI
  **[+0,014;+0,267]pp**. Đây là giảm mất POI, một cải thiện nhỏ.
- So với PQB trước sửa: +2,896pp,95%CI[−3,221;+9,485]pp. CI chứa0,
  **chưa xác nhận gain ổn định qua các cohort**. Không lấy94,191% vòng một
  trừ90,068% vòng hai làm chênh lệch model: hai datasets khác nhau.
- So với baseline: −9,466pp,95%CI[−12,981;−6,157]pp; **không đạt gate−2pp**.
- So với static matched-initial: **bằng nhau** trong mọi phép kiểm tra utility.
  Thư viện mới bảo đảm không giảm Recall nhưng không tạo gain strict trong
  cohort này. Γ0 và public-prior cũng gần tương đương; chưa có evidence
  protected belief cải thiện actual Recall toàn phiên.

Trong cùng cohort, quyết định chỉ còn một joint action giảm từ52,462% ở
portfolio xuống3,914% ở motion; tỷ lệ khung đổi Q tăng20,394%→40,023%.
1421/1584 quyết định test có exact planning gain dương, nhưng gain trung
bình chỉ2,218×10⁻⁵ và vẫn là proxy. Nhiều vị trí Q khác nhau trả cùng pool:
**di chuyển nhiều hơn không tự động có thêm POI**.

### Attacker S1–S3, S9–S10 trên cohort vòng hai

Mỗi ô **Hit100% / MAE mét**. Hit nhỏ và MAE lớn nghĩa attacker khó xác định
GPS hơn. Các attacker cho hai losses được chọn riêng trên SELECTION; không
khẳng định một attacker tối ưu cho mọi phương pháp hoặc mọi loss.

| Scenario | Geo-I baseline | PQB trước sửa | Motion giữ POI Γ1 |
|---|---:|---:|---:|
| S1 | 1.053% / 807.6 | 0.134% / 2290.0 | 0.112% / 2222.1 |
| S2 | 2.778% / 645.6 | 1.389% / 2229.3 | 0.000% / 2200.5 |
| S3 | 1.837% / 668.6 | 0.470% / 2290.0 | 0.045% / 2222.1 |
| S9 | 0.694% / 968.7 | 0.000% / 2691.2 | 0.000% / 2560.5 |
| S10 | 0.000% / 712.3 | 1.389% / 2369.8 | 0.000% / 2429.0 |

Bản mới khó định vị hơn baseline ở các phép đo này. So với PQB trước sửa,
Hit100 thường tốt hơn nhưng MAE không tăng ở mọi scenario; ví dụ S9 có
MAE2560,5m so với2691,2m. Hit0% trên144 test trip-draws **không** có nghĩa
xác suất tấn công bằng0. S2 chỉ là coordinate proxy của cửa sổ dừng;
S4–S6 dùng posterior-odds analysis có điều kiện, không thay bằng identity-
classification benchmark hoặc nói đủ để bảo vệ account/biển số.

### Chi phí, budget và kiểm tra

Mọi arm gửi155 point requests/trip:31 khung×5Q, cùng L30 mỗi category.
Primary reply payload JSON khoảng50,181KB/trip, baseline50,675KB. Đây là
serialization của POI indices trong benchmark, không gồm HTTP/TLS/status
metadata và không là đo latency, pin hoặc tổng RAM trên thiết bị.

11 đoạn ở600s reserve Γ×11/12=0,916667 khi Γ1; prefix ratio≤2,234257.
Replay không dùng secret keys hoặc fit lại attacker. Đã kiểm tra độc lập:

- 11,880 finite-kernel/static decisions; normalization, bounded scores, support và prefix allowance.
- 19,440 bước giữ pool; 51,712 comparisons actual Recall≥static trên toàn TRAIN/SELECTION/TEST.
- 226,800 directed track transitions; coincident-lane coordinates/replies checked separately.
- Mỗi cohort:312,480 native1Hz fixes,240 counterfactual native trips,
  360 archived source files;60 families/480 native trips.
- Toàn test suite: **1058 passed,9 skipped**; các skips giữ nguyên theo cấu hình test suite, không đổi thành pass.

[Protocol](../../artifacts/benchmarks/dominance_motion_20261012_v1/protocol.json),
[results](../../artifacts/benchmarks/dominance_motion_20261012_v1/results.json),
[paired controls](../../artifacts/benchmarks/dominance_motion_20261012_v1/secondary_readout.json),
[independent verification](../../artifacts/benchmarks/dominance_motion_20261012_v1/independent_validation.json).

## 4. Quyết định và điểm còn cần cải thiện

Giữ cả hai bản ở trạng thái **prototype, chưa thay canonical engine**.
Tiến bộ chính đã kiểm chứng: PMF hữu hạn, cận toàn phiên chặt hơn,
support của augmented transcript, actual-utility theorem so với same-initial
static control và giảm mất POI do chuyển động. Gain7,215pp vòng một có
CI dương; vòng hai cho thấy mức gain đó chưa khái quát ổn định.

Khoảng trống còn lại là **pool ban đầu chưa bao phủ POI cần tại GPS thật**,
đặc biệt purpose bán kính và public states ngoài36 prototypes. Quy tắc giữ
mọi POI có thể khóa pool tại một vùng tối ưu cục bộ. Khi các pool có cùng
kích thước và U trước⊆U sau, chúng phải bằng nhau; không thể tăng Recall
chỉ bằng đổi hình học Q. Không chỉnh sample/Gamma theo TEST để che giới hạn.

Hướng có căn cứ toán học cho vòng tiếp theo: mở rộng công khai miền
reference cases và kiểm tra public worst-case coverage; hoặc bảo đảm
monotonicity trên **cache tích lũy có lifecycle công khai**, thay vì bắt
mỗi khung Q tự giữ toàn bộ POI. Đổi cache cần cohort/protocol mới, cùng
policy cho baseline, đo status expiry/chi phí/bulk-catalogue control;
không áp lại kết quả ở đây sang contract khác. Mô hình dùng belief chỉ
nên được nhận là cải thiện service khi vượt Γ0/public-prior controls trong
actual metrics, không từ surrogate variance identity đơn thuần.
