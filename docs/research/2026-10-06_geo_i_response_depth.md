# Geo-I: tăng utility bằng độ sâu phản hồi công khai

**Geo-I / REM, L=30 đã qua primary gate trên fresh synthetic TEST:** Recall
macro tăng **89.714%→92.688%, tức +2.974 điểm phần trăm**, CI95%
**[2.308,3.646] pp**; trả thêm **31.102% reply JSON bytes**. L là số POI tối
đa server trả **mỗi category cho mỗi Q**, không phải ID cấu hình30/67 của
benchmark trước. Vẫn có **K=5 Q** và **local top-k=5**. Đây là cấu hình dịch vụ trả
thêm POI để tăng utility, có chi phí bandwidth rõ ràng. Hai vòng thay planner Q
trước đó đều không qua gate và vẫn được giữ trong evidence.

Luồng Geo-I giữ nguyên: protected GPS supplier đọc theo lịch60s, REM và phép kiểm tra reuse bảo vệ
anchor Z, planner `legacy_l10` chọn5 Q. Development thử L30/L40/L60; fresh chỉ
đối chiếu **L30 đã chốt với L20**, tính **mỗi loại POI cho từng Q**. Query purpose, GPS và đích thật vẫn
chỉ dùng ở local để gộp/xếp hạng top5. Không tạo lại Q, đọc thêm GPS, đổi
ngân sách hay đổi thời điểm gửi. UtilityEvaluator dùng GPS ground truth
synthetic **tại mỗi public event** và đích thật để xếp hạng local chính xác;
đây là oracle đánh giá utility, chưa xác nhận ranking chỉ với sensor fix60s
hoặc đo mức tiêu thụ pin của GPS.

Các mức trả về là prefix của cùng một thứ tự POI công khai. Pool lớn chứa
pool nhỏ, nên static reference Recall không giảm; utility tăng không chứng
minh privacy tăng. L và kích thước response vẫn quan sát được.

| L dịch vụ trên development | Recall macro | Gain so với L20 | Reply bytes so với L20 |
|---|---:|---:|---:|
|20|92.188%|—|1.000×|
|**30**|**94.762%**|**+2.573 pp**|**1.311×: thêm31.094%**|
|40|96.459%|+4.270 pp|1.622×|
|60|97.799%|+5.610 pp|2.243×|

Rule ghi trước scoring chọn **mức nhỏ nhất** có macro gain≥2 pp, nearest
không giảm và reply-byte ratio≤2.5. L30 là mức nhỏ nhất đạt. Development
SELECTION có6 family độc lập, mỗi family3 draw và8 session. CI95% paired
whole-family của gain L30 là[1.793,3.509] pp; đây vẫn là CI exploratory.
Empty references giữ N/A; within-radius có10,008/27,216 category references
defined, không phải mọi category đều có POI trong bán kính.

Nguồn đã được verifier độc lập kiểm tra toàn bộ:
[`development artifact`](../../artifacts/benchmarks/qplanner_response_depth_development_20261006_v1/README.md).
L20 khớp utility/reply/bytes cũ; mọi độ sâu giữ cùng Q/clock/anchor/ledger.
Reply tăng từ123,907,067 lên162,434,661 bytes trên Selection, với22,680
requests không đổi. Đây là compact application JSON trong simulator, chưa
đo HTTP/TLS, latency, pin hoặc chi phí dịch vụ.

Fresh đã freeze L30 trước generation/scoring, chỉ so với L20 trên cùng legacy
Q. TEST có **24 family độc lập ×3 draw ×8 session**. Cả3 điều kiện đã khóa đều
đạt: gain≥2 pp, CI95% có cận dưới>0 và gain từng draw **2.867/2.929/3.126 pp**.
Draw vẫn nằm trong family khi bootstrap10.000 lần, không phải subject độc lập.

| Purpose trên fresh TEST, current replies | L20 Recall | L30 Recall | Gain, pp |
|---|---:|---:|---:|
| Nearest |92.449%|94.680%|+2.230|
| Fastest |92.449%|94.678%|+2.229|
| Within radius |81.694%|86.879%|+5.185|
| Minimum detour |92.266%|94.516%|+2.250|

Within-radius giữ N/A: **42,258/108,684** category references defined và
**17,577/18,114** windows defined. Các purpose khác có toàn bộ references
defined. Mean của25% family thấp nhất tăng **83.565%→88.046%**, delta+4.481 pp.
Sensitivity: toàn session0 (“cold”) tăng+3.122 pp; đoạn400–600s tăng+2.492 pp;
static epoch cache tăng98.808%→99.136%, chỉ+0.328 pp. **Current-only là primary**;
cache/cold/tail/purpose là kết quả phụ, chưa chỉnh nhiều phép so sánh.

Chi phí **TEST thật**: reply **494,713,236→648,578,577 bytes** (+31.102%),
**90,570 requests** và **13,678,985 request bytes** không đổi. Không suy chi phí
TEST từ tỷ lệ development. Protocol, readout và paired-readout hashes đã khớp
certificate independent PASS trước khi đọc số. Xem
[`fresh artifact`](../../artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1/README.md),
[`recommended configuration`](../../artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1/recommended_configuration.json)
và [figure PDF](../../artifacts/benchmarks/qplanner_response_depth_generalization_20261006_v1/figures_v2/response_depth_utility_cost.pdf).

Kết quả xác nhận **utility/cost tuning tĩnh trên cohort synthetic mới cùng
bản đồ**. Đây chưa phải thuật toán Q mới, theorem privacy, đánh bại baseline
được cùng L/cost, hay kết quả đủ chuẩn JISA. Bulk retrieval, thiếu dynamic
service/real GPS/deployment và utility oracle dùng GPS từng event vẫn là các
giới hạn; mức protected GPS supplier60s không phải đo GPS energy.
