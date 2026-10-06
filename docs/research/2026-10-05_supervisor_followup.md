# Cập nhật sau buổi gặp GVHD ngày 03/10/2026

Đã hiện thực mở rộng query purpose, chạy các vòng kiểm tra S7, S4–S6 và thử
Geo-I không delay cho S9/S10. Kết quả có tiến triển rõ về utility và có tín hiệu
privacy tốt hơn ở một số phép thử. Chưa đủ bằng chứng để kết luận vượt mọi
phương pháp hoặc giải quyết đầy đủ các scenario.

[Note cuộc gặp](../meetings/2026-10-03/research_notes.md). Ngày chạy: 05/10/2026.
Dataset, benchmark và PDF trước đây được giữ nguyên. Các kết quả dưới đây thuộc
những cohort riêng, không ghép thành một leaderboard.
S1–S10 dùng taxonomy trong [`data/threat_scenarios.py`](../../data/threat_scenarios.py);
bản grounding tháng 8 có tám harm và cách đánh số khác.

## 1. So sánh bằng cả metric chung và metric gốc

Đã bổ sung API chấm metric gốc, đọc lại **5.336 dòng benchmark** và kiểm tra độc
lập **431 dòng EIE**. Với cùng point estimator, EIE chính là MAE: đổi tên không
tạo thêm bằng chứng. Readout tách từng protocol, giữ lỗi và coverage.

- EIE/Hit đánh giá khả năng suy luận của attacker; Recall và chi phí đánh giá
  dịch vụ. Cần đọc cùng nhau.
- Entropy/ASR/DER của tập chứa vị trí thật không áp dụng nguyên dạng cho tập Q
  chỉ chứa điểm giả. Báo **N/A** khi sai hợp đồng hoặc thiếu posterior/nhãn.
- Chi phí đi tới đích theo Eq. 13 của TransProtect cần cùng đường đi và prior
  đích. Cache gốc thiếu nên chưa chấm lại được metric này cho bảng paper cũ.
  Phép mở rộng theo từng Q phải mang tên riêng, không chọn Q gần GPS thật nhất.
- Trên cùng paper-v2, BR dùng Geo-I có EIE cao hơn TransProtect ở năm scenario,
  nhưng chưa cao hơn DLS ở S1/S3/S9. GeoI-Slack hiện tại có protocol khác nên
  được báo riêng.

Đã bổ sung phép đo thuộc nhóm Eq. 13 trên Q đóng băng của GeoI-Endpoint20,
với prior đều trên 418 đích POI công khai. Trung bình độ méo chi phí của cả
năm Q là **1.386,74 m**, coverage 170/170 mốc; bảng chi phí được giữ lại.
Đây là **chi phí của tọa độ truy vấn**, không phải privacy gain hoặc quãng
đường người dùng phải đi thêm. Chưa có TransProtect cùng cohort mới để so
trực tiếp. [Evidence](../../artifacts/benchmarks/native_cost_diagnostic_20261005/README.md).

[Định nghĩa, bảng số và giới hạn](2026-10-05_native_metrics.md) ·
[Evidence](../../artifacts/benchmarks/native_metrics_20261005/verification.json).

## 2. Mở rộng POI query và tập trung S7

Luồng đang chạy:

**GPS local → Geo-I tạo/giữ Z → ước lượng và chọn Q → request chung → hợp POI
nhận được → xếp hạng theo mục đích riêng tư tại thiết bị.**

Có bốn mục đích: **gần nhất**, **đến nhanh nhất**, **trong bán kính**, và **ít vòng
đường tới đích**. Purpose, category, radius và destination chỉ dùng local.
Server nhận request lấy mọi loại POI với L cố định từ mỗi Q; Q/lịch gửi không
thay đổi theo mục đích. L là số POI phản hồi theo từng loại, K=5 là số Q;
thiết bị chọn tối đa top-5 cho người dùng.

| Phép thử | Kết quả | Ý nghĩa |
|---|---|---|
| Chỉ thay cách lọc local; giữ Q, phản hồi, GPS và L10 | Detour Recall **75,19% → 93,70%** | Ranking theo purpose tốt hơn ranking khoảng cách cho mục đích đi vòng |
| Snapshot, bốn purpose × sáu loại POI | L10; Recall trung bình **95,44%** | Radius riêng đạt 88,89%; chưa đạt 90% cho mọi purpose |
| Sequence, ba intent template, query ở 0/20/40s | L20; Recall **99,05%** | Attacker đọc prefix ở 20/40s: nội dung tường minh **100%**, request chung **33,33%**, bằng chance |

Recall còn bỏ sót POI vi phạm điều kiện query. Trong các trường hợp reference
radius có ít hơn năm item, bộ lọc cũ trả 170 item ngoài bán kính; bộ lọc mới
trả 0 item vi phạm. Vì vậy cần thêm phép đo vi phạm ràng buộc và kết quả rỗng.

Thử nghiệm S7 giữ cùng luồng vị trí/lịch giữa các intent. Nó kiểm tra **rò rỉ
nội dung request**, chưa chứng minh không thể suy nhu cầu từ tuyến đường,
tài khoản hoặc thời điểm bật dịch vụ. Các intent là template tổng hợp.
Fastest dùng free-flow; chưa có congestion, giá, rating hay giờ mở cửa.

[Giải thích và code](2026-10-05_query_purpose.md) ·
[Evidence/validation](../../artifacts/benchmarks/query_purpose_20261005/README.md).

## 3. S4–S6: tăng chất lượng attacker và tách đúng target

**S4** là liên kết danh tính; đo người và phương tiện riêng. **S5** là bước
tiếp theo; **S6** là đích đến. Không gọi cả ba là nhận diện danh tính.

| Target | Raw | GeoI-Slack | Cách đọc |
|---|---:|---:|---|
| S4 cùng người tổng hợp | Balanced accuracy 69,44% | 43,06% | Tín hiệu liên kết trung bình giảm; còn một family AUC=0,861 |
| S4 cùng xe tổng hợp | Balanced accuracy 62,50% | 52,78% | Cần nhiều identity/family hơn để xác nhận |
| S5 proxy GPS sau 20s | MAE 93,5 m; Hit100 66,67% | 860,2 m; 0% | Proxy không gian; chưa thay được next-edge accuracy |
| S6 GPS cuối chuyến | MAE 132,8 m; Hit100 33,33% | 942,0 m; 0% | Prefix S3 gần cuối; chưa phải toàn bộ S6 fork/history |

Đã lưu cả lần raw attacker yếu và vòng cải thiện bằng hình dạng track/ngưỡng
chọn trên selection. Với S4, chỉ 20% cặp cùng danh tính nên accuracy thông
thường dễ gây hiểu lầm; dùng balanced accuracy/AUC. Điểm dưới 50% không tự
chứng minh privacy tốt hơn ngẫu nhiên.

**S5 đúng ID cạnh chưa qua gate**: test edge chưa xuất hiện trong train, raw
classifier cũng 0%. Cần khôi phục mapping cạnh và dùng decoder ứng viên đường
công khai. Nhãn người/xe chỉ là thiết kế SUMO; account/IP/biển số chưa được che.

[Chi tiết attacker và giới hạn](2026-10-05_identity_future.md) ·
[Evidence](../../artifacts/benchmarks/identity_future_20261005/validation_recheck.json).

## 4. S9/S10: GeoI-Endpoint20 không delay

Đã chạy hai vòng. Vòng tăng nhiễu với L10 trượt ngưỡng Recall 90%; giữ nguyên
kết quả này. Vòng tiếp theo chọn epsilon nhỏ hơn kết hợp phản hồi sâu hơn;
khóa lựa chọn trước khi đọc bốn family kiểm tra mới trong nghiên cứu.

**GeoI-Endpoint20:** ε kiểm tra/tạo Z=0,0025/m; B=0,06/m; H=12; cận mỗi phiên
0,0575/m; K=5; L=20; ngưỡng tái dùng 200 m; slack=0,03; warmup/delay=0.
Không biết trước khi nào chuyến kết thúc nên tăng nhiễu trên mọi lần bảo vệ
GPS. Giờ hoạt động vẫn quan sát được. Cận privacy dựa trên kernel Geo-I lý
tưởng; sampler số thực hữu hạn là phép xấp xỉ, không phải chứng minh mới.

Attacker được học riêng theo mỗi phương pháp, gồm **kNN, Extra Trees**, thống
kê tập Q, đặc trưng chuỗi, Viterbi và ngoại suy. GPS thật là positive control:
Hit100=100%, MAE=0. Control này xác nhận pipeline, không chứng minh bank đã là
attacker mạnh nhất có thể.

| Cấu hình | Recall@5 | MAE S9 / S10 | Gửi / mốc đầu vào | Delay |
|---|---:|---:|---:|---:|
| GeoI-Slack L20 | 99,59% | 970 / 1.844 m | 170/170 | 0s |
| Warmup/delay60 L10 | 78,08% | 1.179 / 1.180 m | 122/170 | 60s |
| **GeoI-Endpoint20** | **96,11%** | **1.852 / 2.297 m** | **170/170** | **0s** |

So cùng L20, S9 MAE tăng 882 m, CI95 [416; 1.216]. S10 tăng 453 m nhưng
CI95 [-382; 1.236] qua 0: **lợi thế S10 chưa chắc chắn**. Hit100 đều 0 nên
không dùng cột đó để khẳng định vượt trội. Recall giảm 3,48 điểm % so với
L20 thường; byte gần bằng nhau và khoảng 1,51 lần L10. Recall thấp nhất của
một lần chạy là 85,92%.

[Phân tích và tham số](2026-10-05_endpoint_noise.md) ·
[Evidence và tái kiểm tra](../../artifacts/benchmarks/endpoint_noise_20261005/README.md).

## 5. Mức bằng chứng và việc còn cần hoàn thiện

Các lần chạy mới dùng mạng công khai tái dựng từ 9.138 polyline, 22.106 state,
418 POI. Native SUMO đọc được mạng; luật rẽ/lane gốc chưa được xác minh. Không
so trực tiếp các số này với benchmark mạng cũ. Mẫu nhỏ, identity/intent tổng
hợp và những vòng dùng lại dữ liệu đều được ghi rõ. Bốn family kiểm tra
endpoint mới trong vòng 2 vẫn thuộc dataset có sẵn, không là dữ liệu thực địa.

Ưu tiên tiếp theo:

1. S7: dữ liệu nhu cầu thật tương quan với route và budget byte cho cover query.
2. S4: kiểm tra riêng family còn rò và lịch sử liên kết nhiều chuyến; cộng
   ngân sách giữa những phiên có thể liên kết.
3. S5: decoder cạnh tại fork; S6: bộ fork/history đúng taxonomy.
4. S10: nhiều family và seed độc lập sau khi khóa bank; kiểm tra prefix/multi-trip.
5. Comparator: cùng cohort/map/prior/attacker, giữ cost tables và posterior để
   chấm đầy đủ metric gốc có hợp đồng phù hợp.

Kiểm thử toàn repo: **436 passed, 9 skipped**; canonical runner đã sửa để chạy
đúng fixtures/parametrization. `pip check` qua. Các validator độc lập kiểm tra
hash, split, ngân sách, schema công khai và phép tính kết quả. Quick benchmark
SUMO cũ chưa chạy được do thiếu `data/raw/Beijing.osm[.gz]`; các vòng mới chạy
bằng mạng tái dựng được ghi rõ ở trên. Chín integration tests cũng cần cache
SUMO gốc nên được skip. Không sửa nguồn dữ liệu để cải thiện số.
