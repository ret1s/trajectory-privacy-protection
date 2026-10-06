# Query purposes và S7 — 05/10/2026

Mã chạy thật: [`benchmark/query_purpose.py`](../../../benchmark/query_purpose.py). Bốn mục đích hiện có: gần nhất theo khoảng cách đường, nhanh nhất theo thời gian đi, trong bán kính đường, và ít vòng đường nhất tới đích. Radius, đích và mục đích chỉ vào bước xếp hạng tại thiết bị. Không giả lập giá/rating/giờ mở cửa chưa có trong dữ liệu.

Request từ từng Q luôn lấy mọi loại POI theo độ sâu L cố định; lịch và Q không thay đổi theo mục đích riêng tư. Đây là cách phủ nội dung truy vấn trong mô hình Geo-I hiện tại, không thêm lại phương pháp PublicCover đã bỏ. Chỉ vị trí đã bảo vệ được dùng để tạo Q. Mức bảo vệ nội dung là có điều kiện theo cùng luồng tọa độ/lịch; vẫn có rò rỉ từ tài khoản, kích hoạt phiên và tương quan vị trí–nhu cầu.

## Kết quả phát triển

| Phép thử | Kết quả | Giới hạn |
|---|---|---|
| Snapshot: 36 record S7 một lần, bốn purpose × sáu loại POI | Chọn L=10; Recall test 95,44%, request+response JSON 17.741 byte/mốc | Family-disjoint 6/3/3; dữ liệu phát triển cũ, mạng tái dựng |
| Counterfactual cùng Q: đoán purpose | Nội dung tường minh 100%; request chung 25%, bằng chance 1/4 | Chỉ nội dung khi luồng vị trí/lịch giống nhau |
| Counterfactual cùng Q: đoán loại POI / loại+purpose | Request chung 16,67% / 4,17%, bằng 1/6 / 1/24 | Không phải kiểm định suy luận nhu cầu của người thật |
| Sequence: 36 mẫu, ba query đầu tại 0/20/40s | Chọn L=20; Recall test 99,05% | Mục đích ở bước sau được bổ sung bằng template tổng hợp |
| Attacker đọc toàn bộ prefix sequence | 0s: cả hai 33,33%; 20/40s: tường minh 100%, request chung 33,33% | Ba intent template; query đầu giống nhau, chuỗi sau khác nhau |

Snapshot dùng seed theo record độc lập; diagnostic `correlated_synthetic_intent_attacks` trong JSON là **tên trường lịch sử**, thực tế vẫn là các intent template đặt trên cùng đường, không có dữ liệu nhu cầu thật tương quan với vị trí. Giá trị 44,44% trong mẫu test nhỏ không chứng minh rò rỉ thực tế; paired sequence kiểm tra lại bằng seed theo session/indices giống nhau giữa intent, cho public prefix giống hệt. Không lấy con số này làm claim bảo vệ mọi suy luận nhu cầu.

Snapshot Recall từng purpose: distance 97,41%, fastest 97,41%, radius 88,89%, detour 93,70%. Mean overall không bảo đảm mọi purpose đạt90%. Loại không có top-k chuẩn giữ `null` và được bỏ khỏi mẫu số, không coi là thành công100%. Map có tốc độ8m/s cố định nên fastest trùng distance; unit test dùng đường tốc độ khác nhau để kiểm tra hai mục tiêu thực sự khác.

[So với bộ lọc local chỉ theo khoảng cách](purpose_comparison.json), giữ cùng Q/phản hồi/L=10: detour Recall75,19% →93,70%, tăng18,52 điểm%. Với radius,170/201 item của bộ lọc cũ trong51 tuple có reference đầy đủ(<5 item) nằm ngoài bán kính; bộ lọc mới trả0 item vi phạm. Recall không phạt những item thừa này, nên cần đo ràng buộc query riêng. Đây là đối chứng nội bộ về ranking, không phải paper bên ngoài.

Mọi L=10/20/40/80 đã thử được giữ; lựa chọn chỉ dùng selection, không chọn theo test. Cấu hình L là cố định cho cohort, không chọn L theo nhu cầu riêng tư. Sequence attacker được fit/chọn/đánh giá đúng L=20 đã chọn. Diagnostic lần đầu chấm attacker ở L=10 được giữ riêng, không dùng như bằng chứng của cấu hình L=20 cuối.

Mạng riêng gồm22.106 trạng thái/418 POI/907 ô; hình học công khai nhưng các nối đường/luật rẽ tái dựng, chưa tương đương mạng benchmark gốc. Seed/split được khóa trước khi chạy. [Validation](validation.json) tính độc lập Recall từ ID, lựa chọn L từ selection và kiểm tra schema công khai/ngân sách/sự giống nhau giữa các prefix. Metadata và source hashes có trong protocol/readout từng thư mục.

```sh
python -m experiments.query_purpose_loop --out /private/tmp/new-s7-snapshot
python -m experiments.query_intent_sequence --out /private/tmp/new-s7-sequence
python -m experiments.query_purpose_comparison
python -m experiments.verify_query_purpose
python -m pytest -q tests/test_query_purpose.py tests/test_query_intent.py
```

Các lệnh dùng cache nguồn công khai ở `/private/tmp/trajectory-research-20261005-public-map`, tự dựng nếu chưa có. Không ghi đè kết quả hoàn thành. [Snapshot readout](snapshot/readout.json), [sequence readout](sequence/readout.json), [phân tích thuật toán](../../../docs/research/2026-10-05_query_purpose.md).
