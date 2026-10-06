# Mở rộng query purpose và bảo vệ S7

Yêu cầu từ buổi gặp 03/10: dịch vụ không chỉ tìm POI gần nhất; mục đích query là dữ liệu cần bảo vệ. [Note cuộc gặp](../meetings/2026-10-03/research_notes.md).

## Luồng mô hình mới

GPS → cơ chế Geo-I theo lịch/ngân sách → Z → ước lượng và kiểm tra đường → Q → request lấy POI của mọi loại theo L cố định → hợp phản hồi còn hiệu lực → **QuerySpec riêng tư** chọn top-k trên thiết bị.

QuerySpec không được đưa vào bước chọn Q hoặc request. Tọa độ đích của query đi vòng, radius và loại POI chỉ là input local. Mô hình cần giữ lịch truy vấn độc lập nhu cầu để tránh công bố nhu cầu qua thời gian bắt đầu/tần suất.

## Bốn mục đích được hiện thực

| Mục đích | Điểm xếp hạng tại thiết bị | Điều kiện |
|---|---|---|
| Gần nhất | `d(GPS, POI)` | Khoảng cách đường có hướng |
| Đến nhanh nhất | `time(GPS, POI)` | Thời gian free-flow theo tốc độ đường; chưa có congestion realtime |
| Trong bán kính | POI có `d(GPS, POI) <= radius`, rồi xếp theo d | Radius riêng tư; trả rỗng nếu không có ứng viên phù hợp |
| Ít vòng đường tới đích | `d(GPS, POI) + d(POI, destination) - d(GPS, destination)` | Đích riêng tư chỉ ở thiết bị; cần các đường đi có hướng tồn tại |

POI khả dụng chỉ được chọn từ ID thiết bị đã nhận trong epoch hiện tại. Evaluator/server có trạng thái đầy đủ để tính chuẩn; client không được đọc toàn bộ status mask. K=5 là số Q; L=10/20/40/80 là độ sâu phản hồi thử; top-k=5 là số kết quả tối đa cho mục đích thực. Cận GPS không nhân lên theo số purpose hay L, nhưng byte phản hồi tăng.

Không gọi bốn mục đích này là mọi nhu cầu LBS có thể có. Chưa có dữ liệu giá, chất lượng, giờ mở cửa, thời gian xếp hàng hay năng lực trạm sạc; các mục đích tương ứng cần dữ liệu và protocol bổ sung. Không tạo metadata giả để báo coverage đầy đủ.

## S7 được đánh giá thế nào?

Đối chứng nội dung tường minh thêm purpose/category/radius/destination vào payload; đối chứng bỏ category nhưng vẫn lộ purpose kiểm tra phần còn hở; bản đầy đủ giữ request chung. Các đối chứng nội dung là kiểm tra sức attacker, không là tái lập paper. Utility dùng cùng tập phản hồi để tránh quy hiệu ứng metadata cho retrieval.

Attacker nhìn request/reply JSON, tọa độ, thời gian và kích thước; bank **kNN 1/5/15**, **ExtraTrees** hai cấu hình và **logistic** được học/chọn riêng. Không cho attacker đọc GPS, intent ground truth hay ID evaluator trong feature. Khóa 6 family train, 3 chọn, 3 test; đây là audit phát triển nội bộ trên dữ liệu đã được dùng trước, không phải confirmation độc lập mới.

Trong snapshot, request chung đạt chance 25% cho bốn purpose và 16,67% cho sáu loại POI; explicit đạt 100%. Trong sequence, query đầu pharmacy giống nhau nên ở 0s attacker chỉ đạt 33,33%; đến 20/40s explicit lộ chuỗi ý định và attacker đạt 100%, request chung vẫn 33,33%. Cùng session/indices dùng chung seed trước khi xét intent: các public prefix giống hệt. Đây là kiểm tra kênh nội dung query; chưa chứng minh intent không thể suy từ một tuyến đường đặc trưng, tài khoản hoặc click sau khi nhận kết quả.

L=10 được chọn cho snapshot với Recall 95,44%; sequence cần L=20 và đạt 99,05% trên ba family test. Snapshot riêng radius chỉ 88,89%; không gọi mọi mục đích đều đạt 90%. Thời gian đến nhanh nhất trùng khoảng cách trong mạng demo tốc độ cố định, nên phân biệt hai ranking được kiểm tra thêm trên fixture đường có tốc độ khác nhau.

## Vì sao cần mở rộng cách lọc local?

[So sánh ghép cặp](../../artifacts/benchmarks/query_purpose_20261005/purpose_comparison.json) giữ nguyên Q, phản hồi, GPS và L=10; chỉ thay quy tắc xếp hạng local. Với mục đích ít vòng đường tới đích, tái dùng bộ lọc khoảng cách cũ đạt Recall 75,19%, bộ lọc theo detour đạt 93,70%: tăng 18,52 điểm phần trăm. Bootstrap theo ba family cho khoảng 95% [15,56; 23,33] điểm phần trăm; chỉ áp dụng cohort phát triển nhỏ này, không là so sánh paper bên ngoài.

Recall một mình còn bỏ sót vi phạm điều kiện query. Trong 51 tuple radius có ít hơn 5 POI chuẩn (gồm cả rỗng), reference đã chứa toàn bộ POI khả dụng trong bán kính. Bộ lọc khoảng cách cũ vẫn trả 201 item, trong đó 170 item chắc chắn nằm ngoài bán kính; bộ lọc radius mới không trả item vi phạm. Hai cách có thể cùng Recall 88,89% vì Recall chỉ đếm đúng POI cần tìm, không phạt các POI thừa ngoài điều kiện. Vì vậy mở rộng purpose cần thêm **constraint-violation rate** và coverage/empty-result count, không chỉ Recall.

Không thay đổi traffic hay ngân sách GPS khi đổi cách xếp hạng trong phép ghép cặp này. L sâu hơn để tăng candidate coverage là một thay đổi riêng cần báo byte.

[Evidence và lệnh chạy lại](../../artifacts/benchmarks/query_purpose_20261005/README.md). [Code](../../benchmark/query_purpose.py), [attacker](../../evaluation/query_intent.py), [independent verifier](../../experiments/verify_query_purpose.py). Nguồn GPS/model/benchmark cũ không bị sửa.
