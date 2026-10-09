# Cap từng phiên và bảo vệ riêng danh tính người/phương tiện

Ngày 10/10/2026. Quyết định thiết kế theo yêu cầu người dùng: giữ Geo-I/REM,
quay lại cap hiệu lực 0,23 m⁻¹ mỗi phiên; không giới hạn dịch vụ ở tám phiên.
Tách S4 thành một lớp bảo vệ danh tính chuyên biệt với hai target. Đây là quyết
định cho cấu hình làm việc tiếp theo, không thay đổi cấu hình của các benchmark
đã lưu và không xác nhận đã triển khai anonymous transport.

## 1. Chính sách tọa độ đã chọn

Một phiên là một chuyến/lượt sử dụng được bắt đầu và kết thúc theo giao thức
công khai. Trong phiên có nhiều lần đọc GPS và gửi Q. Mở lại tiến trình để tiếp
tục cùng một chuyến không được xem là một phiên mới để tự nạp cap.

Đặt C_s là cap hiệu lực của một phiên; H là tham số phân bổ; U = 2H − 1.

- u = C_s/U cho mỗi phép thử tái sử dụng và mỗi mẫu REM mới.
- B_nominal = 2H u là tham số danh nghĩa mà engine hiện có nhận.
- C_s = 0,23 m⁻¹; H = 12; U = 23; u = 0,01 m⁻¹; B_nominal = 0,24 m⁻¹.
- Lịch đọc tối thiểu 60 s; dự toán trước GPS: lần đầu 1 đơn vị, lần sau 2.
- Chi phí thực: đầu tiên 1; giữ Z sau khi thử 1; thử rồi tạo mới 2; không đọc 0.
- Hết cap đọc trong phiên thì tiếp tục từ lịch sử bảo vệ. Không có quy tắc từ
  chối chuyến thứ chín của thiết kế được chọn.

Đây là cấu hình per-session đã tồn tại trong GeoI-Slack/session-reset. Kết hợp
per-session cap với L30 và lớp S4 mới chưa có readout độc lập mới. Không đổi
`recommended_configuration.json` của study L30 cũ hoặc cấp lại nhãn số liệu.

Chặn toán học lý tưởng trong một phiên là exp(C_s D_infinity). Nếu attacker nối
M phiên của cùng người hoặc xe, chặn tổng vẫn là exp(M C_s D_infinity) khi cap
bằng nhau. Đổi pseudonym không xóa chi phí hợp thành. Không phát biểu cap 0,23
cho cả đời hay cho tám phiên của cấu hình mới. C_s,H,lịch đọc cần khảo sát độ
nhạy; tính đúng công thức không chứng minh 0,23 là tối ưu phổ quát.

## 2. Hai danh tính khác nhau

- I_person: người thật/chủ thể, giữ xuyên chuyến và có thể đổi xe/thiết bị.
- I_vehicle: phương tiện vật lý, có thể được nhiều người sử dụng.
- I_device: thiết bị hoặc tài khoản là một nguồn phụ trợ; không được tự đồng
  nhất nó với người hoặc xe.

S4-person hỏi hai phiên có cùng người không. S4-vehicle hỏi hai phiên có cùng
phương tiện vật lý không. Ground truth chỉ ở evaluator; không suy ID thật từ
SUMO instance ID. Evaluator hiện dùng riêng `person_id` và `physical_vehicle_id`
(`experiments/identity_future_eval.py`, phần linkage), nên hai target đã có.

Đối chứng bắt buộc:

| Cặp phiên | Cùng người | Cùng xe |
|---|---:|---:|
| Một người dùng lại cùng xe | 1 | 1 |
| Một người đổi xe | 1 | 0 |
| Hai người dùng chung xe | 0 | 1 |
| Hai người, hai xe | 0 | 0 |

Đổi điện thoại không tự đổi hai nhãn này. Dữ liệu phải có hỗ trợ thực cho các
cặp, không chỉ gán hai target bằng cùng nhãn.

## 3. Lớp S4 đề xuất: tách định danh và chống nối phiên

Geo-I bảo vệ tọa độ; lớp S4 xử lý các kênh định danh ngoài tọa độ. Thiết kế này
là hướng cần triển khai/kiểm chứng, không phải một bảo đảm identity đã đạt.

1. **Tách ID trong ứng dụng.** Không gửi person/vehicle/device ID, VIN, account,
   cookie, API key cá nhân, seed hay stable track ID trong request POI. Nếu cần
   mã đối chiếu phản hồi, chỉ dùng mã ngẫu nhiên cho request; không dẫn xuất từ
   ID thật và không tái sử dụng qua chuyến. Dữ liệu cá nhân/cache ở thiết bị.
2. **Tách nguồn mạng khỏi nội dung.** Dùng relay/gateway độc lập kiểu Oblivious
   HTTP, với HTTPS, mã hóa và điều kiện các bên không thông đồng. Server POI
   không đồng thời thấy IP nguồn và nội dung request. Không đưa authentication
   định danh vào payload để tự phá mục tiêu này. Chưa có triển khai relay/OHTTP
   trong simulator hoặc đo HTTP/TLS/độ trễ.
3. **Kiểm tra liên kết quỹ đạo còn lại.** Q vẫn có hình dạng và timing. Shuffle Q
   hoặc đổi pseudonym không đủ xóa home/work/routine hoặc fingerprint tốc độ.
   `PrivateOrderCoverClient` đã bỏ slot-order bền, nhưng không chứng minh S4.
   Giữ layer này là đề xuất cho tới khi attacker chỉ nhìn hình học vẫn được
   đánh giá. Mix-zone/cooperative mixing là hướng ablation có điều kiện nhiều
   người, không tự thêm vào mô hình một người và không trigger theo GPS thật.

Cấu trúc: GPS → Geo-I/REM → b → Q → đóng gói loại định danh → anonymous
transport → POI server; đáp án quay về lọc/sắp xếp local. ψ không chọn Q hoặc
trigger lưu lượng. Mọi thay đổi lịch/chèn lưu lượng sau này phải được khai báo
và định trước từ thông tin công khai; không dùng GPS thô để chọn mix-zone.

Nền tham khảo:

- [RFC 9458, Oblivious HTTP, §7](https://www.rfc-editor.org/rfc/rfc9458.html#section-7):
  tách nguồn mạng và request cần điều kiện trust/metadata; không bảo đảm vô điều
  kiện trước traffic analysis hoặc nội dung tự nhận diện.
- [Beresford–Stajano, Mix Zones, 2004](https://www.cl.cam.ac.uk/~arb33/papers/BeresfordStajano-MixZones-PerSec2004.pdf):
  đổi pseudonym đi cùng vùng không quan sát, không chỉ đổi tên trên một tuyến
  liên tục quan sát được.
- [Cooperative Location Privacy in Vehicular Networks, 2020](https://arxiv.org/abs/2012.06666):
  liên kết pseudonym phụ thuộc hình học, mật độ, thời điểm và mô hình chuyển động.

## 4. Kiểm chứng cần làm trước khi gọi S4 được bảo vệ

Hai đầu ra độc lập: same_person, same_vehicle. Fit attacker trên train, chọn
bank/ngưỡng trên selection, khóa thiết kế trước test mới; nhóm toàn bộ những
phiên cùng thực thể/route family vào cùng split.

Ablation: per-session REM/L30; + loại định danh; + shuffle/mã request mới;
+ anonymous transport theo đúng observer contract; + cơ chế chống liên kết hình
học nếu được triển khai. Hai observer: nội dung không ID, và mạng trực tiếp có
account/IP bền làm positive control. Không dùng việc simulator vốn không gửi
ID làm một mức giảm AUC mới. Không giả lập bỏ IP rồi gọi đã triển khai OHTTP.

Báo AUC và khả năng đảo điểm/định hướng trên selection, balanced accuracy,
Recall@5 theo purpose, số request, bytes, latency và chi phí mới. Dữ liệu cùng
người khác xe/khác người cùng xe là điều kiện đánh giá, không phải test sau này
được phép sửa để có điểm đẹp. Thành công về một target không thay cho target kia.

## 5. Chuyển tiếp tài liệu và bằng chứng

Report 6 trang giới thiệu chính sách mới và S4 đề xuất. Timeline/GPS/Z/b/Q đã
lưu dùng u = 0,00125 thuộc Epoch8 cũ; không nhân ngân sách rồi giữ nguyên Z.
Bảng L20–L30 89,71 → 92,69% cũng thuộc Epoch8 cũ. S4 lịch sử là per-session,
nhưng không có lớp anonymous transport và không xác nhận per-session L30 mới.
Benchmark S5/S6 Epoch8/L20 và Endpoint20 vẫn giữ nguyên nguồn/cấu hình.

`core/session_budget.py`, các study và canonical thesis là snapshot cơ chế và
bằng chứng trước quyết định này, được giữ để tái lập. Không xóa khả năng chạy
Epoch8 hoặc sửa số đo frozen. Cấu hình chọn tiếp theo ghi ở
`2026-10-10_session_cap_identity.json`; trạng thái là design_selected,
implementation_and_new_benchmark_pending, không phải deployment đã xác nhận.
