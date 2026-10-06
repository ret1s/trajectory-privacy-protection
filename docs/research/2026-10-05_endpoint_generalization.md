# Kiểm tra GeoI-Endpoint20 trên 28 nhóm, tách seed theo từng chuyến

Đây là bước kiểm tra tiếp theo của
[vòng phát triển endpoint](2026-10-05_endpoint_noise.md). **Giữ nguyên mô hình
đã chọn**, không tìm lại cấu hình tốt nhất trên các nhóm mới. Geo-I vẫn là nền
tảng: ε kiểm tra/tạo Z = 0,0025/m, B thực tế 0,06/m, cận phiên 0,0575/m,
H=12, K=5, L=20, lịch đọc GPS 60 giây, ngưỡng 200 m, slack 0,03. Không
warmup/delay và không dùng GPS tương lai để nhận diện đoạn cuối.

## Những điểm đã sửa trong cách kiểm tra

Vòng trước dùng hai seed chung khi reset các chuyến. Vòng này dùng một khóa
ngẫu nhiên của evaluator, HMAC theo **session ID + rep**, tạo substream khác
cho từng chuyến/lần lặp. Các phương pháp chỉ dùng chung substream khi so đúng
cùng chuyến/rep. Không tái khởi tạo cùng seed cho hai chuyến khác nhau. Khóa,
seed, ID session và GPS thật không vào đặc trưng attacker hay bản tin công khai.
Đây là PRNG của thí nghiệm; phát biểu privacy hình thức vẫn nói về kernel lý
tưởng và nguồn ngẫu nhiên riêng tư khi triển khai.

Khóa trước **toàn bộ 28 nhóm 1205–1232**, hai session đầu mỗi nhóm, hai rep:
56 chuyến và 112 lần chạy mỗi phương pháp. Không dừng sớm, thay seed, đổi nhóm
hoặc chọn lại defense sau khi xem số. Nhóm 1201–1204 đã xem ở vòng trước được
loại khỏi tập kiểm tra này ngay từ protocol. Đây là phần chưa dùng trong nghiên
cứu hiện tại của dataset SUMO có sẵn, không phải xác nhận thực địa độc lập.

Fit lại shadow attacker trên 701–708 với RNG mới; chọn decoder trên 709–710,
khóa trước khi đọc nhãn kiểm tra. Bank giữ **kNN, Extra Trees**, thống kê tập
điểm, đặc trưng chuỗi, Viterbi và các ngoại suy hình học. Fit riêng theo phương
pháp và chọn riêng theo MAE/Hit. Defense Endpoint20 không được chọn lại;
Recall tập chọn đạt 97,08%, qua ngưỡng trung bình 90% đã định trước.

## Kết quả chính: từng chuyến riêng

MAE cao và Hit thấp thuận lợi cho privacy trước bank đã khóa. Recall cao tốt
cho dịch vụ. Gộp các session/rep trong mỗi nhóm rồi gộp đều 28 nhóm. 112 lần
chạy không được coi là 112 người dùng độc lập.

| Phương pháp | S9 Hit100 / MAE m | S10 Hit100 / MAE m | S9 / S10 Hit500 | Recall@5 | Byte/mốc |
|---|---:|---:|---:|---:|---:|
| GPS thật | 100% / 0 | 100% / 0 | 100% / 100% | 100% | 932 |
| GeoI-Slack, L10 | 3,57% / 621 | 3,57% / 769 | 41,96% / 30,36% | 96,81% | 4.578 |
| GeoI-Slack, L20 | 1,79% / 891 | 0% / 901 | 16,96% / 12,50% | 98,94% | 6.935 |
| **GeoI-Endpoint20** | **0% / 1.483** | **0% / 1.337** | **7,14% / 6,25%** | **96,64%** | **6.936** |

Mỗi phương pháp gửi **1.308/1.308 mốc**, delay bằng 0. Byte đo JSON yêu cầu +
ID POI phản hồi, chưa gồm HTTP/TLS. Máy chủ nhận năm Q; GPS hiện tại chỉ dùng
tại thiết bị để chọn top-5.

So **cùng L20**, endpoint ở S10 có MAE tăng **436 m**, bootstrap ghép theo
nhóm 95% **[313; 563]**. Khoảng này không còn qua 0 như pilot bốn nhóm. S9
tăng **592 m**, khoảng **[441; 738]**. Đây là tín hiệu ổn định hơn về sai số
ước lượng trong cùng bộ sinh và bank attacker, không phải bằng chứng vượt mọi
attacker hoặc các paper nguyên bản.

S10 Hit100 cùng bằng 0 nên không thể dùng cột này để nhận có lợi thế. Hit500
giảm 6,25 điểm %, khoảng **[-14,29; 0]**, vẫn chưa chắc chắn ngoài mẫu. S9
Hit500 giảm 9,82 điểm %, khoảng **[-17,86; -2,68]**. Các decoder theo từng
ngưỡng có thể khác nhau; không diễn giải các Hit thành một CDF duy nhất.

Giá của tăng nhiễu: Recall thấp hơn L20 thường **2,30 điểm %**, khoảng
**[-3,09; -1,55]**; byte gần bằng nhau. So L10, Recall gần bằng nhau
(chênh -0,17 điểm %, khoảng [-0,97; +0,59]), nhưng byte tăng khoảng 51,5%.

Chất lượng dịch vụ có đuôi thấp: Recall trung vị 98,18%, percentile 10 là
92,12%, **6/112 lần chạy dưới 90%**, thấp nhất **77,61%**. Trung bình 96,64%
không bảo đảm mọi chuyến tốt; chưa thay ngưỡng hay bỏ các chuyến này để nâng số.

Không nhóm nào có Hit100 đối với Endpoint20. Nếu giả sử nhóm là Bernoulli
độc lập từ bộ sinh này, cận trên một phía 95% cho xác suất một nhóm mới có ít
nhất một hit là **10,15%**; khoảng hai phía có cận trên 12,34%. Tính theo 28
nhóm, không dùng 112 lần lặp để thu hẹp giả tạo. Vẫn chỉ một thành phố, bản đồ
tái dựng và số rep nhỏ; khoảng bootstrap không hiệu chỉnh nhiều so sánh.

## Kiểm tra phụ: attacker liên kết hai chuyến

Giả sử attacker biết account/định danh để liên kết hai chuyến lặp. Nó dùng cả
hai chuỗi công khai để lấy trung bình/cross-trip các ước lượng, hoặc học từ
đặc trưng của chuyến cần tấn công kèm thống kê gộp cặp. Fit/chọn bank phụ riêng
trên train/selection. **Mỗi endpoint vẫn chấm tới GPS thật của chính chuyến
đó**, không đổi đáp án thành một đích chung hoặc trung bình hai đích.

Lần dựng đầu phát hiện trên train rằng hai vị trí FCD cuối khác khoảng 0–6,6 m
dù cùng tuyến. Giả thiết đích chính xác giống nhau đã bị bác bỏ; lần đó dừng
trước test. Protocol, code snapshot và lỗi được giữ trong artifact. Protocol
sau sửa riêng định nghĩa kiểm tra phụ, giữ defense và toàn bộ 28 nhóm.

| Bank liên kết | S9 MAE m | S10 MAE m | S10 Hit500 |
|---|---:|---:|---:|
| GeoI-Slack, L20 | 929 | 856 | 17,86% |
| GeoI-Endpoint20 | 1.358 | 1.602 | 8,93% |

ΔMAE S10 là **746 m**, khoảng [567; 916]. Không dùng MAE của bank liên kết
cao hơn bank từng chuyến để nói “liên kết làm privacy tốt hơn”: thêm thông tin
không làm attacker tối ưu yếu đi, nhưng lựa chọn trên tập nhỏ có thể tổng quát
hóa kém. Đây là hạn chế của bank/selection; kiểm tra phụ không thay kết quả
chính và không chứng minh mọi attacker liên kết đều thất bại.

Cận 0,0575/m là **mỗi phiên**; hai phiên liên kết có cận ghép tối đa
**0,115/m**, không giữ nguyên 0,0575/m cho cả lịch sử. Giờ hoạt động và account
vẫn không được che bởi thay đổi nhiễu này.

## Evidence và giới hạn

[Evidence và lệnh tái kiểm tra](../../artifacts/benchmarks/endpoint_generalization_20261005/README.md)
giữ protocol, attacker selection, toàn bộ 448 lần chạy, error của mọi decoder,
RNG evaluator riêng, nguồn/model gzip và lần lỗi trước test. Kiểm tra độc lập
đã tính lại **5.232 mốc dịch vụ, 53.760 dự đoán đơn chuyến và 170.240 dự đoán
liên kết**, kiểm tra 112 substream khác nhau, target riêng, byte, clock và cap.

Mạng là mạng công khai tái dựng từ 9.138 polyline; nguồn GPS không sửa. Không
xác minh các luật rẽ/làn gốc. Chỉ đo utility tìm POI gần nhất của dịch vụ hiện
có; chưa gán các số này cho mọi query purpose. Raw đạt 100% xác nhận pipeline
endpoint hoạt động, không xác nhận bank là tối ưu. Kết quả mới tăng độ tin cậy
của **cấu hình đã khóa trong phạm vi này**, vẫn cần kiểm tra thành phố khác,
attacker mạnh hơn và phần đuôi utility.

Bank hiện tại dùng đặc trưng không đổi khi hoán vị Q. Nó chưa khai thác riêng
thứ tự slot Q hoặc candidate ID ổn định mà `PublicTranscript` có thể công bố.
Mảng tọa độ trong runner vẫn giữ thứ tự, nên không tự coi lớp truyền thông đã
che thứ tự/ID. Cần thêm attacker dùng thông tin này; kết quả hiện tại không
đại diện mọi attacker nhìn thấy toàn bộ giao thức. Kiểm tra mới trên test đã
xem phải được gọi là chẩn đoán, không phải xác nhận holdout mới.
