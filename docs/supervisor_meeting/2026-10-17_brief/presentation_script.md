# Kịch bản trình bày: kiến trúc và chứng minh privacy

Buổi 17/10/2026. Bản rút gọn chỉ trình bày kiến trúc và phần chứng minh privacy,
đi theo bốn trang của PDF. Dự kiến khoảng 10–12 phút. Không trình bày benchmark,
chứng minh utility hay walkthrough dataset trong buổi này.

## Trang 1 — Kiến trúc và câu hỏi cần chứng minh

“Thưa thầy/cô, phần cơ chế chính em đã giới thiệu trước đó. Hôm nay em tập
trung giải thích mô hình được đặc tả bằng xác suất như thế nào và vì sao có
cận riêng tư cho cả chuỗi truy vấn.

Geo-I là yêu cầu bảo vệ. REM là cơ chế trực tiếp lấy mẫu Z trên mạng đường
để đáp ứng yêu cầu đó. Khi di chuyển, noisy test quyết định giữ Z hay tạo
Z mới, và mọi thao tác này được hạch toán chi phí privacy.

Từ lịch sử thông tin đã bảo vệ, mô hình duy trì belief về những vị trí
người dùng có thể đang ở. Belief chỉ điều chỉnh xác suất chọn các đoạn Q.
Mỗi đoạn có ba khung, mỗi khung có năm tọa độ gửi server.

Server trả POI; thiết bị hợp cache và xử lý nhu cầu thật tại local. Mũi tên
GPS đi thẳng xuống xếp hạng local không đi qua kênh gửi server.

Điểm cần chứng minh không phải là server có nhìn thấy GPS thật hay không.
Ngay cả Q giả cũng có thể chứa dấu vết qua cách lựa chọn. Em cần giới hạn
mức transcript được phép thay đổi khi quỹ đạo thật thay đổi.”

**Chỉ hình:** GPS → lớp bảo vệ → belief → chọn đoạn Q → server. Sau đó chỉ
khung thiết bị, đường GPS local và nhu cầu riêng. Z và belief không gửi đi.

**Lưu ý:** đây là cấu hình chính có cap Geo-I. Bản thử cập nhật liên tục
chưa thay vào sơ đồ này và chưa được gán kết quả thực nghiệm trong report.

## Trang 2 — Viết thuật toán theo chiều xuôi

“Ở công thức (1), REM gán mỗi ứng viên đường một trọng số giảm theo khoảng
cách đến GPS. Chia cho tổng trọng số để được một phân phối xác suất, rồi
lấy mẫu Z. Không có bước thêm nhiễu trên mặt phẳng rồi mới dùng REM để snap.

Ở công thức (2), em cộng nhiễu Laplace vào khoảng cách đến Z cũ rồi kiểm
tra ngưỡng. Vì vậy quyết định giữ hay tạo lại Z cũng có bảo vệ. Nếu dùng
một ngưỡng chính xác thì bản thân nhánh giữ/đổi có thể tiết lộ khoảng cách.

Ở công thức (3), belief được dự đoán theo thời gian, rồi cập nhật bằng
likelihood của quan sát đã bảo vệ. Nếu không có quan sát mới thì chỉ dự
đoán. Đây là một mô hình gần đúng; em không giả định nó biết chính xác GPS.

Ở công thức (4), em dùng belief để tính điểm kỳ vọng cho từng đoạn Q,
rồi lượng tử hóa điểm vào khoảng từ không đến một. Chữ z thường ở đây là
điểm số, khác tọa độ Z viết hoa.

Ở công thức (5), xác suất nền w chỉ nhận một điều chỉnh có giới hạn.
Đoạn trên điểm trung bình được ưu tiên hơn. Ví dụ hai đoạn vốn 50/50,
điểm 1 và 0, allowance một phần hai, thì xác suất thành 55/45.

Như vậy GPS ảnh hưởng Q thông qua lịch sử đã bảo vệ và belief. Mỗi đoạn
được lấy mẫu một lần, sau đó gửi lần lượt các khung đã chọn.”

**Cách đọc ký hiệu:** dấu ∼ là lấy mẫu; Σ là cộng; điều kiện sau dấu | là
thông tin đang được giữ cố định. Một đoạn Q là một đầu ra chung, không phải
năm lần lấy mẫu độc lập hay một bảo đảm k-anonymity.

## Trang 3 — Geo-I truyền xuống transcript như thế nào?

“Công thức (6) dùng bất đẳng thức tam giác. GPS dịch chuyển d mét thì
khoảng cách đến một ứng viên cố định chỉ thay đổi tối đa d mét. Chặn cả tử
và mẫu của REM cho tỷ số xác suất không vượt e mũ u nhân d.

Ở bước reuse, phép kiểm tra cũng tiêu một đơn vị. Nếu cần tạo mới thì thêm
một đơn vị REM. Vì thế em phải tính cả nhánh, không chỉ đếm số lần đổi Z.

Công thức (7) nhân các xác suất có điều kiện qua lịch sử. Cap M đơn vị
chặn tổng chi phí, từ đó có cận Cs nhân độ lệch lớn nhất giữa hai trace.
Các trace được so trên cùng lịch có thể đọc, không đánh chỉ số theo số lần
đọc đã xảy ra vì số lần đó phụ thuộc nhánh.

Công thức (8) giải thích con số 0,23. Nó là cap hiệu dụng từ cấu hình
B=0,24 và H=12, tương ứng u=0,01 và M=23. Đây không phải hằng số của Geo-I
hay bán kính nhiễu. Cap bảo vệ tổng tiết lộ nhưng có thể ngừng cập nhật GPS.

Phần chiều ngược dùng công thức (9). Server thấy một transcript, nhưng
transcript đó có thể được tạo ra từ nhiều lịch sử Z. Em cộng xác suất qua
tất cả các lịch sử đó. Vì mỗi lịch sử có cận Geo-I và các bước sau chỉ dùng
thông tin đã bảo vệ, tổng cũng giữ cận. Đây là tính chất hậu xử lý.

Chứng minh không yêu cầu phải đảo được Q về Z, cũng không cần chứng minh
không ai có thể đảo. Nó chặn trực tiếp tỷ số xác suất của đầu ra.”

**Điểm nhấn:** mọi K trong phép cộng phải là cùng kernel khi giữ lịch sử
đã bảo vệ và bối cảnh công khai. Không được bổ sung GPS thô để thay đổi
request, support, lịch gửi hoặc lỗi quan sát được.

## Trang 4 — Cận riêng của Q và ý nghĩa suy luận

“Công thức (10) bắt đầu từ điểm số nằm trong khoảng không đến một. Xác
suất của mỗi đoạn nằm giữa w nhân một trừ alpha và w nhân một cộng alpha.
Vì vậy hai belief bất kỳ chỉ làm xác suất cùng đoạn chênh trong hệ số một
cộng epsilon. Tập ứng viên và w phải chung; belief không được loại đoạn.

Công thức (11) phân bổ epsilon giảm theo j nhân j cộng một. Tổng triệt tiêu
không vượt Gamma dù chưa biết chuyến đi dài bao lâu. Ledger không được reset
qua restart; gửi khung còn lại không nhận thêm allowance.

Công thức (12) dùng chain rule để nhân các cận từng đoạn thành cận cho cả
transcript. Không cần độc lập giữa các đoạn. Cận mọi thời lượng ở Gamma một
nhỏ hơn 2,437372, dưới các giả thiết của mô hình.

Công thức (13) lấy cận nhỏ hơn giữa Geo-I theo khoảng cách và cận Q.
Hai ngân sách có đơn vị khác nhau nên không cộng trực tiếp.

Công thức (14) là cách đọc ngược bằng Bayes. Tỷ lệ tin tưởng sau quan sát
bằng tỷ lệ tin tưởng ban đầu nhân bằng chứng từ transcript. Định lý chặn
hệ số bằng chứng đó. Hai quỹ đạo là phép so sánh của định lý, không phải
hai đáp án attacker bắt buộc phải biết sẵn.

Cuối cùng, em giữ rõ các điều kiện: cùng clock và provider law, không có
ID/GPS ngoài kênh, state không rollback, entropy theo giả thiết. Phần Geo-I
không gian vẫn dựa trên kernel lý tưởng và phục hồi upstream đầy đủ.
Bản thử bỏ cap Geo-I phải thay cận không gian bằng cận prefix; không được
tiếp tục gắn 0,23 cho mọi thời lượng.”

**Kết luận nói:** “Em đã nối đặc tả của từng bước với một cận cho toàn
transcript. Bảo đảm là giới hạn lượng bằng chứng quan sát bổ sung, không
phải lời khẳng định tuyệt đối không thể suy ra vị trí.”

## Những câu cần tránh

- “Có năm Q giả thì mặc định đã bảo vệ danh tính.”
- “Belief là posterior GPS thật chính xác.”
- “0,23 là epsilon của mỗi lần đọc hoặc là sai số 0,23 mét.”
- “Tỷ số 2,44 là xác suất attacker đoán đúng.”
- “Cận Geo-I float và full-state restore đã được chứng nhận trong mã.”
- “Bản cập nhật liên tục vẫn có cap Geo-I 0,23 cho cả phiên.”
