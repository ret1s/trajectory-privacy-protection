# Nội dung trao đổi với GVHD — cập nhật 10/09/2026

Đã có một vòng thực nghiệm mới hoàn chỉnh để trình bày: mở rộng dữ liệu, thử một
biến thể thuật toán, khóa cách chọn, chấm trên tuyến mới và kiểm tra độc lập.
**Chưa nên trình bày là thuật toán cuối hoặc kết quả vượt SOTA.**

## Ba nội dung chính

1. **Dữ liệu đáng tin hơn.** Thêm 12 nhóm tuyến SUMO+OSM, 264 chuyến và 172.443 điểm;
   393 bản ghi phủ đủ 10 kịch bản/30 trường hợp con ở cả tập chọn và tập xác nhận.
   Có kiểm tra dữ liệu gốc, sửa lỗi đổi làn gây bước nhảy và lưu SQLite có phiên bản,
   nhật ký. Không dùng GeoLife. Một số ca còn ít mẫu; hai cửa sổ dừng bị trùng giữa
   hai tập đã được công bố và kiểm tra độ nhạy.
2. **Phát triển cách chọn điểm giả.** Giữ neo riêng tư và ràng buộc đường; thử bộ lọc
   phân biệt dừng/di chuyển từ lịch sử đã bảo vệ để ước lượng những POI người dùng
   có thể cần. Chỉ thay bộ lọc, giữ nguyên ngân sách và số điểm giả để so sánh công bằng.
3. **Có kết quả và phản ví dụ rõ.** Bộ lọc mới giữ Recall top-10 khoảng 96,31%, giảm
   Hit đã chọn từ 3,73% xuống 2,92%, nhưng MAE cũng giảm 577,4→555,1 m (bất lợi).
   Không cải thiện đồng đều qua các nhóm. Không cấu hình nào qua ngưỡng chất lượng
   90% ở mọi ca trên tập chọn, nên chưa thay phương pháp mặc định.

## Cách trình bày trong khoảng 5 phút

**Mở đầu — vấn đề:**

“Em muốn tránh việc thuật toán chỉ tốt trên những tuyến đã dùng để chỉnh nó.
Vòng này em giữ nguyên tập học, sinh sáu nhóm tuyến mới để chọn và sáu nhóm khác
để xác nhận. Mục tiêu là biết chất lượng và riêng tư có chuyển sang tuyến mới được không.”

**Dữ liệu — phân biệt dữ liệu với bảo vệ:**

“SUMO tạo chuyển động thật trong mô phỏng; lớp kịch bản bổ sung cửa sổ quan sát,
truy vấn và đáp án cho các dạng tấn công. Em đã có dữ liệu cho cả 10 kịch bản,
nhưng vòng thuật toán này chỉ đánh giá S1–S3. Có dữ liệu S4–S10 chưa có nghĩa đã
chứng minh bảo vệ được các mục tiêu đó.”

**Thuật toán — ý tưởng ngắn gọn:**

“Nếu người dùng đang dừng thì vị trí có thể ít thay đổi; nếu đang đi thì vùng vị
trí có thể dịch chuyển. Em thử ước lượng hai khả năng này chỉ từ dữ liệu đã bảo vệ,
rồi chọn các điểm giả để kết quả trả về vẫn chứa POI cần thiết. Em không đưa tốc độ
thật hay nhãn kịch bản cho bộ lọc.”

**Kết quả — giải thích đúng các con số:**

“Recall cho biết còn lấy được bao nhiêu POI cần tìm. Hit100 là tỷ lệ đối thủ đoán
trong bán kính 100 m; thấp hơn là tốt. MAE là sai số vị trí trung bình; cao hơn là tốt.
Biến thể mới giảm Hit nhưng MAE cũng giảm: ít lần đoán rất gần hơn, song mức sai trung
bình nhỏ hơn. Vì vậy chưa thể nói nó bảo vệ tốt hơn theo mọi cách đo.”

“Lấy top-10 thay vì top-5 từ máy chủ giúp Recall lên khoảng 96%, nhưng số byte mã
POI tăng khoảng 65%. Em ghi riêng chi phí này, không coi toàn bộ mức tăng là công
của thuật toán. Truy vấn thưa 60 giây là ca yếu trên tập chọn; việc tập xác nhận
đẹp hơn không cho phép em chọn lại sau khi đã xem kết quả.”

**Kết lại — xin góp ý cho bước tiếp:**

“Em đề nghị ưu tiên kiểm tra mô hình chuyển động ở truy vấn thưa, và tăng số mẫu
cho các ca tương lai/điểm đầu-cuối còn ít. Sau đó mới mở rộng tấn công và so sánh
ngoài trên một tập xác nhận mới. Em muốn chốt với thầy tiêu chí đóng góp chính:
cải thiện một đánh đổi có bằng chứng, trong một khung đánh giá rõ ràng và tái lập được.”

## Mở tài liệu nào?

- Luận văn chính: `artifacts/reports/graduation_thesis.pdf`.
- Mục **3.9.6 / Bảng 3.8**: dữ liệu mới, điều kiện kiểm tra và số mẫu từng ca.
- Mục **5.13**: bộ lọc hai trạng thái, dữ liệu đầu vào và công thức.
- Mục **6.14 / Bảng 6.30–6.31**: giao thức, kết quả, chi phí và phản biện.
- Chương **7** đã cô đọng lại theo kết quả mới, tránh lặp lịch sử thử nghiệm.
- Số liệu chi tiết: `artifacts/benchmarks/fresh_switching/readout.json`.
- Hồ sơ xác minh/việc cần làm tiếp: `docs/reviews/verification_fresh_switching.md`.

270 kiểm thử đã qua. Kiểm tra độc lập dựng lại 265.068 dự đoán, 97.488 truy vấn,
toàn bộ chuyển đường dummy và kiểm tra chạy lại đầy đủ/tiền tố trên mẫu đã công bố.
Đây là bằng chứng về tính đúng và giới hạn của vòng thử, không phải chứng nhận
đủ điều kiện nhận ở một hội nghị cụ thể.
