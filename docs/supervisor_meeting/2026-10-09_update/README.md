# Bảo vệ riêng tư vị trí bằng Geo-I/REM

Bộ **8 slide**, khoảng **8–10 phút**, tập trung giải thích các cơ chế bổ sung để hoàn thiện bảo vệ scenario và giữ chất lượng dịch vụ. Giữ phong cách tài liệu khoa học: nền trắng, chữ serif và sơ đồ luồng. Slide chỉ chứa nội dung trình chiếu.

Bản ngắn giữ ba thay đổi chính: ngân sách liên phiên, bốn mục đích xếp hạng local và phản hồi L30 để phục hồi utility. Có sơ đồ phương pháp, nhánh thử nghiệm S9/S10, một ví dụ nhiều bước và một trang bằng chứng chính. Tiêu đề 32 px, nội dung chủ yếu 20–22 px, ghi nguồn 15 px trên canvas 1280 × 720. Bản **18 trang được giữ nguyên trong [detailed/](detailed/README.md)** để tra cứu khi cần. Trường `detailed_slide_numbers` đối chiếu từng slide ngắn với bản chi tiết.

- [slides.pdf](slides.pdf): bản trình chiếu 16:9.
- [slides.html](slides.html): bản offline, chữ và hình vector. Dùng ←/→, Home/End để chuyển trang và F để toàn màn hình.
- [presentation_script.md](presentation_script.md): lời trình bày theo từng slide, cách chỉ hình, chuyển ý và nội dung chuẩn bị trả lời.
- [speaker_notes.txt](speaker_notes.txt): ghi chú và các nguồn đối chiếu riêng.

Thứ tự nội dung:

1. Slide 1–2: thay đổi chính; kiến trúc bản trước và hiện tại đặt cạnh nhau theo bốn tầng, tô nổi các phần thay đổi.
2. Slide 3: ngân sách dùng chung qua tám phiên và kiểm tra trước GPS.
3. Slide 4: S7, bốn mục đích local và ứng viên POI L30.
4. Slide 5: S9/S10, delay lịch sử và Endpoint20 được đo riêng.
5. Slide 6: mẫu thực tế qua 0/20/60/120/600 s và phân biệt không đọc với đọc rồi giữ Z.
6. Slide 7: bằng chứng utility/chi phí L30, endpoint L20 và kết luận pilot S5/S6.
7. Slide 8: phạm vi S1–S10 và ưu tiên tiếp theo, gồm khoảng trống S8.

**Phần delay cần đọc đúng phiên bản.** Bản trước có warmup 60 s, chờ công bố 60 s và hủy Q còn trong hàng đợi khi đóng phiên. Endpoint20 đã đo sau đó có warmup/delay bằng 0, tăng nhiễu trên mọi lần đọc được phép. Mô hình Epoch8/L30 hiện tại cũng gửi ngay. Bộ slide đối chiếu bằng chứng đã có; không thêm hoặc đo lại một biến thể delay cho mô hình hiện tại.

Kết quả S1–S3 lịch sử, S4/S8 diagnostic, S5–S6 pilot, Endpoint20 L20 và xác nhận utility L30 thuộc các giao thức riêng. Không ghép chúng thành kết quả của một cấu hình bảo vệ đầy đủ S1–S10. Các số liệu, đầu ra bảo vệ và benchmark gốc được giữ nguyên.

Ví dụ chuyển động lấy cơ học từ `freshqp-001`, TRAIN, draw 1, phiên đầu. Bản ngắn trích 0/20/60/120/600 s từ đầu ra đã lưu, cùng GPS nguồn/FCD và phản hồi L30. Tổng ngân sách tính đủ các lần đọc không hiển thị. Nhánh đọc rồi giữ Z dùng ô phụ từ phiên 6, không nối vào hành trình đầu. Chỉ số Q trên hình là nhãn theo dõi nội bộ; máy chủ không nhận ID track ổn định. Không tự đặt giá trị nhiễu Laplace chưa được lưu.

Dữ liệu và tái lập:

- `benchmark_tables.json`, `benchmark_evidence.json`: số liệu đầy đủ, định nghĩa, phạm vi, nguồn và hash.
- `endpoint_focus.json`: timeline delay trước đây và phép thử có đối chứng delay, trích nguyên nguồn đã lưu.
- `multistep_sample.json`, `build_multistep_walkthrough.py`: timeline nhiều bước, kích hoạt thành phần, phản hồi POI và kiểm tra nguồn GPS; chỉ đọc lại đầu ra, không chạy sampler.
- `sample_walkthrough.json`, `utility_sample.json`, `endpoint_sample.json`: các trích xuất trước vẫn được giữ để đối chiếu.
- `build_slides.py` và các module `*_visuals.py`: dựng HTML và ghi chú; không đánh giá lại mô hình. PDF xuất bằng Chrome sau khi kiểm tra hình/chữ.
- `source_pins.json`, `manifest.json`: nguồn và hash của bản phát hành.

Mẫu là GPS tổng hợp SUMO; đích detour là đích chuẩn chỉ dùng tại thiết bị trong đánh giá. Các bản đồ dùng cùng tỷ lệ mét trên hai trục; crop và bỏ nét đường trùng chỉ phục vụ hiển thị. Bộ mô hình, thuật toán và bản luận văn chính không thay đổi trong lần biên soạn slide này.

© OpenStreetMap contributors — [nguồn và giấy phép](https://www.openstreetmap.org/copyright).
