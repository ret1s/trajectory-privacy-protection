# Bảo vệ riêng tư vị trí bằng Geo-I/REM

Bộ **18 slide** tập trung vào cơ chế theo từng kịch bản, benchmark chi tiết và ví dụ thuật toán qua nhiều mốc chuyển động. Giữ phong cách tài liệu khoa học: nền trắng, chữ serif, công thức và bảng số liệu. Slide chỉ chứa nội dung trình chiếu.

Bố cục được thu gọn từ 22 trang: tiêu đề 32 px, nội dung chủ yếu 20–22 px, ghi nguồn 15 px trên canvas 1280 × 720. Gộp chính sách S9/S10 với timeline delay; S1–S6 vào một bảng cơ chế; S7/S8 cùng trang; GPS cục bộ và POI đổi trạng thái thành hai bảng song song. Các bảng benchmark vẫn tách theo tập dữ liệu. Sơ đồ có các tham số và điều kiện kích hoạt ngay bên cạnh. Trường `previous_slide_numbers` lưu ánh xạ đủ 22 chủ đề của bản trước.

- [slides.pdf](slides.pdf): bản trình chiếu 16:9.
- [slides.html](slides.html): bản offline, chữ và hình vector. Dùng ←/→, Home/End để chuyển trang và F để toàn màn hình.
- [presentation_script.md](presentation_script.md): lời trình bày theo từng slide, cách chỉ hình, chuyển ý và nội dung chuẩn bị trả lời.
- [speaker_notes.txt](speaker_notes.txt): ghi chú và các nguồn đối chiếu riêng.

Thứ tự nội dung:

1. Slide 2: S9/S10, đối chiếu cơ chế delay trước đây với Endpoint20 đã đo sau đó; ví dụ hàng đợi và công bố.
2. Slide 3–4: cơ chế và giới hạn đối với S1–S8.
3. Slide 5–12: giao thức, các bảng benchmark riêng theo dữ liệu/cấu hình, chất lượng dịch vụ và chi phí.
4. Slide 13–18: kiến trúc hiện tại, thay đổi so với bản trước và các ví dụ nhiều bước.

**Phần delay cần đọc đúng phiên bản.** Bản trước có warmup 60 s, chờ công bố 60 s và hủy Q còn trong hàng đợi khi đóng phiên. Endpoint20 đã đo sau đó có warmup/delay bằng 0, tăng nhiễu trên mọi lần đọc được phép. Mô hình Epoch8/L30 hiện tại cũng gửi ngay. Bộ slide đối chiếu bằng chứng đã có; không thêm hoặc đo lại một biến thể delay cho mô hình hiện tại.

Kết quả S1–S3 lịch sử, S4/S8 diagnostic, S5–S6 pilot, Endpoint20 L20 và xác nhận utility L30 thuộc các giao thức riêng. Không ghép chúng thành kết quả của một cấu hình bảo vệ đầy đủ S1–S10. Các số liệu, đầu ra bảo vệ và benchmark gốc được giữ nguyên.

Ví dụ chuyển động lấy cơ học từ `freshqp-001`, TRAIN, draw 1, phiên đầu. Các mốc 0/20/60/120/180/240/300/600 s đều dùng đầu ra đã lưu, cùng GPS nguồn/FCD và phản hồi L30. Bảng ngân sách tính cả các lần đọc giữa 300 và 600 s. Nhánh đọc rồi giữ Z dùng ô phụ từ phiên 6, không nối vào hành trình đầu. Chỉ số Q trên hình là nhãn theo dõi nội bộ; máy chủ không nhận ID track ổn định. Không tự đặt giá trị nhiễu Laplace chưa được lưu.

Dữ liệu và tái lập:

- `benchmark_tables.json`, `benchmark_evidence.json`: số liệu đầy đủ, định nghĩa, phạm vi, nguồn và hash.
- `endpoint_focus.json`: timeline delay trước đây và phép thử có đối chứng delay, trích nguyên nguồn đã lưu.
- `multistep_sample.json`, `build_multistep_walkthrough.py`: timeline nhiều bước, kích hoạt thành phần, phản hồi POI và kiểm tra nguồn GPS; chỉ đọc lại đầu ra, không chạy sampler.
- `sample_walkthrough.json`, `utility_sample.json`, `endpoint_sample.json`: các trích xuất trước vẫn được giữ để đối chiếu.
- `build_slides.py` và các module `*_visuals.py`: dựng HTML và ghi chú; không đánh giá lại mô hình. PDF xuất bằng Chrome sau khi kiểm tra hình/chữ.
- `source_pins.json`, `manifest.json`: nguồn và hash của bản phát hành.

Mẫu là GPS tổng hợp SUMO; đích detour là đích chuẩn chỉ dùng tại thiết bị trong đánh giá. Các bản đồ dùng cùng tỷ lệ mét trên hai trục; crop và bỏ nét đường trùng chỉ phục vụ hiển thị. Bộ mô hình, thuật toán và bản luận văn chính không thay đổi trong lần biên soạn slide này.

© OpenStreetMap contributors — [nguồn và giấy phép](https://www.openstreetmap.org/copyright).
