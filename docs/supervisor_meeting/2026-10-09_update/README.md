# Bảo vệ riêng tư vị trí bằng Geo-I/REM

Bộ **8 slide**, khoảng **8–10 phút**, tập trung giải thích các cơ chế bổ sung để hoàn thiện bảo vệ scenario và giữ chất lượng dịch vụ. Giữ phong cách tài liệu khoa học: nền trắng, chữ serif và sơ đồ luồng. Slide chỉ chứa nội dung trình chiếu.

Bản ngắn đi từ bảo vệ query content tới cơ chế liên phiên và dự đoán tương lai, rồi giải thích cap tám phiên, so kiến trúc cũ–mới, minh họa toàn bộ luồng và đối chiếu benchmark. Tiêu đề 32 px, nội dung chủ yếu 20–22 px, ghi nguồn 15 px trên canvas 1280 × 720. Bản **18 trang được giữ nguyên trong [detailed/](detailed/README.md)** để tra cứu khi cần. Trường `detailed_slide_numbers` đối chiếu từng slide ngắn với bản chi tiết.

- [slides.pdf](slides.pdf): bản trình chiếu 16:9.
- [slides.html](slides.html): bản offline, chữ và hình vector. Dùng ←/→, Home/End để chuyển trang và F để toàn màn hình.
- [presentation_script.md](presentation_script.md): lời trình bày theo từng slide, cách chỉ hình, chuyển ý và nội dung chuẩn bị trả lời.
- [speaker_notes.txt](speaker_notes.txt): ghi chú và các nguồn đối chiếu riêng.

Thứ tự nội dung:

1. S7: tách mục đích riêng khỏi request chung; phân biệt cơ chế đã có với phần mới.
2. L30: độ gần để thu ứng viên, bốn tiêu chí local để chọn câu trả lời; giới hạn độ phủ.
3. S4–S6: nghiên cứu nền, thành phần hỗ trợ, nguy cơ còn lại; S8 là hướng mở, không có slide riêng.
4. Cap tám phiên: định nghĩa phiên/epoch, phép chia ngân sách, chi phí nhánh và hợp thành.
5. Kiến trúc cũ–mới: cùng bốn tầng; đóng khung mô hình tại thiết bị, máy chủ và đầu ra riêng bên ngoài.
6. Ví dụ đầy đủ: timeline 0/20/60/120/600 s, bản đồ 60 s và luồng GPS → Z → b → Q → POI → đáp án local.
7. Benchmark utility: Recall bản trước để đặt bối cảnh; bảng bốn mục đích L20–L30 trên cùng Q và chi phí byte.
8. Benchmark privacy: bằng chứng bổ sung S4/S5/S6 và nhánh S9/S10; phân biệt các cohort và giới hạn.

Cap tám phiên bổ sung giới hạn tích lũy thông tin tọa độ cho Geo-I; nó không tự ẩn account/IP hoặc bảo đảm không liên kết danh tính. Tám là cấu hình hữu hạn, không là tám lần GPS hoặc ngân sách vô hạn cả đời. S8 được lược khỏi trọng tâm buổi nói, vẫn giữ trong định nghĩa scenario và giới hạn nghiên cứu.

**Luồng S7:** nhu cầu thật giữ local; gửi 5 Q × L30 mỗi loại; nhận phản hồi tại thiết bị; gộp và bỏ trùng ID; dùng GPS cùng nhu cầu để lọc và sắp xếp danh sách phù hợp. Tối đa 150 bản ghi mỗi loại trước bỏ trùng, không mặc định 150 POI duy nhất. Phần giải thích cơ chế không yêu cầu top-5. Triển khai và benchmark hiện tại vẫn dùng k = 5 và Recall@5; hình minh họa chỉ trích phần đầu kết quả đã lưu. Không bỏ giới hạn trong mã hoặc thay kết quả đánh giá.

**So sánh với báo cáo trước:** Recall 95,44% của GeoI-Slack L10 ở báo cáo 26/09–03/10 là số lịch sử cho tìm gần nhất, cache và cách gộp năm scenario. Recall macro 92,69% hiện tại dùng bốn purpose, cap liên phiên và cohort khác; không diễn giải chênh hai số là tăng/giảm. Phép so có đối chứng giữ cùng Q/Z/cap/lịch, L20 89,71% → L30 92,69%, kèm CI và byte. S4–S6 nay có bằng chứng bổ sung nhưng chưa có xác nhận privacy mới cho cùng L30.

**Phần delay cần đọc đúng phiên bản.** Bản trước có warmup 60 s, chờ công bố 60 s và hủy Q còn trong hàng đợi khi đóng phiên. Endpoint20 đã đo sau đó có warmup/delay bằng 0, tăng nhiễu trên mọi lần đọc được phép. Mô hình Epoch8/L30 hiện tại cũng gửi ngay. Bộ slide đối chiếu bằng chứng đã có; không thêm hoặc đo lại một biến thể delay cho mô hình hiện tại.

Kết quả S1–S3 lịch sử, S4/S8 diagnostic, S5–S6 pilot, Endpoint20 L20 và xác nhận utility L30 thuộc các giao thức riêng. Không ghép chúng thành kết quả của một cấu hình bảo vệ đầy đủ S1–S10. Các số liệu, đầu ra bảo vệ và benchmark gốc được giữ nguyên.

Ví dụ chuyển động lấy cơ học từ `freshqp-001`, TRAIN, draw 1, phiên đầu. Bản ngắn trích 0/20/60/120/600 s từ đầu ra đã lưu, cùng GPS nguồn/FCD và phản hồi L30 tại 60 s: 420 bản ghi → 252 POI duy nhất → đáp án khác nhau theo mục đích local. Tổng ngân sách tính đủ các lần đọc không hiển thị. Nhánh đọc rồi giữ Z thuộc phiên 6, không nối vào hành trình đầu. Chỉ số Q trên hình là nhãn theo dõi nội bộ; máy chủ không nhận ID track ổn định. Không tự đặt giá trị nhiễu Laplace chưa được lưu.

Dữ liệu và tái lập:

- `benchmark_tables.json`, `benchmark_evidence.json`: số liệu đầy đủ, định nghĩa, phạm vi, nguồn và hash.
- `endpoint_focus.json`: timeline delay trước đây và phép thử có đối chứng delay, trích nguyên nguồn đã lưu.
- `multistep_sample.json`, `build_multistep_walkthrough.py`: timeline nhiều bước, kích hoạt thành phần, phản hồi POI và kiểm tra nguồn GPS; chỉ đọc lại đầu ra, không chạy sampler.
- `sample_walkthrough.json`, `utility_sample.json`, `endpoint_sample.json`: các trích xuất trước vẫn được giữ để đối chiếu.
- `build_slides.py` và các module `*_visuals.py`: dựng HTML và ghi chú; không đánh giá lại mô hình. PDF xuất bằng Chrome sau khi kiểm tra hình/chữ.
- `source_pins.json`, `manifest.json`: nguồn và hash của bản phát hành.

Mẫu là GPS tổng hợp SUMO; đích detour là đích chuẩn chỉ dùng tại thiết bị trong đánh giá. Các bản đồ dùng cùng tỷ lệ mét trên hai trục; crop và bỏ nét đường trùng chỉ phục vụ hiển thị. Bộ mô hình, thuật toán và bản luận văn chính không thay đổi trong lần biên soạn slide này.

© OpenStreetMap contributors — [nguồn và giấy phép](https://www.openstreetmap.org/copyright).

Kiểm chứng truy hồi đa mục đích ngày 09/10: không candidate nào qua gate; giữ L30. Script slide 2 có phần trao đổi về bốn template L10 (đích mẫu công khai), không coi đây là kiến trúc đã áp dụng. [Logic và kết quả đầy đủ](../../research/2026-10-09_multi_purpose_retrieval.md).
