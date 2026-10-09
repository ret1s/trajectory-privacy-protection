# Bảo vệ riêng tư vị trí bằng Geo-I/REM

Bộ slide trình bày phương pháp, ví dụ minh họa và đánh giá thực nghiệm. Bố cục theo phong cách tài liệu khoa học: nền trắng, chữ serif, ký hiệu toán với chỉ số dưới/trên và bảng có đường kẻ ngang. Thông tin biên soạn và nguồn dữ liệu nằm trong ghi chú riêng.

- [slides.pdf](slides.pdf): 14 slide 16:9 để mở hoặc trình chiếu.
- [slides.html](slides.html): bản trình chiếu offline, sơ đồ và biểu đồ vector.
- [presentation_script.md](presentation_script.md): kịch bản trình bày tiếng Việt theo từng slide, kèm cách chỉ hình, câu chuyển ý và phần trả lời khi được hỏi.

Mở HTML bằng trình duyệt. Dùng **←/→** để chuyển trang, **F** để toàn màn hình và **Home/End** để đến trang đầu/cuối. Slide chỉ chứa nội dung trình chiếu; đọc script ở cửa sổ riêng. Trong script, phần **Lời trình bày** là lời nói chính; các phần còn lại hướng dẫn thao tác hoặc chuẩn bị trả lời.

Slide 1-5 giới thiệu luồng thuật toán. Slide **6-8** là ba ví dụ có dữ liệu và bản đồ:

1. GPS → Z → b → năm Q: sample cố định `freshqp-001`, TRAIN, draw1, chuyến1 ở0/20/60giây. Thêm nhánh đọc GPS rồi vẫn giữ Z của chuyến6. Có chi phí ngân sách từng bước.
2. Cùng năm Q, nhu cầu local khác nhau: 420bản ghi phản hồi, bỏ trùng còn252POI. Café gần nhất trả5; trong1000m trả1; detour chọn thứ tự khác. Bản đồ zoom GPS và các POI trả về.
3. Đầu/cuối vẫn gửi ngay: sample lịch sử `u701_00`, rep0, hai cấu hình L20; nămQ ở0/384giây, không delay hoặc hủy cuối. Đây là Endpoint20 riêng, không gán cho currentL30.

Slide9 ghi phạm viS1-S10; slide10-14 trình bày các benchmark utility, đối chứngREM/Planar, GPS local thưa, trạng tháiPOI vàEndpoint20. Kết quảL30 mới và các phép thử lịch sử có nhãn riêng; không gộp chúng thành một cấu hình đã giải quyết toàn bộ scenario.

Ví dụ lấy từ GPS SUMO và tape đã lưu, không tạo lại đầu ra bảo vệ hoặc sửa benchmark. GPS/Z và các phép kiểm tra là dữ liệu minh họa local/evaluator; server chỉ thấyQ và request. Giá trị nhiễu Laplace ban đầu không được lưu nên không tự đặt một giá trị để minh họa. Đích detour là oracle local của evaluator; deployment cần đích đã biết do người dùng cung cấp. Các map giữ cùng tỷ lệ mét trên hai trục; crop/deduplicate nét đường chỉ để hiển thị, không thay graph có hướng hay miền REM.

`sample_walkthrough.json`, `utility_sample.json`, `endpoint_sample.json` giữ dữ liệu sample và nguồn/SHA. `build_sample_walkthrough.py` tái lập phần trích timeline/belief; `utility_sample_extractor.py` trích phản hồi và thứ tự POI. `sample_visuals.py` dựng các bản đồ SVG từ dữ liệu này. `slide_content.json` giữ nội dung và số liệu gốc chưa làm tròn. `benchmark_evidence.json` và `algorithm_scope.json` ghi phạm vi, định nghĩa và nguồn. `source_pins.json` lưu SHA256 toàn bộ nguồn dùng cho deck. `build_slides.py` dựng lại HTML và ghi chú kỹ thuật nguồn; kịch bản `.md` được biên soạn riêng để trình bày, còn `speaker_notes.txt` giữ ghi chú đối chiếu chi tiết; không chạy hoặc thay đổi benchmark. PDF được xuất từ HTML bằng Chrome, với sơ đồ/chữ vector.

© OpenStreetMap contributors — [ghi nguồn và giấy phép](https://www.openstreetmap.org/copyright).
