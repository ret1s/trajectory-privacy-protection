# Brief trao đổi với GVHD — 11/09/2026

Đọc bản PDF: `artifacts/reports/supervisor_brief_2026-09-11.pdf`.
Bản HTML tự chứa: `artifacts/reports/supervisor_brief_2026-09-11.html`.
Đây là bản trình bày ngắn đi kèm, không thay thế graduation thesis.

## Cách trình bày trong khoảng 10–12 phút

- **Trang 1 (30 giây):** “Em trình bày phần làm rõ yêu cầu buổi trước trước,
  rồi mới báo kết quả thuật toán. Dữ liệu thật nằm trên thiết bị; bên dịch vụ chỉ
  nhận dữ liệu sau bảo vệ. Phạm vi hiện tại là xe con trong đô thị.”
- **Trang 2–3 (2 phút):** “Mỗi kịch bản xác định một điều đối thủ muốn biết,
  điều kiện dữ liệu và phương pháp bảo vệ tương ứng. DLS chủ yếu làm khó chọn
  điểm thật ở S1; RDG, TransProtect, semantic correlation và fake queries cùng
  hướng chống nối đường S3, nhưng dùng những cơ chế khác nhau. Mix zones liên
  quan nối phiên S4; che biên liên quan điểm đầu/cuối S9–S10. Đây là ánh xạ mục
  tiêu, không phải bảng kết quả các paper đã vượt qua ca SUMO của em.” Chọn hai
  hàng để giải thích, không đọc hết bảng. “S8 chưa có cơ chế dummy trực tiếp
  được xác minh trong khảo sát; có dữ liệu không đồng nghĩa đã bảo vệ được.”
- **Trang 4 (1 phút):** Theo bốn ô của sơ đồ: thiết kế tuyến → SUMO → gắn nhãn
  kịch bản → kiểm tra/lưu. “Có 12 nhóm tuyến mới, 264 chuyến và 393 bản ghi; một
  chuyến dùng được cho nhiều phép thử. SUMO tạo chuyển động, lớp kịch bản mới
  tạo bài toán riêng tư. Một số ca vẫn ít mẫu.”
- **Trang 5 (1 phút):** “Chọn đối chứng theo cơ chế, không chỉ theo năm:
  ngữ cảnh đường, ngữ nghĩa, chèn truy vấn; thêm DLS và RDG không học sâu.
  Mã S1/S3 ở từng hàng cho biết mục tiêu chính. AnotherMe giữ tham chiếu quỹ
  đạo ảo, chưa gán thành lời giải cho S4–S6. Cột cuối nói rõ cái nào mới được chọn,
  cái nào đã thích nghi; chưa phải tất cả đã chạy xong.”
- **Trang 6 (1 phút):** “Không gộp mọi chỉ số thành một điểm. Đối thủ đoán sai
  bao xa, có còn đoán rất gần không; ứng dụng lấy lại được bao nhiêu POI đúng;
  phải tốn thêm bao nhiêu. Giữ chỉ số gốc để đối chiếu paper, thêm cùng nhiệm vụ
  và đối thủ để so sánh công bằng. K, k và L là ba số khác nhau.”
- **Trang 7 (1 phút):** Đi theo sơ đồ neo nhiễu → ước lượng vùng có thể đang ở
  → chọn tập dummy có độ phủ POI → gộp kết quả trên thiết bị. “Biến thể mới phân
  biệt đang dừng/đang đi. Lõi đang đo S1-S3; các mục tiêu khác cần phép thử và
  module riêng, chưa gọi là đã hoàn tất.”
- **Trang 8 (1 phút):** “Dữ liệu học, tập chọn và xác nhận tách vai trò. Khóa
  đối thủ, cấu hình và ngưỡng trước khi xem xác nhận. Bốn cột kết quả là bốn
  biến thể nội bộ; không phải bốn mô hình SOTA.”
- **Trang 9–10 (2 phút):** “Tối ưu độ phủ giúp chất lượng POI tốt hơn hình học.
  Hai chế độ giảm Hit100 từ 3,73% xuống 2,92%, nhưng MAE cũng giảm nên vẫn là
  đánh đổi. Tăng L tốn thêm khoảng 64,7% byte mã POI. Theo nhóm tuyến có cả tốt
  lẫn xấu. Chưa cấu hình nào qua ngưỡng 90% ở ca yếu nhất trên tập chọn.”

**Câu chốt:** “Kết quả lần này là khung đánh giá cụ thể hơn và một giả thuyết
thuật toán đã được thử có kiểm soát. Em chưa kết luận thắng các đối chứng hay
bảo vệ đủ mười kịch bản. Bước tiếp theo là tái lập đối chứng, tăng mẫu khó và
giải quyết ca truy vấn thưa trước khi mở rộng các mục tiêu còn lại.”

## Tái tạo bản xuất

Yêu cầu: môi trường Python hiện có của repo, Playwright, Chrome và Poppler;
plugin Data Analytics 1.0.2. Không cần sinh lại SUMO hoặc huấn luyện mô hình.
Chỉ tạo dự án runtime tạm, không chép hạ tầng plugin vào codebase.

```sh
venv/bin/python docs/supervisor_meeting/2026-09-11_brief/build_report.py --complete
brief_build_dir=$(mktemp -d /private/tmp/msc-supervisor-brief.XXXXXX)
brief_plugin_dir=/Users/geohanz/.codex/plugins/cache/openai-curated-remote/data-analytics/1.0.2
brief_content_dir=/Users/geohanz/Project/msc/docs/supervisor_meeting/2026-09-11_brief
node "$brief_plugin_dir/scripts/prepare-data-app.mjs" --surface report --output "$brief_build_dir/app" --snapshot "$brief_content_dir/data.json"
cp "$brief_content_dir/data.json" "$brief_build_dir/app/src/data.json"
cp "$brief_content_dir/ReportContent.jsx" "$brief_content_dir/report.css" "$brief_content_dir/artifact.json" "$brief_content_dir/presentation_pages.json" "$brief_build_dir/app/src/content/report/"
node "$brief_plugin_dir/scripts/data-app.mjs" build --project-dir "$brief_build_dir/app" --separate-data
node "$brief_plugin_dir/scripts/data-app.mjs" export-offline --project-dir "$brief_build_dir/app" --output "$brief_build_dir/app/.data-app-offline/exports/supervisor_brief.html"
cp "$brief_build_dir/app/.data-app-offline/exports/supervisor_brief.html" artifacts/reports/supervisor_brief_2026-09-11.html
venv/bin/python -m http.server 4183 --bind 127.0.0.1 --directory "$brief_build_dir/app/dist"
```

Nếu 4183 đang được dùng, chọn cổng khác và truyền cùng URL cho lệnh dưới.
Chạy trong terminal thứ hai sau khi server sẵn sàng:

```sh
venv/bin/python docs/supervisor_meeting/2026-09-11_brief/render_pdf.py --url http://127.0.0.1:4183/ --output artifacts/reports/supervisor_brief_2026-09-11.pdf
venv/bin/python docs/supervisor_meeting/2026-09-11_brief/verify_brief.py
```

Sau khi đổi nội dung/bố cục, vẫn phải render và xem lại **tất cả các trang PDF**.
`source_notes.md` ghi giới hạn truy cập paper và phạm vi kiểm tra thực tế;
`evidence.json` lưu hash nguồn. Không tự dùng report này để thay kết quả thực nghiệm.
