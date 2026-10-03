# Một chuyến, từng bước của GeoI-Slack

Mở [4 slide PDF](walkthrough.pdf) để trình bày hoặc [HTML offline](walkthrough.html) để đổi thời gian, loại POI và trạng thái S9/S10. [Lời dẫn](../PREPARATION.md#ví-dụ-để-chiếu-và-phân-tích-thuật-toán) giải thích từng slide. Các PNG là ảnh 1920 × 1080 để chèn vào slide khác.

Chọn cố định chuyến đầu theo ID (`u701_00`) từ dataset SUMO có sẵn, giữ nguyên 385 điểm GPS; chạy tại 21 mốc. Seed và tham số được ghim trước khi tính kết quả. Đây là dữ liệu mô phỏng, không phải GPS thu từ người thật.

Cache mạng làn nguyên bản không còn trong workspace. Mẫu dùng hình học đường công khai đã lưu và POI OSM để tái dựng **một mạng demo riêng**: gom đầu đường trong 35 m, nối các đường chung nút, tốc độ 8 m/s, trạng thái cách 20 m. Các luật rẽ/nối đường không được chứng nhận trùng mạng benchmark. Việc nhập mạng bằng SUMO thành công chỉ kiểm tra định dạng. Không dùng kết quả mẫu để sửa benchmark hoặc xếp hạng phương pháp.

Chạy đúng `PacedSlackProgressLaneDummy`, `BoundaryProtectedStream`, mô hình ước lượng và dịch vụ POI hiện tại. Không chỉnh GPS/model gốc. Bật S9/S10 đặt mỗi khoảng là 60 giây; đối chiếu với bản tắt. Cùng seed không đồng nghĩa chỉ thay đổi độ trễ: S9 còn đổi vị trí GPS được dùng ở lần bảo vệ đầu tiên. Cận ngân sách 0,23/m là cận của kernel lý tưởng; bộ lấy mẫu số thực là xấp xỉ.

`walkthrough.json` là dữ liệu **phân tích**, có GPS thật, Z, ước lượng, phép kiểm tra và nguồn truy vấn trước khi gửi. `public_transcript.json` là dữ liệu **công khai**, chỉ có timestamp công bố, ID và tọa độ truy vấn. Không nhầm toàn bộ dữ liệu minh họa với những gì attacker quan sát.

Recall tính trên các loại có POI chuẩn, trung bình đều 21 mốc đầu vào; mốc chưa có phản hồi trả rỗng và được chấm 0. Phản hồi lấy trạng thái POI ở thời điểm **công bố**, cache trong epoch 60 giây, rồi xếp hạng bằng GPS hiện tại tại thiết bị. Bản bật có 14 lần gửi, 4 bộ hủy, tổng chi phí 0,10/m và Recall toàn thời gian 71,43%. Recall chỉ ở các mốc có gửi là 100%; mẫu nhỏ này chưa đo attacker.

## Dựng lại và kiểm tra

Python cần NumPy, SciPy, NetworkX, Shapely, pyproj, sumolib, matplotlib, PyMuPDF, QuickJS và eclipse-sumo. `netconvert` phải dùng được; cache tạm mặc định nằm ở `/private/tmp/trajectory-walkthrough`, có thể đổi `--workdir` khi dựng. Validator dùng cache tạm mặc định.

```sh
python docs/supervisor_meeting/2026-09-26_brief/walkthrough/build_walkthrough.py
python docs/supervisor_meeting/2026-09-26_brief/walkthrough/render_walkthrough.py
python docs/supervisor_meeting/2026-09-26_brief/walkthrough/validate_walkthrough.py
```

[Validation](validation.json) kiểm tra hash nguồn, GPS giữ nguyên, chi phí/ngân sách, trọng số ước lượng, slack, đường đi có hướng, Recall từ ID và hàng đợi đầu/cuối. Kiểm tra không phụ thuộc GPS đoạn đầu/mốc không đọc dùng bản sao tạm trong bộ nhớ, không thay dataset. [Kiểm tra JavaScript](explorer_validation.json) chạy 252 trạng thái với DOM giả; chưa kiểm tra hiển thị HTML trong trình duyệt vì không có trình duyệt bật trong môi trường. Cả bốn slide PDF đã được xem trực quan, kiểm tra biên trang và font nhúng.

© OpenStreetMap contributors — [ODbL và ghi nguồn](https://www.openstreetmap.org/copyright). POI giữ ID OSM; không tạo tên doanh nghiệp giả.
