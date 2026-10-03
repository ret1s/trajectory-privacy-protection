# Hai tài liệu cho buổi trình bày

- [Report](report_explained.pdf): **11 trang**, gồm **5 trang chính**, 5 trang sample và 1 trang nguồn.
- [Preparation guide](preparation_guide.pdf): **10 trang**, gồm 5 trang chuẩn bị và cùng 5 trang sample.

| Nội dung | Report | Guide |
|---|---|---|
| Related works và coverage | 1 | 1 |
| Metrics gốc và cách đọc Hit100/MAE/Recall/byte | 2 | 2 |
| Kiến trúc, cấu hình GeoI-Paced/GeoI-Slack và đóng góp | 3 | 3 |
| Attacker theo scenario; Geo-I v2 so với adapter | 4 | 4 |
| Geo-I hiện tại: 14 ca và kết quả chính | 5 | 5 |
| Sample S1/S2/S3/S9/S10 | 6–10 | 6–10 |
| Nguồn và truy nguyên | 11 | Đọc cùng report |

**Cách đọc ví dụ S1:** Hit100 = 33,33% là tỷ lệ attacker đoán cách vị trí thật không quá 100 m; MAE = 299 m là sai số suy luận trung bình. Đây là hai chỉ số riêng, không phải phép chia. Thấp hơn tốt hơn với Hit100; cao hơn tốt hơn với MAE. Bảng 3.1 ghi Hit100 (%) / MAE (m) / Recall@5 (%); ví dụ 33,33% / 299 / 95,83%. Số thứ ba là tỷ lệ POI chuẩn tìm lại đúng. Bảng 14 ca vẫn có cột riêng.

Mạch nói: **paper nào bao phủ nhiệm vụ nào → vì sao cần bộ đo chung → mô hình Geo-I hoạt động thế nào → attacker được thấy gì và suy gì → kết quả benchmark**.

Có thể trình bày phần kết quả như sau: “Ở benchmark v2, BR trên nền Geo-I giảm Hit100 S2/S3 so với ba adapter, nhưng utility còn đánh đổi. Bản hiện tại đạt Recall 95,44%, 13/14 ca trên dữ liệu phát triển. Cache có lợi ích đo được mà không thêm query; slack cải thiện trung bình nhưng còn cần xác nhận. Vì hai phiên bản khác protocol, em không dùng các số này để tuyên bố thắng toàn bộ paper nguyên bản.”

Hai cấu hình hiện tại cùng ε/ngân sách: GeoI-Paced là mốc nội bộ; GeoI-Slack thêm slack 0,03 trong bộ chọn. Geo-I/REM bảo vệ neo; belief, mạng làn và POI chỉ dùng neo đã bảo vệ. GPS thật còn dùng để xếp hạng kết quả cuối tại thiết bị. Cache không sửa query. BR-Boundary bỏ đầu trước lõi và buffer đuôi sau lõi là mở rộng riêng: chưa có benchmark kết hợp với dịch vụ hiện tại.

Metrics không đồng nhất một phần vì đầu ra khác nhau: vị trí thay thế, quỹ đạo giả, tập dummy hoặc query chèn. Threat model, dataset và nhiệm vụ cũng khác. Bộ đo chung yêu cầu attacker suy cùng đáp án từ toàn bộ transcript được phép; chấm Recall trên cùng dịch vụ và tính đủ chi phí.

S10 chỉ có **A: một chuyến; B: nhiều chuyến cùng đích**. B là nhãn trình bày cho nguồn cũ S10.C; không có C độc lập. Mỗi sample có mã record, cửa sổ FCD, bản đồ, timeline và giải thích attacker có thể khai thác dấu hiệu gì. A/B/C mô tả điều kiện, không phải thứ tự độ khó. [Bản đồ tương tác](sample_maps.html) vẫn giữ đủ 29 ca của khung rộng hơn.

Bảng coverage, định lượng và benchmark được dùng chung để hai bản không lệch số. Sửa lời dẫn guide trong `preparation_guide.tex`; sửa nội dung chung trong `geoi_content.py`; dựng report trước guide. Lệnh và provenance ở [README](README.md).
