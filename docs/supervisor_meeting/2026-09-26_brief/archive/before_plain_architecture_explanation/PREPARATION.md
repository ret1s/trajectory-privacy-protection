# Hai tài liệu cho buổi trình bày

- [Report](report_explained.pdf): **6 trang**, gồm **5 trang chính** và 1 trang nguồn.
- [Preparation guide](preparation_guide.pdf): **5 trang**. Tạm bỏ phụ lục mẫu; trình bày ở mức scenario.

| Nội dung | Report | Guide |
|---|---|---|
| Related works: bảng chung scenario–metrics; lý do dùng bộ đo chung | 1 | 1 |
| Sơ đồ luồng kiến trúc theo số thứ tự và layer | 2 | 2 |
| Vai trò component, cấu hình GeoI-Paced/GeoI-Slack và đóng góp | 3 | 3 |
| Attacker theo scenario; Geo-I v2 so với adapter | 4 | 4 |
| Geo-I hiện tại: kết quả gộp theo scenario | 5 | 5 |
| Nguồn và truy nguyên | 6 | Đọc cùng report |

**Cách đọc ví dụ S1:** Hit100 = 33,33% là tỷ lệ attacker đoán cách vị trí thật không quá 100 m; MAE = 299 m là sai số suy luận trung bình. Thấp hơn tốt hơn với Hit100; cao hơn tốt hơn với MAE. Bảng 3.1 ghi Hit100 (%) / MAE (m) / Recall@5 (%); ví dụ 33,33% / 299 / 95,83%. Dấu / tách ba chỉ số, không phải phép chia. Bảng bản hiện tại gộp theo scenario và giữ cột riêng.

Mạch nói: **paper nào bao phủ nhiệm vụ nào → vì sao cần bộ đo chung → mô hình Geo-I hoạt động thế nào → attacker được thấy gì và suy gì → kết quả benchmark**.

**“v2” là phiên bản benchmark, không phải tên attacker.** Cùng tập loại attacker đánh giá các phương pháp; **Shadow kNN** được học riêng từ output từng phương pháp. Chọn bộ suy luận trên tập chọn, giữ cố định khi test. Tên các bộ suy luận đã in đậm ở trang 4–5; phép thử bản hiện tại dùng thêm **kNN**, **ExtraTrees** và các bộ hình học/đường.

Có thể trình bày phần kết quả như sau: “Khung nghiên cứu có S1–S10; hiện đo S1, S2, S3, S9, S10. Ở benchmark v2, BR trên nền Geo-I giảm Hit100 S2/S3 so với ba adapter, nhưng utility còn đánh đổi. Bản hiện tại đạt Recall 95,44%, cả năm scenario đạt 90% ở mức trung bình. S1 vẫn có mức thấp nhất 80,18%. Cache có lợi ích đo được; slack còn cần xác nhận. Hai phiên bản khác protocol nên báo riêng.”

Hai cấu hình hiện tại cùng ε/ngân sách: GeoI-Paced là mốc nội bộ; GeoI-Slack thêm slack 0,03 trong bộ chọn. Geo-I/REM bảo vệ neo; belief, mạng làn và POI chỉ dùng neo đã bảo vệ. GPS thật còn dùng để xếp hạng kết quả cuối tại thiết bị. Cache không sửa query. BR-Boundary bỏ đầu trước lõi và buffer đuôi sau lõi là mở rộng riêng: chưa có benchmark kết hợp với dịch vụ hiện tại.

**Cách nói theo sơ đồ:** bước 1 kiểm tra lịch/ngân sách; bước 2 giữ hoặc tạo neo; 3a ước lượng vùng vị trí và 3b tạo miền đường khả thi; bước 4 chọn K query; server trả POI; bước 5 hợp phản hồi và xếp hạng cục bộ. Không đọc GPS mới thì bỏ bước 2. Boundary S9 đứng trước lõi, S10 đứng trước công bố; “BR-Boundary v1” là tên wrapper.

Phần 1 gộp coverage và metrics vào một bảng. Bốn lý do cần bộ đo chung: khác đầu ra; khác nhiệm vụ/quyền quan sát; utility khác tác vụ; chi phí khác phạm vi. Ta chấm cùng đáp án từ toàn bộ output attacker được phép thấy bằng Hit100/MAE, cùng dịch vụ POI bằng Recall@5 và request + response bằng byte/sự kiện.

Mỗi dòng của benchmark hiện tại là trung bình đều các điều kiện trong scenario. Recall tổng trung bình đều năm scenario. Không coi đạt ngưỡng trung bình là mọi điều kiện đều đạt; số đo chi tiết vẫn giữ trong nguồn để truy nguyên.

Bảng coverage, định lượng và benchmark được dùng chung để hai bản không lệch số. Sửa lời dẫn guide trong `preparation_guide.tex`; sửa nội dung chung trong `geoi_content.py`; dựng report trước guide. Lệnh và provenance ở [README](README.md).
