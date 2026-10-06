# Gặp GVHD ngày 03/10/2026

Nguồn: note người nghiên cứu cung cấp ngày 05/10/2026. GVHD đánh giá tiến độ hiện tại khá tốt. Bốn yêu cầu tiếp theo:

1. So sánh bằng cả bộ metrics chung đã đề xuất và các metrics gốc của phương pháp đối chứng. Giữ đúng nhiệm vụ/đầu ra/đơn vị của từng metric. Mục tiêu là kiểm tra ưu thế trên nhiều phép đo; kết luận vượt trội cần kết quả thực nghiệm, không đặt trước.
2. Mở rộng utility từ top-k theo khoảng cách sang nhiều mục đích truy vấn. Giữ mục đích, loại POI và các điều kiện riêng tư ở thiết bị; thử lấy kết quả bằng các request phủ nhiều nhu cầu để xử lý S7.
3. Ưu tiên S7, rồi S4–S6; chạy vòng phát triển, chọn cấu hình và đánh giá riêng. Theo taxonomy hiện hành: S4 liên kết danh tính người/thiết bị, S5 dự đoán bước tiếp theo, S6 dự đoán đích đến. Liên kết phương tiện được đo riêng với định danh người; không gọi cả S4–S6 là danh tính.
4. Cải thiện S9/S10: tăng chất lượng attacker và thử bảo vệ điểm đầu/cuối bằng nhiễu, không dùng delay. Giữ bản cũ làm đối chứng để thấy đánh đổi. Không lấy attacker yếu làm bằng chứng bảo vệ tốt.

## Cách triển khai vòng nghiên cứu

- Khóa dữ liệu, seed, tiêu chí chọn và metrics trước khi xem kết quả test. Giữ mọi cấu hình thử, kể cả không đạt.
- Attacker chỉ đọc payload công khai; truth, mục đích, danh tính và GPS thật tách riêng cho evaluator. Học/chọn attacker theo từng phương pháp.
- Thêm đối chứng GPS/raw hoặc request ghi rõ nội dung để kiểm tra sức tấn công. Kiểm tra nhãn chưa từng xuất hiện trong train và sự tương quan giữa mẫu cùng family.
- Không sửa các benchmark/PDF cũ. Kết quả mới đặt trong thư mục riêng, có hash nguồn và lệnh chạy lại.
- Cache mạng gốc không có trong workspace. Thử nghiệm cần chạy lại trên mạng công khai tái dựng rộng đủ cohort, ghi rõ luật nối/luật rẽ và tốc độ là giả định; không thay bằng chứng từ mạng SUMO gốc.

Các báo cáo thực nghiệm sau triển khai sẽ được liên kết từ bản cập nhật nghiên cứu ngày 05/10.
