# Kiểm tra phần độ nhạy GPS cục bộ của luận văn

Ngày kiểm tra: 07/10/2026. **PASS trong phạm vi nội dung, số liệu và khả năng
đọc của mục GPS cục bộ.** Bản được xem là
[`build/thesis/reviewed_20261007.pdf`](../../build/thesis/reviewed_20261007.pdf),
74 trang, SHA256
`25b330967aea24a4a931bba910c97383366a036dbdf8e357d6a579c6b3eb955c`.
Đã xem trực tiếp PNG của toàn bộ trang PDF 54–58; mục 6.4 nằm ở PDF 54, 56,
57 (số in 48, 50, 51). Hình 6.1 của phần fresh nằm ở PDF 55. Đây không phải
kiểm tra toàn bộ luận văn hoặc chứng chỉ về GPS thiết bị thật.

Nguồn phần mới là [`current_sensor.tex`](../../thesis/current_sensor.tex), SHA256
`c209ae379849379e893f5d17358bbb09aeab84c4eb052b3dcee6d86b5c6534ec`.
Fragment số liệu
[`sensor_rows.tex`](../../thesis/current_extensions_generated/sensor_rows.tex)
có SHA256 `64aa31320e596ad32a76e966f21912b4efcee7b798eba5d0c9a36f05b60aae3b`.
Cả chín macro dùng trong module đều có định nghĩa. Bảng 6.5, 6.6 và 6.7 có
đủ bảy cấu hình, số liệu khớp
[readout](../../artifacts/benchmarks/local_gps_robustness_20261007_v1/readout.json)
và [certificate PASS](../../artifacts/benchmarks/local_gps_robustness_20261007_v1/validation.json).
Hash protocol/readout hiện tại khớp certificate. Không chấm thêm hay thay đổi
source/artifact đã chốt trong quá trình kiểm tra.

Các bảng, dấu tiếng Việt, công thức Gaussian, đơn vị m/s và footnote đường dẫn
đọc được, không bị cắt hoặc đè. Bảng 6.7 ngắt nhãn dài thành hai dòng, vẫn nằm
trong vùng trang. Có một điểm trình bày nhẹ đã báo người tích hợp: hình fresh
ở PDF 55 chen giữa đoạn mở mục 6.4 và phần thân. Có thể chặn float trước mục
mới để liền mạch; đây không phải sai số liệu hoặc lỗi khoa học.

Nội dung giữ rõ hai đồng hồ GPS, nguồn riêng tư đã đóng băng từ GPS tổng hợp
chính xác, 11 fix ảo cho riêng xếp hạng cục bộ và giả định đích đã biết tại
thiết bị. N/A chỉ dành cho reference thật rỗng; đáp án thiếu khi reference có
đáp án giữ Recall bằng 0. Các đếm ngoài miền bán kính thật là POI lặp theo
event/draw, không phải tỷ lệ người dùng. Kết quả âm của ngoại suy được nêu rõ:
sai số vị trí trung bình giảm nhưng Recall4 và Recall3 thấp hơn giữ fix ở cả
ba mức nhiễu và hai mức L. L30 vẫn tăng Recall 2,24–2,61 điểm phần trăm trên
các cấu hình thưa/nhiễu; không chuyển kết quả này thành cải tiến Geo-I, xác
nhận mới, mô hình GPS thực, tiết kiệm pin hoặc đo bộ nhớ client.

[README của chẩn đoán](../../artifacts/benchmarks/local_gps_robustness_20261007_v1/README.md)
đã kiểm tra đủ 11 liên kết tương đối; không có đích thiếu. `git diff --check`
đạt trên trạng thái được kiểm tra.

## Xác nhận bản dựng cuối

Đã xem lại trực tiếp PDF 54–57 của
[`reviewed_20261007_final.pdf`](../../build/thesis/reviewed_20261007_final.pdf),
74 trang, SHA256
`8b36dae14084ce0a762488f762e2a2417bfd7bbb2b24b1ea0e03271311ff54b7`.
Hình 6.1 nằm trước phần mở mục 6.4 ở PDF 55; không còn chen giữa phần mở và
thân mục. Nội dung GPS cục bộ tiếp tục ở PDF 56–57. Bảng 6.5–6.7 vẫn đủ hàng,
đọc được và giữ nguyên số liệu; dấu tiếng Việt, mốc thời gian, nhãn đơn vị và
footnote không tràn. Đoạn mở và đoạn về đích có ngắt sang trang kế tiếp nhưng
không mất nội dung. **PASS bố cục và nội dung của phần GPS cục bộ trên bản
dựng cuối này.** Bản 74 trang đầu ở trên được giữ làm lịch sử của lần review
trước sửa vị trí float; các source/scorer/readout đã đóng băng không đổi.
