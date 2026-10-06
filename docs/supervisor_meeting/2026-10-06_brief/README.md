# Buổi GVHD tiếp theo — chuẩn bị 06/10/2026

Chưa xác định ngày gặp. Đây là report mới; tài liệu 26/09, walkthrough và thesis
giữ nguyên. Không dùng nhãn sample hoặc phụ lục trong bài nói.

- [Report PDF](report_explained.pdf) · [HTML](report_explained.html) · [LaTeX](report_explained.tex)
- [Script trình bày PDF](preparation_guide.pdf) · [HTML](preparation_guide.html)
- [Evidence và nguồn paper](evidence.json) · [Kiểm tra bản xuất](review.json)
- [Sơ đồ vector](figures/architecture.svg)

Report có 7 phần: related works/metrics; kiến trúc; cấu hình/attacker;
đối chứng cùng protocol; S7/utility; S4–S6; S9/S10 và hướng xác nhận.
Script khoảng 10–12 phút, gồm câu trả lời ngắn khi GVHD hỏi.

Các bảng dùng readout riêng theo cohort/protocol, không tạo thứ hạng từ những
phép thử không tương thích. Vòng phát triển mới không là confirmation chưa
từng xem. Chi phí phản hồi chứa metadata POI không được so trực tiếp với bảng
cũ chỉ chứa POI ID. Metrics gốc thiếu đầu vào giữ N/A; EIE cùng point estimate
trùng MAE, không là bằng chứng độc lập.

## Dựng lại

Từ repo root, dùng Python environment có matplotlib và PyMuPDF:

```bash
python docs/supervisor_meeting/2026-10-06_brief/prepare_report.py
python docs/supervisor_meeting/2026-10-06_brief/plot_architecture.py
```

`prepare_report.py` đặt trạng thái `updating` để tránh tuyên bố đã kiểm tra khi
chưa review. Sau khi kiểm tra nội dung, đặt `content.json:build_status` thành
`complete`, chạy `build_report.py`. Biên dịch hai `.tex` bằng Tectonic/XeLaTeX
từ thư mục này vào một thư mục tạm; chỉ copy PDF đã review vào đây. Sau cùng:

```bash
python docs/supervisor_meeting/2026-10-06_brief/verify_report.py
```

HTML dùng cùng nội dung với PDF và SVG inline, không cần tải font/script ngoài.
PDF/kiến trúc đã render để kiểm tra. Phiên này không có browser CUA được bật,
nên kiểm tra HTML bằng cấu trúc/liên kết, không tuyên bố đã xem screenshot browser.

## Sửa cách giải thích H sau audit

H12 là tham số quy đổi ngân sách với U=23 đơn vị; filter hiện tại cho 12–22
lần đọc GPS tùy nhánh, không là giới hạn cứng 12 lần. Sửa prose ngày 06/10;
không sửa các điểm benchmark. [Audit](../../research/2026-10-06_h_accounting_correction.md)
và [bản lưu trước khi sửa](revisions/r1_before_formal_audit/README.md).
