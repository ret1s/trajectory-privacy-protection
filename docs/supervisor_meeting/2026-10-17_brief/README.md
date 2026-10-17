# Report GVHD ngày 17/10/2026: kiến trúc và privacy

Bản rút gọn chỉ có **hai phần, bốn trang A4**:

1. Kiến trúc hiện tại: đầu vào, từng lớp xử lý, ranh giới thiết bị/server,
   luồng thông tin đã bảo vệ và xử lý yêu cầu local.
2. Đặc tả xác suất và chứng minh privacy: REM/noisy reuse → belief → đoạn Q;
   cận Geo-I và hạch toán cap; cộng qua lịch sử ẩn để truyền cận xuống wire;
   cận riêng của Q, composition, minimum và cách đọc ngược bằng Bayes.

- [report_explained.pdf](report_explained.pdf): bản trình bày khoa học rút gọn.
- [report_explained.tex](report_explained.tex): nguồn LaTeX và hình kiến trúc vector.
- [presentation_script.md](presentation_script.md): kịch bản tương ứng bốn trang.
- [privacy_constants.tex](privacy_constants.tex): cấu hình và cận số được sinh tự động.
- [report_sources.json](report_sources.json): ghim nguồn, cận hữu tỷ và phạm vi.
- [report_manifest.json](report_manifest.json): kiểm tra PDF và hash đầu ra.

Bỏ benchmark, chứng minh utility, bảng scenario và demo dataset khỏi PDF.
Các tệp sample_walkthrough.json, report_tables.tex và report_figures/sample_paths.pdf
của bản sáu trang cũ được giữ làm tư liệu, không được input vào bản PDF này.
Không benchmark lại, lấy mẫu lại hoặc chỉnh core khi sửa report.

Bản chính GeoI-BQ v1 vẫn có cap Geo-I hiệu dụng 0,23/m; report giải thích
công thức tổng quát Cs=(2H−1)B/(2H). Bản fixed-unit continual đang là thử
nghiệm riêng: cận upstream tăng theo prefix, không nhận cap 0,23 cho mọi
thời lượng. Không nhập kết quả thử nghiệm chưa được xác minh vào report.

Từ repository root, với Python chuẩn, Tectonic và Python có pdfplumber:

```sh
python3 docs/supervisor_meeting/2026-10-17_brief/build_report.py
/private/tmp/trajectory-report-tools-20261006/tectonic --keep-logs docs/supervisor_meeting/2026-10-17_brief/report_explained.tex
python3 docs/supervisor_meeting/2026-10-17_brief/build_report.py --check
/Users/geohanz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 docs/supervisor_meeting/2026-10-17_brief/validate_report.py
```

Render bằng Poppler, xem đủ bốn trang rồi mới chạy validator với
`--visual-reviewed`. Các đường runtime là môi trường local, không là yêu
cầu đường dẫn trên máy khác. Bản six-page trước được giữ trong lịch sử Git.
