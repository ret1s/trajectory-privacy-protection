# Report review: phương pháp, chứng minh và benchmark GeoI-BQ

Bản ngày 27/10/2026 viết theo [brief hướng luận văn](../../../thesis/WRITING_BRIEF.md).
Đây là report khoa học để review trước khi tích hợp vào luận văn, không thay
`thesis/main.tex` hoặc bản report privacy rút gọn 17/10.

- [PDF](report_explained.pdf): 13 trang A4, gồm 11 trang nội dung và hai trang phụ lục.
- [LaTeX](report_explained.tex): nguồn nội dung và sơ đồ vector.
- [Nguồn và phạm vi](report_sources.json): hashes nguồn, cấu hình và bảng được xuất.
- [Receipt kiểm tra](report_manifest.json): nguồn, nội dung, layout và tình trạng visual QA.

## Đọc theo thứ tự

| Trang | Nội dung |
|---|---|
| 1 | Bài toán, cơ sở kế thừa, attacker, scenario và ba RQ. |
| 2 | Kiến trúc theo lớp, input/output, client/server và cấu hình chính. |
| 3 | REM → belief → điểm → phân phối chọn đoạn; ví dụ số và TRAIN tape nhiều bước. |
| 4 | Mệnh đề 1–2: REM, noisy test, chi phí và hợp thành cap Geo-I. |
| 5 | Mệnh đề 3–4: bounded Q, nhiều đoạn, hậu xử lý, định lý cuối và Bayes. |
| 6 | Utility: điểm proxy, actual Recall/cache, cận cả phiên và conditional transfer. |
| 7 | Protocol, sáu phương án, attacker bank, metrics, CI và gate. |
| 8 | Đủ 30 ô privacy của sáu phương án × năm scenario; attacker được chọn. |
| 9 | Đủ bốn purposes, CI/gate và chênh lệch với từng control. |
| 10 | Chi phí, ba chế độ cache, độ phủ catalogue và GPS local 60 s. |
| 11 | Public design L30–L60, cùng-Q depth ablation, thảo luận và kết luận. |
| 12 | Phụ lục A: adaptation paper/metric lịch sử, N/A/failure và giới hạn so sánh. |
| 13 | Phụ lục B: pilot continual 30 phút, utility/CI/privacy/cost; chưa thay cấu hình chính. |

Tập trung review mạch 1–7 trước; trang 4–6 dành cho buổi đọc chứng minh,
trang 8–11 cho phần hiểu benchmark. Phụ lục giữ các cohort riêng biệt.

## Nguồn và tái lập

`build_report.py` chỉ xác thực 108 nguồn tham chiếu đã đóng băng, đọc kết quả
đã lưu và định dạng 20 bảng. Script tính lại ví dụ số mock và cận hữu tỷ;
không fit attacker, lấy mẫu bảo vệ, bootstrap mới hoặc rescore benchmark.
Không đổi thuật toán, dữ liệu hay số trong các artifact nguồn.

Từ repository root, dùng Python chuẩn để export; Tectonic 0.17.0 để dựng PDF;
Python có `pdfplumber` để validate. Đường runtime sau là môi trường local hiện tại:

```sh
python3 docs/supervisor_meeting/2026-10-27_brief/build_report.py
/private/tmp/trajectory-report-tools-20261027/tectonic --keep-logs docs/supervisor_meeting/2026-10-27_brief/report_explained.tex
python3 docs/supervisor_meeting/2026-10-27_brief/build_report.py --check
/Users/geohanz/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 docs/supervisor_meeting/2026-10-27_brief/validate_report.py
```

Render và xem đủ 13 trang bằng Poppler trước khi chạy validator với
`--visual-reviewed`. Không gán cờ này chỉ dựa trên text extraction.

Mệnh đề 1–4 dùng cách chia trong report 17/10; tài liệu toán đầy đủ nhóm
các kết quả khác nhau. Report giữ cận Geo-I lý tưởng và cận Q riêng,
không coi sampler số thực/full restore đã được chứng nhận. Các con số utility
là empirical hoặc conditional đúng như phần diễn giải, không là floor cho mọi GPS.
