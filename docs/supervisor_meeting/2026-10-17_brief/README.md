# Report GVHD ngày 17/10/2026

- [report_explained.pdf](report_explained.pdf): tài liệu khoa học6 trang A4 để đọc/trình bày.
- [report_explained.tex](report_explained.tex): nguồn LaTeX chỉnh sửa được, serif10pt và hình vector.
- [presentation_script.md](presentation_script.md): kịch bản theo từng trang, khoảng10–12 phút.
- [report_sources.json](report_sources.json): hash nguồn và các số đã trích, không benchmark lại.
- [sample_walkthrough.json](sample_walkthrough.json): log TRAIN cố định của đúng bản GeoI-BQ v1.

Nội dung: phương pháp và cơ chế bổ sung; privacy cả phiên; posterior/utility
và audit; ví dụ nhiều bước; benchmark so với cùng-depth controls và bare
Geo-I Z; diễn giải đóng góp/trade-off.

Ngày gặp là thứ Bảy17/10; bằng chứng được chốt14/10. Bản report09/10 được
giữ nguyên để đối chiếu cấu trúc; không gộp số khác cấu hình/cohort thành
phép so sánh matched. Không đổi core, dữ liệu, sampler hoặc attacker khi
viết report.

Từ repository root:

```sh
/private/tmp/trajectory-query-segment-venv/bin/python docs/supervisor_meeting/2026-10-17_brief/build_report.py
/private/tmp/trajectory-report-tools-20261006/tectonic --keep-logs docs/supervisor_meeting/2026-10-17_brief/report_explained.tex
/private/tmp/trajectory-query-segment-venv/bin/python docs/supervisor_meeting/2026-10-17_brief/build_report.py --check
/private/tmp/trajectory-slides-20261010-venv/bin/python docs/supervisor_meeting/2026-10-17_brief/validate_report.py
```

Các đường runtime trong `/private/tmp` là environment hiện tại; trên máy
khác dùng Python với dependencies của repo và Tectonic/XeLaTeX.
Kiểm tra source pins và render đủ các trang trước khi phát hành PDF.
`report_manifest.json` ghi hash tài liệu cùng kiểm tra bố cục và nội dung.
Sau khi đã xem đủ sáu trang render, chạy validator với `--visual-reviewed`
để ghi nhận kiểm tra trực quan; không dùng flag này để thay việc xem PDF.

© OpenStreetMap contributors; [nguồn và giấy phép](https://www.openstreetmap.org/copyright).
