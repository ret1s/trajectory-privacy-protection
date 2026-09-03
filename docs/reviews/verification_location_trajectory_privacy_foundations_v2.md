# Verification — Location & Trajectory Privacy Foundations v2

Ngày kiểm tra: 2026-09-03
Phạm vi: delta sau khi tái cấu trúc repository

Trạng thái: **PASS**

## Thay đổi

- Nguồn vẫn là `docs/study_guides/location_trajectory_privacy_foundations.tex`.
- Báo cáo Internship 2 nay ở `archive/internship_2/report.pdf`.
- Bibliography cũ dùng để đối chiếu nay ở
  `archive/internship_2/thesis/refs.bib`; `thesis/main.tex` đã chứa metadata đúng.
- Không thay đổi nội dung nền tảng, công thức, taxonomy hay claim boundary.

## Build và visual QA

- XeLaTeX: exit code `0`; không overfull box, citation/reference chưa resolve.
- `pdfinfo`: 43 trang A4, không mã hoá, không JavaScript.
- Render toàn bộ 43 trang bằng Poppler ở 110 DPI.
- 42/43 trang byte-identical với v1; chỉ physical page 39 thay đổi đúng theo hai
  đường dẫn mới. Trang này đã xem ở kích thước gốc, không clipping/overlap.
- SHA-256 bản giao:
  `8ac273b7ca4c502b6e8ca6a2d3213eb66744566f3bce05fd676510f2ed861018`.

Các kiểm tra nội dung và giới hạn nghiên cứu trong
`verification_location_trajectory_privacy_foundations_v1.md` vẫn giữ nguyên.
