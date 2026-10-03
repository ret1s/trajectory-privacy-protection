# Hai bản cho buổi 03/10/2026

- [Report](report_explained.pdf): **11 trang**, trong đó 5 trang chính, 5 trang sample, 1 trang nguồn. Có [HTML](report_explained.html) và [LaTeX](report_explained.tex).
- [Preparation guide](preparation_guide.pdf): **8 trang**, trong đó 3 trang lời dẫn/cách giải thích và cùng 5 trang sample. Sửa lời nói trong [preparation_guide.tex](preparation_guide.tex).

Sơ đồ chia rõ **input → ba layer lõi → output**, thêm **layer 4 boundary** trước/sau lõi; khung xanh bao vùng mô hình của ta, input và bên nhận nằm ngoài khung.

Kiến trúc đã sửa thành mô hình **Geo-I/REM → belief → road-aware → chọn dummy theo POI**, cùng ngân sách/tái dùng neo và lớp BR-Boundary. Bảng 30/67 là kết quả nhánh truy vấn công khai, không phải kết quả của lõi Geo-I hay Geo-I + boundary.

Cả hai theo đúng thứ tự: **related works ba năm và bảng metrics → lý do chọn bộ đo → kiến trúc gắn với S1/S2/S3/S9/S10 → kết quả đối chứng**.

| Nội dung | Report | Guide |
|---|---|---|
| Papers gần đây, metrics gốc và lý do chọn bộ đo | 1–2 | 1 |
| Kiến trúc BR-Dummy: Geo-I, road/POI-aware, boundary S9/S10 | 3 | 2 |
| Benchmark S1–S3 trên 4 nhóm và endpoint trên 32 nhóm | 4–5 | 3 |
| 14 sample: S1, S2, S3, S9, S10 | 6–10 | 4–8 |
| Nguồn và truy nguyên | 11 | Đọc cùng report |

S10 chỉ có **A: một chuyến; B: nhiều chuyến cùng đích**. B là nhãn trình bày cho mã nguồn cũ S10.C; không có C độc lập trong phạm vi hiện tại. A/B/C mô tả điều kiện thử, không phải thứ tự độ khó.

Phụ lục dùng chung từ `scenario_appendix.tex`, được sinh bởi `build_report.py`; không sửa trực tiếp tệp này. Mỗi sample có record ID, chuyến, số mẫu/cửa sổ FCD, map, timeline và giải thích dấu hiệu attacker có thể khai thác. [Bản đồ tương tác](sample_maps.html) vẫn giữ đủ 29 ca của khung rộng hơn.

## Dựng lại

```sh
python docs/supervisor_meeting/2026-09-26_brief/plot_model_architecture.py
python docs/supervisor_meeting/2026-09-26_brief/build_report.py
tectonic --keep-logs --outdir docs/supervisor_meeting/2026-09-26_brief docs/supervisor_meeting/2026-09-26_brief/report_explained.tex
python docs/supervisor_meeting/2026-09-26_brief/prepare_guide_evidence.py
tectonic --keep-logs --outdir docs/supervisor_meeting/2026-09-26_brief docs/supervisor_meeting/2026-09-26_brief/preparation_guide.tex
```

Sửa nội dung report trong `concise_presentation_content.py`; các module bằng chứng gốc vẫn kiểm tra hash trước khi dựng. Chạy report trước guide để cập nhật phụ lục và provenance. Nếu môi trường thiếu font cmsy5/cmsy6, dùng đường dẫn font và lệnh đã ghi trong `report_validation.json`.

Không chạy thêm model hoặc thay số đo trong lần viết lại này. Bản trước khi rút gọn lưu tại `archive/before_concise_2026-10-03/`.
