# Hai bản dùng cho buổi 03/10/2026

- **Bản trình bày:** [report_explained.pdf](report_explained.pdf), 27 trang. Giữ nguyên bản cập nhật 30/09 với S10.A/B.
- **Bản chuẩn bị cá nhân:** [preparation_guide.pdf](preparation_guide.pdf), 27 trang; [nguồn LaTeX](preparation_guide.tex).
- **Bản đồ dùng khi giải thích:** [sample_maps.html](sample_maps.html).

Bản chuẩn bị đi theo nội dung report, thêm ví dụ, lời trình bày gợi ý, câu chuyển ý,
câu hỏi khó và cách trả lời có căn cứ. Đây là tài liệu tự học trước buổi họp;
không phải kết luận đã được GVHD chấp nhận và không có thực nghiệm mới.

## Đọc theo nhu cầu

| Trang bản chuẩn bị | Nội dung |
|---|---|
| 1–2 | Câu chuyện nghiên cứu và thứ tự trình bày |
| 3–9 | Dịch vụ, ba cấp dữ liệu, scenario và sample |
| 10–15 | Related works, sáu đối chứng, metrics và điểm Q |
| 16–20 | Phương pháp, thuật toán phủ, lịch tải, bảo đảm và giao thức thực nghiệm |
| 21–24 | Bảng kết quả, khoảng tin cậy, trade-off và đóng góp |
| 25 | Câu hỏi khó và câu trả lời ngắn |
| 26–27 | Kịch bản trình bày 15 phút, câu tự kiểm tra và nguồn |

Các bảng thực nghiệm sinh từ readout đã kiểm tra, trong
[preparation_evidence.tex](preparation_evidence.tex).
[preparation_evidence.json](preparation_evidence.json) ghi hash nguồn và phân biệt
các ví dụ minh họa với số đo. Nhãn trình bày theo cùng `report_case_labels.py` của
report chính; dữ liệu thực nghiệm gốc không thay đổi.

## Dựng lại

```sh
python docs/supervisor_meeting/2026-09-26_brief/prepare_guide_evidence.py
tectonic --keep-logs --outdir docs/supervisor_meeting/2026-09-26_brief docs/supervisor_meeting/2026-09-26_brief/preparation_guide.tex
```

Chỉ sửa nội dung trong `preparation_guide.tex`; không cần chạy `build_report.py`.
Nguồn/hash hai font bổ sung và lệnh biên dịch đã kiểm tra được ghi trong
`preparation_validation.json`. Bản report chính được kiểm tra giữ nguyên SHA-256.
