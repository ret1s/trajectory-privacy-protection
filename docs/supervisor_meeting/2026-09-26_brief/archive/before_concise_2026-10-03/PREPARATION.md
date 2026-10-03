# Hai bản dùng cho buổi 03/10/2026

- **Bản trình bày:** [report_explained.pdf](report_explained.pdf), 31 trang. Sắp xếp lại ngày 02/10 theo thứ tự nói; số liệu, dataset, attacker và transcript giữ nguyên bản 30/09.
- **Bản chuẩn bị cá nhân:** [preparation_guide.pdf](preparation_guide.pdf), 30 trang; [nguồn LaTeX](preparation_guide.tex).
- **Bản đồ dùng khi giải thích:** [sample_maps.html](sample_maps.html).

Buổi họp nói ba trọng tâm theo thứ tự: kiến trúc giải pháp và kết quả benchmark trên năm scenario;
lập luận vì sao kết quả đứng vững và vì sao dùng bộ đo đề xuất; nếu còn thời gian, vì sao dataset
chia thành các ca A/B/C. Chi tiết đối chứng đã cập nhật và bảng metrics gốc là lớp bổ sung, mở khi GVHD hỏi.

## Thứ tự trong report chính

| Phần | Mục | Trang |
|---|---|---|
| Tổng quan: bài toán, đối thủ, kết quả rút gọn, ba điểm xin chốt | 1 | 1–2 |
| Phần I: kiến trúc, đối chứng bốn nhóm, S9/S10 trên 32 nhóm | 2–4 | 3–8 |
| Phần II: bốn lớp bằng chứng, bộ đo chung, điểm tổng hợp | 5–7 | 9–14 |
| Phần III: ba cấp dữ liệu và 29 ca A/B/C | 8–9 | 15–25 |
| Phụ lục: khảo sát, sáu đối chứng, metrics gốc, nguồn | 10–13 | 26–31 |

## Đọc bản chuẩn bị theo nhu cầu

| Trang | Nội dung |
|---|---|
| 1–2 | Mục tiêu buổi họp, hai tài liệu, phân bổ 15 phút |
| 3–10 | Trọng tâm 1: dịch vụ, kiến trúc, thuật toán phủ, lịch tải, utility, bảng năm scenario, bảng endpoint, trade-off |
| 11–18 | Trọng tâm 2: bốn lớp bằng chứng, giao thức thực nghiệm, sáu điều kiện so công bằng, metrics, điểm Q, vì sao không dùng metrics gốc, đóng góp |
| 19–24 | Trọng tâm 3: ba đơn vị dữ liệu, scenario và từng ca A/B/C |
| 25–27 | Lớp bổ sung: đối chứng đã thay đổi so với 19/09, metrics gốc, related works |
| 28–30 | Câu hỏi khó, kịch bản 15 phút, tự kiểm tra và nguồn |

Bản chuẩn bị là tài liệu tự học trước buổi họp; không phải kết luận đã được GVHD chấp nhận và không có thực nghiệm mới.
Các bảng thực nghiệm sinh từ readout đã kiểm tra, trong [preparation_evidence.tex](preparation_evidence.tex);
[preparation_evidence.json](preparation_evidence.json) ghi hash nguồn và phân biệt ví dụ minh họa với số đo.
Bản 30/09 của cả hai tài liệu được lưu trong `archive/before_2026-10-03_restructure/`.

## Dựng lại

```sh
python docs/supervisor_meeting/2026-09-26_brief/prepare_guide_evidence.py
tectonic --keep-logs --outdir docs/supervisor_meeting/2026-09-26_brief docs/supervisor_meeting/2026-09-26_brief/preparation_guide.tex
```

Chỉ sửa nội dung trong `preparation_guide.tex`; không cần chạy `build_report.py`. Số trang report trong các dòng
"Đọc cùng report chính" được chép từ bản 31 trang hiện tại; nếu report đổi phân trang, cập nhật lại các dòng này.
Nguồn/hash hai font bổ sung và lệnh biên dịch đã kiểm tra được ghi trong `preparation_validation.json`.
