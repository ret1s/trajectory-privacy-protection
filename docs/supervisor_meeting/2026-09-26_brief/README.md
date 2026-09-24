# Báo cáo ngày 26/09/2026

Bản hiện tại **28 trang**, cập nhật thực nghiệm đến 24/09. LaTeX và HTML dùng
chung nội dung; PDF đã biên dịch và kiểm tra hình thức.

- [PDF](report_explained.pdf), [LaTeX](report_explained.tex), [HTML](report_explained.html).
- Mục 1–2, trang 1–11: ba cấp dữ liệu và đủ 30 mẫu A/B/C, map đặt cạnh đặc tả.
- Mục 3–7, trang 12–19: related works, sáu đối chứng, metrics và ví dụ điểm Q.
- Mục 8, trang 20–22: kiến trúc hiện tại, neo/ngân sách, mục tiêu phủ và cache theo epoch.
- Mục 9, trang 23–25: kết quả POI khả dụng, đủ 15 ca, privacy nguồn, ablation và độ nhạy.
- Mục 10, trang 26: đóng góp có bằng chứng và điều kiện sử dụng.
- Mục 11, trang 27–28: provenance, phạm vi tái lập và tài liệu tham khảo.

## Cấu hình và cách đọc kết quả

Cấu hình làm việc là `response_paced_slack03` với cache phản hồi trong epoch,
K5/L10 và cận tọa độ phiên 0,23 m⁻¹. Recall trung bình theo ca ở p=0,8 là 95,60%,
fixed K5 là 82,07%; riêng cache tăng 0,24 điểm phần trăm với cùng transcript và
chi phí truyền. Bảng ghi cả S1.C chưa đạt, đối chứng bulk và rủi ro S9/S10.

Bộ sample minh họa là `urban_fresh_v2` (393 record). Thực nghiệm mới dùng
`research_loop_expanded_v1` (415 record, lấy 173 record từ 102 chuyến cho năm
scenario). Không ghép số hoặc mẫu số giữa hai bộ. Đối chứng cố định/cache/bulk
trong bảng mới không phải sáu phương pháp từ paper. Privacy được lấy từ đúng
các tọa độ đã bảo vệ; đây chưa phải đo lại một đối thủ đầy đủ sử dụng epoch,
metadata và phản hồi của dịch vụ mới.

Chi tiết thí nghiệm: [kết quả dịch vụ](../../research/live_service_results.md),
[checkpoint](../../research/algorithm_loop_status.md),
[định vị contribution](../../research/contribution_positioning_20260924.md).

## Nguồn nội dung và dữ liệu kèm theo

- `build_report.py`: cấu trúc chung, khảo sát, metrics và bộ dựng LaTeX/HTML.
- `live_method_content.py`: nội dung phương pháp/kết quả hiện tại; đọc trực tiếp
  artifact và kiểm tra hash trước khi đưa số vào bảng.
- `dataset_content.py`, `scenario_guide.json`, `scenario_explanations.json`,
  `case_readings.json`: đặc tả và diễn giải các sample.
- `plot_live_architecture.py`: sơ đồ hiện tại trong `figures/architecture_live.*`.
- `data_samples.json`, `printed_samples.json`, `scenario_inventory.csv`: 30
  mẫu, tọa độ trích, mẫu số và nhãn phía đánh giá.
- `sample_maps.html` / `sample_maps.pdf`: atlas riêng; giữ đủ 30 map A/B/C.
- `method_evidence.json`: hash nguồn phương pháp, kết quả và sơ đồ hiện tại.
- `report_validation.json`: kiểm tra số liệu, liên kết, PDF và phạm vi QA.
- `sources.json`: 18 nguồn, URL và mức xác minh.
- `score_example.json`: chỉ là ví dụ Q, không phải kết quả model.
- `archive/`: nội dung/phạm vi bằng chứng trước khi cập nhật dịch vụ khả dụng.
  Các artifact paper-v2, fresh-switching và boundary audit vẫn giữ ở nguồn gốc;
  không trình bày như kết quả của cấu hình hiện tại.

## Sinh lại

Từ gốc repository, với Python có matplotlib và Tectonic:

```sh
python docs/supervisor_meeting/2026-09-26_brief/plot_live_architecture.py
python docs/supervisor_meeting/2026-09-26_brief/build_report.py
tectonic --untrusted --keep-logs --outdir docs/supervisor_meeting/2026-09-26_brief docs/supervisor_meeting/2026-09-26_brief/report_explained.tex
```

Không chỉnh trực tiếp `.tex`/`.html`: lần build sau sẽ ghi lại từ nguồn chung.
Không cần chạy lại SUMO hay thí nghiệm để dựng báo cáo. Nếu thay hình sample,
dùng `plot_case_panels.py` / `plot_sample_maps.py` rồi kiểm tra nguồn bản đồ.

Lần xuất này dùng Tectonic với cache có sẵn và hai font Computer Modern `cmsy5`
/ `cmsy6` lấy từ CTAN do máy chủ bundle bị timeout; font đã được nhúng trong PDF.
Phụ thuộc và lệnh build thực tế được ghi trong `report_validation.json`.

## Kiểm tra của bản cập nhật

- Đối chiếu bảng utility, đủ 15 ca, privacy và bootstrap với artifact nguồn.
- Giữ 30 panel A/B/C, các tham chiếu và liên kết tệp/hình hợp lệ.
- Xác nhận LaTeX/HTML cùng nội dung và các hash trong `method_evidence.json`.
- Biên dịch PDF, kiểm tra không có tràn hộp hoặc thiếu glyph; render và rà
  trang mở đầu, sơ đồ, bảng kết quả và lập luận. Giữ mẫu số và điều kiện sử dụng
  cạnh kết quả, không nối thêm các chương lịch sử lặp lại.
